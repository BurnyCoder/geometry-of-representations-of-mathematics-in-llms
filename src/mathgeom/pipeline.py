"""Named pipeline phases called by run.py: concepts → arxiv → extract → map → shapes → lattice.

Global context: each phase reads the previous phase's files and writes its own, so phases can
be re-run independently. Small outputs (CSV/JSON) live in data/ and results/<config>/;
activations (tens–hundreds of MB) live in artifacts/acts/<model_key>/ (junction to D:\\AI).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from mathgeom.config import Config, load_yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- data phases
def phase_concepts(cfg: Config) -> None:
    from mathgeom.data.concepts import build_concepts

    build_concepts(cfg)


def phase_arxiv(cfg: Config) -> None:
    """Field × field Ochiai distance from arXiv cross-listing; saved with the dataset revision."""
    from mathgeom.data.arxiv_fields import field_distance_matrix, load_categories

    cats, rev = load_categories(cfg.artifacts / "arxiv_categories.json")
    fields = {f["id"]: f["arxiv"] for f in load_yaml("fields.yaml")["math"]}
    D, stats = field_distance_matrix(cats, fields)
    out = {"revision": rev, "n_papers": len(cats), "names": stats["names"], "distance": D.round(6).tolist(), "papers_per_field": stats["papers_per_field"]}
    (cfg.data_dir / "field_distance.json").write_text(json.dumps(out, indent=1))
    log.info("field distances over %d papers (rev %s): %s", len(cats), rev[:8], dict(zip(stats["names"], stats["papers_per_field"])))


# ---------------------------------------------------------------- extraction
def family_items(fams: dict) -> list[tuple[str, str, str]]:
    """Flatten families.yaml into (family_name, concept, domain) triples (deduplicated per family)."""
    out = []
    for name, f in fams["chains"].items():
        out += [(name, c, f["domain"]) for c in f["items"]]
    for name, f in fams["references"].items():
        out += [(name, c, f["domain"]) for c in f["items"]]
    for name, f in fams["lattices"].items():
        out += [(name, c, f["domain"]) for c in list(f["items"]) + list(f.get("flagged") or {})]
    go = fams["group_objects"]
    objs = set(go["objects"]) | {p for v in go["objects"].values() for p in v} | {x for q in go["parallelograms"] for x in q}
    out += [("group_objects", c, go["domain"]) for c in sorted(objs)]
    tr = fams["transfer"]
    out += [("transfer", c, tr["domain"]) for c in sorted({x for k in ("identity", "commutative") for pr in tr[k] for x in pr})]
    ac = fams["analogy_control"]
    out += [("analogy_control", c, ac["domain"]) for c in [x for pr in ac["pairs"] for x in pr] + ac["distractors"]]
    return out


def phase_extract(cfg: Config) -> None:
    """For every model (trained + random-init), extract concept and family representations."""
    import torch

    from mathgeom.extract import Item, extract
    from mathgeom.models.load import load_model
    from mathgeom.prompts.templates import TEMPLATE_SETS

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fams = load_yaml("families.yaml")
    fam = family_items(fams)
    field_names = {f["id"]: f["id"].replace("_", " ") for k in ("math", "control") for f in load_yaml("fields.yaml")[k]}
    conds = cfg.raw.get("conditions", ["field_neutral", "family"])
    for m in cfg.models:
        for rnd in ([False, True] if m.get("random_control", True) else [False]):
            key = m["key"] + ("__random" if rnd else "")
            odir = cfg.artifacts / "acts" / key
            if all((odir / f"{c}.npz").exists() for c in conds):
                log.info("skip %s (cached)", key)
                continue
            model, tok = load_model(m["id"], random_init=rnd, seed=cfg.seed, device_map=m.get("device_map", "cuda"))
            odir.mkdir(parents=True, exist_ok=True)
            for cond in conds:
                if (odir / f"{cond}.npz").exists():
                    continue
                if cond in ("field_neutral", "field_revealing"):
                    items = [Item(r.concept, "mathematics" if r.kind == "math" else "science", field_names[r.field]) for r in concepts.itertuples()]
                    tpl = TEMPLATE_SETS[cond]
                else:  # family / no_domain
                    items = [Item(c, d) for _, c, d in fam]
                    tpl = TEMPLATE_SETS[cond]
                arrs = extract(model, tok, items, tpl, tag=f"{key}/{cond}")
                np.savez(odir / f"{cond}.npz", **arrs)
                log.info("saved %s/%s: %s", key, cond, {k: v.shape for k, v in arrs.items()})
            del model
            torch.cuda.empty_cache()
    (cfg.artifacts / "acts" / "family_index.json").write_text(json.dumps(fam))


# ---------------------------------------------------------------- analyses
def _load(cfg: Config, key: str, cond: str) -> dict:
    return dict(np.load(cfg.artifacts / "acts" / key / f"{cond}.npz"))


def _primary_layer(n_layers_plus1: int, rel: float) -> int:
    """Index into hidden_states for a relative depth (1..L are block outputs)."""
    return int(round(rel * (n_layers_plus1 - 1)))


def phase_map(cfg: Config) -> None:
    """H-A1 for every model: field-centroid RSA + LOO clustering, trained vs random-init vs layer-0."""
    from mathgeom.analyses.map_fields import (
        concept_level_regression,
        field_centroids,
        field_rsa,
        loo_centroid_correct,
        mantel_field_perm,
        paired_accuracy_bootstrap,
        paired_bootstrap,
        split_half_reliability,
    )
    from mathgeom.geometry.preprocess import all_but_the_top, cosine_rdm
    from mathgeom.stats.rdm_regression import upper

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    field_D, all_names = np.array(fd["distance"]), fd["names"]
    split = cfg.raw.get("analysis_split", "confirmatory")
    rng = np.random.default_rng(cfg.seed)
    results = {}
    for m in cfg.models:
        tr, rnd = _load(cfg, m["key"], "field_neutral"), _load(cfg, m["key"] + "__random", "field_neutral")
        concepts["n_tokens"] = tr["n_tokens"]
        L1 = tr["last"].shape[1]
        Lp = _primary_layer(L1, cfg.primary_rel_depth)
        res = {"primary_layer": Lp, "n_layers": L1 - 1}
        subsets = {
            "math_primary": (concepts.kind == "math") & concepts.primary & (concepts.split == split),
            "math_all": (concepts.kind == "math") & (concepts.split == split),
        }
        for subset, mask in subsets.items():
            df = concepts[mask].reset_index(drop=True)
            idx = np.flatnonzero(mask.values)
            names = [n for n in all_names if n in set(df.field)]
            sel = [all_names.index(n) for n in names]
            target = field_D[np.ix_(sel, sel)]
            lab = df.field.values
            reps = {"trained": tr["last"][idx, Lp], "random_init": rnd["last"][idx, Lp], "layer0": tr["last"][idx, 0]}
            r = {"n": len(df), "n_fields": len(names), "fields": names}
            r["rsa"] = {k: field_rsa(v, lab, names, target) for k, v in reps.items()}
            null = mantel_field_perm(upper(cosine_rdm(field_centroids(reps["trained"], lab, names))), target, cfg.n_perm, rng)
            r["rsa_perm_p"] = float((1 + (null >= r["rsa"]["trained"]).sum()) / (1 + len(null)))
            r["rsa_boot"] = paired_bootstrap(reps, lab, names, target, cfg.n_boot, cfg.seed)
            correct = {k: loo_centroid_correct(v, lab) for k, v in reps.items()}
            r["loo_acc"] = paired_accuracy_bootstrap(correct, lab, cfg.n_boot, cfg.seed)
            r["loo_acc"]["chance"] = 1 / len(names)
            # layer curve with max-over-layers permutation null (same field relabelings for every layer)
            curve_tr = [field_rsa(tr["last"][idx, l], lab, names, target) for l in range(1, L1)]
            curve_rnd = [field_rsa(rnd["last"][idx, l], lab, names, target) for l in range(1, L1)]
            vecs = [upper(cosine_rdm(field_centroids(tr["last"][idx, l], lab, names))) for l in range(1, L1)]
            maxnull = np.max(np.stack([mantel_field_perm(v, target, 2000, np.random.default_rng(cfg.seed + 7)) for v in vecs]), 0)
            r["layer_curve"] = {"trained": curve_tr, "random_init": curve_rnd, "max_null95": float(np.percentile(maxnull, 95)),
                                "best_layer": int(np.argmax(curve_tr) + 1), "best_p_maxstat": float((1 + (maxnull >= max(curve_tr)).sum()) / (1 + len(maxnull)))}
            if subset == "math_primary":  # H-A2: tree-likeness w.r.t. the pre-registered super-tree
                branch = {f["id"]: f["branch"] for f in load_yaml("fields.yaml")["math"]}
                tree = _tree_distance(names, branch)
                hnull = np.array([field_rsa(reps["trained"], lab, names, _tree_distance(names, dict(zip(names, rng.permutation([branch[n] for n in names]))))) for _ in range(2000)])
                hb = paired_bootstrap(reps, lab, names, tree, cfg.n_boot // 2, cfg.seed)
                r["hierarchy"] = {"rho": {k: field_rsa(v, lab, names, tree) for k, v in reps.items()}, "shuffled_tree_null95": float(np.percentile(hnull, 95)),
                                  "p_shuffled_tree": float((1 + (hnull >= field_rsa(reps["trained"], lab, names, tree)).sum()) / (1 + len(hnull))), "boot": hb}
            r["variants"] = {"abtt": field_rsa(all_but_the_top(reps["trained"]), lab, names, target), "pool": field_rsa(tr["pool"][idx, Lp], lab, names, target)}
            r["split_half"] = split_half_reliability(tr["last_a"][idx, Lp], tr["last_b"][idx, Lp], lab, names)
            r["concept_level_regression"] = concept_level_regression(reps["trained"], df, field_D[np.ix_([all_names.index(f) for f in lab], [all_names.index(f) for f in lab])], reps["layer0"], reps["random_init"], max(cfg.n_perm // 10, 200), cfg.seed)
            res[subset] = r
            b = r["rsa_boot"]
            log.info("[%s] %s L%d: RSA trained=%.3f (perm p=%.4f, CI %s) random=%.3f layer0=%.3f | Δrandom CI %s | LOO acc trained=%.3f random=%.3f chance=%.3f",
                     m["key"], subset, Lp, r["rsa"]["trained"], r["rsa_perm_p"], b["trained"]["ci"], r["rsa"]["random_init"], r["rsa"]["layer0"],
                     b["delta_vs_random_init"]["ci"], r["loo_acc"]["trained"]["acc"], r["loo_acc"]["random_init"]["acc"], r["loo_acc"]["chance"])
        # specificity: same pipeline (clustering only; no graded target exists) on math vs non-math control
        for dom in ("math", "control"):
            mask = (concepts.kind == dom) & concepts.primary & (concepts.split == split)
            idx = np.flatnonzero(mask.values)
            lab = concepts.field.values[idx]
            correct = {"trained": loo_centroid_correct(tr["last"][idx, Lp], lab), "random_init": loo_centroid_correct(rnd["last"][idx, Lp], lab), "layer0": loo_centroid_correct(tr["last"][idx, 0], lab)}
            res[f"clustering_{dom}"] = {"n": len(idx), "n_fields": len(set(lab)), "chance": 1 / len(set(lab)), **paired_accuracy_bootstrap(correct, lab, cfg.n_boot, cfg.seed)}
            log.info("[%s] clustering %s: trained=%.3f random=%.3f layer0=%.3f (chance %.3f)", m["key"], dom, correct["trained"].mean(), correct["random_init"].mean(), correct["layer0"].mean(), 1 / len(set(lab)))
        results[m["key"]] = res
    _save(cfg, "map_fields.json", results)


def _tree_distance(names: list[str], branch: dict[str, str]) -> np.ndarray:
    """Super-tree distance between fields: 1 within a branch, 2 across branches (0 on diagonal)."""
    b = np.array([branch[n] for n in names])
    D = np.where(b[:, None] == b[None, :], 1.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return D


def phase_shapes(cfg: Config) -> None:
    """H-B: shape statistics per family per model at the primary layer + layer curves of key stats."""
    from mathgeom.analyses.shapes import family_stats, residual_beyond_reference

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    results = {}
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            acts = _load(cfg, key, "family")["last"]
            L1 = acts.shape[1]
            Lp = _primary_layer(L1, cfg.primary_rel_depth)
            rows = lambda name: [i for i, (f, _, _) in enumerate(fam) if f == name]  # noqa: E731
            res = {"primary_layer": Lp}
            for name, f in fams["chains"].items():
                X = acts[rows(name)]
                r = family_stats(X[:, Lp], n_perm=2000, seed=cfg.seed)
                r["layer_curve"] = [{k: v for k, v in family_stats(X[:, l], n_perm=200, seed=cfg.seed).items() if not k.endswith("null95")} for l in range(1, L1, max(1, (L1 - 1) // 12))]
                if "reference" in f:
                    kind = "circle" if f["expected"] == "circle" else "line"
                    r["beyond_reference"] = residual_beyond_reference(X[:, Lp], acts[rows(f["reference"])][:, Lp], kind, seed=cfg.seed)
                res[name] = r
                log.info("[%s] %s: line=%.2f circle=%.2f robinson=%.2f(p=%.3f) closure=%.2f loo=%.2f(p=%.3f)", key, name, r["line_rho"], r["circle_rho"], r["robinson"], r["robinson_p"], r["closure"], r["loo_order"], r["loo_order_p"])
            # negative control: random 11-item lists of math concepts drawn from the concept table
            res["negative_control"] = _negative_control(cfg, key, Lp)
            results[key] = res
    _save(cfg, "shapes.json", results)


def _negative_control(cfg: Config, key: str, Lp: int, n_lists: int = 100) -> dict:
    """Shape stats on random 11-item 'chains' of math concepts: rejection rate must be ≈5%."""
    from mathgeom.analyses.shapes import family_stats

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    X = _load(cfg, key, "field_neutral")["last"][:, Lp]
    pool = np.flatnonzero((concepts.kind == "math").values)
    rng = np.random.default_rng(cfg.seed + 1)
    ps = {"robinson_p": [], "pca_order_p": [], "line_rho_p": []}
    for _ in range(n_lists):
        st = family_stats(X[rng.choice(pool, 11, replace=False)], n_perm=300, seed=int(rng.integers(1 << 31)))
        for k in ps:
            ps[k].append(st[k])
    return {k: {"reject_rate_0.05": float(np.mean(np.array(v) < 0.05)), "n_lists": n_lists} for k, v in ps.items()}


def phase_lattice(cfg: Config) -> None:
    """H-B2: group-like and self-distributive axiom lattices, group objects, transfer, analogy control."""
    from mathgeom.analyses.lattice import lattice_stats
    from mathgeom.geometry.offsets import analogy_rank, offset_consistency, random_pair_null
    from mathgeom.geometry.preprocess import center

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    pos = {(f, c): i for i, (f, c, _) in enumerate(fam)}
    results = {}
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            acts = _load(cfg, key, "family")["last"]
            Lp = _primary_layer(acts.shape[1], cfg.primary_rel_depth)
            res = {"primary_layer": Lp}
            for name, lat in fams["lattices"].items():
                names = list(lat["items"])
                X = acts[[pos[(name, c)] for c in names]]
                bits = np.array([lat["items"][c] for c in names])
                res[name] = lattice_stats(X[:, Lp], names, bits, lat["axioms"], seed=cfg.seed)
                res[name]["layer_curve"] = [{"layer": l, "axiom_rdm_beta": lattice_stats(X[:, l], names, bits, lat["axioms"], n_perm=200, seed=cfg.seed)["axiom_rdm_beta"]} for l in range(1, acts.shape[1])]
                log.info("[%s] %s: axiom-RDM beta=%.3f p=%.4f offsets=%s analogy=%s", key, name, res[name]["axiom_rdm_beta"], res[name]["axiom_rdm_p"], {k: round(v["consistency"], 3) for k, v in res[name]["offsets"].items()}, res[name]["analogy"])
            # gender-offset positive control for the offset method
            ac = fams["analogy_control"]
            words = [x for pr in ac["pairs"] for x in pr] + ac["distractors"]
            Xa = center(acts[[pos[("analogy_control", w)] for w in words]][:, Lp].astype(np.float64))
            pairs = [(2 * k, 2 * k + 1) for k in range(len(ac["pairs"]))]
            null = random_pair_null(Xa, np.arange(len(words)), len(pairs), seed=cfg.seed)
            res["analogy_control"] = {"consistency": offset_consistency(Xa, pairs), "null95": float(np.percentile(null, 95)), "king_queen_rank": analogy_rank(Xa, 0, 1, 2, 3, np.arange(len(words)))}
            results[key] = res
    _save(cfg, "lattice.json", results)


def phase_calibration(cfg: Config) -> None:
    """Simulation study of RDM-test calibration (motivates the paired-contrast design)."""
    from mathgeom.analyses.calibration import run_calibration

    res = run_calibration(n_sims=cfg.raw.get("calibration_sims", 200), seed=cfg.seed)
    for k, v in res.items():
        log.info("calibration %s: reject@.05 = %.3f ± %.3f (n=%d)", k, v["reject_rate_0.05"], v["se"], v["n_sims"])
    _save(cfg, "calibration.json", res)


def _save(cfg: Config, name: str, obj) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / name).write_text(json.dumps(obj, indent=1, default=float))
    log.info("wrote %s", cfg.results_dir / name)
