"""Named pipeline phases called by run.py (the full ordered list is run.PHASES: calibration, concepts, arxiv,
extract, map, … , lexical, twins; plus report and paper).

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


AUDIT_FAMILIES = ("nonce_modifier",)  # appended by the independent audit; kept out of every earlier pool so earlier results are unchanged


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
    ex = fams.get("extra")  # appended last so earlier item indices never move (see phase_extract)
    if ex:
        out += [("partial_operation", c, ex["domain"]) for c in ex["partial_operation"]["items"]]
        out += [("lattice_transfer", c, ex["domain"]) for c in sorted({x for pr in ex["lattice_transfer"]["identity"] for x in pr})]
        nm = ex.get("nonce_modifier")  # independent audit (D13): heads, then every "modifier head" in modifier order
        if nm:
            out += [("nonce_modifier", c, ex["domain"]) for c in nm["heads"] + [f"{m} {h}" for m in nm["modifiers"] for h in nm["heads"]]]
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
    idx_path = cfg.artifacts / "acts" / "family_index.json"
    old_fam = [tuple(x) for x in json.loads(idx_path.read_text())] if idx_path.exists() else []

    def n_missing(path, cond):  # family-type caches may lack items appended to families.yaml later
        if not path.exists():
            return None
        if cond in ("field_neutral", "field_revealing"):
            return 0
        n = len(np.load(path)["n_tokens"])
        assert [tuple(x) for x in fam[:n]] == old_fam[:n], "cached family items must be a prefix of family_items()"
        return len(fam) - n

    for m in cfg.models:
        # (key, random_init, seed): the trained model, its seed-`cfg.seed` twin, and optional extra twin seeds
        # (`twin_seeds`, independent audit D12) stored as <key>__random_s<seed>
        runs = [(m["key"], False, cfg.seed)] + ([(m["key"] + "__random", True, cfg.seed)] if m.get("random_control", True) else [])
        runs += [(f"{m['key']}__random_s{s}", True, s) for s in m.get("twin_seeds", [])]
        for key, rnd, seed in runs:
            odir = cfg.artifacts / "acts" / key
            if all(n_missing(odir / f"{c}.npz", c) == 0 for c in conds):
                log.info("skip %s (cached)", key)
                continue
            model, tok = load_model(m["id"], random_init=rnd, seed=seed, device_map=m.get("device_map", "cuda"))
            odir.mkdir(parents=True, exist_ok=True)
            for cond in conds:
                miss = n_missing(odir / f"{cond}.npz", cond)
                if miss == 0:
                    continue
                if cond in ("field_neutral", "field_revealing"):
                    items = [Item(r.concept, "mathematics" if r.kind == "math" else "science", field_names[r.field]) for r in concepts.itertuples()]
                else:  # family / no_domain: only the items not yet cached (batch size 1 ⇒ identical to a full run)
                    items = [Item(c, d) for _, c, d in (fam if miss is None else fam[-miss:])]
                arrs = extract(model, tok, items, TEMPLATE_SETS[cond], tag=f"{key}/{cond}")
                if miss:
                    old = np.load(odir / f"{cond}.npz")
                    arrs = {k: np.concatenate([old[k], v]) for k, v in arrs.items()}
                np.savez(odir / f"{cond}.npz", **arrs)
                log.info("saved %s/%s: %s", key, cond, {k: v.shape for k, v in arrs.items()})
            del model
            torch.cuda.empty_cache()
    (cfg.artifacts / "acts" / "family_index.json").write_text(json.dumps(fam))


# ---------------------------------------------------------------- analyses
class _NpzCache(dict):
    """Reads each array of an .npz on first access and keeps it (most callers need only `last`)."""

    def __init__(self, path):
        super().__init__()
        self._z = np.load(path)

    def __missing__(self, k):
        self[k] = self._z[k]
        return self[k]


def load_acts(cfg: Config, key: str, cond: str) -> dict:
    """Load cached activations; a config may swap the template condition used by every analysis
    (`concept_condition` for "field_neutral", `family_condition` for "family") — used by the
    exploratory template-variant run configs/mvp_variants.yaml."""
    cond = {"field_neutral": cfg.raw.get("concept_condition", "field_neutral"), "family": cfg.raw.get("family_condition", "family")}.get(cond, cond)
    return _NpzCache(cfg.artifacts / "acts" / key / f"{cond}.npz")


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
        tr, rnd = load_acts(cfg, m["key"], "field_neutral"), load_acts(cfg, m["key"] + "__random", "field_neutral")
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
            if subset == "math_primary":  # POST-HOC tree-distance RSA w.r.t. the super-tree (the pre-registered H-A2 measure is the cophenetic test, phase_hierarchy)
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
            acts = load_acts(cfg, key, "family")["last"]
            L1 = acts.shape[1]
            Lp = _primary_layer(L1, cfg.primary_rel_depth)
            rows = lambda name: [i for i, (f, _, _) in enumerate(fam) if f == name]  # noqa: E731
            res = {"primary_layer": Lp}
            for name, f in fams["chains"].items():
                X = acts[rows(name)]
                r = family_stats(X[:, Lp], n_perm=2000, seed=cfg.seed, n_perm_loo=10_000)  # 10k LOO permutations: audit D9
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


def phase_chainfreq(cfg: Config) -> None:
    """Plan H-B frequency control (second audit, docs/deviations.md): is each line-type chain ordered
    beyond wordfreq Zipf frequency and model token count? Cross-validated confound regression."""
    from wordfreq import zipf_frequency

    from mathgeom.analyses.shapes import order_beyond_covariates

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    results = {}
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            raw = load_acts(cfg, key, "family")
            Lp = _primary_layer(raw["last"].shape[1], cfg.primary_rel_depth)
            results[key] = {}
            for name, f in fams["chains"].items():
                if f["expected"] != "line":
                    continue
                rows = [i for i, (fn, _, _) in enumerate(fam) if fn == name]
                Z = np.c_[[zipf_frequency(fam[i][1], "en") for i in rows], raw["n_tokens"][rows]]
                results[key][name] = order_beyond_covariates(raw["last"][rows, Lp], Z, seed=cfg.seed)
            log.info("[chainfreq %s] %s", key, {k: (round(v["loo_order_beyond"], 3), round(v["p"], 4)) for k, v in results[key].items()})
    _save(cfg, "chain_frequency.json", results)


def _negative_control(cfg: Config, key: str, Lp: int, n_lists: int = 100) -> dict:
    """Shape stats on random 11-item 'chains' of math concepts: rejection rate must be ≈5%."""
    from mathgeom.analyses.shapes import family_stats

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    X = load_acts(cfg, key, "field_neutral")["last"][:, Lp]
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
            acts = load_acts(cfg, key, "family")["last"]
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
            if "extra" in fams:  # second plan audit, exploratory (docs/deviations.md)
                res["audit2"] = _lattice_audit2(cfg, fams, pos, key, Lp)
            results[key] = res
    _save(cfg, "lattice.json", results)


def _lattice_audit2(cfg: Config, fams: dict, pos: dict, key: str, Lp: int) -> dict:
    """Exploratory H-B2 plan items (second audit): flagged-item probes, pooled / length-projected
    group-like offsets, and the partial-operation lattice. All at the primary layer."""
    from mathgeom.analyses.lattice import lattice_stats, length_direction
    from mathgeom.geometry.preprocess import center, euclid_rdm

    acts = load_acts(cfg, key, "family")
    g = fams["lattices"]["group_like"]
    names = list(g["items"]) + list(g["flagged"])
    rows = [pos[("group_like", c)] for c in names]
    D = euclid_rdm(center(acts["last"][rows, Lp].astype(np.float64)))
    d = lambda a, b: float(D[names.index(a), names.index(b)])  # noqa: E731
    near = lambda a: [names[j] for j in np.argsort(D[names.index(a)]) if names[j] != a][:3]  # noqa: E731
    out = {"associative_quasigroup": {"d_group": d("associative quasigroup", "group"), "d_quasigroup": d("associative quasigroup", "quasigroup"),
                                      "ratio": d("associative quasigroup", "group") / d("associative quasigroup", "quasigroup"), "nearest": near("associative quasigroup")},
           "inverse_semigroup": {"nearest": near("inverse semigroup")}}
    gnames = list(g["items"])
    gi = [pos[("group_like", c)] for c in gnames]
    bits = np.array([g["items"][c] for c in gnames])
    allr = sorted(i for (f, _), i in pos.items() if f not in AUDIT_FAMILIES)  # family items as of this analysis (later appends excluded)
    w = length_direction(acts["last"][allr, Lp].astype(np.float64), acts["n_tokens"][allr].astype(np.float64))
    Xg = acts["last"][gi, Lp].astype(np.float64)
    out["group_like_offsets"] = {
        "pool": lattice_stats(acts["pool"][gi, Lp], gnames, bits, g["axioms"], seed=cfg.seed)["offsets"],
        "length_projected": lattice_stats(Xg - np.outer(Xg @ w, w), gnames, bits, g["axioms"], seed=cfg.seed)["offsets"],
    }
    po = fams["extra"]["partial_operation"]
    pn = list(po["items"])
    out["partial_operation"] = lattice_stats(acts["last"][[pos[("partial_operation", c)] for c in pn], Lp], pn, np.array([po["items"][c] for c in pn]), po["axioms"], seed=cfg.seed)
    log.info("[%s] audit2: AQ d(group)/d(quasigroup)=%.3f nearest=%s; inverse semigroup nearest=%s; partial-op offsets=%s", key, out["associative_quasigroup"]["ratio"],
             out["associative_quasigroup"]["nearest"], out["inverse_semigroup"]["nearest"], {k: round(v["consistency"], 3) for k, v in out["partial_operation"]["offsets"].items()})
    return out


def phase_baselines(cfg: Config) -> None:
    """Plan baselines: field map from fastText and Qwen3-Embedding vs the LLM (paired bootstrap on the same
    concepts), plus infini-gram counts as a frequency covariate (saved to data/infinigram_counts.json)."""
    from scipy.stats import spearmanr

    from mathgeom.analyses.baselines import fasttext_vectors, infinigram_counts, qwen3_embeddings
    from mathgeom.analyses.map_fields import field_rsa, loo_centroid_correct, paired_accuracy_bootstrap, paired_bootstrap

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    split = cfg.raw.get("analysis_split", "confirmatory")
    counts = infinigram_counts(concepts.concept.tolist(), cfg.data_dir / "infinigram_counts.json")
    logc = np.log10(1 + concepts.concept.map(counts).values.astype(float))
    # download once: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip → artifacts/baselines/
    ft, ft_ok = fasttext_vectors(cfg.artifacts / "baselines" / "wiki-news-300d-1M.vec.zip", concepts.concept.tolist())
    emb_path = cfg.artifacts / "baselines_qwen3emb.npy"
    if not emb_path.exists():
        np.save(emb_path, qwen3_embeddings(concepts.concept.tolist()))
    qe = np.load(emb_path)
    ok = ~np.isnan(logc)  # API failures after retries leave a concept without a count
    out = {"infinigram": {"index": "v4_dolma-v1_7_llama", "rho_zipf_vs_log_count": float(spearmanr(concepts.zipf.values[ok], logc[ok])[0]),
                          "frac_zero": float(np.mean(concepts.concept.map(counts).values[ok] == 0)), "n_missing": int((~ok).sum())}}
    for m in cfg.models:
        X = load_acts(cfg, m["key"], "field_neutral")["last"]
        Lp = _primary_layer(X.shape[1], cfg.primary_rel_depth)
        base = ((concepts.kind == "math") & concepts.primary & (concepts.split == split)).values
        res = {}
        for sub, mask in {"all_primary": base, "fasttext_covered": base & ft_ok}.items():
            lab = concepts.field.values[mask]
            names = [n for n in fd["names"] if n in set(lab)]
            sel = [fd["names"].index(n) for n in names]
            target = np.array(fd["distance"])[np.ix_(sel, sel)]
            reps = {"llm_trained": X[mask, Lp], "qwen3_embedding": qe[mask]}
            if sub == "fasttext_covered":
                reps["fasttext"] = ft[mask]
            r = {"n": int(mask.sum()), "rsa": {k: field_rsa(v, lab, names, target) for k, v in reps.items()},
                 "boot": paired_bootstrap(reps, lab, names, target, cfg.n_boot // 2, cfg.seed),
                 "loo_acc": paired_accuracy_bootstrap({k: loo_centroid_correct(v, lab) for k, v in reps.items()}, lab, cfg.n_boot // 2, cfg.seed)}
            res[sub] = r
            log.info("[baselines %s/%s] n=%d rsa %s", m["key"], sub, r["n"], {k: round(v, 3) for k, v in r["rsa"].items()})
        out[m["key"]] = res
    log.info("infini-gram: rho(zipf, log count)=%.3f, zero-count fraction %.3f", out["infinigram"]["rho_zipf_vs_log_count"], out["infinigram"]["frac_zero"])
    _save(cfg, "baselines.json", out)


def phase_smds(cfg: Config) -> None:
    """Secondary H-B (plan): SMDS line vs discrete-circle fits per family, each vs label-shuffled refits."""
    from mathgeom.analyses.smds_fit import smds_family

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    out = {}
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            A = load_acts(cfg, key, "family")["last"]
            Lp = _primary_layer(A.shape[1], cfg.primary_rel_depth)
            res = {"primary_layer": Lp}
            for name in fams["chains"]:
                idx = [i for i, (f, _, _) in enumerate(fam) if f == name]
                res[name] = smds_family(A[idx, Lp], seed=cfg.seed)
            log.info("[smds %s] %s", key, {n: (res[n]["preferred"], round(res[n]["euclidean"]["p"], 3), round(res[n]["discrete_circular"]["p"], 3)) for n in fams["chains"]})
            out[key] = res
    _save(cfg, "smds.json", out)


def phase_hierarchy(cfg: Config) -> None:
    """H-A2 as pre-registered: cophenetic correlation (UPGMA) vs super-tree, intrinsic tree-likeness vs a
    spectrum-matched Gaussian null, and relative Gromov δ — trained vs random-init, field centroids."""
    from mathgeom.analyses.hierarchy import cophenetic_rdm, relative_delta, spectrum_matched_gaussian, supertree_match
    from mathgeom.analyses.map_fields import field_centroids
    from mathgeom.geometry.preprocess import cosine_rdm

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    split = cfg.raw.get("analysis_split", "confirmatory")
    mask = ((concepts.kind == "math") & concepts.primary & (concepts.split == split)).values
    lab = concepts.field.values[mask]
    names = sorted(set(lab))
    branch = {f["id"]: f["branch"] for f in load_yaml("fields.yaml")["math"]}
    tree = _tree_distance(names, branch)
    rng = np.random.default_rng(cfg.seed)
    out = {}
    for m in cfg.models:
        tr, rnd = load_acts(cfg, m["key"], "field_neutral")["last"], load_acts(cfg, m["key"] + "__random", "field_neutral")["last"]
        Lp = _primary_layer(tr.shape[1], cfg.primary_rel_depth)
        reps = {"trained": tr[mask, Lp], "random_init": rnd[mask, Lp], "layer0": tr[mask, 0]}
        cents = {k: field_centroids(v, lab, names) for k, v in reps.items()}
        r = {"primary_layer": Lp, "n_fields": len(names)}
        r["coph_match"] = {k: supertree_match(cosine_rdm(C), tree) for k, C in cents.items()}
        shuf = np.array([supertree_match(cosine_rdm(cents["trained"]), _tree_distance(names, dict(zip(names, rng.permutation([branch[n] for n in names]))))) for _ in range(2000)])
        r["coph_match_p_shuffled_tree"] = float((1 + (shuf >= r["coph_match"]["trained"]).sum()) / (1 + len(shuf)))
        r["coph_match_null95"] = float(np.percentile(shuf, 95))
        boots = []
        strata = [np.flatnonzero(lab == f) for f in names]
        for _ in range(1000):  # paired concept bootstrap of trained − random-init super-tree match
            idx = np.concatenate([rng.choice(s, len(s)) for s in strata])
            boots.append(supertree_match(cosine_rdm(field_centroids(reps["trained"][idx], lab[idx], names)), tree)
                         - supertree_match(cosine_rdm(field_centroids(reps["random_init"][idx], lab[idx], names)), tree))
        r["coph_match_delta_vs_random_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        r["coph_c"] = {k: cophenetic_rdm(cosine_rdm(C))[1] for k, C in cents.items()}
        r["delta_rel"] = {k: relative_delta(C) for k, C in cents.items()}
        g = [spectrum_matched_gaussian(cents["trained"], rng) for _ in range(300)]
        gc = np.array([cophenetic_rdm(cosine_rdm(x))[1] for x in g])
        gd = np.array([relative_delta(x) for x in g])
        r["gaussian_null"] = {"coph_c_95": float(np.percentile(gc, 95)), "coph_c_p": float((1 + (gc >= r["coph_c"]["trained"]).sum()) / (1 + len(gc))),
                              "delta_rel_5": float(np.percentile(gd, 5)), "delta_rel_p": float((1 + (gd <= r["delta_rel"]["trained"]).sum()) / (1 + len(gd)))}
        log.info("[hierarchy %s] coph-match %s (p_shuf=%.4f, Δrnd CI %s) | coph c %s (gauss95 %.3f) | δ_rel %s (gauss5 %.3f)", m["key"],
                 {k: round(v, 3) for k, v in r["coph_match"].items()}, r["coph_match_p_shuffled_tree"], [round(x, 3) for x in r["coph_match_delta_vs_random_ci"]],
                 {k: round(v, 3) for k, v in r["coph_c"].items()}, r["gaussian_null"]["coph_c_95"], {k: round(v, 3) for k, v in r["delta_rel"].items()}, r["gaussian_null"]["delta_rel_5"])
        out[m["key"]] = r
    _save(cfg, "hierarchy.json", out)


def phase_structures(cfg: Config) -> None:
    """H-B2 exploratory/secondary (pre-registered tests 4–5): axiom transfer to ring-like pairs and
    group objects (topological / Lie / algebraic groups) relative to their parents."""
    from mathgeom.analyses.lattice import axiom_pairs, parallelogram_test, parent_rank, transfer_test

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    pos = {(f, c): i for i, (f, c, _) in enumerate(fam)}
    # pool of the first analysis; items appended by the second audit stay out so earlier nulls/ranks are unchanged
    pool = np.array([i for i, (f, _, _) in enumerate(fam) if f not in ("analogy_control", "number_words_3_12", "number_words_0_10", "digits_0_330", "partial_operation", "lattice_transfer", *AUDIT_FAMILIES)])
    lat = fams["lattices"]["group_like"]
    gl = list(lat["items"])
    bits = np.array([lat["items"][c] for c in gl])
    go, tr = fams["group_objects"], fams["transfer"]
    results = {}
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            A = load_acts(cfg, key, "family")["last"]
            Lp = _primary_layer(A.shape[1], cfg.primary_rel_depth)
            X = A[:, Lp].astype(np.float64)
            Xg = X[[pos[("group_like", c)] for c in gl]]
            res = {"primary_layer": Lp, "transfer": {}, "group_objects": {}}
            for axis_name in ("commutative", "identity"):
                train = axiom_pairs(bits, lat["axioms"].index(axis_name))
                test = [(pos[("transfer", a)], pos[("transfer", b)]) for a, b in tr[axis_name]]
                res["transfer"][axis_name] = transfer_test(Xg, train, X, test, pool, seed=cfg.seed)
            if "extra" in fams:  # second audit (exploratory): identity direction → lattice-like held-out pairs
                test = [(pos[("lattice_transfer", a)], pos[("lattice_transfer", b)]) for a, b in fams["extra"]["lattice_transfer"]["identity"]]
                res["transfer"]["identity_lattice"] = transfer_test(Xg, axiom_pairs(bits, lat["axioms"].index("identity")), X, test, pool, seed=cfg.seed)
            quads = [tuple(pos[("group_objects", c)] for c in q) for q in go["parallelograms"]]
            res["group_objects"]["parallelograms"] = parallelogram_test(X, quads, pool, seed=cfg.seed)
            res["group_objects"]["parent_ranks"] = {o: parent_rank(X, pos[("group_objects", o)], [pos[("group_objects", p)] for p in ps], pool)
                                                    for o, ps in go["objects"].items()}
            res["group_objects"]["pool_size"] = int(len(pool) - 1)
            log.info("[structures %s] transfer %s | parallelograms %s | parent ranks %s", key,
                     {k: (round(v["cos"], 3), round(v["p"], 4)) for k, v in res["transfer"].items()},
                     (round(res["group_objects"]["parallelograms"]["cos"], 3), round(res["group_objects"]["parallelograms"]["p"], 4)), res["group_objects"]["parent_ranks"])
            results[key] = res
    _save(cfg, "structures.json", results)


def phase_compare(cfg: Config) -> None:
    """H-A3: paired concept-bootstrap contrast of field-map strength between two trained models
    (e.g. Qwen2.5-Math-1.5B vs its base Qwen2.5-1.5B), each at its own pre-registered layer."""
    from mathgeom.analyses.map_fields import field_rsa, paired_bootstrap

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    split = cfg.raw.get("analysis_split", "confirmatory")
    mask = ((concepts.kind == "math") & concepts.primary & (concepts.split == split)).values
    lab = concepts.field.values[mask]
    names = [n for n in fd["names"] if n in set(lab)]
    sel = [fd["names"].index(n) for n in names]
    target = np.array(fd["distance"])[np.ix_(sel, sel)]
    out = {}
    for a, b in cfg.raw.get("comparisons", []):
        reps = {}
        for k in (a, b):
            X = load_acts(cfg, k, "field_neutral")["last"]
            reps[k] = X[mask, _primary_layer(X.shape[1], cfg.primary_rel_depth)]
        boot = paired_bootstrap(reps, lab, names, target, cfg.n_boot, cfg.seed)
        out[f"{a}_vs_{b}"] = {"rho": {k: field_rsa(v, lab, names, target) for k, v in reps.items()}, "boot": boot}
        log.info("[compare] %s vs %s: rho %s; delta CI %s", a, b, out[f"{a}_vs_{b}"]["rho"], boot[f"delta_vs_{b}"]["ci"])
    if out:  # configs without `comparisons` (e.g. mvp) have nothing to compare
        _save(cfg, "compare.json", out)


def phase_sae(cfg: Config) -> None:
    """Exploratory: field map and field-selective features in Qwen-Scope SAE codes (models with `sae:`)."""
    from mathgeom.analyses.map_fields import field_rsa, loo_centroid_correct
    from mathgeom.analyses.sae import encode, field_selectivity, load_sae

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    split = cfg.raw.get("analysis_split", "confirmatory")
    out = {}
    for m in [m for m in cfg.models if m.get("sae")]:
        X = load_acts(cfg, m["key"], "field_neutral")["last"]
        Lp = _primary_layer(X.shape[1], cfg.primary_rel_depth)
        sae = load_sae(m["sae"]["repo"], Lp - 1)  # hidden_states[Lp] = output of block Lp-1
        res = {"sae_repo": m["sae"]["repo"], "sae_block": Lp - 1, "k": m["sae"]["k"]}
        for dom in ("math", "control"):
            mask = ((concepts.kind == dom) & concepts.primary & (concepts.split == split)).values
            lab = concepts.field.values[mask]
            codes = encode(X[mask, Lp], sae, m["sae"]["k"])
            r = {"loo_acc_sae": float(loo_centroid_correct(codes, lab).mean()), "selective": field_selectivity(codes, lab),
                 "mean_active": float((codes > 0).sum(1).mean()), "n_features_used": int((codes > 0).any(0).sum())}
            if dom == "math":
                names = [n for n in fd["names"] if n in set(lab)]
                sel = [fd["names"].index(n) for n in names]
                r["rsa_sae"] = field_rsa(codes, lab, names, np.array(fd["distance"])[np.ix_(sel, sel)])
                math_used = (codes > 0).any(0)
            else:
                ctrl_used = (codes > 0).any(0)
            res[dom] = r
        res["features_math_only"] = int((math_used & ~ctrl_used).sum())
        res["features_control_only"] = int((ctrl_used & ~math_used).sum())
        res["features_shared"] = int((math_used & ctrl_used).sum())
        log.info("[sae %s] rsa_sae=%.3f loo_acc math=%.3f control=%.3f; math-only %d, shared %d, control-only %d features",
                 m["key"], res["math"]["rsa_sae"], res["math"]["loo_acc_sae"], res["control"]["loo_acc_sae"], res["features_math_only"], res["features_shared"], res["features_control_only"])
        out[m["key"]] = res
    _save(cfg, "sae.json", out)


def phase_judges(cfg: Config) -> None:
    """Abstraction ratings for all math concepts from the judge panel → data/judge_ratings.csv."""
    from scipy.stats import spearmanr

    from mathgeom.data.judges import icc2, rate

    if not cfg.raw.get("judges"):
        log.info("judges: none configured, skipping")
        return
    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    math = concepts[concepts.kind == "math"].reset_index(drop=True)
    path = cfg.data_dir / "judge_ratings.csv"
    df = pd.read_csv(path) if path.exists() else math[["concept", "field"]].copy()
    for jid in cfg.raw.get("judges", []):
        if jid not in df:
            df[jid] = rate(jid, df.concept.tolist())
            df.to_csv(path, index=False)
    J = df[cfg.raw["judges"]].values
    pair = [spearmanr(J[:, a], J[:, b])[0] for a in range(J.shape[1]) for b in range(a + 1, J.shape[1])]
    i1, ik = icc2(J)
    res = {"judges": cfg.raw["judges"], "n": len(df), "mean_pairwise_spearman": float(np.mean(pair)), "pairwise_spearman": [float(x) for x in pair], "icc2_1": i1, "icc2_k": ik,
           "field_mean_rating": df.assign(mean=J.mean(1)).groupby("field")["mean"].mean().round(3).to_dict()}
    log.info("judges: mean pairwise rho=%.3f ICC(2,1)=%.3f ICC(2,k)=%.3f", res["mean_pairwise_spearman"], i1, ik)
    _save(cfg, "judges.json", res)


def phase_mathlib(cfg: Config) -> None:
    """H-A4 judge validation (plan; pre-specified in docs/deviations.md, second audit): Spearman ρ of the
    mean judge abstraction rating with Mathlib import depth, plus Mathlib coverage per field."""
    import subprocess

    from scipy.stats import spearmanr

    from mathgeom.data.mathlib_depth import declarations, import_depth, match_concepts, module_imports

    rev = cfg.raw.get("mathlib_rev")
    if not rev:
        log.info("mathlib: no mathlib_rev configured, skipping")
        return
    root = cfg.artifacts / "mathlib4"
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    if not root.exists():  # sparse, blob-less clone: only Mathlib/ sources and docs/ name maps (~120 MB)
        subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", "https://github.com/leanprover-community/mathlib4", str(root)], check=True)
        git("sparse-checkout", "set", "Mathlib", "docs")
    if git("rev-parse", "HEAD") != rev:
        git("fetch", "--depth", "1", "origin", rev)
        git("checkout", "--detach", rev)
    depth = import_depth(module_imports(root))
    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    ratings = pd.read_csv(cfg.data_dir / "judge_ratings.csv")
    judges = [c for c in ratings.columns if "/" in c]
    df = concepts[concepts.kind == "math"].merge(ratings[["concept", "field", *judges]], on=["concept", "field"]).reset_index(drop=True)
    match = match_concepts(df.concept.tolist(), [root / "docs" / "overview.yaml", root / "docs" / "undergrad.yaml"], declarations(root))
    for k in ("decl", "module", "source"):
        df[k] = df.concept.map(lambda c, k=k: match.get(c, {}).get(k))
    df["depth"] = df.module.map(depth)
    df["mean_rating"] = df[judges].mean(1)
    df[["concept", "field", "primary", "split", "decl", "module", "source", "depth", "mean_rating"]].to_csv(cfg.data_dir / "mathlib_depth.csv", index=False)
    m = df.dropna(subset=["depth"]).reset_index(drop=True)
    rho = lambda a, b: float(spearmanr(a, b)[0])  # noqa: E731
    rng = np.random.default_rng(cfg.seed)
    boot = [rho(m.mean_rating.values[i], m.depth.values[i]) for i in (rng.integers(0, len(m), len(m)) for _ in range(cfg.raw.get("n_boot", 2000)))]
    Z = np.c_[np.ones(len(m)), m.zipf.values]
    res_ = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
    out = {"mathlib_rev": rev, "n_modules": len(depth), "n_math_concepts": len(df), "n_matched": len(m),
           "n_by_source": m.source.value_counts().to_dict(), "depth_median": float(m.depth.median()),
           "coverage_by_field": df.assign(hit=df.depth.notna()).groupby("field")["hit"].mean().round(3).to_dict(),
           "rho_mean_rating_depth": rho(m.mean_rating, m.depth), "p": float(spearmanr(m.mean_rating, m.depth)[1]),
           "ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
           "rho_per_judge": {j: rho(m[j], m.depth) for j in judges},
           "rho_partial_zipf": rho(res_(m.mean_rating.values), res_(m.depth.values.astype(float)))}
    log.info("mathlib: %d/%d concepts matched (%s); rho(rating, depth)=%.3f CI [%.3f, %.3f]", out["n_matched"], out["n_math_concepts"], out["n_by_source"],
             out["rho_mean_rating_depth"], *out["ci95"])
    _save(cfg, "mathlib.json", out)


def phase_axes(cfg: Config) -> None:
    """H-A4 (secondary): do the top PCs of math concepts align with pure/applied and judge abstraction?
    Spearman of each PC score with the covariate after residualizing both on zipf & token count."""
    from scipy.stats import spearmanr

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    ratings = pd.read_csv(cfg.data_dir / "judge_ratings.csv")
    judges = [c for c in ratings.columns if "/" in c]  # judge columns are HF repo ids
    abstr = ratings.set_index("concept")[judges].mean(1)
    split = cfg.raw.get("analysis_split", "confirmatory")
    mask = ((concepts.kind == "math") & concepts.primary & (concepts.split == split)).values
    df = concepts[mask].reset_index(drop=True)
    pa = (df.pa == "applied").astype(float).values
    ab = df.concept.map(abstr).values
    out = {}
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            acts = load_acts(cfg, key, "field_neutral")
            Lp = _primary_layer(acts["last"].shape[1], cfg.primary_rel_depth)
            X = acts["last"][mask, Lp].astype(np.float64)
            X = X - X.mean(0)
            _, S, Vt = np.linalg.svd(X, full_matrices=False)
            pcs = X @ Vt[:5].T
            Z = np.c_[np.ones(len(df)), df.zipf.values, acts["n_tokens"][mask]]
            res_ = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
            r = {"explained_var": (S[:5] ** 2 / (S**2).sum()).round(4).tolist(), "pcs": []}
            for k in range(5):
                pc = res_(pcs[:, k])
                rp, pp = spearmanr(pc, res_(pa))
                ra, pa_ = spearmanr(pc, res_(ab))
                r["pcs"].append({"pc": k + 1, "rho_applied": float(rp), "p_applied": float(pp), "rho_abstraction": float(ra), "p_abstraction": float(pa_)})
            # best single linear readout (5-fold CV ridge) of abstraction rating, for reference
            from sklearn.linear_model import RidgeCV
            from sklearn.model_selection import cross_val_predict

            pred = cross_val_predict(RidgeCV(alphas=np.logspace(-1, 4, 12)), X, res_(ab), cv=5)
            r["cv_ridge_rho_abstraction"] = float(spearmanr(pred, res_(ab))[0])
            ig_path = cfg.data_dir / "infinigram_counts.json"
            if ig_path.exists():  # robustness: also partial out pretraining-scale frequency (infini-gram, Dolma v1.7)
                ig = json.loads(ig_path.read_text())
                logc = np.log10(1 + df.concept.map(ig).fillna(0).values.astype(float))
                Z2 = np.c_[Z, logc]
                res2 = lambda v: v - Z2 @ np.linalg.lstsq(Z2, v, rcond=None)[0]  # noqa: E731
                r["pcs_with_infinigram"] = [{"pc": k + 1, "rho_applied": float(spearmanr(res2(pcs[:, k]), res2(pa))[0]),
                                             "rho_abstraction": float(spearmanr(res2(pcs[:, k]), res2(ab))[0])} for k in range(2)]
            out[key] = r
            log.info("[axes %s] PC1-5 rho(applied)=%s rho(abstraction)=%s cv-ridge rho=%.3f", key, [round(x["rho_applied"], 2) for x in r["pcs"]], [round(x["rho_abstraction"], 2) for x in r["pcs"]], r["cv_ridge_rho_abstraction"])
    _save(cfg, "axes.json", out)


def phase_causal(cfg: Config) -> None:
    """H-B causal test on the ring and space chains (trained model only; gated on baseline accuracy)."""
    import torch

    from mathgeom.analyses.causal import chain_causal_test
    from mathgeom.models.load import load_model

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    out = {}
    for m in cfg.models:
        acts = load_acts(cfg, m["key"], "family")["last"]
        Lp = _primary_layer(acts.shape[1], cfg.primary_rel_depth)
        model, tok = load_model(m["id"], device_map=m.get("device_map", "cuda"))
        layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
        res = {"layer": Lp}
        for name in cfg.raw.get("causal_families", ["ring_chain", "space_chain"]):
            f = fams["chains"][name]
            idx = [i for i, (g, _, _) in enumerate(fam) if g == name]
            res[name] = chain_causal_test(model, tok, layers[Lp - 1], f["items"], f["domain"], acts[idx, Lp].astype(np.float64), seed=cfg.seed)
            log.info("[causal %s] %s: %s", m["key"], name, {k: v for k, v in res[name].items() if k != "scales"} | {"scales": {s: (round(v["acc_chain_subspace"], 3), round(v["acc_random_subspace_mean"], 3)) for s, v in res[name]["scales"].items()}})
        out[m["key"]] = res
        del model
        torch.cuda.empty_cache()
    _save(cfg, "causal.json", out)


def phase_posthoc(cfg: Config) -> None:
    """POST-HOC (docs/deviations.md): all-but-the-top removal-depth sweep and per-PC field variance (η²),
    run after the pre-registered ABTT variant (k = d/100) reversed the field-map correlation."""
    from mathgeom.analyses.map_fields import field_rsa
    from mathgeom.geometry.preprocess import all_but_the_top, center

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    split = cfg.raw.get("analysis_split", "confirmatory")
    mask = ((concepts.kind == "math") & concepts.primary & (concepts.split == split)).values
    lab = concepts.field.values[mask]
    names = [n for n in fd["names"] if n in set(lab)]
    sel = [fd["names"].index(n) for n in names]
    target = np.array(fd["distance"])[np.ix_(sel, sel)]
    out = {}
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            X = load_acts(cfg, key, "field_neutral")["last"]
            Lp = _primary_layer(X.shape[1], cfg.primary_rel_depth)
            X = X[mask, Lp].astype(np.float64)
            Xc = center(X)
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Xc @ Vt[:20].T
            eta = [float(pd.Series(P[:, i]).groupby(lab).transform("mean").var() / P[:, i].var()) for i in range(20)]
            out[key] = {"abtt_sweep": {str(k): field_rsa(all_but_the_top(X, k), lab, names, target) for k in (0, 1, 2, 3, 5, 10, 20, 40)},
                        "explained_var_top5": (S[:5] ** 2 / (S**2).sum()).tolist(), "eta2_field_pc1_20": eta}
            out[key]["abtt_sweep"]["0"] = field_rsa(X, lab, names, target)
            log.info("[posthoc %s] abtt sweep %s", key, {k: round(v, 3) for k, v in out[key]["abtt_sweep"].items()})
    # POST-HOC 3: matched-k clustering — LOO accuracy among random 4-field subsets of math vs the 4 control fields
    from mathgeom.analyses.map_fields import loo_centroid_correct

    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            X = load_acts(cfg, key, "field_neutral")["last"]
            Lp = _primary_layer(X.shape[1], cfg.primary_rel_depth)
            base = (concepts.primary & (concepts.split == split)).values
            ctrl = base & (concepts.kind == "control").values
            k = concepts.field.values[ctrl].tolist()
            n_ctrl_fields = len(set(k))
            acc_ctrl = float(loo_centroid_correct(X[ctrl, Lp], concepts.field.values[ctrl]).mean())
            mfields = sorted(set(concepts.field.values[base & (concepts.kind == "math").values]))
            rs = np.random.default_rng(cfg.seed + 3)
            accs = []
            for _ in range(200):
                pick = rs.choice(mfields, n_ctrl_fields, replace=False)
                mm = base & np.isin(concepts.field.values, pick)
                accs.append(float(loo_centroid_correct(X[mm, Lp], concepts.field.values[mm]).mean()))
            out.setdefault(key, {})["matched_clustering"] = {"k_fields": n_ctrl_fields, "control_acc": acc_ctrl, "math_subsets_mean": float(np.mean(accs)),
                                                            "math_subsets_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))], "frac_math_subsets_ge_control": float(np.mean(np.array(accs) >= acc_ctrl))}
            log.info("[posthoc %s] matched %d-field clustering: control %.3f vs math subsets %.3f", key, n_ctrl_fields, acc_ctrl, np.mean(accs))

    # POST-HOC 2: cyclic order in the top-2-PC plane, every family, every layer (debug of the positive control)
    from mathgeom.geometry.manifold import pc_plane_cycle_score

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    rng = np.random.default_rng(cfg.seed)
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            acts = load_acts(cfg, key, "family")["last"]
            cyc = {}
            for name in fams["chains"]:
                idx = [i for i, (f, _, _) in enumerate(fam) if f == name]
                o = np.arange(len(idx))
                per_layer = []
                for l in range(1, acts.shape[1]):
                    X = acts[idx, l].astype(np.float64)
                    sc = pc_plane_cycle_score(X, o)
                    null = np.array([pc_plane_cycle_score(X, rng.permutation(len(idx))) for _ in range(500)])
                    per_layer.append({"layer": l, "score": sc, "p": float((1 + (null >= sc).sum()) / 501)})
                cyc[name] = per_layer
            out.setdefault(key, {})["pc_plane_cycle"] = cyc
            log.info("[posthoc %s] PC-plane cycle score at primary layer: %s", key, {n: round(v[_primary_layer(acts.shape[1], cfg.primary_rel_depth) - 1]["score"], 2) for n, v in cyc.items()})
    # POST-HOC 4 (reports/prereg_addendum_3d.md, written before running): closed cycles in any plane of the
    # top-5 PCs, every layer, with a max-over-planes permutation null; a separate random stream keeps every
    # score above unchanged. Bare numerals (digits_0_330) are the lexical control for angles.
    from mathgeom.geometry.manifold import best_plane_cycle

    rng4 = np.random.default_rng(cfg.seed + 7)
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            acts = load_acts(cfg, key, "family")["last"]
            Lp = _primary_layer(acts.shape[1], cfg.primary_rel_depth)
            res = {}
            for name in ("days", "months", "angles", "digits_0_330"):
                idx = [i for i, (f, _, _) in enumerate(fam) if f == name]
                res[name] = [{"layer": l, **best_plane_cycle(acts[idx, l].astype(np.float64), np.arange(len(idx)), k=5, n_perm=1000, rng=rng4)}
                             for l in range(1, acts.shape[1])]
            out[key]["pc_planes_cycle"] = res
            log.info("[posthoc %s] closed cycle in best top-5 PC plane at layer %d: %s", key, Lp,
                     {n: (round(v[Lp - 1]["score"], 2), "PC%d-PC%d" % tuple(np.array(v[Lp - 1]["plane"]) + 1), round(v[Lp - 1]["p"], 3)) for n, v in res.items()})
    _save(cfg, "posthoc.json", out)


def phase_lexical(cfg: Config) -> None:
    """Independent audit (reports/prereg_addendum_lexical.md), post-hoc: word-form baselines for H-A1, H-B2, the
    group objects and H-B (D1–D6), the nonce-modifier calibration (D13), the block-only simulation of the chain test
    (D7; mvp only), and the bootstrap bias / BCa check of H-A1 and H-A3 (D11), which needs no new extraction either.
    The random-init twin was meant to measure form-only similarity but scrambles even exact word composition, so
    each test is rerun on representations that encode the names' spelling and nothing else (analyses/lexical.py)."""
    from mathgeom.analyses import lexical as lx
    from mathgeom.analyses.lattice import axiom_pairs, lattice_stats, parallelogram_test
    from mathgeom.analyses.map_fields import bca_summary, field_centroids, field_rsa, loo_centroid_correct, mantel_field_perm, paired_bootstrap
    from mathgeom.analyses.shapes import family_stats
    from mathgeom.geometry.manifold import perm_p
    from mathgeom.geometry.offsets import offset_consistency, random_pair_null
    from mathgeom.geometry.preprocess import center, cosine_rdm
    from mathgeom.stats.rdm_regression import upper

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    pos = {(f, c): i for i, (f, c, _) in enumerate(fam)}
    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    fidx = np.flatnonzero(((concepts.kind == "math") & concepts.primary & (concepts.split == cfg.raw.get("analysis_split", "confirmatory"))).values)
    lab, cnames = concepts.field.values[fidx], [concepts.concept.values[i] for i in fidx]
    fnames = [n for n in fd["names"] if n in set(lab)]
    sel = [fd["names"].index(n) for n in fnames]
    target = np.array(fd["distance"])[np.ix_(sel, sel)]
    g = fams["lattices"]["group_like"]
    gl, axioms = list(g["items"]), g["axioms"]
    bits = np.array([g["items"][c] for c in gl])
    sub = [c for c in gl if not c.startswith("commutative")]  # the 8 names without the word "commutative"
    sbits = np.array([g["items"][c] for c in sub])
    excl = ("analogy_control", "number_words_3_12", "number_words_0_10", "digits_0_330", "partial_operation", "lattice_transfer", *AUDIT_FAMILIES)
    pool = [i for i, (f, _, _) in enumerate(fam) if f not in excl]  # the pool of phase_structures
    quads = [tuple(pos[("group_objects", c)] for c in q) for q in fams["group_objects"]["parallelograms"]]
    sc = fams["chains"]["space_chain"]["items"]
    space_blocks = {"T0_to_normal": sc[: sc.index("normal space") + 1], "metrizable_to_Hilbert": sc[sc.index("metrizable space"):]}
    nm = fams["extra"]["nonce_modifier"]
    BASE = ("char3", "word_bag", "layer0_bag")
    out = {}
    for m in cfg.models:
        key = m["key"]
        acts = {"trained": load_acts(cfg, key, "family"), "twin": load_acts(cfg, key + "__random", "family")}
        Lp = _primary_layer(acts["trained"]["last"].shape[1], cfg.primary_rel_depth)

        def rep(kind, rows):  # representation of family rows: model states at the pre-registered layer, or a baseline
            if kind in acts:
                return acts[kind]["last"][rows, Lp].astype(np.float64)
            if kind == "layer0_bag":
                return lx.layer0_bag(acts["trained"], rows)
            return {"char3": lx.char3_vectors, "word_bag": lx.word_bag}[kind]([fam[i][1] for i in rows])

        res = {"primary_layer": Lp}
        # D1: field map on the baselines, and the paired bootstrap of trained − baseline
        fa = load_acts(cfg, key, "field_neutral")
        reps = {"trained": fa["last"][fidx, Lp], "char3": lx.char3_vectors(cnames), "word_bag": lx.word_bag(cnames), "layer0_bag": lx.layer0_bag(fa, fidx)}
        rng = np.random.default_rng(cfg.seed)
        res["field_map"] = {"rsa": {k: field_rsa(v, lab, fnames, target) for k, v in reps.items()},
                            "mantel_p": {k: perm_p(field_rsa(reps[k], lab, fnames, target), mantel_field_perm(upper(cosine_rdm(field_centroids(reps[k], lab, fnames))), target, cfg.n_perm, rng))
                                         for k in BASE},
                            "boot": paired_bootstrap(reps, lab, fnames, target, cfg.n_boot // 2, cfg.seed),
                            "loo_acc": {k: float(loo_centroid_correct(v, lab).mean()) for k, v in reps.items()}}
        # D2–D3: group-like axiom offsets on the baselines, the duplicate/disjoint split, the sub-lattice, the held-out pair
        gr, sr = [pos[("group_like", c)] for c in gl], [pos[("group_like", c)] for c in sub]
        base_stats = {b: lattice_stats(rep(b, gr), gl, bits, axioms, n_perm=200, seed=cfg.seed) for b in BASE}
        res["group_like"] = {
            "offsets": {b: s["offsets"] for b, s in base_stats.items()},
            "analogy_top1": {b: s["analogy"]["top1"] for b, s in base_stats.items()},
            "decomposition": {k: {ax: lx.disjoint_offset_test(rep(k, gr), gl, axiom_pairs(bits, i), seed=cfg.seed + i) for i, ax in enumerate(axioms)} for k in acts},
            "sublattice": {k: lx.offsets_vs_null(rep(k, sr), sbits, axioms, n_perm=5000, seed=cfg.seed) for k in (*acts, *BASE)},
            "abelian_heldout": {k: lx.heldout_commutative_test(rep(k, gr), gl, bits) for k in (*acts, *BASE)},
        }
        # D4: group-object parallelograms on the baselines (same pool, quadruples and seed as phase_structures)
        rows_all = sorted(set(pool) | {r for q in quads for r in q})
        at = {r: j for j, r in enumerate(rows_all)}
        res["group_objects"] = {b: parallelogram_test(rep(b, rows_all), [tuple(at[r] for r in q) for q in quads], np.array([at[r] for r in pool]), seed=cfg.seed) for b in BASE}
        # D5–D6: chains on the baselines (a word bag of one-word items is degenerate, so it is left out) and within-block order
        keep = ("line_rho", "circle_rho", "closure", "robinson", "robinson_p", "pca_order", "pca_order_p", "loo_order", "loo_order_p")
        res["chains"] = {name: {b: {k: v for k, v in family_stats(rep(b, [pos[(name, c)] for c in f["items"]]), n_perm=2000, seed=cfg.seed).items() if k in keep}
                                for b in ("char3", "layer0_bag")} for name, f in fams["chains"].items()}
        drows = [pos[("ring_chain", c)] for c in fams["chains"]["ring_chain"]["items"] if c.endswith("domain")]
        res["ring_within_domain"] = {k: lx.exact_order_test(rep(k, drows)) for k in (*acts, "char3", "layer0_bag")}
        res["space_blocks"] = {bn: {k: lx.exact_order_test(rep(k, [pos[("space_chain", c)] for c in items])) for k in (*acts, "char3", "layer0_bag")}
                               for bn, items in space_blocks.items()}
        # D13: nonce modifiers (only where the appended items have been extracted)
        last = pos[("nonce_modifier", f"{nm['modifiers'][-1]} {nm['heads'][-1]}")]
        if all(len(a["n_tokens"]) > last for a in acts.values()):
            res["nonce_modifier"] = {}
            for mod in nm["modifiers"]:
                rows = [pos[("nonce_modifier", h)] for h in nm["heads"]] + [pos[("nonce_modifier", f"{mod} {h}")] for h in nm["heads"]]
                pairs = [(i, i + len(nm["heads"])) for i in range(len(nm["heads"]))]
                res["nonce_modifier"][mod] = {}
                for k in (*acts, "char3", "layer0_bag"):
                    X = center(rep(k, rows))
                    obs, null = offset_consistency(X, pairs), random_pair_null(X, np.arange(len(rows)), len(pairs), 2000, cfg.seed)
                    res["nonce_modifier"][mod][k] = {"consistency": obs, "null95": float(np.percentile(null, 95)), "p": perm_p(obs, null)}
        # D11: bootstrap bias and BCa of the H-A1 field map vs the twin (the same resamples as phase_map)
        twin_field = load_acts(cfg, key + "__random", "field_neutral")["last"][fidx, Lp]
        res["bootstrap"] = bca_summary({"trained": reps["trained"], "random_init": twin_field}, lab, fnames, target, cfg.n_boot, cfg.seed)
        out[key] = res
        log.info("[lexical %s] field rho %s | char3 offsets %s | abelian %s | group objects %s | ring within-domain p %.4f | nonce %s", key,
                 {k: round(v, 3) for k, v in res["field_map"]["rsa"].items()}, {a: round(o["consistency"], 2) for a, o in res["group_like"]["offsets"]["char3"].items()},
                 {k: round(v["cos"], 2) for k, v in res["group_like"]["abelian_heldout"].items()}, {b: round(v["cos"], 3) for b, v in res["group_objects"].items()},
                 res["ring_within_domain"]["trained"]["robinson_p"], {md: round(v["trained"]["consistency"], 2) for md, v in res.get("nonce_modifier", {}).items()})
    for a, b in cfg.raw.get("comparisons", []):  # D11 for H-A3 (paired contrast of two trained models)
        reps = {}
        for k in (a, b):
            X = load_acts(cfg, k, "field_neutral")["last"]
            reps[k] = X[fidx, _primary_layer(X.shape[1], cfg.primary_rel_depth)]
        out.setdefault("compare_bootstrap", {})[f"{a}_vs_{b}"] = bca_summary(reps, lab, fnames, target, cfg.n_boot, cfg.seed)
    if cfg.name == "mvp":  # D7: how diagnostic is the chain test? Head-noun clusters only, at the confirmatory model's scale
        m0 = cfg.models[0]["key"]
        A = load_acts(cfg, m0, "family")["last"]
        X = center(A[[pos[("ring_chain", c)] for c in fams["chains"]["ring_chain"]["items"]], out[m0]["primary_layer"]].astype(np.float64))
        out["block_only_simulation"] = [lx.block_only_simulation(snr, n_sims=40, seed=100, scale=float(np.linalg.norm(X, axis=1).mean())) for snr in (0.5, 1.0, 2.0)]
        log.info("[lexical] block-only simulation: %s", out["block_only_simulation"])
    _save(cfg, "lexical.json", out)


def phase_twins(cfg: Config) -> None:
    """Independent audit D12 (post-hoc): how much do the twin statistics depend on the random draw? For the trained
    model and each random-init seed (seed `cfg.seed` is the twin of every other analysis): field ρ and LOO field
    accuracy (primary concepts), the group-like offsets, and the ring/space chain statistics; plus the H-A1 paired
    bootstrap Δ (trained − twin) per seed."""
    from mathgeom.analyses.lattice import lattice_stats
    from mathgeom.analyses.map_fields import field_rsa, loo_centroid_correct, paired_bootstrap
    from mathgeom.analyses.shapes import family_stats

    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    pos = {(f, c): i for i, (f, c, _) in enumerate(fam)}
    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    fidx = np.flatnonzero(((concepts.kind == "math") & concepts.primary & (concepts.split == cfg.raw.get("analysis_split", "confirmatory"))).values)
    lab = concepts.field.values[fidx]
    fnames = [n for n in fd["names"] if n in set(lab)]
    sel = [fd["names"].index(n) for n in fnames]
    target = np.array(fd["distance"])[np.ix_(sel, sel)]
    g = fams["lattices"]["group_like"]
    gl = list(g["items"])
    gbits = np.array([g["items"][c] for c in gl])
    out = {}
    if not any(m.get("twin_seeds") for m in cfg.models):  # e.g. `run.py all` on a config without extra seeds
        log.info("[twins] no `twin_seeds` in %s; nothing to do", cfg.name)
        return
    for m in [m for m in cfg.models if m.get("twin_seeds")]:
        tags ={"trained": m["key"], f"seed{cfg.seed}": m["key"] + "__random", **{f"seed{s}": f"{m['key']}__random_s{s}" for s in m.get("twin_seeds", [])}}
        res, field = {}, {}
        for tag, key in tags.items():
            fa, A = load_acts(cfg, key, "field_neutral")["last"], load_acts(cfg, key, "family")["last"]
            Lp = _primary_layer(A.shape[1], cfg.primary_rel_depth)
            field[tag] = fa[fidx, Lp]
            ls = lattice_stats(A[[pos[("group_like", c)] for c in gl], Lp], gl, gbits, g["axioms"], n_perm=200, seed=cfg.seed)
            chains = {c: {k: v for k, v in family_stats(A[[pos[(c, x)] for x in fams["chains"][c]["items"]], Lp], n_perm=2000, seed=cfg.seed, n_perm_loo=2000).items()
                          if k in ("robinson", "robinson_p", "loo_order", "loo_order_p")} for c in ("ring_chain", "space_chain")}
            res[tag] = {"model": key, "field_rho": field_rsa(field[tag], lab, fnames, target), "field_loo_acc": float(loo_centroid_correct(field[tag], lab).mean()),
                        "offsets": {a: {k: o[k] for k in ("consistency", "p")} for a, o in ls["offsets"].items()}, "chains": chains}
            log.info("[twins %s] field rho %.3f, offsets %s, ring/space robinson %.2f/%.2f", tag, res[tag]["field_rho"], {a: round(o["consistency"], 2) for a, o in res[tag]["offsets"].items()},
                     chains["ring_chain"]["robinson"], chains["space_chain"]["robinson"])
        res["field_boot"] = paired_bootstrap(field, lab, fnames, target, cfg.n_boot // 2, cfg.seed)  # Δ trained − each seed
        seeds = [t for t in tags if t != "trained"]
        stat = {"field_rho": lambda e: e["field_rho"], "field_loo_acc": lambda e: e["field_loo_acc"],
                **{f"offset_{a}": (lambda e, a=a: e["offsets"][a]["consistency"]) for a in res["trained"]["offsets"]},
                **{f"robinson_{c}": (lambda e, c=c: e["chains"][c]["robinson"]) for c in ("ring_chain", "space_chain")}}
        res["summary"] = {"twin_range": {n: [min(f(res[t]) for t in seeds), max(f(res[t]) for t in seeds)] for n, f in stat.items()},
                          "trained_exceeds_every_twin": {n: bool(f(res["trained"]) > max(f(res[t]) for t in seeds)) for n, f in stat.items()},
                          "field_delta_ci_lowest_lower": min(res["field_boot"][f"delta_vs_{t}"]["ci"][0] for t in seeds), "n_seeds": len(seeds)}
        out[m["key"]] = res
    _save(cfg, "twins.json", out)


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
