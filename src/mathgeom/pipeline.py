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
    """Load cached activations; a config may swap the template condition used by every analysis
    (`concept_condition` for "field_neutral", `family_condition` for "family") — used by the
    exploratory template-variant run configs/mvp_variants.yaml."""
    cond = {"field_neutral": cfg.raw.get("concept_condition", "field_neutral"), "family": cfg.raw.get("family_condition", "family")}.get(cond, cond)
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
        X = _load(cfg, m["key"], "field_neutral")["last"]
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
            A = _load(cfg, key, "family")["last"]
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
        tr, rnd = _load(cfg, m["key"], "field_neutral")["last"], _load(cfg, m["key"] + "__random", "field_neutral")["last"]
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
    pool = np.array([i for i, (f, _, _) in enumerate(fam) if f not in ("analogy_control", "number_words_3_12", "number_words_0_10", "digits_0_330")])
    lat = fams["lattices"]["group_like"]
    gl = list(lat["items"])
    bits = np.array([lat["items"][c] for c in gl])
    go, tr = fams["group_objects"], fams["transfer"]
    results = {}
    for m in cfg.models:
        for key in (m["key"], m["key"] + "__random"):
            A = _load(cfg, key, "family")["last"]
            Lp = _primary_layer(A.shape[1], cfg.primary_rel_depth)
            X = A[:, Lp].astype(np.float64)
            Xg = X[[pos[("group_like", c)] for c in gl]]
            res = {"primary_layer": Lp, "transfer": {}, "group_objects": {}}
            for axis_name in ("commutative", "identity"):
                train = axiom_pairs(bits, lat["axioms"].index(axis_name))
                test = [(pos[("transfer", a)], pos[("transfer", b)]) for a, b in tr[axis_name]]
                res["transfer"][axis_name] = transfer_test(Xg, train, X, test, pool, seed=cfg.seed)
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
            X = _load(cfg, k, "field_neutral")["last"]
            reps[k] = X[mask, _primary_layer(X.shape[1], cfg.primary_rel_depth)]
        boot = paired_bootstrap(reps, lab, names, target, cfg.n_boot, cfg.seed)
        out[f"{a}_vs_{b}"] = {"rho": {k: field_rsa(v, lab, names, target) for k, v in reps.items()}, "boot": boot}
        log.info("[compare] %s vs %s: rho %s; delta CI %s", a, b, out[f"{a}_vs_{b}"]["rho"], boot[f"delta_vs_{b}"]["ci"])
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
        X = _load(cfg, m["key"], "field_neutral")["last"]
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
            acts = _load(cfg, key, "field_neutral")
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
        acts = _load(cfg, m["key"], "family")["last"]
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
            X = _load(cfg, key, "field_neutral")["last"]
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
            X = _load(cfg, key, "field_neutral")["last"]
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
            acts = _load(cfg, key, "family")["last"]
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
    _save(cfg, "posthoc.json", out)


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
