"""Export data for the explorable article on GitHub Pages (site/index.html, site/app.js).

Global context: the site shows the same unsupervised projections as the static figures
(report/figdata.py) — per layer, so readers can scrub through depth — plus everything needed for
in-browser interaction: field distance matrices (hover-linked to the map), concept nearest
neighbours, exact (n−1)-D lattice coordinates (an analogy playground computes ranks exactly in the
browser) and the difference-vector cosine matrix. Layout:
  site/data/index.json   models, fields, scorecard, forest rows, a pruned copy of the results JSON
  site/data/<model>.json per-model geometry (lazy-loaded when the reader picks that model)
All numbers on the page are read from these files; none are typed into the HTML.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mathgeom.config import ROOT, Config, load_yaml
from mathgeom.report import figdata
from mathgeom.report import style as S

SITE = ROOT / "site" / "data"
SHOW_FAMILIES = ["days", "months", "angles", "ring_chain", "space_chain", "number_systems", "ngons", "dimensions"]


def _r(x, d: int = 3):
    return np.round(np.asarray(x, dtype=float), d).tolist()


def _prune(obj, drop=("layer_curve", "concept_level_regression", "negative_control", "smds")):
    """Results JSON without per-layer family curves and large diagnostics (the site needs headline values)."""
    if isinstance(obj, dict):
        return {k: _prune(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_prune(v, drop) for v in obj]
    return obj


def export_model(cfg: Config, key: str, mp_key: dict) -> dict:
    Lp = mp_key["primary_layer"]
    setup = figdata.field_setup(cfg, mp_key)
    names = setup["names"]
    base = figdata.mds(setup["arxiv"])
    out = {"key": key, "name": S.MODEL.get(key, key), "primary_layer": Lp, "fields": {"order": setup["order"]}}
    # field map + distance matrix at every layer, aligned to the primary-layer layout (no flips when scrubbing)
    from mathgeom.analyses.map_fields import field_centroids
    from mathgeom.geometry.preprocess import cosine_rdm
    from mathgeom.pipeline import load_acts

    for side, k in (("trained", key), ("random_init", f"{key}__random")):
        X = load_acts(cfg, k, "field_neutral")["last"][setup["mask"]]
        rdms = [cosine_rdm(field_centroids(X[:, l], setup["labels"], names)) for l in range(1, X.shape[1])]
        ref = figdata.align(figdata.mds(rdms[Lp - 1]), base)
        out["fields"][side] = {"layout": [_r(figdata.align(figdata.mds(D), ref)) for D in rdms], "rank": [_r(figdata.rank_unit(D), 3) for D in rdms]}
        pts = figdata.pca(X[:, Lp], 3)
        out.setdefault("concepts", {})[side] = {"xyz": _r(pts, 3), "nn": figdata.concept_neighbours(X[:, Lp], 10).tolist()}
    # families: PC planes at every layer, aligned to the pre-registered layer; the twin's are also rotated
    # onto the trained plane so the trained ↔ twin morph is a minimal motion
    chains = load_yaml("families.yaml")["chains"]
    out["families"] = {}
    for f in SHOW_FAMILIES:
        tr = figdata.planes_by_layer(figdata.family_acts(cfg, key, f), Lp)
        tw = figdata.planes_by_layer(figdata.family_acts(cfg, f"{key}__random", f), Lp, onto=tr[Lp - 1])
        out["families"][f] = {"items": figdata.family_items(cfg, f), "expected": chains[f]["expected"], "trained": _r(tr), "random_init": _r(tw)}
    # group-like lattice: per-layer planes, exact coordinates and the offset cosine matrix at the primary layer
    lat = load_yaml("families.yaml")["lattices"]["group_like"]
    items, axioms = list(lat["items"]), lat["axioms"]
    bits = np.array([lat["items"][c] for c in items])
    lt = {"items": items, "axioms": axioms, "bits": bits.tolist()}
    onto = None
    for side, k in (("trained", key), ("random_init", f"{key}__random")):
        X = figdata.family_acts(cfg, k, "group_like", items)
        om = figdata.offset_matrix(X[:, Lp], bits, axioms)
        planes = figdata.planes_by_layer(X, Lp, onto=onto)  # twin rotated onto the trained plane (morph)
        onto = planes[Lp - 1]
        lt[side] = {"planes": _r(planes), "exact": _r(figdata.exact_coords(X[:, Lp]), 5), "cos": _r(om["cos"])}
        lt["pairs"], lt["group"], lt["shared"] = [list(p) for p in om["pairs"]], om["group"].tolist(), om["shared"].astype(int).tolist()
    out["lattice"] = lt
    return out


def export_site(cfg: Config, mp: dict) -> None:
    """Per-model geometry files for this config's models, then the shared index (all configs so far)."""
    SITE.mkdir(parents=True, exist_ok=True)
    for key in [k for k in mp if not k.endswith("__random")]:
        (SITE / f"{key}.json").write_text(json.dumps(export_model(cfg, key, mp[key]), separators=(",", ":")))
    write_index(cfg)


def write_index(cfg: Config) -> None:
    from mathgeom.report.figures.overview import forest_rows
    from mathgeom.report.paper import _results
    from mathgeom.report.render import scorecard_rows

    r = _results("all")
    keys = [k for k in S.MODEL if k in r.get("map_fields", {}) and (SITE / f"{k}.json").exists()]
    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    first = r["map_fields"][keys[0]]
    setup = figdata.field_setup(cfg, first)
    mask = setup["mask"]
    ratings = pd.read_csv(cfg.data_dir / "judge_ratings.csv")
    judges = [c for c in ratings.columns if "/" in c]
    abstr = ratings.set_index("concept")[judges].mean(1)
    ml = pd.read_csv(cfg.data_dir / "mathlib_depth.csv").set_index("concept") if (cfg.data_dir / "mathlib_depth.csv").exists() else None
    cdf = concepts[mask]
    idx = {
        "models": [{"key": k, "name": S.MODEL[k], "primary_layer": r["map_fields"][k]["primary_layer"], "n_layers": len(r["map_fields"][k]["math_primary"]["layer_curve"]["trained"])} for k in keys],
        "fields": {"names": setup["names"], "labels": [S.FIELD.get(n, n) for n in setup["names"]], "pa": setup["pa"], "branch": setup["branch"],
                   "branch_labels": S.BRANCH, "order": setup["order"], "arxiv_rank": _r(figdata.rank_unit(setup["arxiv"])),
                   "arxiv_layout": _r(figdata.unit_rms(figdata.mds(setup["arxiv"]))), "edges": figdata.neighbour_edges(setup["arxiv"]), "n_papers": setup["n_papers"]},
        "concepts": {"concept": cdf.concept.tolist(), "field": cdf.field.tolist(), "abstraction": _r(cdf.concept.map(abstr).values, 2),
                     "mathlib": [None if ml is None or c not in ml.index or pd.isna(ml.loc[c, "depth"]) else {"decl": ml.loc[c, "decl"], "depth": int(ml.loc[c, "depth"])} for c in cdf.concept]},
        "mathlib_points": [] if ml is None else [{"concept": c, "field": S.FIELD.get(row.field, row.field), "decl": row.decl, "depth": int(row.depth), "rating": round(float(row.mean_rating), 3)}
                                                  for c, row in ml.dropna(subset=["depth"]).iterrows()],  # all matched concepts (the set the ρ is computed on)
        "family_labels": S.FAMILY,
        "scorecard": scorecard_rows(r),
        "forest": [{"title": t, "rows": rows} for t, rows in forest_rows(r)],
        "results": {k: _prune(v, ("layer_curve", "negative_control") if k in ("shapes", "lattice") else ("concept_level_regression",))
                    for k, v in r.items() if k not in ("numbers", "manifest", "sae", "smds", "calibration")},
    }
    for k in list(idx["results"].get("posthoc", {})):  # keep only the cycle scores the site plots
        pc = idx["results"]["posthoc"][k].get("pc_plane_cycle", {})
        idx["results"]["posthoc"][k] = {"pc_plane_cycle": {f: [round(e["score"], 3) for e in pc[f]] for f in ("days", "months", "angles") if f in pc}}
    (SITE / "index.json").write_text(json.dumps(idx, separators=(",", ":"), default=float))
