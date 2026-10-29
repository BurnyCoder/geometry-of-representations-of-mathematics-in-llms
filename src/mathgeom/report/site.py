"""Export data for the interactive GitHub Pages explorer (site/index.html + site/data.json).

Global context: the explorer lets readers rotate/hover the same geometry the statistics are
computed on — field-centroid maps (arXiv vs trained vs random-init), the concept cloud, the
ordered families and the group-like axiom lattice — for the trained model and its random-init
twin side by side. All coordinates are PCA/MDS projections computed here from cached
activations at the pre-registered layer; all numbers shown come from results/<cfg>/*.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mathgeom.config import ROOT, Config, load_yaml


def _pca(X: np.ndarray, k: int) -> np.ndarray:
    X = X.astype(np.float64)
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return (X @ Vt[:k].T).round(4)


def export_site(cfg: Config, mp: dict, sh: dict, la: dict) -> None:
    from mathgeom.analyses.lattice import axiom_pairs
    from mathgeom.analyses.map_fields import field_centroids
    from mathgeom.geometry.preprocess import cosine_rdm
    from mathgeom.report.render import _mds

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    fields_cfg = {f["id"]: f for f in load_yaml("fields.yaml")["math"]}
    fams = load_yaml("families.yaml")
    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    data = {"config": cfg.name, "models": {}}
    split = cfg.raw.get("analysis_split", "confirmatory")
    for key, res in mp.items():
        Lp = res["primary_layer"]
        d = {"primary_layer": Lp, "results": {"map": res, "shapes": sh.get(key), "lattice": la.get(key)}}
        mask = ((concepts.kind == "math") & concepts.primary & (concepts.split == split)).values
        lab = concepts.field.values[mask]
        names = res["math_primary"]["fields"]
        sel = [fd["names"].index(n) for n in names]
        maps = {}
        for k, X in (("trained", np.load(cfg.artifacts / "acts" / key / "field_neutral.npz")["last"]), ("random_init", np.load(cfg.artifacts / "acts" / f"{key}__random" / "field_neutral.npz")["last"])):
            maps[k] = _mds(cosine_rdm(field_centroids(X[mask, Lp], lab, names))).round(4).tolist()
            pts = _pca(X[mask, Lp], 3)
            d[f"concepts_{k}"] = {"xyz": pts.tolist(), "concept": concepts.concept.values[mask].tolist(), "field": lab.tolist()}
        maps["arxiv"] = _mds(np.array(fd["distance"])[np.ix_(sel, sel)]).round(4).tolist()
        d["field_maps"] = {"names": names, "pa": [fields_cfg[n]["pa"] for n in names], "branch": [fields_cfg[n]["branch"] for n in names], **maps}
        fam_out = {}
        for k in (key, f"{key}__random"):
            acts = np.load(cfg.artifacts / "acts" / k / "family.npz")["last"]
            for name in list(fams["chains"]) + ["group_like"]:
                idx = [i for i, (f, _, _) in enumerate(fam) if f == name]
                fam_out.setdefault(name, {"items": [fam[i][1] for i in idx]})["trained" if k == key else "random_init"] = _pca(acts[idx, Lp], 3).tolist()
        lat = fams["lattices"]["group_like"]
        names_l = list(lat["items"])
        bits = np.array([lat["items"][c] for c in names_l])
        order = [fam_out["group_like"]["items"].index(c) for c in names_l]
        for k in ("trained", "random_init"):
            fam_out["group_like"][k] = [fam_out["group_like"][k][i] for i in order]
        fam_out["group_like"]["items"] = names_l
        fam_out["group_like"]["edges"] = [{"axiom": a, "pairs": axiom_pairs(bits, i)} for i, a in enumerate(lat["axioms"])]
        for name, f in fams["chains"].items():
            fam_out[name]["expected"] = f["expected"]
        d["families"] = fam_out
        data["models"][key] = d
    out = ROOT / "site"
    out.mkdir(exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, default=float))
