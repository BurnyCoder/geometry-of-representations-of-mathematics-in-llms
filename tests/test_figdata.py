"""Figures must show exactly what the statistics measured: the shared figure-data preparation is checked
against the results JSON of every config (skipped when activations/results are absent).

Global context: an earlier report bug drew the confirmatory model's activations for the template-variant
config; these checks would have caught it (the variant's field ρ differs from the confirmatory one).
"""

import json

import numpy as np
import pytest
from scipy.stats import spearmanr

from mathgeom.config import ROOT, load_config, load_yaml
from mathgeom.report import figdata
from mathgeom.stats.rdm_regression import upper


def _ready(cfg_name: str) -> bool:
    cfg = load_config(f"configs/{cfg_name}.yaml")
    return (ROOT / "results" / cfg_name / "lattice.json").exists() and (cfg.artifacts / "acts" / "family_index.json").exists()


@pytest.mark.parametrize("cfg_name", ["mvp", "mvp_variants", "robustness", "gemma"])
def test_figure_data_reproduces_reported_statistics(cfg_name):
    if not _ready(cfg_name):
        pytest.skip("results or activations not present")
    from mathgeom.analyses.lattice import parallelograms
    from mathgeom.geometry.manifold import pc_plane_cycle_score
    from mathgeom.geometry.offsets import analogy_rank, offset_consistency

    cfg = load_config(f"configs/{cfg_name}.yaml")
    res = {f: json.loads((ROOT / "results" / cfg_name / f"{f}.json").read_text()) for f in ("map_fields", "lattice", "posthoc")}
    key = next(iter(res["map_fields"]))
    Lp = res["map_fields"][key]["primary_layer"]
    # field map: Spearman of the prepared centroid RDM with arXiv distance == reported ρ
    setup = figdata.field_setup(cfg, res["map_fields"][key])
    rdms = figdata.field_rdms(cfg, key, setup, Lp)
    assert spearmanr(upper(rdms["trained"]), upper(rdms["arxiv"]))[0] == pytest.approx(res["map_fields"][key]["math_primary"]["rsa"]["trained"], abs=1e-9)
    # lattice: within-axiom block means of the offset cosine matrix == reported offset consistency
    lat = load_yaml("families.yaml")["lattices"]["group_like"]
    items = list(lat["items"])
    bits = np.array([lat["items"][c] for c in items])
    X = figdata.family_acts(cfg, key, "group_like", items)[:, Lp]
    om = figdata.offset_matrix(X, bits, lat["axioms"])
    for a, name in enumerate(lat["axioms"]):
        blk = om["cos"][np.ix_(om["group"] == a, om["group"] == a)]
        assert blk[~np.eye(len(blk), dtype=bool)].mean() == pytest.approx(res["lattice"][key]["group_like"]["offsets"][name]["consistency"], abs=1e-9)
    assert not om["shared"][np.ix_(om["group"] == 0, om["group"] == 0)].any()  # same-axiom pairs never share an endpoint
    # exact (n−1)-D coordinates reproduce offsets and analogy ranks of the full vectors
    E = figdata.exact_coords(X)
    for a in range(len(lat["axioms"])):
        from mathgeom.analyses.lattice import axiom_pairs

        pr = axiom_pairs(bits, a)
        assert offset_consistency(E, pr) == pytest.approx(offset_consistency(X - X.mean(0), pr), abs=1e-9)
    cand = np.arange(len(items))
    assert [analogy_rank(E, *q, cand) for q in parallelograms(bits)] == [analogy_rank(X - X.mean(0), *q, cand) for q in parallelograms(bits)]
    # days: PC-plane cycle score of the prepared family == post-hoc result at the primary layer
    D = figdata.family_acts(cfg, key, "days")
    assert pc_plane_cycle_score(D[:, Lp], np.arange(len(D))) == pytest.approx(res["posthoc"][key]["pc_plane_cycle"]["days"][Lp - 1]["score"])
