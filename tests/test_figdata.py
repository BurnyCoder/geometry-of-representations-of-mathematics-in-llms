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


def test_morph_alignment_is_display_only():
    """Rotating the twin's planes onto the trained plane (site morph) changes no distance, only the pose."""
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(0)
    X, onto = rng.normal(size=(9, 5, 40)), rng.normal(size=(9, 2))
    plain, moved = figdata.planes_by_layer(X, 2), figdata.planes_by_layer(X, 2, onto=onto)
    for a, b in zip(plain, moved):
        np.testing.assert_allclose(pdist(a), pdist(b), atol=1e-9)
    # and the moved reference plane is closer to `onto` than the unmoved one
    target = figdata.unit_rms(onto)
    assert np.linalg.norm(moved[1] - target) <= np.linalg.norm(plain[1] - target) + 1e-9


# ------------------------------------------------------------------ 3-D views (display only)
def test_pca_variance_matches_sklearn():
    """Variance shown on every PCA view == sklearn's explained_variance_ratio_ (ground truth)."""
    from sklearn.decomposition import PCA

    X = np.random.default_rng(1).normal(size=(13, 60)) @ np.diag(np.linspace(3, 0.1, 60))
    np.testing.assert_allclose(figdata.pca_var(X, 5), PCA(n_components=5).fit(X).explained_variance_ratio_, atol=1e-9)


@pytest.mark.parametrize("k", [3, 5])
def test_kd_planes_keep_each_layers_geometry(k):
    """k-D layer views (aligned to the reference layer, optionally onto the trained view) are rotations of
    that layer's own top-k PCA scores: pairwise distances agree up to the unit-RMS scale."""
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(2)
    X, onto = rng.normal(size=(11, 4, 30)), rng.normal(size=(11, k))
    P = figdata.planes_by_layer(X, 2, onto=onto, k=k)
    assert P.shape == (3, 11, k)
    for l in range(1, 4):
        np.testing.assert_allclose(pdist(P[l - 1]), pdist(figdata.unit_rms(figdata.pca(X[:, l], k))), atol=1e-9)


def test_3d_pca_views_keep_true_axes():
    """3-D PCA views: each axis is that layer's own PCk (up to sign and a common display scale), so the
    variance printed per axis is exact; signs agree with the reference layer / the trained view."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 4, 40)) @ np.diag(np.linspace(3, 0.2, 40))
    V = figdata.pcs_by_layer(X, 2, k=5)
    for l in range(1, 4):
        P = figdata.pca(X[:, l], 5)
        ratio = V[l - 1] / P  # column-wise ±c with one common c > 0
        c = np.abs(ratio).mean()
        np.testing.assert_allclose(np.abs(ratio), c, rtol=1e-9)
        # per-axis variance of the view is proportional to that layer's explained variance
        np.testing.assert_allclose(V[l - 1].var(0) / V[l - 1].var(0).sum(), figdata.pca_var(X[:, l], 5) / figdata.pca_var(X[:, l], 5).sum(), rtol=1e-9)
    ref = figdata.pca(X[:, 2], 5)
    assert ((V[0] * ref).sum(0) >= 0).all()  # signs aligned to the reference layer
    W = figdata.pcs_by_layer(X[::-1], 2, onto=V[1][::-1], k=5)  # a "twin" aligned onto another view
    assert ((W[1] * V[1][::-1]).sum(0) >= 0).all()


def test_mds_3d_and_stress():
    from scipy.spatial.distance import pdist, squareform

    pts = np.random.default_rng(3).normal(size=(17, 3))
    D = squareform(pdist(pts))
    assert figdata.mds(D, k=3).shape == (17, 3)
    assert figdata.stress1(5 * pts, D) == pytest.approx(0.0, abs=1e-12)  # same distances up to scale: no stress
    assert figdata.stress1(figdata.mds(D, k=3), D) < figdata.stress1(figdata.mds(D, k=2), D)


def test_ideal_fit_recovers_a_rotated_circle():
    """The ideal template (circle in a plane of the k-D view) fitted by rotation onto a noiseless rotated
    circle reproduces it, and is unit-RMS like every view."""
    from scipy.stats import ortho_group

    t = 2 * np.pi * np.arange(7) / 7
    P = figdata.unit_rms(np.c_[np.cos(t), np.sin(t), np.zeros((7, 3))] @ ortho_group.rvs(5, random_state=4))
    fit = figdata.ideal_fit(P, closed=True)
    np.testing.assert_allclose(fit, P, atol=1e-9)
    line = figdata.ideal_fit(P, closed=False)
    assert line.shape == P.shape and np.sqrt((line**2).sum(1).mean()) == pytest.approx(1.0)


def test_view_map_projects_exact_coordinates_onto_the_view():
    """The 5x5 map from exact (PC-basis) coordinates to the displayed view reproduces the view at the
    reference layer exactly, so b - a + c can be drawn in the 3-D lattice without approximation."""
    rng = np.random.default_rng(5)
    X, onto = rng.normal(size=(13, 3, 50)), rng.normal(size=(13, 5))
    view = figdata.pcs_by_layer(X, 1, onto=onto, k=5)[0]
    E = figdata.exact_coords(X[:, 1])
    M = figdata.view_map(E, view)
    np.testing.assert_allclose(E[:, :5] @ M, view, atol=1e-9)


@pytest.mark.skipif(not _ready("mvp"), reason="results or activations not present")
def test_view_variance_matches_the_figures():
    """results/<cfg>/views.json (read by captions) reports the same variance shares the site and figures show."""
    cfg = load_config("configs/mvp.yaml")
    key = "qwen3.5-2b"
    v = figdata.view_variance(cfg, [key])[key]
    d = json.loads((ROOT / "site" / "data" / f"{key}.json").read_text())
    Lp = d["primary_layer"]
    np.testing.assert_allclose(v["angles"]["var5"][Lp - 1], d["families"]["angles"]["var"]["trained"][Lp - 1], atol=1e-4)
    np.testing.assert_allclose(v["group_like"]["var5"][Lp - 1], d["lattice"]["trained"]["var"][Lp - 1], atol=1e-4)
    assert v["angles"]["cum3"][Lp - 1] == pytest.approx(sum(v["angles"]["var5"][Lp - 1][:3]), abs=1e-3)
