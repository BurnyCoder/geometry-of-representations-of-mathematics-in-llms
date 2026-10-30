"""Synthetic tests for shape and axiom-lattice statistics.

Global context: before touching LLM activations we require that each statistic
(i) recovers a planted circle / line / parallelogram embedded in high-dimensional
noise, and (ii) is NOT fooled by random labels — the n ≪ d overfitting trap.
"""

import numpy as np

from mathgeom.geometry.manifold import (
    closure_ratio,
    loo_order_rho,
    pca_order_rho,
    perm_p,
    permutation_null,
    robinson_score,
    template_fit,
)
from mathgeom.geometry.offsets import analogy_rank, offset_consistency, random_pair_null
from mathgeom.geometry.preprocess import all_but_the_top, char_ngram_rdm, euclid_rdm


def _embed(points2d, d=1024, noise=0.05, seed=0):
    """Embed 2D points into d dims via a random orthonormal map, plus isotropic noise."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    return points2d @ Q.T + noise * rng.normal(size=(len(points2d), d))


def test_circle_preferred_for_circle_and_line_for_line():
    n = 12
    th = 2 * np.pi * np.arange(n) / n
    # shapes must be large relative to the high-dim noise norm (≈ 0.05·√(2·1024) ≈ 2.3)
    Xc = _embed(5 * np.c_[np.cos(th), np.sin(th)])
    Xl = _embed(np.c_[np.arange(n), np.zeros(n)])
    o = np.arange(n)
    Dc, Dl = euclid_rdm(Xc), euclid_rdm(Xl)
    assert template_fit(Dc, o, "circle") > template_fit(Dc, o, "line")
    assert template_fit(Dl, o, "line") > template_fit(Dl, o, "circle")
    assert closure_ratio(Dc, o) < 2 < closure_ratio(Dl, o)


def test_robinson_and_pca_order_detect_line_not_noise():
    n = 11
    o = np.arange(n)
    X = _embed(np.c_[np.arange(n), 0.3 * np.sin(np.arange(n) / 3)], noise=0.1)  # gently curved arc
    D = euclid_rdm(X)
    assert robinson_score(D, o) > 0.8
    assert pca_order_rho(X, o) > 0.9
    null = permutation_null(lambda p: robinson_score(D, p), n, n_perm=300)
    assert perm_p(robinson_score(D, o), null) < 0.01


def test_loo_order_not_fooled_by_pure_noise():
    """With n=11, d=1024 pure noise, a supervised fit would 'find' any order; LOO must not."""
    rng = np.random.default_rng(5)
    rhos = [loo_order_rho(rng.normal(size=(11, 1024)), np.arange(11)) for _ in range(40)]
    assert np.mean(rhos) < 0.3


def test_offset_consistency_and_analogy_on_planted_cube():
    rng = np.random.default_rng(7)
    d = 512
    axes = rng.normal(size=(3, d))  # three "axiom" directions
    bits = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])  # 8 cube vertices
    X = bits @ axes + 0.3 * rng.normal(size=(8, d))
    pairs_ax0 = [(i, j) for i in range(8) for j in range(8) if (bits[j] - bits[i]).tolist() == [1, 0, 0]]
    obs = offset_consistency(X, pairs_ax0)
    null = random_pair_null(X, np.arange(8), len(pairs_ax0), n_perm=300)
    assert obs > np.percentile(null, 99)
    # 000 -> 100 :: 010 -> 110
    idx = {tuple(b): k for k, b in enumerate(bits.tolist())}
    r = analogy_rank(X, idx[(0, 0, 0)], idx[(1, 0, 0)], idx[(0, 1, 0)], idx[(1, 1, 0)], np.arange(8))
    assert r == 1


def test_abtt_and_char_ngrams():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 200)) + 50 * rng.normal(size=(1, 200))  # big shared offset
    Y = all_but_the_top(X, 2)
    assert np.allclose(Y.mean(0), 0, atol=1e-8)
    assert all_but_the_top(X.astype(np.float16), 2).dtype == np.float64  # stored activations are float16
    D = char_ngram_rdm(["group", "semigroup", "quasigroup", "manifold"])
    assert D[0, 1] < D[0, 3] and D[1, 2] < D[1, 3]


def test_pc_plane_cycle_score_detects_planted_cycle_only():
    from mathgeom.geometry.manifold import pc_plane_cycle_score

    n = 12
    th = 2 * np.pi * np.arange(n) / n
    X = _embed(5 * np.c_[np.cos(th), np.sin(th)])
    assert pc_plane_cycle_score(X, np.arange(n)) == 1.0
    rng = np.random.default_rng(3)
    assert np.mean([pc_plane_cycle_score(X, rng.permutation(n)) for _ in range(50)]) < 0.4


def test_cycle_score_default_plane_is_the_pc1_pc2_score():
    """plane=(0, 1) is the original post-hoc statistic (existing results must stay byte-identical)."""
    from mathgeom.geometry.manifold import pc_plane_cycle_score

    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 64))
    for o in (np.arange(12), rng.permutation(12)):
        assert pc_plane_cycle_score(X, o) == pc_plane_cycle_score(X, o, plane=(0, 1))


def test_best_plane_cycle_finds_a_cycle_hidden_in_pc3_pc4():
    """Addendum test (reports/prereg_addendum_3d.md): a line dominates PC1-2, a cycle lives in PC3-4."""
    from mathgeom.geometry.manifold import best_plane_cycle, pc_plane_cycle_score

    n = 12
    th = 2 * np.pi * np.arange(n) / n
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.normal(size=(256, 4)))
    # item-space components made orthogonal (circle first), so the circle is exactly PC3-PC4 and the
    # ramp (PC1) and a second distractor (PC2) carry more variance than it
    C, _ = np.linalg.qr(np.c_[np.cos(th), np.sin(th), np.arange(n) - (n - 1) / 2, np.sin(np.arange(n) * 2.1)])
    X = np.c_[2 * C[:, 0], 2 * C[:, 1], 8 * C[:, 2], 5 * C[:, 3]] @ Q.T + 0.005 * rng.normal(size=(n, 256))
    assert pc_plane_cycle_score(X, np.arange(n)) < 1.0  # the PC1-2 plane does not show it
    r = best_plane_cycle(X, np.arange(n), k=5, n_perm=1000, rng=np.random.default_rng(0))
    assert r["score"] == 1.0 and set(r["plane"]) == {2, 3} and r["p"] < 0.01
    assert len(r["planes"]) == 10


def test_best_plane_cycle_rejects_an_open_arc():
    """Addendum amendment 1: a horseshoe (a line bent into a U by PCA) has a perfect angular order but is
    not a closed loop, so its closed-cycle score is 0 in every plane that does not close."""
    from mathgeom.geometry.manifold import best_plane_cycle

    n = 12
    t = np.linspace(-1, 1, n)
    X = _embed(np.c_[4 * t, 3 * t**2])  # U-shaped arc in its top-2 PC plane
    r = best_plane_cycle(X, np.arange(n), k=5, n_perm=500, rng=np.random.default_rng(0))
    pc12 = next(p for p in r["planes"] if p[:2] == [1, 2])
    assert pc12[2] == 1.0 and pc12[3] > 2  # perfect angular order, open loop
    assert r["p"] > 0.05


def test_best_plane_cycle_is_valid_on_structureless_data():
    """The max-over-planes null keeps the false-positive rate at or below 5% (it corrects for picking a
    plane). Addendum amendment 2: the closure rule puts mass at score 0, so the test is conservative
    (1,000 null simulations: 1.8% at n = 12, 4.0% at n = 7); validity, not exact size, is what we check."""
    from mathgeom.geometry.manifold import best_plane_cycle

    rng = np.random.default_rng(6)
    for n in (7, 12):
        ps = [best_plane_cycle(rng.normal(size=(n, 64)), np.arange(n), k=5, n_perm=200, rng=rng)["p"] for _ in range(300)]
        assert np.mean(np.array(ps) < 0.05) <= 0.08, n


def test_smds_needs_shuffle_null_and_then_picks_right_template():
    """Raw SMDS fit can favour the wrong template; the label-shuffled null must rescue the decision."""
    from mathgeom.analyses.smds_fit import smds_family

    n = 12
    th = 2 * np.pi * np.arange(n) / n
    res = smds_family(_embed(5 * np.c_[np.cos(th), np.sin(th)], d=256, noise=0.1), n_shuffle=39)  # min p = 1/40 < .05
    assert res["preferred"] == "discrete_circular" and res["discrete_circular"]["p"] < 0.05 and res["euclidean"]["p"] > 0.05


def test_order_beyond_covariates_credits_real_order_not_frequency():
    """Plan H-B frequency control: an order carried by the representation survives partialling unrelated
    covariates; an 'order' that is only a frequency code does not."""
    from mathgeom.analyses.shapes import order_beyond_covariates

    rng = np.random.default_rng(3)
    t = np.arange(11.0)
    line = _embed(np.c_[t, np.zeros(11)], noise=0.05, seed=3)
    real = order_beyond_covariates(line, rng.normal(size=(11, 2)), n_perm=200, seed=0)
    assert real["loo_order_beyond"] > 0.8 and real["p"] < 0.05
    freq = t + rng.normal(scale=0.5, size=11)  # frequency rises along the chain
    freq_only = _embed(np.c_[freq, np.zeros(11)], noise=0.05, seed=4)  # representation encodes frequency only
    fake = order_beyond_covariates(freq_only, np.c_[freq, rng.normal(size=11)], n_perm=200, seed=0)
    assert fake["p"] > 0.05
    # large-scale noise (like a random-init twin): residualizing on all items at once leaked the held-out
    # item into the training span and gave ρ ≈ 1 for every order; fold-wise residualization must not
    big = order_beyond_covariates(300 * rng.normal(size=(11, 256)), np.c_[freq, rng.normal(size=11)], n_perm=200, seed=0)
    assert big["p"] > 0.05 and big["null95"] < 0.99
