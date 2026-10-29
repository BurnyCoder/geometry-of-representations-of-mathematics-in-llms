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
    D = char_ngram_rdm(["group", "semigroup", "quasigroup", "manifold"])
    assert D[0, 1] < D[0, 3] and D[1, 2] < D[1, 3]
