"""Tests for RDM regression: power on planted structure, calibration under a structured null.

Global context: the field-map hypothesis (H-A1) is only meaningful if the permutation
test is calibrated when nuisance RDMs are themselves structured and correlated with
the target — the failure mode of partial Mantel tests (Guillot & Rousset 2013).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kstest

from mathgeom.stats.rdm_regression import bootstrap_beta, rdm_regression


def _block_rdm(labels):
    """Same-group → 0, different-group → 1 (a categorical 'field' distance)."""
    return (labels[:, None] != labels[None, :]).astype(float)


def _structured_nuisance(rng, n, labels, corr_strength=0.8):
    """A continuous nuisance RDM partly aligned with the field labels (like frequency)."""
    z = corr_strength * labels + rng.normal(size=n)
    return squareform(pdist(z[:, None]))


def test_detects_planted_field_structure():
    rng = np.random.default_rng(1)
    n, d = 60, 50
    labels = np.repeat(np.arange(6), 10)
    centers = rng.normal(size=(6, d)) * 1.5
    X = centers[labels] + rng.normal(size=(n, d))  # points cluster by field
    model_rdm = squareform(pdist(X, "cosine"))
    nuis = _structured_nuisance(rng, n, labels)
    res = rdm_regression(model_rdm, {"field": _block_rdm(labels), "nuis": nuis}, "field", n_perm=500, seed=0)
    assert res.beta > 0.2
    assert res.p < 0.01


def test_freedman_lane_calibrated_when_nuisance_is_correctly_specified():
    """Model RDM = nuisance RDM + pair noise; nuisance correlated with field labels.

    With a correctly specified nuisance the Freedman–Lane test should hold ≈5% type-I error.
    (With a MISspecified nuisance — item-level noise, nonlinear link — our simulations showed
    15–22% rejections; that is why H-A1's primary test is a paired trained-vs-random-init
    contrast, not this regression. See reports/.)
    """
    rng = np.random.default_rng(2)
    n = 40
    labels = np.repeat(np.arange(4), 10)
    pvals = []
    for sim in range(120):
        z = 0.8 * labels + rng.normal(size=n)
        nuis = squareform(pdist(z[:, None]))
        noise = rng.normal(size=(n, n)) * 0.5
        model = nuis + (noise + noise.T) / 2
        np.fill_diagonal(model, 0)
        pvals.append(rdm_regression(model, {"field": _block_rdm(labels), "nuis": nuis}, "field", n_perm=199, seed=sim).p)
    pvals = np.array(pvals)
    assert (pvals < 0.05).mean() < 0.12
    assert kstest(pvals, "uniform").pvalue > 0.001


def test_field_level_mantel_is_calibrated_under_pure_null():
    from mathgeom.analyses.map_fields import field_rsa, mantel_field_perm, field_centroids
    from mathgeom.geometry.preprocess import cosine_rdm
    from mathgeom.stats.rdm_regression import upper

    rng = np.random.default_rng(4)
    F, per, d = 12, 15, 32
    lab = np.repeat(np.arange(F), per).astype(str)
    names = list(np.unique(lab))
    pvals = []
    for _ in range(150):
        target = squareform(pdist(rng.normal(size=(F, 3))))
        X = rng.normal(size=(F * per, d))  # unrelated to fields
        obs = field_rsa(X, lab, names, target)
        null = mantel_field_perm(upper(cosine_rdm(field_centroids(X, lab, names))), target, 199, rng)
        pvals.append((1 + (null >= obs).sum()) / 200)
    assert (np.array(pvals) < 0.05).mean() < 0.1


def test_bootstrap_ci_excludes_zero_with_signal():
    rng = np.random.default_rng(3)
    n, d = 40, 30
    labels = np.repeat(np.arange(4), 10)
    X = rng.normal(size=(4, d))[labels] * 2 + rng.normal(size=(n, d))
    model = squareform(pdist(X, "cosine"))
    lo, hi, _ = bootstrap_beta(model, {"field": _block_rdm(labels)}, "field", n_boot=200, seed=0, groups=labels)
    assert lo > 0
