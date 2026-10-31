"""H-B: shape statistics for ordered concept families at one (model, layer).

Global context: for each family (configs/families.yaml) we compute unsupervised shape
statistics — line vs circle template ρ, Robinson score, closure ratio, PCA order |ρ| — plus
leave-one-out order ρ, each with a label-permutation null computed by the identical procedure
(manifold.py) — except the closure ratio, which the pre-registered rules use as a fixed threshold (< 2). For families with a number reference (n-gons, dimensions, angles) we also
regress the family RDM on the reference family's RDM to see what structure remains beyond the
underlying number code (Kantamneni & Tegmark 2025, arXiv:2502.00873).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from mathgeom.geometry.manifold import (
    closure_ratio,
    loo_order_rho,
    pca_order_rho,
    perm_p,
    permutation_null,
    robinson_score,
    template_fit,
)
from mathgeom.geometry.preprocess import center, euclid_rdm
from mathgeom.stats.rdm_regression import rdm_regression, upper


def family_stats(X: np.ndarray, n_perm: int = 2000, seed: int = 0, n_perm_loo: int | None = None) -> dict:
    """All shape statistics for one family whose rows are in ground-truth order.

    The LOO-order null has min(n_perm, 300) permutations unless `n_perm_loo` is given; the pre-registered
    layer uses 10,000 (independent audit, reports/prereg_addendum_lexical.md D9; the kernel-form ridge made it
    affordable), and then the scale-relative-α sensitivity variant (D10) is added as loo_order_rel."""
    X = center(X.astype(np.float64))
    D = euclid_rdm(X)
    n = len(X)
    o = np.arange(n)
    out = {"n": n}
    stats = {
        "line_rho": lambda p: template_fit(D, p, "line"),
        "circle_rho": lambda p: template_fit(D, p, "circle"),
        "robinson": lambda p: robinson_score(D, p),
        "pca_order": lambda p: pca_order_rho(X, p),
    }
    for name, f in stats.items():
        obs = f(o)
        null = permutation_null(f, n, n_perm=n_perm, seed=seed)
        out[name] = obs
        out[name + "_p"] = perm_p(obs, null)
        out[name + "_null95"] = float(np.percentile(null, 95))
    out["closure"] = closure_ratio(D, o)
    # POST-HOC (docs/deviations.md): d(first,last)/d(first,middle) — <1 for a closed loop, >1 for an
    # open chain; unlike `closure` it is not ≈1 for structureless (equidistant) points by default
    out["closure_far"] = float(D[0, -1] / D[0, n // 2])
    out["closure_far_null95"] = float(np.percentile(permutation_null(lambda p: D[p[0], p[-1]] / D[p[0], p[n // 2]], n, n_perm=n_perm or 1, seed=seed), 5))
    loo = loo_order_rho(X, o)
    loo_null = permutation_null(lambda p: loo_order_rho(X, p), n, n_perm=n_perm_loo or min(n_perm, 300), seed=seed)
    out["loo_order"], out["loo_order_p"] = loo, perm_p(loo, loo_null)
    if n_perm_loo:  # D10 sensitivity: ridge penalty relative to the data scale (α = mean squared norm of the training items)
        rel = loo_order_rho(X, o, alpha="rel")
        out["loo_order_rel"], out["loo_order_rel_p"] = rel, perm_p(rel, permutation_null(lambda p: loo_order_rho(X, p, alpha="rel"), n, n_perm=n_perm_loo, seed=seed))
    return out


def order_beyond_covariates(X: np.ndarray, Z: np.ndarray, n_perm: int = 300, seed: int = 0) -> dict:
    """Plan H-B frequency control: is the chain order recoverable beyond per-item covariates Z
    (log-frequency, token count)?

    Cross-validated confound regression (Snoek, Miletić & Scholte 2019, NeuroImage 184:741,
    doi:10.1016/j.neuroimage.2018.09.074): in each leave-one-out fold, the representation and the ranks
    are regressed on [1, Z] using the training items only, and the held-out item is residualized with
    those coefficients; a ridge fit on the training residuals predicts the held-out residual rank. The
    statistic is the Spearman ρ of held-out predictions with held-out residual ranks, against an
    order-permutation null. Residualizing on all items at once leaks the held-out item into the
    training span and made the statistic ≈ 1 for any order in the random-init twin (docs/deviations.md).
    Simulation: 5% / 2.5% rejections under a frequency-only and a large-scale-noise null; 100% power
    when X encodes the order, also when Z is a noisy copy of the order.
    """
    X = center(X.astype(np.float64))
    n = len(X)
    Z1 = np.c_[np.ones(n), Z]

    def stat(p):  # p = order of items; rank[item] = position
        rank = np.empty(n)
        rank[p] = np.arange(n)
        pred, true = np.empty(n), np.empty(n)
        for i in range(n):
            tr = np.arange(n) != i
            R = np.c_[X, rank] - Z1 @ np.linalg.lstsq(Z1[tr], np.c_[X[tr], rank[tr]], rcond=None)[0]  # fold-wise confound fit
            pred[i], true[i] = Ridge(alpha=10.0).fit(R[tr, :-1], R[tr, -1]).predict(R[i : i + 1, :-1])[0], R[i, -1]
        return float(spearmanr(pred, true)[0])

    obs = stat(np.arange(n))
    null = permutation_null(stat, n, n_perm=n_perm, seed=seed)
    return {"loo_order_beyond": obs, "p": perm_p(obs, null), "null95": float(np.percentile(null, 95)),
            "rho_rank_covariates": [float(spearmanr(np.arange(n), Z[:, k])[0]) for k in range(Z.shape[1])]}


def residual_beyond_reference(X_fam: np.ndarray, X_ref: np.ndarray, kind: str, n_perm: int = 5000, seed: int = 0) -> dict:
    """Does the family RDM fit its predicted template beyond the reference (number) RDM?

    Regress family RDM on [template RDM (target), reference RDM (nuisance)].
    """
    from mathgeom.geometry.manifold import template_rdm

    n = len(X_fam)
    D_f = euclid_rdm(X_fam)
    D_r = euclid_rdm(X_ref)
    res = rdm_regression(D_f, {"template": template_rdm(n, kind), "reference": D_r}, "template", n_perm=n_perm, seed=seed)
    return {"beta_template": res.beta, "p": res.p, "beta_reference": res.betas["reference"], "rho_fam_ref": float(spearmanr(upper(D_f), upper(D_r))[0])}
