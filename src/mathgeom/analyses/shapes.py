"""H-B: shape statistics for ordered concept families at one (model, layer).

Global context: for each family (configs/families.yaml) we compute unsupervised shape
statistics — line vs circle template ρ, Robinson score, closure ratio, PCA order |ρ| — plus
leave-one-out order ρ, each with a label-permutation null computed by the identical procedure
(manifold.py). For families with a number reference (n-gons, dimensions, angles) we also
regress the family RDM on the reference family's RDM to see what structure remains beyond the
underlying number code (Kantamneni & Tegmark 2025, arXiv:2502.00873).
"""

from __future__ import annotations

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
from mathgeom.geometry.preprocess import center, euclid_rdm
from mathgeom.stats.rdm_regression import rdm_regression, upper


def family_stats(X: np.ndarray, n_perm: int = 2000, seed: int = 0) -> dict:
    """All shape statistics for one family whose rows are in ground-truth order."""
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
    loo_null = permutation_null(lambda p: loo_order_rho(X, p), n, n_perm=min(n_perm, 300), seed=seed)
    out["loo_order"], out["loo_order_p"] = loo, perm_p(loo, loo_null)
    return out


def residual_beyond_reference(X_fam: np.ndarray, X_ref: np.ndarray, kind: str, n_perm: int = 5000, seed: int = 0) -> dict:
    """Does the family RDM fit its predicted template beyond the reference (number) RDM?

    Regress family RDM on [template RDM (target), reference RDM (nuisance)].
    """
    from mathgeom.geometry.manifold import template_rdm

    n = len(X_fam)
    D_f = euclid_rdm(X_fam)
    D_r = euclid_rdm(X_ref)
    res = rdm_regression(D_f, {"template": template_rdm(n, kind), "reference": D_r}, "template", n_perm=n_perm, seed=seed)
    from scipy.stats import spearmanr

    return {"beta_template": res.beta, "p": res.p, "beta_reference": res.betas["reference"], "rho_fam_ref": float(spearmanr(upper(D_f), upper(D_r))[0])}
