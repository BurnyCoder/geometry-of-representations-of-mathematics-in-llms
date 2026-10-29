"""Regression on representational dissimilarity matrices (RDMs) with permutation inference.

Global context: H-A1 asks whether the LLM's concept RDM tracks a ground-truth field
distance *after* nuisance RDMs (lexical overlap, frequency, layer-0 embedding, ...)
are accounted for. Partial Mantel tests are known to inflate type-I error when
nuisance matrices are structured (Guillot & Rousset 2013, Methods Ecol. Evol.,
https://doi.org/10.1111/2041-210X.12018), so we use multiple regression on distance
matrices (Lichstein 2007, https://doi.org/10.1007/s11258-006-9126-3) with the
Freedman–Lane permutation scheme (Freedman & Lane 1983; compared in Anderson &
Legendre 1999, https://doi.org/10.1080/00949659908811936): fit the reduced
(nuisance-only) model, permute *concept labels* of its residual matrix (rows and
columns together, preserving the matrix dependence structure), add the fitted values
back, refit the full model and record the target t-statistic.

Local context: everything works on the upper triangle of n×n symmetric matrices;
values are rank-transformed (Spearman-style) and z-scored before OLS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def upper(m: np.ndarray) -> np.ndarray:
    """Return the strict upper triangle of a square matrix as a flat vector."""
    i, j = np.triu_indices(m.shape[0], k=1)  # row/col indices of off-diagonal pairs
    return m[i, j]


def _zrank(v: np.ndarray) -> np.ndarray:
    """Rank-transform then z-score a vector (Spearman-style robustness to monotone warps)."""
    r = rankdata(v)  # average ranks for ties
    s = r.std()
    return (r - r.mean()) / (s if s > 0 else 1.0)  # constant predictors stay all-zero


def _vec_to_mat(v: np.ndarray, n: int) -> np.ndarray:
    """Inverse of `upper`: rebuild a symmetric n×n matrix (zero diagonal) from a flat vector."""
    m = np.zeros((n, n), dtype=v.dtype)
    i, j = np.triu_indices(n, k=1)
    m[i, j] = v
    m[j, i] = v
    return m


@dataclass
class RDMRegressionResult:
    """Outcome of `rdm_regression` for the target predictor."""

    beta: float  # standardized coefficient of the target RDM in the full model
    t: float  # its t-statistic (pivotal; used for permutation inference)
    p: float  # Freedman–Lane permutation p-value (one-sided: beta > 0)
    n_perm: int
    betas: dict[str, float]  # all standardized coefficients (nuisances included)
    r2: float  # full-model R² on ranked upper-triangle values


def _ols_t(X: np.ndarray, Y: np.ndarray, col: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS of each column of Y on X; return (beta[col], t[col]) per column of Y.

    X: (m, k) design with intercept; Y: (m, B) batch of responses.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)  # (k, k); pinv tolerates collinear nuisances
    B = XtX_inv @ X.T @ Y  # (k, B) coefficients for the whole batch at once
    resid = Y - X @ B  # (m, B)
    dof = X.shape[0] - X.shape[1]
    sigma2 = (resid**2).sum(0) / dof  # residual variance per response
    se = np.sqrt(sigma2 * XtX_inv[col, col])  # standard error of the target coefficient
    return B[col], B[col] / se


def rdm_regression(
    target_rdm: np.ndarray,
    predictors: dict[str, np.ndarray],
    target: str,
    n_perm: int = 10_000,
    seed: int = 0,
    batch: int = 250,
) -> RDMRegressionResult:
    """Regress `target_rdm` (model RDM) on named predictor RDMs; test `target` by Freedman–Lane.

    Args:
        target_rdm: n×n model dissimilarity matrix (the "response").
        predictors: name → n×n matrix; must contain `target` plus any nuisances.
        target: name of the predictor of interest (e.g. "field_distance").
        n_perm: number of concept-label permutations of reduced-model residuals.
        seed: RNG seed (reproducibility).
        batch: permutations evaluated per vectorized OLS call (memory/speed trade-off).
    """
    n = target_rdm.shape[0]
    names = [target] + [k for k in predictors if k != target]  # target is column 1 (after intercept)
    y = _zrank(upper(target_rdm))
    cols = [_zrank(upper(predictors[k])) for k in names]
    m = y.shape[0]
    X_full = np.column_stack([np.ones(m)] + cols)  # intercept + target + nuisances
    X_red = np.column_stack([np.ones(m)] + cols[1:])  # intercept + nuisances only

    beta_obs, t_obs = _ols_t(X_full, y[:, None], col=1)
    all_b = np.linalg.pinv(X_full) @ y
    r2 = 1 - ((y - X_full @ all_b) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    # Freedman–Lane: reduced-model fit + residuals, held as matrices so we can relabel concepts.
    b_red = np.linalg.pinv(X_red) @ y
    fitted = X_red @ b_red
    resid_mat = _vec_to_mat(y - fitted, n)
    iu = np.triu_indices(n, k=1)

    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        Ystar = np.empty((m, b))
        for c in range(b):
            p = rng.permutation(n)  # relabel concepts: rows and columns move together
            Ystar[:, c] = fitted + resid_mat[np.ix_(p, p)][iu]
        _, t_perm = _ols_t(X_full, Ystar, col=1)
        exceed += int((t_perm >= t_obs[0]).sum())
        done += b

    return RDMRegressionResult(
        beta=float(beta_obs[0]),
        t=float(t_obs[0]),
        p=(1 + exceed) / (1 + n_perm),  # Phipson & Smyth 2010 correction, never exactly 0
        n_perm=n_perm,
        betas={k: float(v) for k, v in zip(names, all_b[1:])},
        r2=float(r2),
    )


def bootstrap_beta(
    target_rdm: np.ndarray,
    predictors: dict[str, np.ndarray],
    target: str,
    n_boot: int = 2000,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Percentile 95% CI of the target's standardized beta, resampling *concepts*.

    Pairs formed by a concept with its own duplicate are dropped (distance 0 is not data).
    If `groups` is given (e.g. field label per concept), resampling is stratified within
    groups so every bootstrap sample keeps all fields.
    """
    n = target_rdm.shape[0]
    rng = np.random.default_rng(seed)
    names = [target] + [k for k in predictors if k != target]
    out = np.empty(n_boot)
    for b in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:  # stratified resample: same count per group as observed
            idx = np.concatenate([rng.choice(np.flatnonzero(groups == g), (groups == g).sum()) for g in np.unique(groups)])
        i, j = np.triu_indices(len(idx), k=1)
        keep = idx[i] != idx[j]  # discard self-pairs created by duplicates
        a, c = idx[i][keep], idx[j][keep]
        y = _zrank(target_rdm[a, c])
        X = np.column_stack([np.ones(keep.sum())] + [_zrank(predictors[k][a, c]) for k in names])
        out[b] = (np.linalg.pinv(X) @ y)[1]
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), out
