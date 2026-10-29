"""Secondary H-B analysis (plan): Supervised MDS shape fits with label-shuffled refits as the null.

Global context: SMDS (Tiblias et al. 2025, arXiv:2510.01025; package `smds`, Apache-2.0,
https://github.com/UKPLab/supervised-multidimensional-scaling) fits a linear projection whose
pairwise distances match a hypothesised template. With n ≤ 12 items in ≥1,024 dims a supervised
projection can fit *any* labelling, so — as the plan required — the observed fit is compared with
the same procedure refitted on shuffled labels; the fit itself is never interpreted alone.
Templates: `euclidean` (|i−j|, a line) and `discrete_circular` (cyclic distance). The package's
continuous `circular` template normalizes labels to [0, 1] and would place the first and last item
at the same point, so it is not used. Score = SMDS `score()` = 1 − scale-normalized stress.
"""

from __future__ import annotations

import warnings

import numpy as np


def smds_score(X: np.ndarray, y: np.ndarray, manifold: str) -> float:
    from smds import SupervisedMDS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SupervisedMDS(manifold=manifold, n_components=2)
        m.fit(X, y)
        return float(m.score(X, y))


def smds_family(X: np.ndarray, n_shuffle: int = 50, seed: int = 0) -> dict:
    """Observed SMDS fit per template vs its label-shuffled refit null (one-sided permutation p)."""
    X = X.astype(np.float64) - X.mean(0)
    y = np.arange(len(X))
    rng = np.random.default_rng(seed)
    out = {}
    for man in ("euclidean", "discrete_circular"):
        obs = smds_score(X, y, man)
        null = np.array([smds_score(X, rng.permutation(y), man) for _ in range(n_shuffle)])
        out[man] = {"score": obs, "null95": float(np.percentile(null, 95)), "p": float((1 + (null >= obs).sum()) / (1 + n_shuffle))}
    out["preferred"] = preferred(out)
    return out


def preferred(res: dict) -> str:
    """Template that beats its shuffled null (p < .05) by the larger margin; 'none' if neither does."""
    sig = [k for k in ("euclidean", "discrete_circular") if res[k]["p"] < 0.05]
    return max(sig, key=lambda k: res[k]["score"] - res[k]["null95"]) if sig else "none"
