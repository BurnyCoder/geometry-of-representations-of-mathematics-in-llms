"""Shape tests for ordered concept families (H-B): line/arc vs circle, order recovery.

Global context: Engels et al. 2024 (arXiv:2405.14860) found circles for days/months;
Karkada et al. 2026 (arXiv:2602.15029) predict ordered latent variables give 1D curved
manifolds. With n ≤ 12 points in d ≥ 1536 dims any supervised linear map can impose
any order, so every statistic here is either unsupervised (PCA, RDM-vs-template
correlation, Robinson/closure scores) or leave-one-out, and is compared against a
label-permutation null computed with exactly the same procedure.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from mathgeom.stats.rdm_regression import upper


def template_rdm(n: int, kind: str) -> np.ndarray:
    """Ideal dissimilarity for positions 0..n-1: 'line' = |i−j|, 'circle' = cyclic distance."""
    i = np.arange(n)
    lin = np.abs(i[:, None] - i[None, :]).astype(float)
    if kind == "line":
        return lin
    if kind == "circle":
        return np.minimum(lin, n - lin)
    raise ValueError(kind)


def template_fit(D: np.ndarray, order: np.ndarray, kind: str) -> float:
    """Spearman ρ between the observed RDM (items in true order) and a template RDM."""
    Do = D[np.ix_(order, order)]
    return float(spearmanr(upper(Do), upper(template_rdm(len(order), kind)))[0])


def robinson_score(D: np.ndarray, order: np.ndarray) -> float:
    """Fraction of triples i<j<k (in true order) with D[i,k] ≥ max(D[i,j], D[j,k]).

    A perfect 1D ordering (Robinson matrix) gives 1; random labels give ≈ 1/3.
    """
    Do = D[np.ix_(order, order)]
    trip = list(combinations(range(len(order)), 3))
    ok = [Do[i, k] >= max(Do[i, j], Do[j, k]) for i, j, k in trip]
    return float(np.mean(ok))


def closure_ratio(D: np.ndarray, order: np.ndarray) -> float:
    """d(first, last) / mean adjacent distance. ≈1 for a closed circle; large for an open arc/line."""
    Do = D[np.ix_(order, order)]
    adj = np.mean([Do[t, t + 1] for t in range(len(order) - 1)])
    return float(Do[0, -1] / adj)


def pca_order_rho(X: np.ndarray, order: np.ndarray) -> float:
    """|Spearman ρ| between true rank and projection on the first principal component (unsupervised)."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[0]
    rank = np.empty(len(order))
    rank[order] = np.arange(len(order))  # rank[item] = its position in the true order
    return float(abs(spearmanr(proj, rank)[0]))


def loo_order_rho(X: np.ndarray, order: np.ndarray, alpha: float = 10.0) -> float:
    """Leave-one-out ridge prediction of rank from representation; Spearman ρ of held-out predictions."""
    n = len(order)
    rank = np.empty(n)
    rank[order] = np.arange(n)
    pred = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        m = Ridge(alpha=alpha).fit(X[tr], rank[tr])
        pred[i] = m.predict(X[i : i + 1])[0]
    return float(spearmanr(pred, rank)[0])


def permutation_null(
    stat: Callable[[np.ndarray], float], n: int, n_perm: int = 2000, seed: int = 0
) -> np.ndarray:
    """Distribution of `stat(order)` over random orders (same procedure as the observed statistic)."""
    rng = np.random.default_rng(seed)
    return np.array([stat(rng.permutation(n)) for _ in range(n_perm)])


def perm_p(observed: float, null: np.ndarray) -> float:
    """One-sided permutation p-value with the +1 correction (Phipson & Smyth 2010)."""
    return float((1 + (null >= observed).sum()) / (1 + len(null)))


def pc_plane_cycle_score(X: np.ndarray, order: np.ndarray, plane: tuple[int, int] = (0, 1)) -> float:
    """POST-HOC: fraction of items whose angular successor in a PC plane (default the top-2-PC plane)
    is the true next item (cyclically; best of the two directions). 1 = perfect cycle in the plane, as
    in the 2-D projections of Engels et al. 2024 (arXiv:2405.14860); ≈ 1/(n-1)·2 for random arrangements."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[list(plane)].T
    n = len(order)
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)  # pos[item] = index in the true cyclic order
    ang_order = np.argsort(np.arctan2(P[:, 1], P[:, 0]))
    succ = {ang_order[i]: ang_order[(i + 1) % n] for i in range(n)}
    fwd = np.mean([pos[succ[i]] == (pos[i] + 1) % n for i in range(n)])
    bwd = np.mean([pos[succ[i]] == (pos[i] - 1) % n for i in range(n)])
    return float(max(fwd, bwd))


def best_plane_cycle(X: np.ndarray, order: np.ndarray, k: int = 5, n_perm: int = 1000, rng: np.random.Generator | None = None,
                     max_closure: float = 2.0) -> dict:
    """POST-HOC (reports/prereg_addendum_3d.md): the *closed-cycle* score in every plane (PCi, PCj),
    i < j ≤ k — the cycle score, counted only if the in-plane closure ratio (closure_ratio on the plane's
    distances, preregistered threshold 2) shows the loop closes, because an open arc (PCA's horseshoe of a
    line) also has a perfect angular order — its maximum over planes, and a permutation p-value of that
    maximum. Each null draw re-scores a random order in the *same* planes with the same rule and keeps its
    maximum, so choosing the best plane is accounted for. The PC basis does not depend on the order, so it
    is computed once and only orders are re-scored."""
    rng = rng if rng is not None else np.random.default_rng(0)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:k].T
    n, planes = len(order), list(combinations(range(k), 2))
    succ = np.empty((len(planes), n), dtype=int)  # angular successor of every item, per plane
    Dp = np.empty((len(planes), n, n))  # in-plane Euclidean distances, per plane
    for q, (i, j) in enumerate(planes):
        ang = np.argsort(np.arctan2(Z[:, j], Z[:, i]))
        succ[q, ang] = np.roll(ang, -1)
        Dp[q] = np.linalg.norm(Z[:, None, [i, j]] - Z[None, :, [i, j]], axis=-1)

    def scores(orders: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # orders [B, n] → cycle, closure [B, planes]
        pos = np.argsort(orders, axis=1)  # position of each item in the order
        nxt = pos[:, succ]  # [B, planes, n]: position of each item's angular successor
        cyc = np.maximum((nxt == ((pos + 1) % n)[:, None, :]).mean(-1), (nxt == ((pos - 1) % n)[:, None, :]).mean(-1))
        adj = Dp[:, orders[:, :-1], orders[:, 1:]].mean(-1).T  # [B, planes] mean adjacent distance (as closure_ratio)
        clo = Dp[:, orders[:, 0], orders[:, -1]].T / adj
        return cyc, clo

    cyc, clo = (a[0] for a in scores(np.asarray(order)[None]))
    closed = np.where(clo < max_closure, cyc, 0.0)
    ncyc, nclo = scores(np.stack([rng.permutation(n) for _ in range(n_perm)]))
    null = np.where(nclo < max_closure, ncyc, 0.0).max(1)
    b = int(np.argmax(closed))
    return {"score": float(closed[b]), "plane": [planes[b][0], planes[b][1]], "p": float((1 + (null >= closed[b]).sum()) / (n_perm + 1)),
            "planes": [[i + 1, j + 1, round(float(c), 4), round(float(z), 4)] for (i, j), c, z in zip(planes, cyc, clo)]}
