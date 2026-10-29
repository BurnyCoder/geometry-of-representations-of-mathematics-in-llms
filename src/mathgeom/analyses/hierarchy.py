"""H-A2: is the field geometry tree-like, and does its tree match the pre-registered super-tree?

Global context: the preregistration measures H-A2 by *cophenetic correlation*. We build an
average-linkage (UPGMA) dendrogram of the field centroids (cosine RDM, as in H-A1) and use its
cophenetic distances (Sokal & Rohlf 1962, https://doi.org/10.2307/1217208; scipy
`cluster.hierarchy.cophenet`) in two ways:
  (1) super-tree match — Spearman ρ between the dendrogram's cophenetic distances and the
      super-tree distance (1 within a branch, 2 across), against shuffled-tree and random-init nulls;
  (2) intrinsic tree-likeness — the cophenetic correlation c of the dendrogram with the RDM itself,
      against a spectrum-matched Gaussian cloud (same number of points, same covariance
      eigenvalues; high-dimensional point clouds can look tree-like by default).
We also report Gromov's four-point δ-hyperbolicity, relative to the diameter (2δ/diam; 0 = a
tree metric), as computed in Khrulkov et al. 2019 (arXiv:1904.02239), on Euclidean centroid
distances — exact over all quadruples (17 fields → 2,380 quadruples).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from mathgeom.stats.rdm_regression import upper


def cophenetic_rdm(D: np.ndarray) -> tuple[np.ndarray, float]:
    """UPGMA dendrogram of a distance matrix → (cophenetic distance matrix, cophenetic correlation c)."""
    y = squareform(D, checks=False)
    Z = linkage(y, method="average")
    c, coph = cophenet(Z, y)
    return squareform(coph), float(c)


def supertree_match(D: np.ndarray, tree: np.ndarray) -> float:
    """Spearman ρ between the representation's cophenetic distances and the super-tree distance."""
    C, _ = cophenetic_rdm(D)
    return float(spearmanr(upper(C), upper(tree))[0])


def relative_delta(P: np.ndarray) -> float:
    """Gromov four-point δ-hyperbolicity of a point set (Euclidean), relative to the diameter: 2δ/diam."""
    D = squareform(pdist(P))
    delta = 0.0
    for x, y, z, w in combinations(range(len(P)), 4):
        s = sorted((D[x, y] + D[z, w], D[x, z] + D[y, w], D[x, w] + D[y, z]))
        delta = max(delta, (s[2] - s[1]) / 2)
    return float(2 * delta / D.max())


def spectrum_matched_gaussian(C: np.ndarray, rng) -> np.ndarray:
    """Gaussian points (same n) whose covariance has the same eigenvalues as the centered points C."""
    Cc = C - C.mean(0)
    _, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    k = len(S)
    return (rng.normal(size=(len(C), k)) * (S / np.sqrt(len(C) - 1))) @ Vt
