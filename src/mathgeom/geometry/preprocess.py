"""Activation preprocessing and RDM construction.

Global context: residual streams are anisotropic — a few "rogue" dimensions with large mean and
variance dominate cosine similarity (Timkey & van Schijndel 2021, arXiv:2109.04404, who recommend
standardization; see also the massive activations of Sun et al. 2024, arXiv:2402.17762). We mean-center
per (model, layer) over the concept set, which removes their shared mean but not their variance, and
offer all-but-the-top (Mu, Bhat & Viswanath 2018, arXiv:1702.01417) as a variant.
We deliberately do NOT whiten: Golechha et al., "Intricacies of Feature Geometry in Large Language
Models" (ICLR 2025 blog track, https://iclr.cc/virtual/2025/poster/31349), showed whitening makes
random concepts look orthogonal/polytope-structured.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist, squareform


def center(X: np.ndarray) -> np.ndarray:
    """Subtract the mean representation across concepts (rows)."""
    return X - X.mean(0, keepdims=True)


def all_but_the_top(X: np.ndarray, n_remove: int | None = None) -> np.ndarray:
    """Center, then project out the top principal directions (Mu, Bhat & Viswanath 2018, Alg. 1).

    n_remove defaults to d/100 as recommended in the paper, but never more than n-2.
    """
    Xc = center(X.astype(np.float64))  # activations are stored as float16; LAPACK needs float32/64
    d = X.shape[1]
    k = n_remove if n_remove is not None else max(1, d // 100)
    k = min(k, X.shape[0] - 2)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)  # rows of Vt = principal directions
    U = Vt[:k]
    return Xc - (Xc @ U.T) @ U


def cosine_rdm(X: np.ndarray) -> np.ndarray:
    """n×n cosine-distance matrix (1 − cos) between rows."""
    return squareform(pdist(X.astype(np.float64), metric="cosine"))


def euclid_rdm(X: np.ndarray) -> np.ndarray:
    """n×n Euclidean distance matrix — used for small ordered families/lattices.

    Cosine distance after centering on a small family's own mean is degenerate for collinear
    points (all points on one side of the centroid share a direction), which our synthetic
    tests exposed; shape work in the literature (Engels et al. PCA plots, SMDS stress) is
    Euclidean as well.
    """
    return squareform(pdist(X.astype(np.float64), metric="euclidean"))


def abs_diff_rdm(v: np.ndarray) -> np.ndarray:
    """|v_i − v_j| for a scalar covariate (e.g. log-frequency, token count)."""
    v = np.asarray(v, dtype=float)
    return np.abs(v[:, None] - v[None, :])


def same_label_rdm(labels) -> np.ndarray:
    """0 if two items share a label (e.g. head noun), else 1."""
    lab = np.asarray(labels)
    return (lab[:, None] != lab[None, :]).astype(float)


def char_ngrams(s: str, n: int = 3) -> Counter:
    """Counts of the character n-grams of a lower-cased string padded with one space on each side,
    so that word boundaries form n-grams (" gr", "up "). Shared by the lexical-overlap RDM below and the
    character-3-gram baseline of the independent audit (analyses/lexical.py)."""
    t = f" {s.lower()} "
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


def char_ngram_rdm(strings: list[str], n: int = 3) -> np.ndarray:
    """1 − Jaccard similarity of character n-gram sets — a lexical-overlap nuisance RDM.

    Captures e.g. 'semigroup'/'quasigroup'/'group' sharing the substring 'group'.
    """
    grams = [set(char_ngrams(s, n)) for s in strings]
    k = len(strings)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            inter = len(grams[i] & grams[j])
            union = len(grams[i] | grams[j])
            D[i, j] = D[j, i] = 1 - inter / union
    return D
