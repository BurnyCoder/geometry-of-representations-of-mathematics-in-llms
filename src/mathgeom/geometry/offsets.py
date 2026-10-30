"""Axiom-lattice tests (H-B2): offset consistency (parallelograms) and analogy completion.

Global context: if a model represents "adding the identity axiom" as one direction,
then difference vectors such as monoid−semigroup, loop−quasigroup and
unital magma−magma should be parallel — the parallelogram "crystal" structure of
Li et al. 2024 (arXiv:2410.19750), itself the activation-space analogue of word-vector
analogies (Mikolov et al. 2013, arXiv:1301.3781).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def offset_consistency(X: np.ndarray, pairs: list[tuple[int, int]], keep: list[tuple[int, int]] | None = None) -> float:
    """Mean pairwise cosine between normalized difference vectors X[b]−X[a] over `pairs`.

    `keep` (indices into `pairs`) restricts the mean to some of the cosines — the independent audit's
    lexically disjoint pairs (reports/prereg_addendum_lexical.md D2); None = all pairs of pairs."""
    diffs = _unit(np.stack([X[b] - X[a] for a, b in pairs]))
    sims = [diffs[i] @ diffs[j] for i, j in (keep if keep is not None else combinations(range(len(pairs)), 2))]
    return float(np.mean(sims))


def random_pair_null(X: np.ndarray, pool: np.ndarray, n_pairs: int, n_perm: int = 2000, seed: int = 0,
                     keep: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Null for `offset_consistency`: same number of random (a,b) pairs drawn from a concept pool
    (and, with `keep`, the same positions of cosines averaged as in the observed statistic)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for t in range(n_perm):
        pr = [tuple(rng.choice(pool, 2, replace=False)) for _ in range(n_pairs)]
        out[t] = offset_consistency(X, pr, keep)
    return out


def analogy_rank(X: np.ndarray, a: int, b: int, c: int, d: int, candidates: np.ndarray) -> int:
    """Rank (1 = best) of item d among candidates for the query X[b] − X[a] + X[c] (cosine).

    Inputs a, b, c are excluded from the candidate set (standard analogy protocol).
    """
    q = _unit(X[b] - X[a] + X[c])
    cand = np.array([k for k in candidates if k not in (a, b, c)])
    sims = _unit(X[cand]) @ q
    order = cand[np.argsort(-sims)]
    return int(np.flatnonzero(order == d)[0]) + 1
