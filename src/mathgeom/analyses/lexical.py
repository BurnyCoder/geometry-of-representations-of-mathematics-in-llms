"""Independent audit: word-form (lexical) baselines for the geometric tests (reports/prereg_addendum_lexical.md).

Global context: the pre-registration used the random-init twin as the measure of "form-only similarity",
but a random network can scramble even exact word-level composition (in the Qwen3.5-2B twin, magma → semigroup
and commutative magma → commutative semigroup have a difference cosine of 0.04, a bag of words gives 1; the
Qwen2.5 twins keep 0.30–0.41 and Gemma's twin 0.77–0.81). So "beats the twin" does not rule out that a result
follows from the names alone. The baselines here encode the names without context — spelling (char3, word_bag)
or context-free token meaning (layer0_bag) — and are analysed with the statistic, centring, null type and seed
of the trained model's test (permutation counts are smaller where noted: the chains' leave-one-out null has
300 permutations, the field-map Δ bootstrap 1,000 resamples):
  char3       counts of character 3-grams (preprocess.char_ngrams)
  word_bag    counts of lower-cased words
  layer0_bag  the sum of the model's trained input embeddings over the name's tokens (no context)
plus the decompositions that locate where the axiom offsets come from (lexically duplicated pairs), an
exact within-block order test for the chains, and a simulation of how diagnostic the chain test is.
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from mathgeom.analyses.lattice import axiom_pairs
from mathgeom.geometry.manifold import loo_order_rho, pca_order_rho, perm_p, permutation_null, robinson_score
from mathgeom.geometry.offsets import _unit, offset_consistency, random_pair_null
from mathgeom.geometry.preprocess import center, char_ngrams, euclid_rdm

RING_BLOCKS = [0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]  # head nouns of the ring chain: rng | ring ×2 | domain ×6 | field ×2


# ------------------------------------------------------------------ baselines
def _counts(bags: list[dict]) -> np.ndarray:
    """Rows = items, columns = the union vocabulary of the bags (sorted for determinism)."""
    vocab = sorted(set().union(*bags))
    col = {v: j for j, v in enumerate(vocab)}
    M = np.zeros((len(bags), len(vocab)))
    for i, b in enumerate(bags):
        for v, c in b.items():
            M[i, col[v]] = c
    return M


def char3_vectors(names: list[str]) -> np.ndarray:
    """Character-3-gram count vectors (word boundaries padded, lower-cased)."""
    return _counts([char_ngrams(s, 3) for s in names])


def word_bag(names: list[str]) -> np.ndarray:
    """Counts of lower-cased whitespace-separated words."""
    bags = []
    for s in names:
        b: dict[str, int] = {}
        for w in s.lower().split():
            b[w] = b.get(w, 0) + 1
        bags.append(b)
    return _counts(bags)


def layer0_bag(acts: dict, rows) -> np.ndarray:
    """Sum of the trained input embeddings over each name's tokens: hidden_states[0] mean-pooled over the
    concept's tokens (`pool[:, 0]`) times the token count. Position-free, so it does not depend on the template."""
    return acts["pool"][rows, 0].astype(np.float64) * acts["n_tokens"][rows][:, None]


# ------------------------------------------------------------------ axiom offsets
def _strip(s: str) -> str:
    return s.replace("commutative ", "")


def lexical_duplicates(names: list[str], pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Index pairs (i, j) of same-axiom pairs that coincide after deleting "commutative " from all four names,
    e.g. magma → semigroup and commutative magma → commutative semigroup: their difference vectors share the
    head-noun change, so any representation dominated by the head noun makes them parallel."""
    key = [(_strip(names[a]), _strip(names[b])) for a, b in pairs]
    return [(i, j) for i, j in combinations(range(len(pairs)), 2) if key[i] == key[j]]


def disjoint_offset_test(X: np.ndarray, names: list[str], pairs: list[tuple[int, int]], n_perm: int = 2000, seed: int = 0) -> dict:
    """Addendum D2: offset consistency over the lexically disjoint cosine pairs only, against random structure
    pairs evaluated at the same positions; the duplicated cosines are reported separately."""
    dup = lexical_duplicates(names, pairs)
    keep = [ij for ij in combinations(range(len(pairs)), 2) if ij not in dup]
    obs = offset_consistency(X, pairs, keep)
    null = random_pair_null(X, np.arange(len(names)), len(pairs), n_perm, seed, keep)
    return {"n_dup": len(dup), "n_disjoint": len(keep), "dup_mean": offset_consistency(X, pairs, dup) if dup else None,
            "disjoint_mean": obs, "null95": float(np.percentile(null, 95)), "p": perm_p(obs, null)}


def offsets_vs_null(X: np.ndarray, bits: np.ndarray, axioms: list[str], n_perm: int = 2000, seed: int = 0) -> dict:
    """Offset consistency per axiom with the random-pair null of lattice_stats (same seeds: seed + axis)."""
    out = {}
    for ax, name in enumerate(axioms):
        pr = axiom_pairs(bits, ax)
        if len(pr) < 2:
            continue
        obs = offset_consistency(X, pr)
        null = random_pair_null(X, np.arange(len(X)), len(pr), n_perm, seed + ax)
        out[name] = {"n_pairs": len(pr), "consistency": obs, "null95": float(np.percentile(null, 95)), "p": perm_p(obs, null)}
    return out


def heldout_commutative_test(X: np.ndarray, names: list[str], bits: np.ndarray, axis: int = 3) -> dict:
    """Addendum D3: does the direction of the pairs that add the word "commutative" predict abelian group − group,
    the one commutativity pair that shares no word with it? Exact null (amendment 1): every ordered pair of
    distinct items with the same commutativity bit, i.e. differences that do not involve commutativity — pairs
    that change commutativity together with another axiom would carry the direction too."""
    pr = axiom_pairs(bits, axis)
    test = next(p for p in pr if names[p[1]] == "abelian group")
    d = _unit(np.mean([_unit(X[b] - X[a]) for a, b in pr if (a, b) != test], 0))
    obs = float(_unit(X[test[1]] - X[test[0]]) @ d)
    null = np.array([_unit(X[b] - X[a]) @ d for a in range(len(X)) for b in range(len(X)) if a != b and bits[a, axis] == bits[b, axis]])
    return {"cos": obs, "null95": float(np.percentile(null, 95)), "p": perm_p(obs, null), "n_null": len(null)}


# ------------------------------------------------------------------ chains
def exact_order_test(X: np.ndarray) -> dict:
    """Addendum D5/D6: for a short block whose rows are in true order, the Robinson score and the PC1 |Spearman ρ|
    (manifold.robinson_score, manifold.pca_order_rho) with exact p-values over all n! orders: the fraction of
    orders scoring at least the observed value (the identity order included)."""
    X = center(X.astype(np.float64))
    D = euclid_rdm(X)
    orders = [np.array(p) for p in permutations(range(len(X)))]  # orders[0] is the true order
    rob = np.array([robinson_score(D, o) for o in orders])
    pc1 = np.array([pca_order_rho(X, o) for o in orders])
    eps = 1e-12
    return {"n": len(X), "robinson": float(rob[0]), "robinson_p": float(np.mean(rob >= rob[0] - eps)),
            "pc1": float(pc1[0]), "pc1_p": float(np.mean(pc1 >= pc1[0] - eps))}


def block_only_simulation(snr: float, n_sims: int = 40, n_perm_robinson: int = 2000, n_perm_loo: int = 300, seed: int = 0,
                          d: int = 2048, scale: float = 1.0) -> dict:
    """Addendum D7: a representation that only clusters the ring chain's items by head noun (equidistant cluster
    centres, no order between or within clusters) — how often does the pre-registered chain test (Robinson p < .05
    AND LOO order p < .05, with the pre-registered permutation counts) reject? `scale` sets the mean row norm
    (the pre-registered ridge α = 10 is not scale-invariant, so the simulation uses the real data's scale)."""
    n = len(RING_BLOCKS)
    rej, robs, loos = [], [], []
    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        B = _unit(rng.normal(size=(max(RING_BLOCKS) + 1, d)))
        X = center(snr * B[RING_BLOCKS] + rng.normal(size=(n, d)) / np.sqrt(d))
        X *= scale / np.linalg.norm(X, axis=1).mean()
        D = euclid_rdm(X)
        o = np.arange(n)
        rb, lo = robinson_score(D, o), loo_order_rho(X, o)
        p_rb = perm_p(rb, permutation_null(lambda p: robinson_score(D, p), n, n_perm_robinson, seed + s))
        p_lo = perm_p(lo, permutation_null(lambda p: loo_order_rho(X, p), n, n_perm_loo, seed + s))
        rej.append(max(p_rb, p_lo) < 0.05)
        robs.append(rb)
        loos.append(lo)
    return {"snr": snr, "n_sims": n_sims, "reject_rate": float(np.mean(rej)), "mean_robinson": float(np.mean(robs)), "mean_loo": float(np.mean(loos))}
