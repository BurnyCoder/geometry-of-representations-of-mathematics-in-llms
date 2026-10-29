"""H-B2: axiom-lattice geometry (group-like structures, racks/quandles, group objects).

Global context: each structure is a bit vector of axioms (configs/families.yaml). Tests:
(1) axiom RDM — regress the family RDM on Hamming distance of axiom vectors with a
character-trigram lexical-overlap nuisance (semigroup/quasigroup/group share 'group');
(2) offset consistency — for each axiom, mean cosine between difference vectors of all pairs
differing only in that bit, vs a random-pair null from the same family pool;
(3) analogy completion — rank of the 4th vertex for every parallelogram in the lattice.
All are also computed on the random-init model by the caller.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from mathgeom.geometry.offsets import analogy_rank, offset_consistency, random_pair_null
from mathgeom.geometry.preprocess import center, char_ngram_rdm, euclid_rdm
from mathgeom.stats.rdm_regression import rdm_regression


def axiom_pairs(bits: np.ndarray, axis: int) -> list[tuple[int, int]]:
    """All (a, b) with bits[b] = bits[a] + e_axis (differ only by switching that axiom on)."""
    e = np.zeros(bits.shape[1], dtype=int)
    e[axis] = 1
    return [(a, b) for a in range(len(bits)) for b in range(len(bits)) if np.array_equal(bits[b] - bits[a], e)]


def parallelograms(bits: np.ndarray) -> list[tuple[int, int, int, int]]:
    """All (a, b, c, d) with b−a = d−c = a single axiom bit, {a,b} ∩ {c,d} = ∅."""
    out = []
    for axis in range(bits.shape[1]):
        pr = axiom_pairs(bits, axis)
        for (a, b), (c, d) in combinations(pr, 2):
            out.append((a, b, c, d))
    return out


def lattice_stats(X: np.ndarray, names: list[str], bits: np.ndarray, axioms: list[str], n_perm: int = 5000, seed: int = 0) -> dict:
    """All H-B2 statistics for one lattice at one (model, layer)."""
    Xc = center(X.astype(np.float64))
    D = euclid_rdm(Xc)
    ham = (bits[:, None, :] != bits[None, :, :]).sum(-1).astype(float)
    reg = rdm_regression(D, {"hamming": ham, "char3_lexical": char_ngram_rdm(names)}, "hamming", n_perm=n_perm, seed=seed)
    out = {"axiom_rdm_beta": reg.beta, "axiom_rdm_p": reg.p, "lexical_beta": reg.betas["char3_lexical"], "offsets": {}}
    pool = np.arange(len(names))
    for ax, name in enumerate(axioms):
        pr = axiom_pairs(bits, ax)
        if len(pr) < 2:
            continue
        obs = offset_consistency(Xc, pr)
        null = random_pair_null(Xc, pool, len(pr), n_perm=2000, seed=seed + ax)
        out["offsets"][name] = {"n_pairs": len(pr), "consistency": obs, "null_mean": float(null.mean()), "null95": float(np.percentile(null, 95)), "p": float((1 + (null >= obs).sum()) / (1 + len(null)))}
    ranks = [analogy_rank(Xc, a, b, c, d, pool) for a, b, c, d in parallelograms(bits)]
    n_cand = len(names) - 3
    out["analogy"] = {"n": len(ranks), "top1": float(np.mean(np.array(ranks) == 1)), "mean_rank": float(np.mean(ranks)), "chance_mean_rank": (n_cand + 1) / 2, "chance_top1": 1 / n_cand}
    return out
