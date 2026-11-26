"""Independent order, positive-control and endpoint-preserving offset calculations."""
from itertools import combinations
from math import comb, factorial

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def exact_robinson(d):
    """Count all orderings by dynamic programming. Each triple scores exactly once, at its middle item."""
    n = len(d)
    k = comb(n, 3)
    ok = (d[:, None, :] >= d[:, :, None]) & (d[:, None, :] >= d[None, :, :])
    observed = sum(int(ok[i, j, l]) for i, j, l in combinations(range(n), 3))
    hist = np.zeros((1 << n, k + 1), dtype=np.int64)
    hist[0, 0] = 1
    for used in range((1 << n) - 1):
        left = [i for i in range(n) if used & (1 << i)]
        rest = [i for i in range(n) if not used & (1 << i)]
        for middle in rest:
            right = [i for i in rest if i != middle]
            increment = int(ok[:, middle, :][np.ix_(left, right)].sum())
            hist[used | (1 << middle), increment:] += hist[used, :k+1-increment]
    total = int(hist[-1].sum())
    assert total == factorial(n)
    extreme = int(hist[-1, observed:].sum())
    return {"satisfied": observed, "triples": k, "orders": total, "extreme": extreme, "p": extreme / total}


def minimal_upper_bounds(bits, a, b):
    """Minimal upper bounds in the sampled coordinatewise order of required axioms."""
    def le(i, j):
        return all(x <= y for x, y in zip(bits[i], bits[j]))
    bounds = [u for u in bits if le(a, u) and le(b, u)]
    return [u for u in bounds if not any(v != u and le(v, u) for v in bounds)]


def family_check(x, index, spec, saved, lattice):
    """Recover full-space templates, exact short-chain tests, and a matched-edge null from the same cache."""
    def rows(f):
        return np.array([i for i, (name, _, _) in enumerate(index) if name == f])
    out = {}
    for family in ("days", "months", "angles", "digits_0_330", "ring_chain", "space_chain"):
        d = squareform(pdist(x[rows(family)]))
        n = len(d)
        iu = np.triu_indices(n, 1)
        linear = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        line = float(spearmanr(d[iu], linear[iu]).statistic)
        circle = float(spearmanr(d[iu], np.minimum(linear, n-linear)[iu]).statistic)
        closure = float(d[0, -1] / np.diag(d, 1).mean())
        r = {"line_rho": line, "circle_rho": circle, "closure": closure}
        if family in saved:
            np.testing.assert_allclose([r[k] for k in r], [saved[family][k] for k in r], atol=1e-9)
        if family in ("days", "months"):
            r["full_space_control"] = bool(circle > line and closure < 2 and saved[family]["robinson_p"] < .05)
        if family in ("ring_chain", "space_chain"):
            r["robinson_exact"] = exact_robinson(d)
            blocks = [slice(3, 9)] if family == "ring_chain" else [slice(0, 6), slice(6, 11)]
            r["blocks"] = [exact_robinson(d[b, b]) for b in blocks]
        out[family] = r
    profiles = spec["items"]
    indices = [i for i, (g, n, _) in enumerate(index) if g == "group_like" and n in profiles]
    bits = np.array([profiles[index[i][1]] for i in indices])
    v = x[indices]
    diffs = v[None, :, :] - v[:, None, :]
    uv = (diffs / np.maximum(np.linalg.norm(diffs, axis=-1, keepdims=True), 1e-25)).reshape(-1, v.shape[1])
    gram = uv @ uv.T
    rng = np.random.default_rng(20261126)
    orders = np.stack([rng.permutation(len(v)) for _ in range(20000)])
    out["offsets"] = {}
    for axis, name in enumerate(spec["axioms"]):
        i, j = np.where(np.all(bits[None, :, :] - bits[:, None, :] == np.eye(4, dtype=int)[axis], axis=-1))
        s, t = np.triu_indices(len(i), 1)
        edge = i * len(v) + j
        observed = float(gram[edge[s], edge[t]].mean())
        np.testing.assert_allclose(observed, lattice["group_like"]["offsets"][name]["consistency"], atol=1e-9)
        labels = orders[:, i] * len(v) + orders[:, j]
        null = gram[labels[:, s], labels[:, t]].mean(1)
        extreme = int(np.sum(null >= observed - 1e-12))
        out["offsets"][name] = {"cosine": observed, "edges": len(i), "extreme": extreme, "draws": len(null),
                                "matched_p": (1 + extreme) / (1 + len(null))}
    return out
