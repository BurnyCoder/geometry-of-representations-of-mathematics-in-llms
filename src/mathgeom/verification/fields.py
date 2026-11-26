"""Reconstruct the arXiv proxy and H-A tests from published inputs; no model dependencies."""
from collections import Counter
from itertools import combinations
import json

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr


def check_target(cache, spec, saved):
    """Count papers once per field even when multiple component categories co-occur."""
    raw = json.loads(cache.read_text())
    names = [f["id"] for f in spec]
    assert names == saved["names"] and raw["revision"] == saved["revision"]
    categories = {code: i for i, f in enumerate(spec) for code in f["arxiv"]}
    co = np.zeros((len(names), len(names)), dtype=np.int64)
    for codes, n in Counter(raw["categories"]).items():
        fields = sorted({categories[c] for c in codes.split() if c in categories})
        if fields:
            co[np.ix_(fields, fields)] += n
    counts = co.diagonal()
    dist = 1 - co / np.sqrt(counts[:, None] * counts[None, :])
    np.testing.assert_array_equal(counts, saved["papers_per_field"])
    np.testing.assert_array_equal(dist.round(6), saved["distance"])
    assert len(raw["categories"]) == saved["n_papers"]
    return {"papers": len(raw["categories"]), "revision": raw["revision"], "field_counts": counts.tolist(),
            "overlaps": co.tolist(), "distinct_pair_distances": len(set(squareform(dist, checks=False)))}


def normranks(x):
    """Pearson inner products of centered midranks give Spearman correlations, including ties."""
    ranks = rankdata(x, axis=-1)
    centered = ranks - ranks.mean(axis=-1, keepdims=True)
    return centered / np.linalg.norm(centered, axis=-1, keepdims=True)


def reference_design(labels, names, target, branches):
    """Use the same within-field concept resamples and simultaneous field relabelings for each checkpoint."""
    strata = [np.flatnonzero(labels == field) for field in names]
    n = len(names)
    weights = [np.zeros((2000, len(ix))) for ix in strata]
    rng = np.random.default_rng(0)
    for b in range(2000):
        for ix, w in zip(strata, weights):
            w[b] = np.bincount(rng.integers(len(ix), size=len(ix)), minlength=len(ix)) / len(ix)
    iu = np.triu_indices(n, 1)
    ref = normranks(target[iu])
    matrix = squareform(ref)
    rng = np.random.default_rng(0)
    orders = np.stack([rng.permutation(n) for _ in range(10000)])
    perm = matrix[orders[:, iu[0]], orders[:, iu[1]]]
    branch = np.asarray([branches[n] for n in names])
    tree = (branch[:, None] != branch[None, :]).astype(float)
    return strata, weights, iu, ref, perm, tree, orders


def rdm(centers, sizes):
    """Center at the resampled concept mean (different field sample sizes), then compare unit centroids."""
    x = centers - np.average(centers, axis=-2, weights=sizes)[..., None, :]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return 1 - x @ x.swapaxes(-1, -2)


def checkpoint(x, design):
    strata, weights, iu, ref, perm, tree, orders = design
    sizes = [len(ix) for ix in strata]
    groups = [x[ix] for ix in strata]
    d = rdm(np.stack([g.mean(0) for g in groups]), sizes)[iu]
    v = normranks(d)
    rho = float(v @ ref)
    extreme = int(np.sum(perm @ v >= rho - 1e-14))
    c, coph = cophenet(linkage(d, method="average"), d)
    tree_match = float(normranks(coph) @ normranks(tree[iu]))
    trees = normranks(tree[orders[:2000, iu[0]], orders[:2000, iu[1]]])
    tree_p = (1 + int(np.sum(trees @ normranks(coph) >= tree_match - 1e-14))) / 2001
    draws, hierarchy_draws = [], []
    branch_ranks = normranks(tree[iu])
    for start in range(0, len(weights[0]), 40):
        centers = np.stack([w[start:start+40] @ g for w, g in zip(weights, groups)], axis=1)
        batch = rdm(centers, sizes)[:, iu[0], iu[1]]
        draws.extend(normranks(batch) @ ref)
        if start < 1000:
            coph = [cophenet(linkage(v, method="average")) for v in batch]
            hierarchy_draws.extend(normranks(coph) @ branch_ranks)
    return {"rho": rho, "mantel_exceedances": extreme, "mantel_p": (1 + extreme) / 10001,
            "cophenetic_c": float(c), "supertree_rho": tree_match, "supertree_label_p": tree_p}, np.asarray(draws), np.asarray(hierarchy_draws)


def applied_labels(x, frame, n_tokens):
    """Exact field-label sensitivity; keep every concept of a field together in all assignments."""
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    z = np.column_stack((np.ones(len(x)), frame.zipf, n_tokens))
    def residual(v):
        return v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
    pcs = normranks(residual(x @ vt[:5].T).T)
    fields = sorted(frame.field.unique())
    true = tuple(sorted(frame.loc[frame.pa == "applied", "field"].unique()))
    choices = list(combinations(fields, len(true)))
    values = []
    for subset in choices:
        cov = frame.field.isin(subset).to_numpy(dtype=float)
        values.append(np.abs(pcs @ normranks(residual(cov))))
    vals = np.array(values)
    observed = vals[choices.index(true)]
    return {"assignments": len(choices), "applied_fields": list(true), "absolute_rhos": observed.tolist(),
            "pc_specific_exact_p": (vals >= observed - 1e-12).mean(0).tolist(),
            "max_five_pc_p": (vals.max(1)[:, None] >= observed - 1e-12).mean(0).tolist()}
