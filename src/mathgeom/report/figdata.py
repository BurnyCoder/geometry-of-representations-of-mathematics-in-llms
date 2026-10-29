"""Data preparation shared by the static figures and the interactive site (identical projections in both).

Global context: every picture is an *unsupervised* view (PCA or metric MDS) of the same activations
the statistics are computed on; nothing here fits a projection to labels, because with n ≤ 13 items
in d ≥ 1024 a supervised plane would draw circles or parallelograms even for random vectors.
Alignment between panels or layers is display-only: orthogonal Procrustes (rotation/reflection) after
centring and scaling each layout to unit RMS. The statistics shown on figures are read from
results/<cfg>/*.json, never recomputed from these 2-D layouts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from mathgeom.config import Config, load_yaml
from mathgeom.pipeline import load_acts


# ------------------------------------------------------------------ generic geometry
def unit_rms(P: np.ndarray) -> np.ndarray:
    P = P - P.mean(0)
    return P / (np.sqrt((P**2).sum(1).mean()) + 1e-12)


def align(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate/reflect a unit-RMS layout onto a reference (display only; distances are unchanged)."""
    P, ref = unit_rms(P), unit_rms(ref)
    R, _ = orthogonal_procrustes(P, ref)
    return P @ R


def pca(X: np.ndarray, k: int = 2) -> np.ndarray:
    X = X.astype(np.float64) - X.astype(np.float64).mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:k].T


def exact_coords(X: np.ndarray) -> np.ndarray:
    """(n−1)-D coordinates of the centred rows: every pairwise distance and every inner product of the
    centred vectors is preserved exactly (they span ≤ n−1 dimensions), so the browser can recompute
    offsets and analogies without the full d-dimensional vectors."""
    X = X.astype(np.float64) - X.astype(np.float64).mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = max(1, int((S > S[0] * 1e-9).sum()))
    return U[:, :k] * S[:k]


def mds(D: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import MDS

    return MDS(n_components=2, metric="precomputed", random_state=seed, init="classical_mds", n_init=1).fit_transform(D)


def rank_unit(D: np.ndarray) -> np.ndarray:
    """Off-diagonal distances as within-matrix percentile ranks (Spearman RSA compares ranks)."""
    from scipy.stats import rankdata

    iu = np.triu_indices(len(D), 1)
    R = np.zeros_like(D, dtype=float)
    R[iu] = (rankdata(D[iu]) - 1) / max(1, len(iu[0]) - 1)
    return R + R.T


# ------------------------------------------------------------------ field map
def field_setup(cfg: Config, mp_key: dict) -> dict:
    """Concept mask (primary, confirmatory), labels, field order by super-tree branch, arXiv distance."""
    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    fcfg = {f["id"]: f for f in load_yaml("fields.yaml")["math"]}
    split = cfg.raw.get("analysis_split", "confirmatory")
    mask = ((concepts.kind == "math") & concepts.primary & (concepts.split == split)).values
    names = mp_key["math_primary"]["fields"]
    sel = [fd["names"].index(n) for n in names]
    branches = list(dict.fromkeys(fcfg[n]["branch"] for n in fcfg))  # branch order as listed in fields.yaml
    order = sorted(range(len(names)), key=lambda i: (branches.index(fcfg[names[i]]["branch"]), names[i]))
    return {"concepts": concepts, "mask": mask, "labels": concepts.field.values[mask], "names": names,
            "pa": [fcfg[n]["pa"] for n in names], "branch": [fcfg[n]["branch"] for n in names], "branches": branches,
            "order": order, "arxiv": np.array(fd["distance"])[np.ix_(sel, sel)], "n_papers": fd["n_papers"]}


def field_rdms(cfg: Config, key: str, setup: dict, layer: int) -> dict[str, np.ndarray]:
    """Cosine RDM of field centroids (mean-centred concepts) for the trained model and its twin."""
    from mathgeom.analyses.map_fields import field_centroids
    from mathgeom.geometry.preprocess import cosine_rdm

    out = {"arxiv": setup["arxiv"]}
    for name, k in (("trained", key), ("random_init", f"{key}__random")):
        X = load_acts(cfg, k, "field_neutral")["last"][setup["mask"], layer]
        out[name] = cosine_rdm(field_centroids(X, setup["labels"], setup["names"]))
    return out


def field_layouts(rdms: dict[str, np.ndarray], ref: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """2-D metric MDS of each RDM, aligned to the arXiv layout (or to `ref`, e.g. for the layer slider)."""
    base = mds(rdms["arxiv"]) if ref is None else ref
    return {k: align(mds(D), base) for k, D in rdms.items()}


def neighbour_edges(D: np.ndarray, k: int = 2) -> list[tuple[int, int]]:
    """Undirected edges from each item to its k nearest neighbours under the ground-truth distance."""
    E = set()
    for i in range(len(D)):
        for j in np.argsort(D[i])[1 : k + 1]:
            E.add((min(i, j), max(i, j)))
    return sorted(E)


# ------------------------------------------------------------------ families and lattices
def family_index(cfg: Config) -> list[tuple[str, str, str]]:
    return [tuple(x) for x in json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())]


def family_acts(cfg: Config, key: str, family: str, items: list[str] | None = None) -> np.ndarray:
    """[n_items × (L+1) × d] activations of one family (rows in config order, or in `items` order)."""
    fam = family_index(cfg)
    pos = {(f, c): i for i, (f, c, _) in enumerate(fam)}
    rows = [pos[(family, c)] for c in items] if items is not None else [i for i, (f, _, _) in enumerate(fam) if f == family]
    return load_acts(cfg, key, "family")["last"][rows].astype(np.float64)


def family_items(cfg: Config, family: str) -> list[str]:
    return [c for f, c, _ in family_index(cfg) if f == family]


def planes_by_layer(X: np.ndarray, ref_layer: int, onto: np.ndarray | None = None) -> np.ndarray:
    """PC-1/2 plane of the family at every layer, each aligned to the reference layer's plane → [L, n, 2].
    `onto` (e.g. the trained model's plane) first rotates the reference itself onto that layout, so the
    site's trained ↔ twin morph moves points as little as possible (display only, distances unchanged)."""
    ref = pca(X[:, ref_layer]) if onto is None else align(pca(X[:, ref_layer]), onto)
    return np.stack([align(pca(X[:, l]), ref) for l in range(1, X.shape[1])])


def offset_matrix(X: np.ndarray, bits: np.ndarray, axioms: list[str]) -> dict:
    """Cosine matrix of every single-axiom difference vector, grouped by axiom.

    Within an axiom block, pairs never share an endpoint and the block mean (off-diagonal) equals the
    reported offset consistency. Across axioms, two pairs can share a vertex; with isotropic noise such
    pairs have cosine ≈ ±0.5 regardless of structure, so they are flagged (`shared`) for hatching.
    """
    from mathgeom.analyses.lattice import axiom_pairs
    from mathgeom.geometry.preprocess import center

    Xc = center(X.astype(np.float64))
    pairs, group = [], []
    for a, name in enumerate(axioms):
        for p in axiom_pairs(bits, a):
            pairs.append(p), group.append(a)
    V = np.stack([Xc[b] - Xc[a] for a, b in pairs])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    C = V @ V.T
    shared = np.array([[len({*p} & {*q}) > 0 and i != j for j, q in enumerate(pairs)] for i, p in enumerate(pairs)])
    return {"pairs": pairs, "group": np.array(group), "cos": C, "shared": shared}


def concept_neighbours(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Indices of the k nearest concepts (cosine, mean-centred) for every concept."""
    from mathgeom.geometry.preprocess import center

    Xc = center(X.astype(np.float64))
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    S = Xc @ Xc.T
    np.fill_diagonal(S, -np.inf)
    return np.argsort(-S, axis=1)[:, :k]
