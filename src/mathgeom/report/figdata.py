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


def pca_var(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Fraction of total variance on each of the top-k PCs (= sklearn PCA.explained_variance_ratio_);
    every PCA view states it, so a picture is never read as more than the variance it shows."""
    S = np.linalg.svd(X.astype(np.float64) - X.astype(np.float64).mean(0), compute_uv=False)
    return (S[:k] ** 2) / (S**2).sum()


def exact_coords(X: np.ndarray) -> np.ndarray:
    """(n−1)-D coordinates of the centred rows: every pairwise distance and every inner product of the
    centred vectors is preserved exactly (they span ≤ n−1 dimensions), so the browser can recompute
    offsets and analogies without the full d-dimensional vectors."""
    X = X.astype(np.float64) - X.astype(np.float64).mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = max(1, int((S > S[0] * 1e-9).sum()))
    return U[:, :k] * S[:k]


def mds(D: np.ndarray, seed: int = 0, k: int = 2) -> np.ndarray:
    from sklearn.manifold import MDS

    return MDS(n_components=k, metric="precomputed", random_state=seed, init="classical_mds", n_init=1).fit_transform(D)


def stress1(layout: np.ndarray, D: np.ndarray) -> float:
    """Kruskal's stress-1 of a 2-D/3-D layout against the full-space distances D, with the optimal scale
    factor (so display scaling does not matter): √(Σ(b·d − δ)² / Σδ²). 0 = distances kept exactly; the
    standard goodness-of-fit of metric MDS (Kruskal 1964, Psychometrika 29:1–27), which every MDS view states."""
    from scipy.spatial.distance import pdist

    d, delta = pdist(layout), D[np.triu_indices(len(D), 1)]
    b = (d @ delta) / (d @ d)
    return float(np.sqrt(((b * d - delta) ** 2).sum() / (delta**2).sum()))


def ideal_fit(P: np.ndarray, closed: bool) -> np.ndarray:
    """The ideal template in true order (equally spaced circle, or evenly spaced line) embedded in the
    view's k dimensions and rotated/reflected onto the observed unit-RMS view P (orthogonal Procrustes).
    A reading aid drawn identically for the twin; no statistic uses it."""
    n, k = P.shape
    t = 2 * np.pi * np.arange(n) / n
    T = np.c_[np.cos(t), np.sin(t)] if closed else np.c_[np.arange(n) - (n - 1) / 2, np.zeros(n)]
    return align(np.c_[T, np.zeros((n, k - 2))], P)


def view_map(E: np.ndarray, view: np.ndarray) -> np.ndarray:
    """Linear map M (k×k) with E[:, :k] @ M == view, where E are exact PC-basis coordinates and `view` the
    displayed top-k layout of the same layer (a scaled rotation of them). Lets the site place any linear
    combination of items, e.g. the analogy query b − a + c, in the view exactly."""
    k = view.shape[1]
    return np.linalg.lstsq(E[:, :k], view, rcond=None)[0]


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


def field_layouts(rdms: dict[str, np.ndarray], ref: np.ndarray | None = None, k: int = 2) -> dict[str, np.ndarray]:
    """k-D metric MDS of each RDM (2-D maps; 3-D for the hero), aligned to the arXiv layout (or to `ref`)."""
    base = mds(rdms["arxiv"], k=k) if ref is None else ref
    return {name: align(mds(D, k=k), base) for name, D in rdms.items()}


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


def planes_by_layer(X: np.ndarray, ref_layer: int, onto: np.ndarray | None = None, k: int = 2) -> np.ndarray:
    """Top-k PC view of the family at every layer (k = 2: the PC1–2 plane; 3–5: the 3-D views), each
    aligned to the reference layer's view → [L, n, k]. `onto` (e.g. the trained model's view) first
    rotates the reference itself onto that layout, so the site's trained ↔ twin morph moves points as
    little as possible (display only, distances unchanged)."""
    ref = pca(X[:, ref_layer], k) if onto is None else align(pca(X[:, ref_layer], k), onto)
    return np.stack([align(pca(X[:, l], k), ref) for l in range(1, X.shape[1])])


def pcs_by_layer(X: np.ndarray, ref_layer: int, onto: np.ndarray | None = None, k: int = 5) -> np.ndarray:
    """Top-k PC scores at every layer for the 3-D views → [L, n, k]. Every axis stays that layer's own
    principal component, so the variance printed on it (pca_var) is exact; only each axis's sign is flipped
    to agree with the reference layer's scores (or with `onto`, e.g. the trained model's, for the twin), so
    views do not mirror when scrubbing. Unit-RMS scaling is display only."""
    ref = pca(X[:, ref_layer], k) if onto is None else onto
    out = []
    for l in range(1, X.shape[1]):
        P = pca(X[:, l], k)
        s = np.sign((P * ref).sum(0))
        out.append(unit_rms(P * np.where(s == 0, 1, s)))
    return np.stack(out)


VIEW_FAMILIES = ["days", "months", "angles", "digits_0_330", "ring_chain", "space_chain", "number_systems", "ngons", "dimensions", "group_like"]


def view_variance(cfg: Config, keys: list[str]) -> dict:
    """{model or twin key: {family: {"var5": [[share of variance on PC1..5] per layer], "cum3": [PC1–3 share per
    layer]}}}: how much of each family's variance its 3-D view shows (results/<cfg>/views.json, descriptive)."""
    lattice_items = list(load_yaml("families.yaml")["lattices"]["group_like"]["items"])  # the 13 structures the figures show
    out = {}
    for key in keys:
        for k in (key, f"{key}__random"):
            out[k] = {}
            for f in VIEW_FAMILIES:
                X = family_acts(cfg, k, f, lattice_items if f == "group_like" else None)
                v = [pca_var(X[:, l], 5) for l in range(1, X.shape[1])]
                out[k][f] = {"var5": [np.round(x, 4).tolist() for x in v], "cum3": [round(float(x[:3].sum()), 4) for x in v]}
    return out


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


def axiom_angles(X: np.ndarray, bits: np.ndarray, axioms: list[str]) -> dict:
    """The "compass" of the axiom explainer: for every pair of structures that differ only in axiom a,
    the full-space angle (degrees) between its difference vector and the mean direction of the *other*
    same-axiom difference vectors (leave-one-out). 0° = it points where the others point; 90° = unrelated.
    These are the vectors whose mean pairwise cosine is the reported offset consistency. Differences do
    not depend on centring, so exact (n−1)-D coordinates give identical angles (tested)."""
    from mathgeom.analyses.lattice import axiom_pairs

    X = np.asarray(X, dtype=np.float64)
    out = {}
    for a, name in enumerate(axioms):
        pairs = axiom_pairs(bits, a)
        U = np.stack([X[q] - X[p] for p, q in pairs])
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
        deg = []
        for k in range(len(pairs)):
            m = U[np.arange(len(pairs)) != k].sum(0)
            deg.append(float(np.degrees(np.arccos(np.clip(U[k] @ m / (np.linalg.norm(m) + 1e-12), -1.0, 1.0)))))
        out[name] = {"pairs": [tuple(map(int, p)) for p in pairs], "deg": deg}
    return out


def concept_neighbours(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Indices of the k nearest concepts (cosine, mean-centred) for every concept."""
    from mathgeom.geometry.preprocess import center

    Xc = center(X.astype(np.float64))
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    S = Xc @ Xc.T
    np.fill_diagonal(S, -np.inf)
    return np.argsort(-S, axis=1)[:, :k]
