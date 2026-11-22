"""Second independent audit (2026-11-22): recompute the headline statistics from the cached activations with code that
does not import the project's analysis modules (numpy/scipy only), and compare with the shipped results JSON.

Global context: every number in the README, paper and site is rendered from results/<cfg>/*.json, so a bug in an
analysis module would propagate silently. This re-implements, from the definitions in docs/prereg.md, the field-map ρ
(H-A1; trained and twin), the offset consistency of the four group-like axioms (H-B2), the ring/space-chain Robinson
scores (H-B) and the angles' closure ratio and line/circle template ρ, at relative depth 0.5 for all five models.
Skipped when the activations (artifacts/acts, about 6 GB, published with the Hugging Face dataset) are absent.
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from mathgeom.config import ROOT

ACTS = ROOT / "artifacts" / "acts"
MODELS = {"qwen3.5-2b": "mvp", "qwen3.5-0.8b": "robustness", "qwen2.5-1.5b": "robustness", "qwen2.5-math-1.5b": "robustness", "gemma-4-e2b": "gemma"}


def _field_rho(X: np.ndarray, lab: np.ndarray, names: list[str], T: np.ndarray) -> float:
    """Spearman ρ of the cosine distances between field centroids of mean-centred representations and the arXiv distances."""
    X = X.astype(np.float64) - X.astype(np.float64).mean(0)
    C = np.stack([X[lab == f].mean(0) for f in names])
    return spearmanr(pdist(C, "cosine"), T[np.triu_indices(len(names), 1)])[0]


def _offset_consistency(X: np.ndarray, bits: np.ndarray, axis: int) -> float:
    """Mean pairwise cosine of the difference vectors of all pairs that differ only in `axis` (switched on)."""
    e = np.eye(bits.shape[1], dtype=int)[axis]
    pairs = [(a, b) for a in range(len(bits)) for b in range(len(bits)) if (bits[b] - bits[a] == e).all()]
    X = X.astype(np.float64)  # activations are stored as float16; subtract in float64
    D = np.stack([X[b] - X[a] for a, b in pairs])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    return float(np.mean([D[i] @ D[j] for i, j in combinations(range(len(pairs)), 2)]))


def _robinson(X: np.ndarray) -> float:
    """Fraction of triples i < j < k (rows in true order) with d(i, k) ≥ max(d(i, j), d(j, k))."""
    D = squareform(pdist(X.astype(np.float64)))
    return float(np.mean([D[i, k] >= max(D[i, j], D[j, k]) for i, j, k in combinations(range(len(D)), 3)]))


def _angles(X: np.ndarray) -> tuple[float, float, float]:
    """Closure ratio d(first, last) / mean adjacent distance, and Spearman ρ of the RDM with |i − j| and with the cyclic distance."""
    D = squareform(pdist(X.astype(np.float64)))
    n = len(D)
    lin = np.abs(np.arange(n)[:, None] - np.arange(n)[None])
    iu = np.triu_indices(n, 1)
    return D[0, -1] / np.mean(np.diag(D, 1)), spearmanr(D[iu], lin[iu])[0], spearmanr(D[iu], np.minimum(lin, n - lin)[iu])[0]


@pytest.mark.skipif(not all((ACTS / k / "field_neutral.npz").exists() and (ACTS / f"{k}__random" / "family.npz").exists() for k in MODELS),
                    reason="cached activations not available")
def test_headline_statistics_recomputed_from_activations():
    c = pd.read_csv(ROOT / "data" / "concepts.csv")
    fd = json.loads((ROOT / "data" / "field_distance.json").read_text())
    fams = yaml.safe_load((ROOT / "configs" / "families.yaml").read_text())
    fam = json.loads((ACTS / "family_index.json").read_text())
    pos = {(f, x): i for i, (f, x, _) in enumerate(fam)}
    mask = ((c.kind == "math") & c.primary & (c.split == "confirmatory")).values
    lab = c.field.values[mask]
    names = [n for n in fd["names"] if n in set(lab)]
    sel = [fd["names"].index(n) for n in names]
    T = np.array(fd["distance"])[np.ix_(sel, sel)]
    g = fams["lattices"]["group_like"]
    gl, bits = list(g["items"]), np.array(list(g["items"].values()))
    for key, cfg in MODELS.items():
        res = {n: json.loads((ROOT / "results" / cfg / f"{n}.json").read_text()) for n in ("map_fields", "lattice", "shapes")}
        for k, who in ((key, "trained"), (f"{key}__random", "random_init")):
            last = np.load(ACTS / k / "field_neutral.npz")["last"]
            Lp = int(round(0.5 * (last.shape[1] - 1)))  # hidden_states index of relative depth 0.5
            assert Lp == res["map_fields"][key]["primary_layer"]
            assert _field_rho(last[mask, Lp], lab, names, T) == pytest.approx(res["map_fields"][key]["math_primary"]["rsa"][who], abs=1e-9)
            A = np.load(ACTS / k / "family.npz")["last"][:, Lp]
            Xg = A[[pos[("group_like", x)] for x in gl]]
            for ax, name in enumerate(g["axioms"]):
                assert _offset_consistency(Xg, bits, ax) == pytest.approx(res["lattice"][k]["group_like"]["offsets"][name]["consistency"], abs=1e-9)
            for ch in ("ring_chain", "space_chain"):
                assert _robinson(A[[pos[(ch, x)] for x in fams["chains"][ch]["items"]]]) == pytest.approx(res["shapes"][k][ch]["robinson"], abs=1e-12)
            clo, line, circ = _angles(A[[pos[("angles", x)] for x in fams["chains"]["angles"]["items"]]])
            s = res["shapes"][k]["angles"]
            assert (clo, line, circ) == pytest.approx((s["closure"], s["line_rho"], s["circle_rho"]), abs=1e-9)
