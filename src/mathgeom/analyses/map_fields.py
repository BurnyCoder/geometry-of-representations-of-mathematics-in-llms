"""H-A1: do LLM representations of math concepts form a map of fields?

Global context — two estimands, both compared against controls by PAIRED concept bootstrap:

1. Graded map (primary): field-level RSA. Field centroids of mean-centered concept
   representations → k×k cosine RDM → Spearman ρ with the arXiv cross-listing distance
   (data/arxiv_fields.py). Inference unit = field: Mantel permutation of field labels
   (calibrated under the pure null: rejection rate < 10% at α=.05 is asserted over 150 simulations in
   tests/test_rdm_regression.py::test_field_level_mantel_is_calibrated_under_pure_null) and a max-over-layers
   null for the layer curve.
2. Clustering: leave-one-out nearest-centroid field accuracy at the concept level.

Why paired controls rather than partial regression: our simulations (analyses/calibration.py →
results/mvp/calibration.json, 200 runs per scenario) showed that partialling a nuisance RDM that only
noisily measures a confounder (lexical form, frequency) leaves residual confounding (52.5% false positives
at the field level; 22% at the concept level with a misspecified nuisance, 6.5% with a correct one).
So the falsification test is Δ = statistic(trained) − statistic(control) for
the random-init twin (same tokenizer/architecture, no learning) and the layer-0 embedding,
with a stratified concept bootstrap where both models see the same resampled concepts.
Partial-regression coefficients (Freedman–Lane) are reported as secondary/descriptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mathgeom.geometry.preprocess import abs_diff_rdm, center, char_ngram_rdm, cosine_rdm, same_label_rdm
from mathgeom.stats.rdm_regression import rdm_regression, upper


def field_centroids(X: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    """Mean of centered representations per field, in `names` order."""
    Xc = center(X.astype(np.float64))
    return np.stack([Xc[labels == f].mean(0) for f in names])


def field_rsa(X: np.ndarray, labels: np.ndarray, names: list[str], target: np.ndarray) -> float:
    """Spearman ρ between the field-centroid cosine RDM and the target field-distance matrix."""
    C = field_centroids(X, labels, names)
    return float(spearmanr(upper(cosine_rdm(C)), upper(target))[0])


def mantel_field_perm(rdm_vec: np.ndarray, target: np.ndarray, n_perm: int, rng) -> np.ndarray:
    """Null distribution of Spearman ρ under random relabeling of fields (rows+cols of target)."""
    k = target.shape[0]
    iu = np.triu_indices(k, 1)
    r_rdm = np.argsort(np.argsort(rdm_vec))  # ranks once (Spearman = Pearson of ranks; no ties in cosine RDM)
    r_rdm = (r_rdm - r_rdm.mean()) / r_rdm.std()
    out = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(k)
        t = target[np.ix_(p, p)][iu]
        rt = np.argsort(np.argsort(t))
        rt = (rt - rt.mean()) / rt.std()
        out[b] = (r_rdm * rt).mean()
    return out


def loo_centroid_correct(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-concept 0/1: is the concept nearest (cosine) to its own field's leave-one-out centroid?"""
    Xc = center(X.astype(np.float64))
    Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    names = np.unique(labels)
    sums = np.stack([Xc[labels == f].sum(0) for f in names])
    counts = np.array([(labels == f).sum() for f in names])
    correct = np.empty(len(X), dtype=bool)
    for i in range(len(X)):
        own = np.flatnonzero(names == labels[i])[0]
        C = sums.copy()
        C[own] -= Xc[i]  # leave the concept out of its own centroid
        C = C / np.maximum(counts - (np.arange(len(names)) == own), 1)[:, None]
        Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
        correct[i] = np.argmax(Cn @ Xn[i]) == own
    return correct


def bootstrap_draws(reps: dict[str, np.ndarray], labels: np.ndarray, names: list[str], target: np.ndarray, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Field-RSA of every representation on the same stratified concept resamples (one resample per draw,
    shared by all keys, so a given seed gives the same draws for a key whatever the other keys are)."""
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_boot) for k in reps}
    strata = [np.flatnonzero(labels == f) for f in names]
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata])  # same indices for every model
        for k in reps:
            draws[k][b] = field_rsa(reps[k][idx], labels[idx], names, target)
    return draws


def paired_bootstrap(reps: dict[str, np.ndarray], labels: np.ndarray, names: list[str], target: np.ndarray, n_boot: int, seed: int) -> dict:
    """Stratified concept bootstrap of field-RSA for several representations of the SAME concepts.

    Returns per-representation CIs and CIs of differences vs the first key ("trained").
    """
    keys = list(reps)
    draws = bootstrap_draws(reps, labels, names, target, n_boot, seed)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = {k: {"ci": ci(v)} for k, v in draws.items()}
    for k in keys[1:]:
        out[f"delta_vs_{k}"] = {"ci": ci(draws[keys[0]] - draws[k]), "p_le_0": float(np.mean(draws[keys[0]] - draws[k] <= 0))}
    return out


def bca_summary(reps: dict[str, np.ndarray], labels: np.ndarray, names: list[str], target: np.ndarray, n_boot: int, seed: int) -> dict:
    """Independent audit D11: bootstrap bias (bootstrap mean − point estimate) and BCa 95% intervals of field-RSA
    per representation and of its difference from the first key, from the same resamples as paired_bootstrap
    (same seed) and a leave-one-concept-out jackknife. Duplicated concepts make resampled centroids noisier, so
    resampled ρ is pulled toward 0 and the percentile interval sits below the point estimate."""
    from mathgeom.stats.bootstrap import bca_interval

    keys = list(reps)
    draws = bootstrap_draws(reps, labels, names, target, n_boot, seed)
    point = {k: field_rsa(v, labels, names, target) for k, v in reps.items()}
    loo = [np.arange(len(labels)) != i for i in range(len(labels))]
    jack = {k: np.array([field_rsa(v[m], labels[m], names, target) for m in loo]) for k, v in reps.items()}

    def summ(th, bt, jk):
        return {"point": th, "boot_mean": float(bt.mean()), "bias": float(bt.mean() - th), "bca": list(bca_interval(th, bt, jk)),
                "percentile": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}

    out = {k: summ(point[k], draws[k], jack[k]) for k in keys}
    for k in keys[1:]:
        out[f"delta_vs_{k}"] = summ(point[keys[0]] - point[k], draws[keys[0]] - draws[k], jack[keys[0]] - jack[k])
    return out


def paired_accuracy_bootstrap(correct: dict[str, np.ndarray], labels: np.ndarray, n_boot: int, seed: int) -> dict:
    """Stratified bootstrap of LOO accuracy and of accuracy differences vs the first key."""
    rng = np.random.default_rng(seed)
    keys = list(correct)
    strata = [np.flatnonzero(labels == f) for f in np.unique(labels)]
    draws = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata])
        for k in keys:
            draws[k][b] = correct[k][idx].mean()
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = {k: {"acc": float(correct[k].mean()), "ci": ci(draws[k])} for k in keys}
    for k in keys[1:]:
        out[f"delta_vs_{k}"] = {"ci": ci(draws[keys[0]] - draws[k]), "p_le_0": float(np.mean(draws[keys[0]] - draws[k] <= 0))}
    return out


def concept_level_regression(X: np.ndarray, df: pd.DataFrame, target_concept: np.ndarray, layer0: np.ndarray, random_rep: np.ndarray, n_perm: int, seed: int) -> dict:
    """SECONDARY: concept-level RDM regression (Freedman–Lane) of the trained RDM on the field
    distance with nuisances (layer-0, random-init twin, head noun, frequency, token count, lexical).
    Known limitation: residual confounding can inflate type-I error (see module docstring)."""
    preds = {
        "field_distance": target_concept,
        "random_init_rdm": cosine_rdm(center(random_rep.astype(np.float64))),
        "layer0_embedding": cosine_rdm(center(layer0.astype(np.float64))),
        "head_noun": same_label_rdm(df["head"].values),
        "zipf_diff": abs_diff_rdm(df["zipf"].values),
        "ntok_diff": abs_diff_rdm(df["n_tokens"].values),
        "char3_lexical": char_ngram_rdm(df["concept"].tolist()),
    }
    res = rdm_regression(cosine_rdm(center(X.astype(np.float64))), preds, "field_distance", n_perm=n_perm, seed=seed)
    return {"beta": res.beta, "t": res.t, "p": res.p, "r2": res.r2, "betas": res.betas}


def split_half_reliability(Xa: np.ndarray, Xb: np.ndarray, labels: np.ndarray, names: list[str]) -> float:
    """Spearman–Brown-corrected ρ between field-centroid RDMs from the two template halves."""
    r = spearmanr(upper(cosine_rdm(field_centroids(Xa, labels, names))), upper(cosine_rdm(field_centroids(Xb, labels, names))))[0]
    return float(2 * r / (1 + r))
