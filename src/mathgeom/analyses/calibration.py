"""Simulation study: type-I error and power of the RDM tests used (or rejected) in this project.

Global context: before analyzing any LLM data we asked whether partial RDM regression
(Freedman–Lane permutation) can remove lexical/frequency confounds. These simulations
reproduce the finding that drove the preregistered design: with a correctly specified
nuisance the test is calibrated, but when the nuisance only approximates the confounder
(item-level noise, nonlinear link, noisy proxy) false positives inflate — so the primary
tests are paired trained-vs-random-init contrasts. Scenarios:

  concept_linear_null     model RDM = nuisance RDM + pair noise              (correct spec.)
  concept_itemnoise_null  model = |z_i+e_i − z_j−e_j|, nuisance = |z_i − z_j|   (misspecified)
  field_pure_null         representations unrelated to fields               (field-level FL)
  field_confounded_null   representation & target share a latent confounder; nuisance is a
                          noisy proxy of it                                  (field-level FL)
  field_alternative       representation encodes target geometry beyond the confounder
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

from mathgeom.stats.rdm_regression import rdm_regression


def _block(labels):
    return (labels[:, None] != labels[None, :]).astype(float)


def _cos_centroid_rdm(X, lab, F):
    C = np.stack([X[lab == f].mean(0) for f in range(F)])
    return squareform(pdist(C - C.mean(0), "cosine"))


def run_calibration(n_sims: int = 200, n_perm: int = 199, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("concept_linear_null", "concept_itemnoise_null", "field_pure_null", "field_confounded_null", "field_alternative")}
    n, labels = 40, np.repeat(np.arange(4), 10)
    F, per, d = 17, 25, 64
    lab = np.repeat(np.arange(F), per)
    for s in range(n_sims):
        z = 0.8 * labels + rng.normal(size=n)
        nuis = squareform(pdist(z[:, None]))
        noise = rng.normal(size=(n, n)) * 0.5
        m_lin = nuis + (noise + noise.T) / 2
        np.fill_diagonal(m_lin, 0)
        out["concept_linear_null"].append(rdm_regression(m_lin, {"t": _block(labels), "n": nuis}, "t", n_perm=n_perm, seed=s).p)
        m_item = squareform(pdist((z + 0.5 * rng.normal(size=n))[:, None]))
        out["concept_itemnoise_null"].append(rdm_regression(m_item, {"t": _block(labels), "n": nuis}, "t", n_perm=n_perm, seed=s).p)

        zf = rng.normal(size=(F, 3))  # field-level latent confounder (e.g. lexical style)
        target = squareform(pdist(zf + 0.7 * rng.normal(size=(F, 3))))
        nuis_f = _cos_centroid_rdm(np.c_[zf[lab] + rng.normal(size=(len(lab), 3)), 0.1 * rng.normal(size=(len(lab), d))], lab, F)
        X0 = np.c_[zf[lab] + rng.normal(size=(len(lab), 3)), 0.3 * rng.normal(size=(len(lab), d))]
        out["field_confounded_null"].append(rdm_regression(_cos_centroid_rdm(X0, lab, F), {"t": target, "n": nuis_f}, "t", n_perm=n_perm, seed=s).p)
        Xp = rng.normal(size=(len(lab), d))
        out["field_pure_null"].append(rdm_regression(_cos_centroid_rdm(Xp, lab, F), {"t": target, "n": nuis_f}, "t", n_perm=n_perm, seed=s).p)
        emb = MDS(4, metric="precomputed", random_state=0, init="classical_mds", n_init=1).fit_transform(target)
        X1 = np.c_[zf[lab] + rng.normal(size=(len(lab), 3)), 0.5 * emb[lab] + rng.normal(size=(len(lab), 4)), 0.3 * rng.normal(size=(len(lab), d))]
        out["field_alternative"].append(rdm_regression(_cos_centroid_rdm(X1, lab, F), {"t": target, "n": nuis_f}, "t", n_perm=n_perm, seed=s).p)
    res = {}
    for k, v in out.items():
        v = np.array(v)
        r = float((v < 0.05).mean())
        se = float(np.sqrt(r * (1 - r) / len(v)))
        res[k] = {"reject_rate_0.05": r, "se": se, "n_sims": len(v), "n_perm": n_perm}
    return res
