"""H-A4 figure: the concrete ↔ abstract axis of the math-concept cloud, and the Mathlib check of the judges.

Global context: axes.json correlates each principal component of the primary confirmatory math
concepts with the mean judge abstraction rating after residualizing both on wordfreq Zipf frequency
and token count. The scatter reproduces exactly that computation (PC scores residualized the same way)
so the picture and the reported ρ describe the same numbers; the PC sign is flipped for display so
that "more abstract" points right. The Mathlib panel plots data/mathlib_depth.csv (matched concepts only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mathgeom.pipeline import load_acts
from mathgeom.report import style as S


def axis_data(cfg, key: str, Lp: int) -> dict:
    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    ratings = pd.read_csv(cfg.data_dir / "judge_ratings.csv")
    judges = [c for c in ratings.columns if "/" in c]
    abstr = ratings.set_index("concept")[judges].mean(1)
    mask = ((concepts.kind == "math") & concepts.primary & (concepts.split == cfg.raw.get("analysis_split", "confirmatory"))).values
    df = concepts[mask].reset_index(drop=True)
    acts = load_acts(cfg, key, "field_neutral")
    X = acts["last"][mask, Lp].astype(np.float64)
    X -= X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Z = np.c_[np.ones(len(df)), df.zipf.values, acts["n_tokens"][mask]]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731  (as in pipeline.phase_axes)
    pcs = np.stack([res(X @ Vt[k]) for k in range(2)], 1)
    ab = df.concept.map(abstr).values
    pcs[:, 0] *= np.sign(np.corrcoef(pcs[:, 0], res(ab))[0, 1])
    pcs[:, 1] *= np.sign(np.corrcoef(pcs[:, 1], res((df.pa == "applied").astype(float).values))[0, 1])
    return {"pcs": pcs, "abstraction": ab, "concept": df.concept.values, "pa": df.pa.values}


def draw(d: dict, ax_res: dict, ml: pd.DataFrame | None, mlres: dict | None, title: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3 if ml is not None else 2, figsize=(14 if ml is not None else 9.6, 4.7), gridspec_kw={"wspace": 0.28})
    P, ab = d["pcs"], d["abstraction"]
    ax = axes[0]
    sc = ax.scatter(P[:, 0], P[:, 1], c=ab, cmap=S.seq_cmap, s=14, edgecolors=S.SURFACE, linewidths=0.4)
    ext = list(np.argsort(ab)[:4]) + list(np.argsort(ab)[-4:])
    S.label_points(ax, P[ext], [d["concept"][i] for i in ext], size=6.3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("judge abstraction (1–7)", fontsize=7), cb.outline.set_visible(False)
    ax.set_xlabel(f"PC1 (residualized) · |ρ| with abstraction = {abs(ax_res['pcs'][0]['rho_abstraction']):.2f}", fontsize=7.5)
    ax.set_ylabel("PC2 (residualized)", fontsize=7.5)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{title}: concrete → abstract", fontsize=9)
    ax = axes[1]
    for lab, col in (("pure", S.BLUE), ("applied", S.ORANGE)):
        m = d["pa"] == lab
        ax.scatter(P[m, 0], P[m, 1], color=col, s=14, edgecolors=S.SURFACE, linewidths=0.4, label=lab, alpha=0.9)
    ax.set_xlabel("PC1 (residualized)", fontsize=7.5)
    ax.set_ylabel(f"PC2 · |ρ| with applied = {abs(ax_res['pcs'][1]['rho_applied']):.2f} (PC sign is arbitrary)", fontsize=7.5)
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(loc="lower right")
    ax.set_title("pure ↔ applied", fontsize=9)
    if ml is not None:
        ax = axes[2]
        ax.scatter(ml.depth, ml.mean_rating, color=S.BLUE, s=24, edgecolors=S.SURFACE, linewidths=0.6)
        lab = ml.sort_values("depth").iloc[[0, 1, len(ml) // 2, -2, -1]]
        S.label_points(ax, lab[["depth", "mean_rating"]].values.astype(float), lab.concept.tolist(), size=6.3)
        ax.set_xlabel("Mathlib import depth of the defining module", fontsize=7.5)
        ax.set_ylabel("mean judge abstraction", fontsize=7.5)
        ax.grid(True, axis="y")
        ax.set_title(f"Judges vs Mathlib: ρ = {mlres['rho_mean_rating_depth']:.2f}, 95% CI {S.fmt_ci(mlres['ci95'])}, n = {mlres['n_matched']}", fontsize=9)
    return fig


def build(cfg, ax_all: dict, mathlib: dict | None, primary: dict, outdir) -> list[str]:
    out = []
    ml = pd.read_csv(cfg.data_dir / "mathlib_depth.csv").dropna(subset=["depth"]) if mathlib else None
    for key in [k for k in ax_all if not k.endswith("__random")]:
        out.append(S.save(draw(axis_data(cfg, key, primary[key]), ax_all[key], ml, mathlib, S.pretty(key)), outdir, f"abstraction_{key}"))
    return out
