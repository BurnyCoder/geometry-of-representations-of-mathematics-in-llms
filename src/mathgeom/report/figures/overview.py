"""Cross-model figures: the hero ("at a glance"), field map vs depth for all five models, and the
effect-size forest plot. Built by `run.py paper` from the merged results of all configs.

Global context: the hero uses the pre-registered confirmatory model (Qwen3.5-2B) for its pictures but
shows all five models wherever a number is summarized, so no single model is cherry-picked. Every value
is read from results/<cfg>/*.json; confidence intervals are the paired-bootstrap 95% intervals reported
there (unadjusted for multiplicity; the Holm decisions are in the scorecard table).
"""

from __future__ import annotations

import numpy as np

from mathgeom.report import figdata
from mathgeom.report import style as S

KEYS = list(S.MODEL)


def draw_hero(r: dict, planes: dict, conf: str):
    """(a) field-map ρ for all models vs twins; (b) days cycle vs angles line; (c) axiom offsets."""
    import matplotlib.pyplot as plt

    mp, la, sh, ph = r["map_fields"], r["lattice"], r["shapes"], r["posthoc"]
    fig = plt.figure(figsize=(14.5, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.85, 0.85, 1.35], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(KEYS))[::-1]
    for yi, k in zip(y, KEYS):
        m = mp[k]["math_primary"]
        ci = m["rsa_boot"]["trained"]["ci"]
        ax.plot(ci, [yi, yi], color=S.BLUE, lw=2, solid_capstyle="round", alpha=0.5)
        ax.scatter(m["rsa"]["random_init"], yi, color=S.TWIN, s=34, zorder=3)
        ax.scatter(m["rsa"]["trained"], yi, color=S.BLUE, s=44, zorder=4, edgecolors=S.SURFACE)
    ax.axvline(0, color=S.AXIS, lw=0.7)
    ax.set_yticks(y, [S.MODEL[k] + (" ★" if k == conf else "") for k in KEYS], fontsize=7.5)
    ax.set_xlabel("ρ(field-centroid distances, arXiv distances)", fontsize=7.5)
    ax.scatter([], [], color=S.BLUE, label="trained (95% CI)"), ax.scatter([], [], color=S.TWIN, label="random-init twin")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.5)
    ax.set_title("A · Fields form a map in every model", fontsize=9)
    Lp = mp[conf]["primary_layer"]
    from mathgeom.report.figures.families import draw_plane

    for col, (f, title, closed) in enumerate((("days", "B · Days: a circle", True), ("angles", "C · Angles: not a circle", False)), start=1):
        ax = fig.add_subplot(gs[col])
        P, items = planes[f]
        draw_plane(ax, P, items, closed=closed, size=6, highlight=((len(P) - 1, 0),) if f == "angles" else ())
        if f == "days":
            S.note(ax, f"cycle score {ph[conf]['pc_plane_cycle']['days'][Lp - 1]['score']:.2f} (twin {ph[conf + '__random']['pc_plane_cycle']['days'][Lp - 1]['score']:.2f})")
        else:
            S.note(ax, f"d(0°, 330°) = {sh[conf]['angles']['closure']:.2f} × neighbour distance")
        ax.set_title(title, fontsize=9)
    ax = fig.add_subplot(gs[3])
    axioms = list(la[conf]["group_like"]["offsets"])
    y = np.arange(len(axioms))[::-1]
    for yi, a in zip(y, axioms):
        o = la[conf]["group_like"]["offsets"][a]
        ax.plot([o["null_mean"], o["null95"]], [yi, yi], color=S.GRID, lw=5, solid_capstyle="round", zorder=1)
        others = [la[k]["group_like"]["offsets"][a]["consistency"] for k in KEYS if k != conf]
        twins = [la[k + "__random"]["group_like"]["offsets"][a]["consistency"] for k in KEYS]
        ax.scatter(twins, [yi - 0.15] * len(twins), color=S.TWIN, s=14, zorder=2)
        ax.scatter(others, [yi + 0.15] * len(others), facecolors=S.SURFACE, edgecolors=S.BLUE, s=18, zorder=3)
        ax.scatter(o["consistency"], yi, color=S.BLUE, s=48, zorder=4, edgecolors=S.SURFACE)
    ax.axvline(0, color=S.AXIS, lw=0.7)
    ax.set_yticks(y, [f"+ {a}" for a in axioms], fontsize=7.5)
    ax.set_xlabel("offset consistency (mean cosine of same-axiom differences)", fontsize=7.5)
    ax.scatter([], [], color=S.BLUE, label=f"{S.MODEL[conf]} ★"), ax.scatter([], [], facecolors=S.SURFACE, edgecolors=S.BLUE, label="other 4 models")
    ax.scatter([], [], color=S.TWIN, label="5 twins"), ax.plot([], [], color=S.GRID, lw=5, label="null: mean → 95th pct")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=6.3)
    ax.set_title("D · Axioms act as directions", fontsize=9)
    return fig


def draw_depth(r: dict):
    import matplotlib.pyplot as plt

    mp = r["map_fields"]
    fig, axes = plt.subplots(1, len(KEYS), figsize=(14, 3.1), sharey=True)
    for ax, k in zip(axes, KEYS):
        lc = mp[k]["math_primary"]["layer_curve"]
        L = len(lc["trained"])
        x = np.arange(1, L + 1) / L
        ax.axhline(lc["max_null95"], color=S.MUTED, lw=0.8)
        ax.plot(x, lc["random_init"], color=S.TWIN, label="random-init twin")
        ax.plot(x, lc["trained"], color=S.BLUE, label="trained")
        ax.axvline(mp[k]["primary_layer"] / L, color=S.AXIS, lw=0.8)
        ax.axhline(0, color=S.GRID, lw=0.6)
        ax.set_title(S.MODEL[k], fontsize=9)
        ax.set_xlabel("relative depth", fontsize=7.5)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("ρ with arXiv distances", fontsize=7.5)
    axes[0].text(0.52, axes[0].get_ylim()[1] * 0.95, "pre-registered\nlayer", fontsize=6.3, color=S.MUTED, va="top")
    axes[-1].text(0.98, mp[KEYS[-1]]["math_primary"]["layer_curve"]["max_null95"] + 0.01, "max-over-layers\nnull, 95th pct", fontsize=6.3, color=S.MUTED, ha="right", va="bottom")
    axes[-1].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    return fig


def forest_rows(r: dict) -> list[tuple[str, list[tuple[str, float, list, str]]]]:
    """(facet title, [(label, estimate, ci, status)]) — status: 'pre-registered' or 'added'."""
    mp, hi, b, cmp, ml = r["map_fields"], r["hierarchy"], r["baselines"], r["compare"], r["mathlib"]
    d = lambda m: m["rsa"]["trained"] - m["rsa"]["random_init"]  # noqa: E731
    facets = [("Field map: trained − twin (Δρ, H-A1)", [(S.MODEL[k], d(mp[k]["math_primary"]), mp[k]["math_primary"]["rsa_boot"]["delta_vs_random_init"]["ci"], "pre-registered") for k in KEYS])]
    q = b["qwen3.5-2b"]
    facets.append(("Field map: LLM − text baselines (Δρ, Qwen3.5-2B)", [
        ("− Qwen3-Embedding", q["all_primary"]["rsa"]["llm_trained"] - q["all_primary"]["rsa"]["qwen3_embedding"], q["all_primary"]["boot"]["delta_vs_qwen3_embedding"]["ci"], "added"),
        ("− fastText", q["fasttext_covered"]["rsa"]["llm_trained"] - q["fasttext_covered"]["rsa"]["fasttext"], q["fasttext_covered"]["boot"]["delta_vs_fasttext"]["ci"], "added")]))
    c = cmp["qwen2.5-math-1.5b_vs_qwen2.5-1.5b"]
    facets.append(("Math pretraining: Qwen2.5-Math − Qwen2.5 (Δρ, H-A3)", [("1T math tokens", c["rho"]["qwen2.5-math-1.5b"] - c["rho"]["qwen2.5-1.5b"], c["boot"]["delta_vs_qwen2.5-1.5b"]["ci"], "pre-registered")]))
    facets.append(("Hierarchy: trained − twin (Δ cophenetic match, H-A2)", [(S.MODEL[k], hi[k]["coph_match"]["trained"] - hi[k]["coph_match"]["random_init"], hi[k]["coph_match_delta_vs_random_ci"], "pre-registered") for k in KEYS]))
    facets.append(("Judges vs Mathlib depth (ρ, H-A4 validation)", [(f"n = {ml['n_matched']} concepts", ml["rho_mean_rating_depth"], ml["ci95"], "added")]))
    return facets


def draw_forest(r: dict):
    import matplotlib.pyplot as plt

    facets = forest_rows(r)
    heights = [len(rows) + 0.8 for _, rows in facets]
    fig, axes = plt.subplots(len(facets), 1, figsize=(8.6, 0.32 * sum(heights) + 1.2), gridspec_kw={"height_ratios": heights, "hspace": 0.9})
    for ax, (title, rows) in zip(axes, facets):
        y = np.arange(len(rows))[::-1]
        for yi, (lab, est, ci, status) in zip(y, rows):
            sig = S.excludes_zero(ci)
            ax.plot(ci, [yi, yi], color=S.BLUE if sig else S.TWIN, lw=2.2, solid_capstyle="round")
            ax.scatter(est, yi, color=S.BLUE if sig else S.TWIN, s=34, zorder=3, edgecolors=S.SURFACE)
            ax.text(1.01, yi, ("CI excludes 0" if sig else "CI includes 0") + ("" if status == "pre-registered" else " · added"), transform=ax.get_yaxis_transform(),
                    fontsize=6.5, color=S.INK2, va="center")
        ax.axvline(0, color=S.INK2, lw=0.7)
        ax.set_yticks(y, [lab for lab, *_ in rows], fontsize=7.3)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(title, fontsize=8.5)
        ax.grid(True, axis="x")
    return fig


def build(r: dict, cfg_conf, outdir) -> list[str]:
    from mathgeom.report.figures.families import short

    conf = cfg_conf.models[0]["key"]  # the pre-registered confirmatory model
    Lp = r["map_fields"][conf]["primary_layer"]
    planes = {f: (figdata.pca(figdata.family_acts(cfg_conf, conf, f)[:, Lp]), [short(f, c)[:3] if f == "days" else (short(f, c) if i in (0, 3, 6, 11) else "") for i, c in enumerate(figdata.family_items(cfg_conf, f))])
              for f in ("days", "angles")}
    return [S.save(draw_hero(r, planes, conf), outdir, "hero"), S.save(draw_depth(r), outdir, "depth_curves"), S.save(draw_forest(r), outdir, "forest")]
