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


def offset_dots(ax, off, axes_names: list[str], conf: str, extra=None):
    """Dot plot of offset consistency per axis: the confirmatory model (filled), the other models (open),
    all twins (grey) and the confirmatory model's random-pair null (grey bar: mean → 95th percentile).
    `off(key, axis)` returns that lattice's offsets entry; `extra(axis)` an optional diamond value."""
    y = np.arange(len(axes_names))[::-1]
    for yi, a in zip(y, axes_names):
        o = off(conf, a)
        ax.plot([o["null_mean"], o["null95"]], [yi, yi], color=S.GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([off(k + "__random", a)["consistency"] for k in KEYS], [yi - 0.15] * len(KEYS), color=S.TWIN, s=14, zorder=2)
        ax.scatter([off(k, a)["consistency"] for k in KEYS if k != conf], [yi + 0.15] * (len(KEYS) - 1), facecolors=S.SURFACE, edgecolors=S.BLUE, s=18, zorder=3)
        ax.scatter(o["consistency"], yi, color=S.BLUE, s=48, zorder=4, edgecolors=S.SURFACE)
        if extra and extra(a) is not None:
            ax.scatter(extra(a), yi, marker="D", facecolors="none", edgecolors=S.INK2, s=26, zorder=5, linewidths=1.0)
    ax.axvline(0, color=S.AXIS, lw=0.7)
    ax.set_yticks(y, [f"+ {a}" for a in axes_names], fontsize=7.5)
    ax.set_xlabel("offset consistency (mean cosine of same-axis difference vectors)", fontsize=7.5)
    ax.scatter([], [], color=S.BLUE, label=f"{S.MODEL[conf]} ★"), ax.scatter([], [], facecolors=S.SURFACE, edgecolors=S.BLUE, label="other 4 models")
    if extra:
        ax.scatter([], [], marker="D", facecolors="none", edgecolors=S.INK2, label="★ without token-count direction")
    ax.scatter([], [], color=S.TWIN, label="5 twins"), ax.plot([], [], color=S.GRID, lw=5, label="★ random-pair null: mean → 95th pct")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=6.3)


def map_panel3d(ax, P: np.ndarray, fmap: dict, title: str, note: str) -> None:
    """One 3-D field map (3-D metric MDS) with the ground-truth neighbour edges, true proportions, fixed camera."""
    for i, j in fmap["edges"]:
        ax.plot(*P[[i, j]].T, color=S.AXIS, lw=0.7)
    ax.scatter(*P.T, c=[S.BLUE if p == "pure" else S.ORANGE for p in fmap["pa"]], s=18, depthshade=False, edgecolors=S.SURFACE, linewidths=0.6)
    for (x, y, z), n in zip(P, fmap["names"]):
        ax.text(x, y, z, " " + S.FIELD.get(n, n), fontsize=4.6, color=S.INK2)
    S.bare3d(ax, P, zoom=1.15)
    for setter, lab in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), ("MDS 1", "MDS 2", "MDS 3")):
        setter(lab, fontsize=6, color=S.MUTED, labelpad=-12)
    S.note3d(ax, note)
    ax.set_title(title, fontsize=9)


def draw_hero(r: dict, fam3d: dict, lat3d: dict, fmap: dict, conf: str):
    """Results at a glance, 3-D first. Row 1, axioms as directions: (A) the '+ commutativity' arrows of the
    confirmatory model and of its twin drawn from one origin in each model's top-3 PCs; (B) offset
    consistency of all four axioms in all models vs null and twins. Row 2, the field map: (C) rho with arXiv
    for all models vs twins; (D) 3-D MDS maps of arXiv, the model and its twin. Row 3, ordered families in
    their top-3 PCs: (E) days, angles, ring chain, space chain."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from mathgeom.report.figures.families import draw_plane3d
    from mathgeom.report.figures.lattice import _arrows_at_origin

    mp, la, sh, ph = r["map_fields"], r["lattice"], r["shapes"], r["posthoc"]
    Lp = mp[conf]["primary_layer"]
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.15, 1, 1, 1], hspace=0.42, wspace=0.22)
    # row 1 . axioms as directions
    for col, side in enumerate(("trained", "random_init")):
        V, pairs = lat3d[side], lat3d["pairs"]
        lim = max(np.linalg.norm(V[q] - V[p]) for p, q in pairs) * 1.05
        ax = fig.add_subplot(gs[0, col], projection="3d")
        _arrows_at_origin(ax, V, pairs, S.BLUE if side == "trained" else S.TWIN, lim)
        k = conf if side == "trained" else f"{conf}__random"
        c = la[k]["group_like"]["offsets"]["commutative"]["consistency"]
        ax.set_title(("A · Axioms act as directions\n" if col == 0 else "\n") + (f"{S.MODEL[conf]} ★" if side == "trained" else "its random-init twin"), fontsize=9)
        S.note3d(ax, f"'+ commutativity' arrows from one origin\nconsistency {c:.2f}")
    ax = fig.add_subplot(gs[0, 2:])
    lp = la[conf].get("audit2", {}).get("group_like_offsets", {}).get("length_projected", {})
    offset_dots(ax, lambda k, a: la[k]["group_like"]["offsets"][a], list(la[conf]["group_like"]["offsets"]), conf,
                extra=lambda a: lp[a]["consistency"] if a in lp else None)  # diamonds: word-length (token-count) control
    ax.set_title("B · All four axioms, all five models: parallel arrows, unlike the twins", fontsize=9)
    # row 2 . the field map
    ax = fig.add_subplot(gs[1, 0])
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
    ax.set_title("C · Fields form a map in every model", fontsize=9)
    m = mp[conf]["math_primary"]["rsa"]
    for col, (k, title, note) in enumerate((("arxiv", "D · arXiv cross-listing (ground truth)", "grey edges: each field's 2 nearest arXiv neighbours"),
                                             ("trained", f"{S.MODEL[conf]} ★: related fields sit together", f"ρ with arXiv = {m['trained']:.2f}"),
                                             ("random_init", "Its random-init twin: no map", f"ρ with arXiv = {m['random_init']:.2f}")), start=1):
        map_panel3d(fig.add_subplot(gs[1, col], projection="3d"), fmap["lay3"][k], fmap, title, note)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=S.BLUE, label="pure field"), Line2D([], [], marker="o", ls="", color=S.ORANGE, label="applied field")],
               loc="upper center", bbox_to_anchor=(0.62, 0.355), ncol=2, fontsize=6.5)
    # row 3 . ordered families in 3-D
    notes = {"days": f"cycle score {ph[conf]['pc_plane_cycle']['days'][Lp - 1]['score']:.2f} (twin {ph[conf + '__random']['pc_plane_cycle']['days'][Lp - 1]['score']:.2f})",
             "angles": f"d(0°, 330°) = {sh[conf]['angles']['closure']:.2f} × neighbour step (twin {sh[conf + '__random']['angles']['closure']:.2f})",
             "ring_chain": f"Robinson {sh[conf]['ring_chain']['robinson']:.2f} (twin {sh[conf + '__random']['ring_chain']['robinson']:.2f})",
             "space_chain": f"Robinson {sh[conf]['space_chain']['robinson']:.2f} (twin {sh[conf + '__random']['space_chain']['robinson']:.2f})"}
    titles = {"days": "E · Days: a circle", "angles": "Angles: not a circle", "ring_chain": "Ring chain: an ordered path", "space_chain": "Space chain: an ordered path"}
    for col, f in enumerate(("days", "angles", "ring_chain", "space_chain")):
        ax = fig.add_subplot(gs[2, col], projection="3d")
        P, var, labels = fam3d[f]
        draw_plane3d(ax, P, labels, f in ("days", "angles"), var, highlight=((len(P) - 1, 0),) if f == "angles" else (), size=5.0)
        S.note3d(ax, notes[f])
        ax.set_title(titles[f], fontsize=9)
    return fig


def draw_partial_operation(r: dict, conf: str):
    """Negative result: the partial-operation lattice (semigroupoid / small category / groupoid vs
    semigroup / monoid / group) shows no consistent 'make the operation total' direction."""
    import matplotlib.pyplot as plt

    la = r["lattice"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    axes_names = list(la[conf]["audit2"]["partial_operation"]["offsets"])
    offset_dots(ax, lambda k, a: la[k]["audit2"]["partial_operation"]["offsets"][a], axes_names, conf)
    ax.set_title("Partial vs total operations: no consistent direction (exploratory)", fontsize=9)
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


def forest_rows(r: dict) -> list[tuple[str, list[dict]]]:
    """(facet title, rows); each row: label, estimate, 95% CI, status ('pre-registered' / 'added'), n, and
    holm (the Holm decision where the test is in the pre-registered Holm family, else None)."""
    from mathgeom.report.render import holm_decisions

    mp, hi, b, cmp, ml = r["map_fields"], r["hierarchy"], r["baselines"], r["compare"], r["mathlib"]
    row = lambda label, est, ci, status, n, holm=None: {"label": label, "est": est, "ci": ci, "status": status, "n": n, "holm": holm}  # noqa: E731
    holm = lambda k: next(d["reject"] for d in holm_decisions(mp, r["shapes"], r["lattice"], k) if d["name"].startswith("H-A1"))  # noqa: E731
    d = lambda m: m["rsa"]["trained"] - m["rsa"]["random_init"]  # noqa: E731
    facets = [("Field map: trained − twin (Δρ, H-A1)", [row(S.MODEL[k], d(mp[k]["math_primary"]), mp[k]["math_primary"]["rsa_boot"]["delta_vs_random_init"]["ci"], "pre-registered",
                                                        f"{mp[k]['math_primary']['n']} concepts", holm(k)) for k in KEYS])]
    q = b["qwen3.5-2b"]
    facets.append(("Field map: LLM − text baselines (Δρ, Qwen3.5-2B)", [
        row("− Qwen3-Embedding", q["all_primary"]["rsa"]["llm_trained"] - q["all_primary"]["rsa"]["qwen3_embedding"], q["all_primary"]["boot"]["delta_vs_qwen3_embedding"]["ci"], "added", f"{q['all_primary']['n']} concepts"),
        row("− fastText", q["fasttext_covered"]["rsa"]["llm_trained"] - q["fasttext_covered"]["rsa"]["fasttext"], q["fasttext_covered"]["boot"]["delta_vs_fasttext"]["ci"], "added", f"{q['fasttext_covered']['n']} concepts")]))
    c = cmp["qwen2.5-math-1.5b_vs_qwen2.5-1.5b"]
    facets.append(("Math pretraining: Qwen2.5-Math − Qwen2.5 (Δρ, H-A3)", [row("1T math tokens", c["rho"]["qwen2.5-math-1.5b"] - c["rho"]["qwen2.5-1.5b"], c["boot"]["delta_vs_qwen2.5-1.5b"]["ci"],
                                                                              "pre-registered", f"{mp['qwen2.5-1.5b']['math_primary']['n']} concepts")]))
    facets.append(("Hierarchy: trained − twin (Δ cophenetic match, H-A2)", [row(S.MODEL[k], hi[k]["coph_match"]["trained"] - hi[k]["coph_match"]["random_init"], hi[k]["coph_match_delta_vs_random_ci"],
                                                                               "pre-registered", f"{hi[k]['n_fields']} fields") for k in KEYS]))
    facets.append(("Judges vs Mathlib depth (ρ, H-A4 validation)", [row(f"n = {ml['n_matched']} concepts", ml["rho_mean_rating_depth"], ml["ci95"], "added", f"{ml['n_matched']} concepts")]))
    return facets


def draw_forest(r: dict):
    import matplotlib.pyplot as plt

    facets = forest_rows(r)
    heights = [len(rows) + 0.8 for _, rows in facets]
    fig, axes = plt.subplots(len(facets), 1, figsize=(8.6, 0.32 * sum(heights) + 1.2), gridspec_kw={"height_ratios": heights, "hspace": 0.9})
    for ax, (title, rows) in zip(axes, facets):
        y = np.arange(len(rows))[::-1]
        for yi, rw in zip(y, rows):
            sig = S.excludes_zero(rw["ci"])
            ax.plot(rw["ci"], [yi, yi], color=S.BLUE if sig else S.TWIN, lw=2.2, solid_capstyle="round")
            ax.scatter(rw["est"], yi, color=S.BLUE if sig else S.TWIN, s=34, zorder=3, edgecolors=S.SURFACE)
            verdict = "CI excludes 0" if sig else "CI includes 0"
            if rw["holm"] is not None:  # in the pre-registered Holm family: the decision comes from Holm, not from the CI
                verdict += " · Holm: " + ("significant" if rw["holm"] else "not significant")
            ax.text(1.01, yi, verdict + ("" if rw["status"] == "pre-registered" else " · added"), transform=ax.get_yaxis_transform(), fontsize=6.5, color=S.INK2, va="center")
        ax.axvline(0, color=S.INK2, lw=0.7)
        ax.set_yticks(y, [rw["label"] for rw in rows], fontsize=7.3)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(title, fontsize=8.5)
        ax.grid(True, axis="x")
    return fig


def draw_group_objects(r: dict):
    """Group objects (topological / Lie / algebraic groups): parallelogram consistency per model vs its null
    and twin, and how close 'Lie group' sits to each of its two parents (nearest-neighbour rank)."""
    import matplotlib.pyplot as plt

    st = r["structures"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.3), gridspec_kw={"wspace": 0.55})
    y = np.arange(len(KEYS))[::-1]
    ax = axes[0]
    for yi, k in zip(y, KEYS):
        p, t = st[k]["group_objects"]["parallelograms"], st[k + "__random"]["group_objects"]["parallelograms"]
        ax.scatter(p["null95"], yi, marker="|", s=260, color=S.MUTED, linewidths=2, zorder=2)  # only the 95th percentile is stored
        ax.scatter(t["cos"], yi, color=S.TWIN, s=34, zorder=3)
        ax.scatter(p["cos"], yi, color=S.BLUE, s=44, zorder=4, edgecolors=S.SURFACE)
        ax.text(0.68, yi, S.fmt_p(p["p"]), fontsize=6.8, color=S.INK2, va="center")
    ax.set_yticks(y, [S.MODEL[k] for k in KEYS], fontsize=7.5)
    ax.set_xlim(-0.1, 0.8)
    ax.axvline(0, color=S.AXIS, lw=0.7)
    ax.set_xlabel("mean cosine of the 3 parallelograms\n(e.g. Lie group − topological group vs smooth manifold − topological space)", fontsize=7)
    ax.scatter([], [], color=S.BLUE, label="trained"), ax.scatter([], [], color=S.TWIN, label="random-init twin"), ax.scatter([], [], marker="|", s=120, color=S.MUTED, linewidths=2, label="random-quadruple null, 95th pct")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=3, fontsize=6.3)
    ax.set_title("Group objects combine their parents", fontsize=9)
    ax = axes[1]
    for yi, k in zip(y, KEYS):
        for kk, col in ((k, S.BLUE), (k + "__random", S.TWIN)):
            g, m = st[kk]["group_objects"]["parent_ranks"]["Lie group"]
            off = 0.13 if kk == k else -0.13
            ax.plot([g, m], [yi + off, yi + off], color=col, lw=1, alpha=0.5)
            ax.scatter(g, yi + off, color=col, s=26, marker="o", zorder=3)
            ax.scatter(m, yi + off, color=col, s=30, marker="D", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, [S.MODEL[k] for k in KEYS], fontsize=7.5)
    ax.set_xlabel(f"rank among {st[KEYS[0]]['group_objects']['pool_size']} family items (1 = nearest, log scale)", fontsize=7)
    ax.scatter([], [], color=S.INK2, marker="o", label="“group”"), ax.scatter([], [], color=S.INK2, marker="D", label="“smooth manifold”")
    ax.scatter([], [], color=S.BLUE, label="trained"), ax.scatter([], [], color=S.TWIN, label="twin")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=4, fontsize=6.3)
    ax.set_title("“Lie group”: how near each parent sits", fontsize=9)
    return fig


def build(r: dict, cfg_conf, outdir) -> list[str]:
    from mathgeom.report.figures.families import save_turntable, short
    from mathgeom.report.figures.lattice import save_axiom_gif

    conf = cfg_conf.models[0]["key"]  # the pre-registered confirmatory model
    Lp = r["map_fields"][conf]["primary_layer"]
    from mathgeom.analyses.lattice import axiom_pairs
    from mathgeom.config import load_yaml
    from mathgeom.report.figures.families import views3d
    from mathgeom.report.figures.lattice import _views3d

    fams = ("days", "angles", "ring_chain", "space_chain")
    v3 = views3d(cfg_conf, conf, list(fams), Lp)
    lab = lambda f, i, c: short(f, c)[:3] if f == "days" else (short(f, c) if f != "angles" or i in (0, 3, 6, 11) else "")  # noqa: E731
    fam3d = {f: (*v3[(conf, f)], [lab(f, i, c) for i, c in enumerate(figdata.family_items(cfg_conf, f))]) for f in fams}
    lat = load_yaml("families.yaml")["lattices"]["group_like"]
    names = list(lat["items"])
    bits = np.array([lat["items"][c] for c in names])
    X = {"trained": figdata.family_acts(cfg_conf, conf, "group_like", names)[:, Lp], "random_init": figdata.family_acts(cfg_conf, f"{conf}__random", "group_like", names)[:, Lp]}
    lat3d = {s: v[0] for s, v in _views3d(X).items()} | {"pairs": axiom_pairs(bits, lat["axioms"].index("commutative"))}
    setup = figdata.field_setup(cfg_conf, r["map_fields"][conf])
    fmap = {"lay3": figdata.field_layouts(figdata.field_rdms(cfg_conf, conf, setup, Lp), k=3), "names": setup["names"], "pa": setup["pa"],
            "edges": figdata.neighbour_edges(setup["arxiv"])}
    return [S.save(draw_hero(r, fam3d, lat3d, fmap, conf), outdir, "hero"), S.save(draw_depth(r), outdir, "depth_curves"), S.save(draw_forest(r), outdir, "forest"),
            S.save(draw_group_objects(r), outdir, "group_objects"), S.save(draw_partial_operation(r, conf), outdir, "partial_operation"),
            save_turntable(cfg_conf, conf, Lp, outdir),  # README 3-D turntable, confirmatory model only (≈1.5 MB per GIF)
            save_axiom_gif(cfg_conf, conf, r["lattice"], Lp, outdir)]  # README lead animation: axiom arrows in 3-D, to one origin
