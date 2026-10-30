"""H-B2 figures: the group-like axiom lattice (magma … abelian group, 4 axiom bits).

Global context: "adding an axiom is a direction" predicts that the difference vectors of all pairs
differing only in that axiom are parallel (offset consistency, lattice.json). Three views:
(1) the ideal geometry as a schematic (each axiom one fixed step);
(2) the unsupervised PCA plane, one facet per axiom with that axiom's arrows highlighted (emphasis
    instead of four hues, which no colour-vision-safe palette separates on one plot);
(3) the cosine matrix of all single-axiom difference vectors — bright diagonal blocks = consistent
    directions. Cross-axiom pairs sharing a vertex are hatched: with isotropic noise they sit at ≈ ±0.5
    whatever the geometry, so they carry no evidence.
"""

from __future__ import annotations

import json

import numpy as np

from mathgeom.analyses.lattice import axiom_pairs
from mathgeom.config import load_yaml
from mathgeom.report import figdata
from mathgeom.report import style as S

AX_LABEL = {"associative": "+ associativity", "identity": "+ identity", "divisible": "+ divisibility (inverses)", "commutative": "+ commutativity"}
HEAD = "-|>,head_length=0.8,head_width=0.45"  # filled arrowhead, about twice the default length so direction reads at a glance


def draw_schematic(ax, names: list[str], bits: np.ndarray, axioms: list[str]):
    """Ideal lattice on a 4×4 grid: x = 2·associative + divisible, y = 2·identity + commutative."""
    a, i, d, c = (bits[:, axioms.index(k)] for k in ("associative", "identity", "divisible", "commutative"))
    P = np.c_[2 * a + d, 2 * i + c].astype(float)
    for ax_i in range(len(axioms)):
        for p, q in axiom_pairs(bits, ax_i):
            # associativity and identity are 2-unit steps on this grid: bend them so they don't run through the node in
            # between (e.g. magma → semigroup would cross quasigroup and read as two pairs that are not in the lattice)
            bend = "arc3,rad=0.2" if np.abs(P[q] - P[p]).max() > 1 else "arc3"
            ax.annotate("", P[q], P[p], arrowprops=dict(arrowstyle=HEAD, color=S.AXIS, lw=0.9, shrinkA=6, shrinkB=6, mutation_scale=12, connectionstyle=bend))
    ax.scatter(P[:, 0], P[:, 1], s=40, color=S.INK2, zorder=3, edgecolors=S.SURFACE, linewidths=1.0)
    for (x, y), n, cb in zip(P, names, c):  # commutative rows label above, the others below
        two_lines = n.replace("commutative ", "comm.\n").replace(" ", "\n") if " " in n else n  # grid cells are one unit apart
        ax.text(x, y + (0.16 if cb else -0.16), two_lines, fontsize=5.6, ha="center", va="bottom" if cb else "top", color=S.INK2, linespacing=0.95)
    ax.set_xticks([0, 1, 2, 3], ["—", "+div", "+assoc", "+assoc\n+div"], fontsize=6.3)
    ax.set_yticks([0, 1, 2, 3], ["—", "+comm", "+id", "+id\n+comm"], fontsize=6.3)
    ax.set_xlim(-0.6, 3.6), ax.set_ylim(-0.6, 3.6)
    ax.set_title("Ideal: each axiom is one fixed step", fontsize=8.5)


def draw_facets(Ps: dict, names: list[str], bits: np.ndarray, axioms: list[str], notes: dict):
    """Schematic + one PCA facet per axiom; rows = trained, twin."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(axioms) + 1, figsize=(2.65 * (len(axioms) + 1), 5.7))
    draw_schematic(axes[0, 0], names, bits, axioms)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.5, "Each facet shows the same\nunsupervised PCA plane.\nArrows (blue; twin: grey):\nthe pairs that differ\nonly in that axiom.\nParallel arrows =\na consistent direction.",
                    fontsize=7, color=S.INK2, va="center", transform=axes[1, 0].transAxes)
    for row, k in enumerate(("trained", "random_init")):
        P = Ps[k]
        for col, name in enumerate(axioms, start=1):
            ax = axes[row, col]
            for a_i in range(len(axioms)):
                for p, q in axiom_pairs(bits, a_i):
                    hot = a_i == col - 1
                    ax.annotate("", P[q], P[p], arrowprops=dict(arrowstyle=HEAD if hot else "-", color=(S.BLUE if k == "trained" else S.TWIN) if hot else S.GRID,
                                                              lw=1.6 if hot else 0.7, shrinkA=4, shrinkB=4, mutation_scale=12), zorder=2 if hot else 1)
            ax.scatter(P[:, 0], P[:, 1], s=16, color=S.INK2, zorder=3, edgecolors=S.SURFACE, linewidths=0.8)
            if col == 1:
                S.label_points(ax, P, names, size=5.6)
            S.bare(ax)
            S.note(ax, notes[k][name])
            if row == 0:
                ax.set_title(AX_LABEL[name], fontsize=8.5)
            if col == 1:
                ax.set_ylabel("trained" if k == "trained" else "random-init twin", fontsize=8.5, color=S.INK2)
    fig.tight_layout(h_pad=1.0, w_pad=0.4)
    return fig


def draw_facets3d(Vs: dict, names: list[str], bits: np.ndarray, axioms: list[str], notes: dict):
    """The 3-D version of draw_facets: each model's top-3 PCs (true proportions, fixed camera), one facet
    per axiom with that axiom's single-axiom pairs as blue 3-D arrows (style.arrow3d) and the rest grey; rows =
    trained, twin. A 3-D view can still hide or exaggerate parallelism; the cosine matrix is the evidence."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.9 * len(axioms), 6.6))
    for row, k in enumerate(("trained", "random_init")):
        P, var = Vs[k]
        for col, name in enumerate(axioms):
            ax = fig.add_subplot(2, len(axioms), row * len(axioms) + col + 1, projection="3d")
            for a_i in range(len(axioms)):
                for p, q in axiom_pairs(bits, a_i):
                    if a_i == col:  # filled, camera-facing head of fixed size (style.arrow3d), so short arrows keep a visible direction
                        S.arrow3d(ax, P[p], P[q], S.BLUE if k == "trained" else S.TWIN, size=0.09 * np.ptp(P, axis=0).max(), lw=1.4)
                    else:
                        ax.plot(*P[[p, q]].T, color=S.GRID, lw=0.7)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=12, color=S.INK2, depthshade=False)
            if col == 0:
                for (x, y, z), t in zip(P, names):
                    ax.text(x, y, z, " " + t.replace("commutative ", "comm. "), fontsize=4.8, color=S.INK2)
            S.bare3d(ax, P, var)
            S.note3d(ax, notes[k][name])
            if row == 0:
                ax.set_title(AX_LABEL[name], fontsize=8.5)
            if col == 0:
                ax.text2D(-0.06, 0.5, "trained" if k == "trained" else "random-init twin", rotation=90, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color=S.INK2)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.07, wspace=0.05, hspace=0.25)
    return fig


def _views3d(X: dict) -> dict:
    """{side: (top-3 PC view, explained variance of PC1–5)} for one layer; the twin's signs aligned to the trained view."""
    Vt = figdata.pcs_by_layer(np.stack([X["trained"]] * 2, 1), 1, k=3)[0]
    return {"trained": (Vt, figdata.pca_var(X["trained"], 5)),
            "random_init": (figdata.pcs_by_layer(np.stack([X["random_init"]] * 2, 1), 1, onto=Vt, k=3)[0], figdata.pca_var(X["random_init"], 5))}


def _short(n: str) -> str:
    return n.replace("commutative ", "comm. ")


def _arrows_at_origin(ax, V: np.ndarray, pairs: list, color: str, lim: float, view: tuple = S.VIEW3D) -> None:
    """Step 3 of the explainer: every same-axiom difference vector drawn from one common origin."""
    for p, q in pairs:
        S.arrow3d(ax, np.zeros(3), V[q] - V[p], color, size=0.16 * lim, lw=1.6, view=view)
    ax.scatter([0], [0], [0], s=14, color=S.INK2, depthshade=False)
    S.bare3d(ax, np.array([[-lim] * 3, [lim] * 3]), zoom=1.3)
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
    ax.view_init(*view)


def draw_explainer(X: dict, names: list[str], bits: np.ndarray, axioms: list[str], off: dict, show: str, model: str):
    """How the axiom test works, in four steps, on real data (except the ground-truth schematic):
    1 the lattice of structures; 2 one axiom's pairs as arrows in the 3-D view (top-3 PCs) of the model;
    3 the same arrows from one common origin, model vs random-init twin; 4 a compass of full-space angles
    (figdata.axiom_angles) for every axiom, with the reported offset consistency (off[side][axiom])."""
    import matplotlib.pyplot as plt

    V = _views3d(X)
    a_i = axioms.index(show)
    pairs = axiom_pairs(bits, a_i)
    fig = plt.figure(figsize=(17, 6.0))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.7, 1.15, 0.8, 0.8], hspace=0.45, wspace=0.18)
    # 1 · the ground truth
    draw_schematic(fig.add_subplot(gs[:, 0]), names, bits, axioms)
    fig.axes[-1].set_title("1 · Structures are sets of axioms", fontsize=9.5)
    # 2 · arrows in the model's 3-D view
    ax = fig.add_subplot(gs[:, 1], projection="3d")
    P, var = V["trained"]
    for b in range(len(axioms)):
        for p, q in axiom_pairs(bits, b):
            if b != a_i:
                ax.plot(*P[[p, q]].T, color=S.GRID, lw=0.7)
    for p, q in pairs:
        S.arrow3d(ax, P[p], P[q], S.BLUE, size=0.09 * np.ptp(P, axis=0).max(), lw=1.6)
    ax.scatter(*P.T, s=12, color=S.INK2, depthshade=False)
    for (x, y, z), t in zip(P, names):
        ax.text(x, y, z, " " + _short(t), fontsize=5.2, color=S.INK2)
    S.bare3d(ax, P, var, zoom=1.35)
    ax.set_title(f"2 · Adding an axiom is an arrow\n{AX_LABEL[show]} in {model} (top-3 PCs)", fontsize=9.5)
    # 3 · line them up: one common origin, model vs twin
    for row, side in enumerate(("trained", "random_init")):
        Pv = V[side][0]
        lim = max(np.linalg.norm(Pv[q] - Pv[p]) for p, q in pairs) * 1.05
        ax = fig.add_subplot(gs[row, 2], projection="3d")
        _arrows_at_origin(ax, Pv, pairs, S.BLUE if side == "trained" else S.TWIN, lim)
        o = off[side][show]
        ax.set_title(("3 · Line the arrows up\n" if row == 0 else "") + ("model" if side == "trained" else "random-init twin"), fontsize=9.5 if row == 0 else 8.5)
        S.note3d(ax, f"consistency {o['consistency']:.2f}")
    # 4 · the compass: full-space angle of each arrow to the others' mean direction, for every axiom
    ang = {side: figdata.axiom_angles(X[side], bits, axioms) for side in ("trained", "random_init")}
    for k, name in enumerate(axioms):
        ax = fig.add_subplot(gs[k // 2, 3 + k % 2])
        t = np.linspace(0, np.pi, 60)
        ax.plot(np.cos(t), np.sin(t), color=S.GRID, lw=0.6), ax.plot(np.cos(t), -np.sin(t), color=S.GRID, lw=0.6)
        ax.axhline(0, color=S.AXIS, lw=0.6, ls=(0, (2, 2)))
        for side, sgn, col in (("trained", 1, S.BLUE), ("random_init", -1, S.TWIN)):
            for d in ang[side][name]["deg"]:
                r = np.radians(d)
                ax.annotate("", (np.cos(r), sgn * np.sin(r)), (0, 0), arrowprops=dict(arrowstyle=HEAD, color=col, lw=1.3, mutation_scale=9, shrinkA=0, shrinkB=0))
        ax.text(1.05, 0.04, "0°", fontsize=6.3, color=S.MUTED, va="bottom")
        ax.text(0.02, 1.0, "90°", fontsize=6.3, color=S.MUTED, va="bottom")
        ax.set_xlim(-0.45, 1.25), ax.set_ylim(-1.12, 1.12), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        o, w = off["trained"][name], off["random_init"][name]
        ax.set_title(f"{AX_LABEL[name]}", fontsize=8)
        ax.set_xlabel(f"consistency {o['consistency']:.2f} ({S.fmt_p(o['p'])})\ntwin {w['consistency']:.2f}", fontsize=6.6, color=S.INK2)
    fig.text(0.70, 0.97, "4 · Measure in the full space: angle of each arrow to the others' mean", fontsize=9.5, weight="bold", ha="left")
    fig.text(0.70, 0.935, "model above the dashed line (blue), twin below (grey); 0° = same direction, 90° = unrelated", fontsize=7, color=S.INK2, ha="left")
    return fig


def save_axiom_gif(cfg, key: str, la: dict, Lp: int, outdir, frames_per_axiom: int = 24, axioms_shown: list | None = None) -> str:
    """README animation, in 3-D: for each axiom, its pairs appear as arrows in the model's top-3-PC view,
    then slide to one common origin (do they point the same way?), model left, random-init twin right,
    while the camera turns a little (±15°, so the camera-facing arrowheads stay readable)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    lat = load_yaml("families.yaml")["lattices"]["group_like"]
    names, axioms = list(lat["items"]), lat["axioms"]
    bits = np.array([lat["items"][c] for c in names])
    X = {"trained": figdata.family_acts(cfg, key, "group_like", names)[:, Lp], "random_init": figdata.family_acts(cfg, f"{key}__random", "group_like", names)[:, Lp]}
    V = {s: v[0] for s, v in _views3d(X).items()}
    shown = axioms_shown or axioms
    lims = {s: np.abs(V[s]).max() * 1.15 for s in V}  # the box that holds every structure (frames in place)
    F = frames_per_axiom
    fig = plt.figure(figsize=(7.6, 4.2), dpi=100)
    axs = {s: fig.add_subplot(1, 2, i + 1, projection="3d") for i, s in enumerate(("trained", "random_init"))}

    def frame(n):
        a = shown[n // F]
        f = n % F
        u = 0.0 if F == 1 else min(1.0, max(0.0, (f / (F - 1) - 0.25) / 0.5))  # hold, slide, hold
        t = u * u * (3 - 2 * u)  # smoothstep
        view = (S.VIEW3D[0], S.VIEW3D[1] - 15 + 30 * n / max(1, len(shown) * F - 1))
        pairs = axiom_pairs(bits, axioms.index(a))
        for s, ax in axs.items():
            ax.clear()
            # zoom from the box that holds all points to one that this axiom's arrows fill, as they reach the origin
            P = V[s]
            lim = (1 - t) * lims[s] + t * 1.15 * max(np.linalg.norm(P[q] - P[p]) for p, q in pairs)
            ax.scatter(*P.T, s=10, color=S.INK2, alpha=1 - 0.7 * t, depthshade=False)
            if t < 0.15:
                for (x, y, z), nm in zip(P, names):
                    ax.text(x, y, z, " " + _short(nm), fontsize=5, color=S.INK2)
            for p, q in pairs:
                tail = (1 - t) * P[p]
                S.arrow3d(ax, tail, tail + P[q] - P[p], S.BLUE if s == "trained" else S.TWIN, size=0.14 * lim, lw=1.8, view=view)
            S.bare3d(ax, np.array([[-lim] * 3, [lim] * 3]), zoom=1.15)
            ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
            ax.view_init(*view)
            c = la[key if s == "trained" else f"{key}__random"]["group_like"]["offsets"][a]["consistency"]
            ax.set_title(f"{S.pretty(key) if s == 'trained' else 'random-init twin'} · consistency {c:.2f}", fontsize=8.5, color=S.INK if s == "trained" else S.INK2)
        fig.suptitle(f"{AX_LABEL[a]}: each arrow is a pair of structures that differ only in this axiom" + ("" if t < 0.5 else " — moved to one origin (zoomed)"), fontsize=8.5, color=S.INK2)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.0, wspace=0.0)
    anim = FuncAnimation(fig, frame, frames=len(shown) * F, interval=110)
    path = outdir / f"axiom_arrows_{key}.gif"
    anim.save(path, writer=PillowWriter(fps=9))
    plt.close(fig)
    return path.name


def draw_cosines(oms: dict, axioms: list[str], la_key: dict, la_twin: dict, titles: dict, lex: dict | None = None):
    """Cosine matrices of single-axiom difference vectors (trained, twin) + per-axiom dot plot; `lex` = the
    offsets of the names' character 3-grams (independent audit D2, results/<cfg>/lexical.json), drawn as ×."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, gs_axes = plt.subplots(1, 5, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.05, 0.85, 1.15], "wspace": 0.12})
    axes, cax = [gs_axes[0], gs_axes[1], gs_axes[4]], gs_axes[2]
    gs_axes[3].axis("off")
    for ax, k in zip(axes[:2], ("trained", "random_init")):
        om = oms[k]
        n = len(om["group"])
        C = np.ma.masked_where(np.eye(n, dtype=bool), om["cos"])  # self-cosines (= 1) carry no information
        im = ax.imshow(C, cmap=S.div_cmap, vmin=-1, vmax=1, interpolation="nearest")
        for i, j in zip(*np.nonzero(om["shared"])):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", lw=0, ec=S.SURFACE))
        bounds = [i for i in range(1, n) if om["group"][i] != om["group"][i - 1]]
        for b in bounds:
            ax.axhline(b - 0.5, color=S.INK2, lw=0.8), ax.axvline(b - 0.5, color=S.INK2, lw=0.8)
        starts = [0, *bounds, n]
        ax.set_xticks([(s + e - 1) / 2 for s, e in zip(starts[:-1], starts[1:])], [a[:5] + "." for a in axioms], fontsize=7)
        ax.set_yticks([(s + e - 1) / 2 for s, e in zip(starts[:-1], starts[1:])], [a[:5] + "." for a in axioms] if k == "trained" else [], fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(titles[k], fontsize=9)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cosine between difference vectors", fontsize=7)
    cb.outline.set_visible(False)
    ax = axes[2]
    y = np.arange(len(axioms))[::-1]
    for yi, name in zip(y, axioms):
        o, t = la_key["group_like"]["offsets"][name], la_twin["group_like"]["offsets"][name]
        ax.plot([o["null_mean"], o["null95"]], [yi, yi], color=S.GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter(t["consistency"], yi, color=S.TWIN, s=36, zorder=3, label="random-init twin" if yi == y[0] else None)
        lp = la_key.get("audit2", {}).get("group_like_offsets", {}).get("length_projected", {}).get(name)
        if lp:
            ax.scatter(lp["consistency"], yi - 0.18, facecolors=S.SURFACE, edgecolors=S.BLUE, s=30, zorder=3, label="trained, token-count direction removed" if yi == y[0] else None)
        if lex:
            ax.scatter(lex[name]["consistency"], yi + 0.18, marker="x", color=S.ORANGE, s=36, zorder=4, linewidths=1.5, label="names' character 3-grams (no model)" if yi == y[0] else None)
        ax.scatter(o["consistency"], yi, color=S.BLUE, s=42, zorder=4, label="trained" if yi == y[0] else None)
        ax.text(1.0, yi, S.fmt_p(o["p"]), fontsize=6.8, color=S.INK2, va="center", ha="right")
    ax.axvline(0, color=S.AXIS, lw=0.7)
    ax.set_yticks(y, [AX_LABEL[a] for a in axioms], fontsize=7.5)
    ax.set_xlim(-0.25, 1.0)
    ax.set_xlabel("offset consistency (mean cosine of same-axiom difference vectors)", fontsize=7.3)
    ax.plot([], [], color=S.GRID, lw=5, label="random-pair null: mean → 95th pct")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2, fontsize=6.5)
    ax.set_title("Offset consistency per axiom", fontsize=9)
    return fig


def build(cfg, la: dict, outdir) -> list[str]:
    out = []
    lat = load_yaml("families.yaml")["lattices"]["group_like"]
    names, axioms = list(lat["items"]), lat["axioms"]
    bits = np.array([lat["items"][c] for c in names])
    lx_path = cfg.results_dir / "lexical.json"
    lx = json.loads(lx_path.read_text()) if lx_path.exists() else {}
    for key in [k for k in la if not k.endswith("__random")]:
        Lp = la[key]["primary_layer"]
        X = {"trained": figdata.family_acts(cfg, key, "group_like", names)[:, Lp], "random_init": figdata.family_acts(cfg, f"{key}__random", "group_like", names)[:, Lp]}
        Ps = {k: figdata.unit_rms(figdata.pca(v)) for k, v in X.items()}
        notes = {k: {a: f"consistency {la[kk]['group_like']['offsets'][a]['consistency']:.2f} ({S.fmt_p(la[kk]['group_like']['offsets'][a]['p'])})" for a in axioms}
                 for k, kk in (("trained", key), ("random_init", f"{key}__random"))}
        out.append(S.save(draw_facets(Ps, names, bits, axioms, notes), outdir, f"lattice_axes_{key}"))
        Vt = figdata.pcs_by_layer(np.stack([X["trained"], X["trained"]], 1), 1, k=3)[0]  # own PCs; the twin's signs aligned to it
        Vs = {"trained": (Vt, figdata.pca_var(X["trained"], 5)),
              "random_init": (figdata.pcs_by_layer(np.stack([X["random_init"]] * 2, 1), 1, onto=Vt, k=3)[0], figdata.pca_var(X["random_init"], 5))}
        out.append(S.save(draw_facets3d(Vs, names, bits, axioms, notes), outdir, f"lattice3d_{key}"))
        # "how the test works": commutativity in steps 2-3 for every model (a fixed choice, not picked per model)
        off = {"trained": la[key]["group_like"]["offsets"], "random_init": la[f"{key}__random"]["group_like"]["offsets"]}
        out.append(S.save(draw_explainer(X, names, bits, axioms, off, "commutative", S.pretty(key)), outdir, f"axioms_explainer_{key}"))
        oms = {k: figdata.offset_matrix(v, bits, axioms) for k, v in X.items()}
        titles = {"trained": f"{S.pretty(key)} · layer {Lp}", "random_init": "random-init twin"}
        lex = lx.get(key, {}).get("group_like", {}).get("offsets", {}).get("char3")
        out.append(S.save(draw_cosines(oms, axioms, la[key], la[f"{key}__random"], titles, lex), outdir, f"lattice_cosines_{key}"))
    return out
