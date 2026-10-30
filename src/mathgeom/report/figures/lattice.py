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
            ax.annotate("", P[q], P[p], arrowprops=dict(arrowstyle=HEAD, color=S.AXIS, lw=0.9, shrinkA=6, shrinkB=6, mutation_scale=12))
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
    axes[1, 0].text(0.0, 0.5, "Each facet shows the same\nunsupervised PCA plane.\nBlue arrows: the pairs that\ndiffer only in that axiom.\nParallel blue arrows =\na consistent direction.",
                    fontsize=7, color=S.INK2, va="center", transform=axes[1, 0].transAxes)
    for row, k in enumerate(("trained", "random_init")):
        P = Ps[k]
        for col, name in enumerate(axioms, start=1):
            ax = axes[row, col]
            for a_i in range(len(axioms)):
                for p, q in axiom_pairs(bits, a_i):
                    hot = a_i == col - 1
                    ax.annotate("", P[q], P[p], arrowprops=dict(arrowstyle=HEAD if hot else "-", color=S.BLUE if hot else S.GRID,
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
                        S.arrow3d(ax, P[p], P[q], S.BLUE, size=0.09 * np.ptp(P, axis=0).max(), lw=1.4)
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


def draw_cosines(oms: dict, axioms: list[str], la_key: dict, la_twin: dict, titles: dict):
    """Cosine matrices of single-axiom difference vectors (trained, twin) + per-axiom dot plot."""
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
        oms = {k: figdata.offset_matrix(v, bits, axioms) for k, v in X.items()}
        titles = {"trained": f"{S.pretty(key)} · layer {Lp}", "random_init": "random-init twin"}
        out.append(S.save(draw_cosines(oms, axioms, la[key], la[f"{key}__random"], titles), outdir, f"lattice_cosines_{key}"))
    return out
