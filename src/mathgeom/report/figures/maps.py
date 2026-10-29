"""H-A1 figures: the field map (MDS layouts with ground-truth neighbour edges) and the distance matrices.

Global context: ρ is the Spearman correlation between the full 17×17 field-centroid distance matrix and
the arXiv distance matrix (map_fields.json). The MDS layouts are only a picture of those matrices; the
edges connect each field to its two nearest arXiv neighbours, so a model that preserves the map keeps
these edges short, while the random-init twin scatters them across the plane.
"""

from __future__ import annotations

import numpy as np

from mathgeom.report import figdata
from mathgeom.report import style as S


def draw_field_maps(lay: dict, names: list[str], pa: list[str], edges: list[tuple[int, int]], titles: dict, badges: dict):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
    for ax, k in zip(axes, ("arxiv", "trained", "random_init")):
        P = lay[k]
        for i, j in edges:
            ax.plot(*P[[i, j]].T, color=S.AXIS, lw=0.8, zorder=1)
        col = [S.BLUE if p == "pure" else S.ORANGE for p in pa]
        ax.scatter(P[:, 0], P[:, 1], c=col, s=42, zorder=3, edgecolors=S.SURFACE, linewidths=1.2)
        S.label_points(ax, P, [S.FIELD.get(n, n) for n in names], size=6.8)
        ax.set_title(titles[k], fontsize=9)
        S.bare(ax)
        S.note(ax, badges[k])
    handles = [Line2D([], [], marker="o", ls="", color=S.BLUE, label="pure"), Line2D([], [], marker="o", ls="", color=S.ORANGE, label="applied"),
               Line2D([], [], color=S.AXIS, lw=0.8, label="edge to each field's 2 nearest arXiv neighbours")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.1, wspace=0.05)
    return fig


def draw_rdms(rdms: dict, names: list[str], branch: list[str], order: list[int], titles: dict, badges: dict):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), gridspec_kw={"wspace": 0.06})
    o = np.array(order)
    bounds = [i for i in range(1, len(o)) if branch[o[i]] != branch[o[i - 1]]]
    for ax, k in zip(axes, ("arxiv", "trained", "random_init")):
        R = figdata.rank_unit(rdms[k])[np.ix_(o, o)]
        im = ax.imshow(np.ma.masked_where(np.eye(len(o), dtype=bool), R), cmap=S.seq_cmap, vmin=0, vmax=1, interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color=S.SURFACE, lw=2.2), ax.axvline(b - 0.5, color=S.SURFACE, lw=2.2)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(titles[k] + f"\n{badges[k]}", fontsize=9)
    axes[0].set_yticks(range(len(o)), [S.FIELD.get(names[i], names[i]) for i in o], fontsize=6.8)
    starts = [0, *bounds, len(o)]
    for a, b in zip(starts[:-1], starts[1:]):  # branch brackets under the first matrix
        axes[0].annotate(S.BRANCH.get(branch[o[a]], branch[o[a]]), xy=((a + b - 1) / 2, len(o) - 0.3), xycoords="data", ha="right", va="top",
                         rotation=35, fontsize=6.3, color=S.MUTED, annotation_clip=False)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("distance rank within matrix (0 = closest pair)", fontsize=7)
    cb.outline.set_visible(False)
    return fig


def build(cfg, mp: dict, outdir) -> list[str]:
    out = []
    for key, res in mp.items():
        Lp = res["primary_layer"]
        setup = figdata.field_setup(cfg, res)
        rdms = figdata.field_rdms(cfg, key, setup, Lp)
        lay = figdata.field_layouts(rdms)
        r, ci = res["math_primary"]["rsa"], res["math_primary"]["rsa_boot"]
        titles = {"arxiv": "arXiv cross-listing distance (ground truth)", "trained": f"{S.pretty(key)} · layer {Lp}", "random_init": "random-init twin (same tokenizer, no training)"}
        badges = {"arxiv": f"Ochiai distance over {setup['n_papers'] / 1e6:.1f}M arXiv papers",
                  "trained": f"ρ with arXiv = {r['trained']:.2f}, 95% CI {S.fmt_ci(ci['trained']['ci'])}", "random_init": f"ρ with arXiv = {r['random_init']:.2f}, 95% CI {S.fmt_ci(ci['random_init']['ci'])}"}
        edges = figdata.neighbour_edges(setup["arxiv"])
        out.append(S.save(draw_field_maps(lay, setup["names"], setup["pa"], edges, titles, badges), outdir, f"field_maps_{key}"))
        out.append(S.save(draw_rdms(rdms, setup["names"], setup["branch"], setup["order"], titles, badges), outdir, f"field_rdms_{key}"))
    return out
