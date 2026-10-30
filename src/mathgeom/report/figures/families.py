"""H-B figures: cyclic families, ordered chains, the circle forming across depth (PNG + GIF), and angles.

Global context: every panel is the unsupervised top-2-PC plane of one family at one layer (no labels
used to fit it); colour runs light→dark along the ground-truth order, and the path connects consecutive
items (closed for cyclic families). Statistics are read from shapes.json / posthoc.json /
chain_frequency.json. Days and months are the positive control (Engels et al. 2024, arXiv:2405.14860);
the PC-plane cycle score is post-hoc (docs/deviations.md).
"""

from __future__ import annotations

import numpy as np

from mathgeom.config import load_yaml
from mathgeom.report import figdata
from mathgeom.report import style as S

CYCLIC = ["days", "months", "angles"]
CHAINS = ["ring_chain", "space_chain", "number_systems", "ngons", "dimensions"]
DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]  # fixed relative depths for the filmstrip (not chosen by result)


def short(family: str, item: str) -> str:
    if family == "months":
        return item[:3]
    if family in ("angles", "digits_0_330"):
        return item.replace(" degrees", "°") + ("" if "°" in item or family == "digits_0_330" else "")
    return item.replace("-dimensional", "-dim.").replace(" space", "").replace("algebraically closed", "alg. closed")


def draw_plane(ax, P: np.ndarray, labels: list[str], closed: bool, label: bool = True, size: float = 6.5, highlight: tuple = ()):
    seq = np.r_[np.arange(len(P)), 0] if closed else np.arange(len(P))
    ax.plot(P[seq, 0], P[seq, 1], color=S.AXIS, lw=1.0, zorder=1)
    for i, j in highlight:  # e.g. the predicted 330°→0° closing edge
        ax.plot(*P[[i, j]].T, color=S.ORANGE, lw=1.6, zorder=2)
    ax.scatter(P[:, 0], P[:, 1], c=S.ordinal(len(P)), s=34, zorder=3, edgecolors=S.SURFACE, linewidths=1.0)
    if label:
        S.label_points(ax, P, labels, size=size)
    S.bare(ax)


def _cycle(ph: dict, key: str, fam: str, layer: int) -> dict:
    return ph[key]["pc_plane_cycle"][fam][layer - 1]


def draw_grid(cfg, key: str, fams: list[str], Lp: int, stat, closed_of) -> object:
    """2 rows (trained, twin) × len(fams) PC planes at the pre-registered layer."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(fams), figsize=(2.75 * len(fams), 5.9))
    for row, k in enumerate((key, f"{key}__random")):
        for col, f in enumerate(fams):
            X = figdata.family_acts(cfg, k, f)[:, Lp]
            items = figdata.family_items(cfg, f)
            ax = axes[row, col]
            hl = ((len(items) - 1, 0),) if f == "angles" else ()
            draw_plane(ax, figdata.pca(X), [short(f, c) for c in items], closed_of(f), highlight=hl)
            S.note(ax, stat(k, f))
            if row == 0:
                ax.set_title(S.FAMILY.get(f, f), fontsize=8.5)
            if col == 0:
                ax.set_ylabel("trained" if row == 0 else "random-init twin", fontsize=8.5, color=S.INK2)
    fig.tight_layout(h_pad=1.2, w_pad=0.6)
    return fig


def draw_filmstrip(cfg, key: str, ph: dict, L: int, Lp: int):
    """Days and months at fixed relative depths (aligned to the pre-registered layer), plus the cycle score vs depth."""
    import matplotlib.pyplot as plt

    layers = [max(1, int(round(d * L))) for d in DEPTHS]
    fig, axes = plt.subplots(2, len(layers) + 1, figsize=(2.35 * (len(layers) + 1), 4.9), gridspec_kw={"width_ratios": [1] * len(layers) + [1.35]})
    for row, f in enumerate(("days", "months")):
        X = figdata.family_acts(cfg, key, f)
        planes = figdata.planes_by_layer(X, Lp)
        items = [short(f, c) for c in figdata.family_items(cfg, f)]
        for col, l in enumerate(layers):
            ax = axes[row, col]
            draw_plane(ax, planes[l - 1], items, closed=True, label=f == "days" or col == len(layers) - 1, size=6)
            c = _cycle(ph, key, f, l)
            S.note(ax, f"cycle score {c['score']:.2f}")
            if row == 0:
                ax.set_title(f"depth {l / L:.2f} · layer {l}", fontsize=8)
            if col == 0:
                ax.set_ylabel(S.FAMILY[f].split("\n")[0], fontsize=8.5, color=S.INK2)
        ax = axes[row, -1]
        x = np.arange(1, L + 1) / L
        for k, colr, lab in ((key, S.TRAINED, "trained"), (f"{key}__random", S.TWIN, "random-init twin")):
            ax.plot(x, [e["score"] for e in ph[k]["pc_plane_cycle"][f]], color=colr, label=lab)
        ax.axvline(Lp / L, color=S.MUTED, lw=0.7)
        ax.text(Lp / L, 1.04, "pre-registered", fontsize=6.5, color=S.MUTED, ha="center")
        ax.set_ylim(0, 1.1), ax.set_xlim(0, 1)
        ax.set_xlabel("relative depth", fontsize=7.5), ax.set_ylabel("PC-plane cycle score", fontsize=7.5)
        if row == 0:
            ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.72), fontsize=6.5)
    fig.tight_layout(h_pad=1.0, w_pad=0.5)
    return fig


def save_gif(cfg, key: str, ph: dict, L: int, Lp: int, outdir) -> str:
    """Animated PC planes of days and months through every layer, trained (top) and its random-init twin
    (bottom, the control), for the README; post-hoc cycle score per frame."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    rows = (key, f"{key}__random")
    data = {(k, f): (figdata.planes_by_layer(figdata.family_acts(cfg, k, f), Lp), [short(f, c) for c in figdata.family_items(cfg, f)])
            for k in rows for f in ("days", "months")}
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.2), dpi=100)
    lim = {k: max(np.abs(data[(k, f)][0]).max() for f in ("days", "months")) * 1.25 for k in rows}  # one scale per row

    def frame(l):
        for (k, f), ax in zip(data, axes.flat):
            planes, items = data[(k, f)]
            ax.clear()
            P = planes[l]
            seq = np.r_[np.arange(len(P)), 0]
            ax.plot(P[seq, 0], P[seq, 1], color=S.AXIS, lw=1.0)
            ax.scatter(P[:, 0], P[:, 1], c=S.ordinal(len(P)), s=30, zorder=3, edgecolors=S.SURFACE, linewidths=1.0)
            S.bare(ax)
            ax.set_aspect("equal", adjustable="box")  # keep the fixed per-row limits in every frame (no rescaling between frames)
            ax.set_xlim(-lim[k], lim[k]), ax.set_ylim(-lim[k], lim[k])
            S.label_points(ax, P, items, size=6.5)  # after the limits: adjustText works in display coordinates
            ax.set_title(f"{S.FAMILY[f].split(chr(10))[0]} · cycle score {_cycle(ph, k, f, l + 1)['score']:.2f}", fontsize=8.5)
            if f == "days":  # row label: which network this row shows
                ax.set_ylabel(S.pretty(key) if k == key else "random-init twin", fontsize=9, color=S.INK if k == key else S.INK2)
        fig.suptitle(f"layer {l + 1} of {L} (depth {(l + 1) / L:.2f}) · cycle score 1 = every item's angular neighbour is its true successor; an open arc also scores 1 (post-hoc)", fontsize=8.5, color=S.INK2)

    anim = FuncAnimation(fig, frame, frames=L, interval=450)
    path = outdir / f"circle_depth_{key}.gif"
    anim.save(path, writer=PillowWriter(fps=2.5))
    plt.close(fig)
    return path.name


def views3d(cfg, key: str, fams: list[str], Lp: int) -> dict:
    """{(model key, family): (top-3 PC view, explained variance of PC1–5)} at the preregistered layer; each
    axis is that model's own PC (the twin's signs aligned to the trained view), as on the site."""
    out = {}
    for f in fams:
        Xt, Xw = (figdata.family_acts(cfg, k, f)[:, [0, Lp]] for k in (key, f"{key}__random"))
        Vt = figdata.pcs_by_layer(Xt, 1, k=3)[0]
        out[(key, f)] = (Vt, figdata.pca_var(Xt[:, 1], 5))
        out[(f"{key}__random", f)] = (figdata.pcs_by_layer(Xw, 1, onto=Vt, k=3)[0], figdata.pca_var(Xw[:, 1], 5))
    return out


def draw_plane3d(ax, P: np.ndarray, labels: list[str], closed: bool, var, highlight: tuple = (), size: float = 5.6):
    """One family in its top-3 PCs: the path in true order, a grey shadow of it on the floor (depth cue),
    ordinal colour, labels; true proportions and the fixed camera (style.bare3d)."""
    seq = np.r_[np.arange(len(P)), 0] if closed else np.arange(len(P))
    ax.plot(P[seq, 0], P[seq, 1], np.full(len(seq), P[:, 2].min()), color=S.GRID, lw=0.8)
    ax.plot(P[seq, 0], P[seq, 1], P[seq, 2], color=S.AXIS, lw=1.0)
    for i, j in highlight:  # e.g. the predicted 330°→0° closing edge
        ax.plot(*P[[i, j]].T, color=S.ORANGE, lw=1.6)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=S.ordinal(len(P)), s=24, edgecolors=S.SURFACE, linewidths=0.7, depthshade=False)
    for (x, y, z), t in zip(P, labels):
        ax.text(x, y, z, " " + t, fontsize=size, color=S.INK2)
    S.bare3d(ax, P, var)


def planes_note(ph: dict, k: str, f: str, Lp: int) -> str:
    """The post-hoc closed-cycle result (reports/prereg_addendum_3d.md) for a cyclic family, if computed."""
    e = ph.get(k, {}).get("pc_planes_cycle", {}).get(f)
    if not e:
        return ""
    e = e[Lp - 1]
    return f"\nclosed cycle, best plane PC{e['plane'][0] + 1}–PC{e['plane'][1] + 1}: {e['score']:.2f} ({S.fmt_p(e['p'])})"


def draw_grid3d(cfg, key: str, fams: list[str], Lp: int, closed_of, extra=lambda k, f: ""):
    """2 rows (trained, twin) × families, each in its top-3 PCs at the pre-registered layer."""
    import matplotlib.pyplot as plt

    V = views3d(cfg, key, fams, Lp)
    fig = plt.figure(figsize=(2.9 * len(fams), 6.6))
    for row, k in enumerate((key, f"{key}__random")):
        for col, f in enumerate(fams):
            ax = fig.add_subplot(2, len(fams), row * len(fams) + col + 1, projection="3d")
            P, var = V[(k, f)]
            items = figdata.family_items(cfg, f)
            draw_plane3d(ax, P, [short(f, c) for c in items], closed_of(f), var, highlight=((len(items) - 1, 0),) if f == "angles" else ())
            S.note3d(ax, f"PC1–3 hold {100 * var[:3].sum():.0f}% of the variance" + extra(k, f))
            if row == 0:
                ax.set_title(S.FAMILY.get(f, f), fontsize=8.5)
            if col == 0:
                ax.text2D(-0.06, 0.5, "trained" if row == 0 else "random-init twin", rotation=90, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color=S.INK2)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.07, wspace=0.05, hspace=0.28)
    return fig


def save_turntable(cfg, key: str, Lp: int, outdir, frames: int = 36) -> str:
    """README turntable: days, months and angles in their top-3 PCs (trained top, twin bottom) at the
    preregistered layer, rotating once about the vertical axis (matplotlib FuncAnimation + PillowWriter)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    V = views3d(cfg, key, CYCLIC, Lp)
    fig = plt.figure(figsize=(7.2, 5.0), dpi=100)
    axes = []
    for row, k in enumerate((key, f"{key}__random")):
        for col, f in enumerate(CYCLIC):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            P, var = V[(k, f)]
            items = figdata.family_items(cfg, f)
            draw_plane3d(ax, P, [short(f, c) for c in items], True, var, highlight=((len(items) - 1, 0),) if f == "angles" else (), size=5.0)
            ax.set_title(f"{S.FAMILY[f].split(chr(10))[0]} · {S.pretty(key) if k == key else 'twin'}", fontsize=7.5, color=S.INK if k == key else S.INK2)
            axes.append(ax)
    fig.suptitle(f"top-3 principal components at layer {Lp} (preregistered) · trained (top) vs random-init twin (bottom)", fontsize=7.5, color=S.INK2)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.02, wspace=0.0, hspace=0.12)

    def frame(i):
        for ax in axes:
            ax.view_init(S.VIEW3D[0], S.VIEW3D[1] + 360 * i / frames)

    anim = FuncAnimation(fig, frame, frames=frames, interval=80)
    path = outdir / f"turntable_{key}.gif"
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
    return path.name


def draw_angles(cfg, key: str, sh: dict, Lp: int):
    """Pre-registered prediction (a circle where 330° neighbours 0°) next to what the model shows."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    items = figdata.family_items(cfg, "angles")
    lab = [short("angles", c) for c in items]
    t = np.deg2rad(np.arange(len(items)) * 30)
    draw_plane(axes[0], np.c_[np.sin(t), np.cos(t)], lab, closed=True, highlight=((len(items) - 1, 0),))
    axes[0].set_title("Predicted: a circle (330° next to 0°)", fontsize=9)
    draw_plane(axes[1], figdata.pca(figdata.family_acts(cfg, key, "angles")[:, Lp]), lab, closed=False, highlight=((len(items) - 1, 0),))
    a = sh[key]["angles"]
    axes[1].set_title(f"Found: {S.pretty(key)} · layer {Lp}", fontsize=9)
    S.note(axes[1], f"closure d(0°, 330°) / mean neighbour distance = {a['closure']:.2f} (circle: < 2)\n"
                    f"circle ρ = {a['circle_rho']:.2f} vs line ρ = {a['line_rho']:.2f} (circle: circle ρ > line ρ)")
    digits = figdata.family_items(cfg, "digits_0_330")
    draw_plane(axes[2], figdata.pca(figdata.family_acts(cfg, key, "digits_0_330")[:, Lp]), digits, closed=False)
    axes[2].set_title("Reference: bare numbers 0 … 330", fontsize=9)
    S.note(axes[2], f"angles vs numbers: ρ(RDM, RDM) = {a['beyond_reference']['rho_fam_ref']:.2f}")
    t = sh[f"{key}__random"]["angles"]  # control: the same projection of the random-init twin
    draw_plane(axes[3], figdata.pca(figdata.family_acts(cfg, f"{key}__random", "angles")[:, Lp]), lab, closed=False, highlight=((len(items) - 1, 0),))
    axes[3].set_title("Control: random-init twin", fontsize=9)
    S.note(axes[3], f"closure = {t['closure']:.2f}; circle ρ = {t['circle_rho']:.2f} vs line ρ = {t['line_rho']:.2f}\n"
                    "meets the circle rule by chance: a structureless cloud passes it, so the rule is not specific")
    fig.tight_layout(w_pad=1.0)
    return fig


def build(cfg, sh: dict, ph: dict, cf: dict, outdir, gif: bool = True) -> list[str]:
    out = []
    fams_cfg = load_yaml("families.yaml")["chains"]
    for key in [k for k in sh if not k.endswith("__random")]:
        Lp = sh[key]["primary_layer"]
        L = len(ph[key]["pc_plane_cycle"]["days"])

        def cyc_stat(k, f):
            c = _cycle(ph, k, f, Lp)
            return f"cycle {c['score']:.2f} ({S.fmt_p(c['p'])})"

        def chain_stat(k, f):
            r = sh[k][f]
            s = f"Robinson {r['robinson']:.2f} ({S.fmt_p(r['robinson_p'])})"
            if cf and f in cf.get(k, {}):
                s += f"\nbeyond frequency: {S.fmt_p(cf[k][f]['p'])}"
            return s

        out.append(S.save(draw_grid(cfg, key, CYCLIC, Lp, cyc_stat, lambda f: True), outdir, f"cyclic_{key}"))
        out.append(S.save(draw_grid(cfg, key, CHAINS, Lp, chain_stat, lambda f: fams_cfg[f]["expected"] == "circle"), outdir, f"chains_{key}"))
        out.append(S.save(draw_filmstrip(cfg, key, ph, L, Lp), outdir, f"circle_depth_{key}"))
        out.append(S.save(draw_angles(cfg, key, sh, Lp), outdir, f"angles_{key}"))
        out.append(S.save(draw_grid3d(cfg, key, CYCLIC, Lp, lambda f: True, lambda k, f: planes_note(ph, k, f, Lp)), outdir, f"families3d_{key}"))
        out.append(S.save(draw_grid3d(cfg, key, CHAINS, Lp, lambda f: fams_cfg[f]["expected"] == "circle"), outdir, f"chains3d_{key}"))
        if gif:
            out.append(save_gif(cfg, key, ph, L, Lp, outdir))
    return out
