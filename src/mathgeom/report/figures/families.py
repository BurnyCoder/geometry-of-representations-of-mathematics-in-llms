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
    """Animated PC planes of days and months through every layer (README; post-hoc cycle score per frame)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    data = {f: (figdata.planes_by_layer(figdata.family_acts(cfg, key, f), Lp), [short(f, c) for c in figdata.family_items(cfg, f)]) for f in ("days", "months")}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), dpi=100)
    lim = max(np.abs(P).max() for P, _ in data.values()) * 1.25

    def frame(l):
        for ax, (f, (planes, items)) in zip(axes, data.items()):
            ax.clear()
            P = planes[l]
            seq = np.r_[np.arange(len(P)), 0]
            ax.plot(P[seq, 0], P[seq, 1], color=S.AXIS, lw=1.0)
            ax.scatter(P[:, 0], P[:, 1], c=S.ordinal(len(P)), s=30, zorder=3, edgecolors=S.SURFACE, linewidths=1.0)
            for (x, y), t in zip(P, items):
                ax.text(x, y + lim * 0.06, t, fontsize=6.5, ha="center", color=S.INK2)
            ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
            S.bare(ax)
            ax.set_title(f"{S.FAMILY[f].split(chr(10))[0]} · cycle score {_cycle(ph, key, f, l + 1)['score']:.2f}", fontsize=8.5)
        fig.suptitle(f"{S.pretty(key)} · layer {l + 1} of {L} (depth {(l + 1) / L:.2f})", fontsize=8.5, color=S.INK2)

    anim = FuncAnimation(fig, frame, frames=L, interval=450)
    path = outdir / f"circle_depth_{key}.gif"
    anim.save(path, writer=PillowWriter(fps=2.5))
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
    S.note(axes[3], f"closure = {t['closure']:.2f}; circle ρ = {t['circle_rho']:.2f} vs line ρ = {t['line_rho']:.2f}")
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
        if gif:
            out.append(save_gif(cfg, key, ph, L, Lp, outdir))
    return out
