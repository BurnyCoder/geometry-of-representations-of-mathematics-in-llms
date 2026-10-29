"""One visual system for every static figure (matplotlib) — the site mirrors these tokens in CSS.

Global context: colours come from a palette validated for colour-vision deficiency (adjacent and
all-pairs ΔE in OKLab, contrast vs the surface; dataviz skill validator). Scatter-type panels use at
most three categorical colours (the validated all-pairs limit); more identities are shown with small
multiples or emphasis (one highlighted, the rest grey). Order uses a one-hue sequential blue ramp,
signed similarity a blue↔red diverging scale with a grey midpoint. The trained model is always blue,
its random-init twin always grey.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --- tokens (light surface; the site defines the dark steps in site/style.css)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1–3 (validated all-pairs)
TRAINED, TWIN, NULL = BLUE, "#a3a29b", "#c3c2b7"  # trained model / random-init twin / null reference
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"  # blue ↔ grey ↔ red

MODEL = {"qwen3.5-2b": "Qwen3.5-2B", "qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-math-1.5b": "Qwen2.5-Math-1.5B", "gemma-4-e2b": "Gemma-4-E2B"}
FIELD = {"general_topology": "general topology", "group_theory": "group theory", "ring_module_theory": "rings & modules",
         "commutative_algebra": "commutative algebra", "category_theory": "category theory", "graph_theory": "graph theory",
         "set_theory_logic": "set theory & logic", "algebraic_geometry": "algebraic geometry", "number_theory": "number theory",
         "real_complex_analysis": "real & complex analysis", "functional_analysis": "functional analysis", "algebraic_topology": "algebraic topology",
         "differential_geometry": "differential geometry", "lie_representation_theory": "Lie & representation theory",
         "probability_statistics": "probability & statistics", "numerical_analysis": "numerical analysis", "game_theory": "game theory"}
BRANCH = {"algebra_number_theory": "algebra & number theory", "geometry_topology": "geometry & topology", "analysis": "analysis",
          "foundations_discrete": "foundations & discrete", "applied": "applied"}
FAMILY = {"days": "Days of the week", "months": "Months", "ring_chain": "Ring chain\nrng ⊃ ring ⊃ … ⊃ alg. closed field", "space_chain": "Space chain\nT0 ⊃ T1 ⊃ … ⊃ Hilbert",
          "number_systems": "Number systems\nℕ ⊂ ℤ ⊂ … ⊂ octonions", "ngons": "n-gons\ntriangle … dodecagon", "dimensions": "Dimensions\nzero- … ten-dimensional", "angles": "Angles\n0° … 330°"}


def pretty(key: str) -> str:
    """Display name of a model key; the random-init twin keeps its model name plus 'twin'."""
    base = key.removesuffix("__random")
    return MODEL.get(base, base) + (" · random-init twin" if key.endswith("__random") else "")


seq_cmap = LinearSegmentedColormap.from_list("mg_seq", SEQ)
div_cmap = LinearSegmentedColormap.from_list("mg_div", [DIV_NEG, DIV_MID, DIV_POS])


def ordinal(n: int) -> list[str]:
    """n ordered colours from the blue ramp, starting no lighter than step 250 (≥ 2:1 on the surface)."""
    ramp = ListedColormap(SEQ[3:])
    return [mpl.colors.to_hex(ramp(t)) for t in np.linspace(0, 1, n)]


def use_style() -> None:
    """Hairline axes, no top/right spines, opaque surface (README images must survive GitHub dark mode),
    TrueType fonts in PDFs (pdf.fonttype 42) and DejaVu Sans, which ships with matplotlib (ρ Δ ⊃ ℕ ✓)."""
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 8.5, "axes.labelcolor": INK2, "text.color": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.6, "pdf.fonttype": 42, "svg.hashsalt": "mathgeom",
    })


def save(fig, outdir: Path, stem: str, formats: tuple[str, ...] = ("png", "pdf")) -> str:
    """Write <stem>.png (200 dpi) and <stem>.pdf with time/version metadata removed so reruns are byte-stable."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"png": {"Software": None}, "pdf": {"CreationDate": None, "Producer": None, "Creator": None}}
    for ext in formats:
        fig.savefig(outdir / f"{stem}.{ext}", dpi=200, metadata=meta.get(ext), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return stem


def bare(ax) -> None:
    """Projection panels: coordinates are arbitrary (PCA/MDS), so hide ticks and keep a hairline frame."""
    ax.set_xticks([]), ax.set_yticks([])
    for s in ("left", "bottom", "top", "right"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(GRID)
    ax.set_aspect("equal", adjustable="datalim")


def label_points(ax, xy: np.ndarray, labels: list[str], size: float = 7.0, color: str = INK2, **kw) -> None:
    """Direct labels, de-overlapped with adjustText (MIT; https://github.com/Phlya/adjustText)."""
    from adjustText import adjust_text

    texts = [ax.text(x, y, t, fontsize=size, color=color, **kw) for (x, y), t in zip(xy, labels)]
    adjust_text(texts, x=xy[:, 0], y=xy[:, 1], ax=ax, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.5), expand=(1.15, 1.35))


def badge(ax, text: str, loc: str = "lower right", color: str = INK2) -> None:
    """A small statistic badge inside a panel (values always formatted from the results JSON)."""
    xy = {"lower right": (0.98, 0.03, "right", "bottom"), "upper right": (0.98, 0.97, "right", "top"),
          "lower left": (0.02, 0.03, "left", "bottom"), "upper left": (0.02, 0.97, "left", "top")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=7, color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID, lw=0.6))


def note(ax, text: str) -> None:
    """Statistic line under a projection panel (its axes carry no ticks, so this space is free and
    the text never covers data points)."""
    ax.set_xlabel(text, fontsize=7, color=INK2, labelpad=4)


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def excludes_zero(ci) -> bool:
    return ci[0] > 0 or ci[1] < 0
