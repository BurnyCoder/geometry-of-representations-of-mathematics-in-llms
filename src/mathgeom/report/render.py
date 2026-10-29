"""Report phase: regenerate every table and figure from results/<cfg>/*.json (+ cached activations).

Global context: the project rule is "no hand-typed numbers" — README's results block,
reports/<cfg>_results.md, and site/data.json are all written here from the analysis JSON,
so `run.py report` is the single command that refreshes every number and figure.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from mathgeom.config import ROOT, Config, load_yaml

log = logging.getLogger(__name__)
FIG = ROOT / "reports" / "figures"


def _j(cfg: Config, name: str) -> dict:
    p = cfg.results_dir / name
    return json.loads(p.read_text()) if p.exists() else {}


def _ci(c) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def _p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


# ------------------------------------------------------------------ tables
def table_field_map(mp: dict) -> str:
    rows = ["| Model | Concept set | n concepts / fields | ρ_field trained (perm p) [95% CI] | ρ random-init | ρ layer-0 | Δ trained − random-init [95% CI] | LOO field acc trained / random / layer-0 (chance) | split-half reliability |", "|---|---|---|---|---|---|---|---|---|"]
    for key, res in mp.items():
        for sub in ("math_primary", "math_all"):
            r = res[sub]
            b, a = r["rsa_boot"], r["loo_acc"]
            rows.append(
                f"| {key} | {sub} | {r['n']} / {r['n_fields']} | {r['rsa']['trained']:.3f} (p={_p(r['rsa_perm_p'])}) {_ci(b['trained']['ci'])} | {r['rsa']['random_init']:.3f} | {r['rsa']['layer0']:.3f} | "
                f"{r['rsa']['trained'] - r['rsa']['random_init']:+.3f} {_ci(b['delta_vs_random_init']['ci'])} | {a['trained']['acc']:.3f} / {a['random_init']['acc']:.3f} / {a['layer0']['acc']:.3f} ({a['chance']:.3f}) | {r['split_half']:.3f} |"
            )
    return "\n".join(rows)


def table_clustering(mp: dict) -> str:
    rows = ["| Model | Domain | n / fields | LOO acc trained [CI] | random-init | layer-0 | chance | Δ trained − random-init [CI] |", "|---|---|---|---|---|---|---|---|"]
    for key, res in mp.items():
        for dom in ("math", "control"):
            r = res[f"clustering_{dom}"]
            rows.append(f"| {key} | {dom} | {r['n']} / {r['n_fields']} | {r['trained']['acc']:.3f} {_ci(r['trained']['ci'])} | {r['random_init']['acc']:.3f} | {r['layer0']['acc']:.3f} | {r['chance']:.3f} | {r['trained']['acc'] - r['random_init']['acc']:+.3f} {_ci(r['delta_vs_random_init']['ci'])} |")
    return "\n".join(rows)


def table_shapes(sh: dict, fams: dict) -> str:
    rows = ["| Family | Expected | Model | line ρ | circle ρ | closure (d first-last / adjacent) | Robinson (p) | PCA order \\|ρ\\| (p) | LOO order ρ (p) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, f in fams["chains"].items():
        for key, res in sh.items():
            r = res[name]
            rows.append(f"| {name} | {f['expected']} | {key} | {r['line_rho']:.2f} | {r['circle_rho']:.2f} | {r['closure']:.2f} | {r['robinson']:.2f} ({_p(r['robinson_p'])}) | {r['pca_order']:.2f} ({_p(r['pca_order_p'])}) | {r['loo_order']:.2f} ({_p(r['loo_order_p'])}) |")
    return "\n".join(rows)


def table_negative_control(sh: dict) -> str:
    rows = ["| Model | Robinson reject rate | PCA-order reject rate | line-ρ reject rate | n lists |", "|---|---|---|---|---|"]
    for key, res in sh.items():
        nc = res["negative_control"]
        rows.append(f"| {key} | {nc['robinson_p']['reject_rate_0.05']:.2f} | {nc['pca_order_p']['reject_rate_0.05']:.2f} | {nc['line_rho_p']['reject_rate_0.05']:.2f} | {nc['robinson_p']['n_lists']} |")
    return "\n".join(rows)


def table_beyond_reference(sh: dict, fams: dict) -> str:
    rows = ["| Family | Model | ρ(family RDM, reference RDM) | template β beyond reference (p) |", "|---|---|---|---|"]
    for name, f in fams["chains"].items():
        if "reference" not in f:
            continue
        for key, res in sh.items():
            br = res[name]["beyond_reference"]
            rows.append(f"| {name} vs {f['reference']} | {key} | {br['rho_fam_ref']:.2f} | {br['beta_template']:+.3f} ({_p(br['p'])}) |")
    return "\n".join(rows)


def table_lattice(la: dict) -> str:
    rows = ["| Lattice | Model | axiom-RDM β (p) | lexical β | offset consistency per axiom: obs / null95 (p) | analogy top-1 (chance) | mean rank (chance) |", "|---|---|---|---|---|---|---|"]
    for key, res in la.items():
        for name in ("group_like", "self_distributive"):
            r = res[name]
            offs = "; ".join(f"{a}: {o['consistency']:.2f}/{o['null95']:.2f} ({_p(o['p'])})" for a, o in r["offsets"].items())
            an = r["analogy"]
            rows.append(f"| {name} | {key} | {r['axiom_rdm_beta']:+.3f} ({_p(r['axiom_rdm_p'])}) | {r['lexical_beta']:+.3f} | {offs} | {an['top1']:.2f} ({an['chance_top1']:.2f}) | {an['mean_rank']:.1f} ({an['chance_mean_rank']:.1f}) |")
        ac = res["analogy_control"]
        rows.append(f"| gender control | {key} | — | — | gender: {ac['consistency']:.2f}/{ac['null95']:.2f} | king:queen rank {ac['king_queen_rank']} | — |")
    return "\n".join(rows)


# ------------------------------------------------------------------ figures
def fig_layer_curves(mp: dict) -> list[str]:
    import matplotlib.pyplot as plt

    out = []
    for key, res in mp.items():
        r = res["math_primary"]
        lc = r["layer_curve"]
        x = np.arange(1, len(lc["trained"]) + 1)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(x, lc["trained"], "-o", ms=3, label=f"{key} (trained)", color="#1f5fa8")
        ax.plot(x, lc["random_init"], "-s", ms=3, label="random-init twin", color="#b0b0b0")
        ax.axhline(lc["max_null95"], ls="--", color="#c0392b", lw=1, label="95% max-over-layers null")
        ax.axvline(res["primary_layer"], ls=":", color="k", lw=1, label="pre-registered layer")
        ax.set_xlabel("layer (block output index)")
        ax.set_ylabel("ρ(field-centroid RDM, arXiv distance)")
        ax.set_title("H-A1: field map vs depth (primary concepts, confirmatory split)", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        p = FIG / f"field_map_layers_{key}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        out.append(p.name)
    return out


def _mds(D: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import MDS

    return MDS(n_components=2, metric="precomputed", random_state=seed, init="classical_mds", n_init=1).fit_transform(D)


def fig_field_maps(cfg: Config, mp: dict) -> list[str]:
    """Side-by-side 2D MDS: arXiv field distance vs trained field-centroid RDM vs random-init."""
    import matplotlib.pyplot as plt
    from scipy.linalg import orthogonal_procrustes

    from mathgeom.analyses.map_fields import field_centroids
    from mathgeom.geometry.preprocess import cosine_rdm

    concepts = pd.read_csv(cfg.data_dir / "concepts.csv")
    fd = json.loads((cfg.data_dir / "field_distance.json").read_text())
    pa = {f["id"]: f["pa"] for f in load_yaml("fields.yaml")["math"]}
    out = []
    for key, res in mp.items():
        Lp = res["primary_layer"]
        mask = ((concepts.kind == "math") & concepts.primary & (concepts.split == cfg.raw.get("analysis_split", "confirmatory"))).values
        names = res["math_primary"]["fields"]
        sel = [fd["names"].index(n) for n in names]
        lab = concepts.field.values[mask]
        tr = np.load(cfg.artifacts / "acts" / key / "field_neutral.npz")["last"][mask, Lp]
        rn = np.load(cfg.artifacts / "acts" / f"{key}__random" / "field_neutral.npz")["last"][mask, Lp]
        panels = {"arXiv cross-listing distance (ground truth)": np.array(fd["distance"])[np.ix_(sel, sel)],
                  f"{key}, layer {Lp}": cosine_rdm(field_centroids(tr, lab, names)),
                  "random-init twin": cosine_rdm(field_centroids(rn, lab, names))}
        embs = {t: _mds(D) for t, D in panels.items()}
        ref = list(embs.values())[0]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (t, E) in zip(axes, embs.items()):
            E = E - E.mean(0)
            R, _ = orthogonal_procrustes(E, ref - ref.mean(0))  # rotate/reflect to align with ground truth (display only)
            E = E @ R
            for (x, y), n in zip(E, names):
                ax.scatter(x, y, c="#1f5fa8" if pa[n] == "pure" else "#d35400", s=30)
                ax.annotate(n.replace("_", " "), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_title(t, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle("Field maps (2D metric MDS, Procrustes-aligned to the arXiv map; blue = pure, orange = applied)", fontsize=10)
        fig.tight_layout()
        p = FIG / f"field_maps_{key}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p.name)
    return out


def fig_families(cfg: Config, fams: dict, keys: list[str], primary_layers: dict) -> list[str]:
    """PCA (2D) of each chain family, trained vs random-init, consecutive items connected."""
    import matplotlib.pyplot as plt

    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    names = list(fams["chains"])
    out = []
    for key in keys:
        fig, axes = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.4))
        for row, k in enumerate([key, f"{key}__random"]):
            acts = np.load(cfg.artifacts / "acts" / k / "family.npz")["last"]
            for col, name in enumerate(names):
                idx = [i for i, (f, _, _) in enumerate(fam) if f == name]
                X = acts[idx, primary_layers[key]].astype(np.float64)
                X = X - X.mean(0)
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                P = X @ Vt[:2].T
                ax = axes[row, col]
                closed = fams["chains"][name]["expected"] == "circle"
                seq = np.r_[np.arange(len(P)), 0] if closed else np.arange(len(P))
                ax.plot(P[seq, 0], P[seq, 1], "-", color="#bbbbbb", lw=0.8)
                ax.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), cmap="viridis", s=18, zorder=3)
                for j, (_, c, _) in enumerate(fam[i] for i in idx):
                    if j in (0, len(idx) - 1) or len(idx) <= 8:
                        ax.annotate(c[:12], P[j], fontsize=5.5)
                ax.set_xticks([]), ax.set_yticks([])
                if row == 0:
                    ax.set_title(name, fontsize=8)
                if col == 0:
                    ax.set_ylabel("trained" if row == 0 else "random-init", fontsize=8)
        fig.suptitle(f"H-B shape families, PCA of layer {primary_layers[key]} ({key}); colour = ground-truth order", fontsize=9)
        fig.tight_layout()
        p = FIG / f"families_pca_{key}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p.name)
    return out


def fig_lattice(cfg: Config, fams: dict, keys: list[str], primary_layers: dict) -> list[str]:
    """Group-like lattice in PCA-2D with axiom edges coloured by axiom; trained vs random-init."""
    import matplotlib.pyplot as plt

    from mathgeom.analyses.lattice import axiom_pairs

    fam = json.loads((cfg.artifacts / "acts" / "family_index.json").read_text())
    pos = {(f, c): i for i, (f, c, _) in enumerate(fam)}
    lat = fams["lattices"]["group_like"]
    names = list(lat["items"])
    bits = np.array([lat["items"][c] for c in names])
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    out = []
    for key in keys:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, k in zip(axes, [key, f"{key}__random"]):
            X = np.load(cfg.artifacts / "acts" / k / "family.npz")["last"][[pos[("group_like", c)] for c in names], primary_layers[key]].astype(np.float64)
            X = X - X.mean(0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            P = X @ Vt[:2].T
            for a, (axname, col) in enumerate(zip(lat["axioms"], colors)):
                for i, j in axiom_pairs(bits, a):
                    ax.annotate("", P[j], P[i], arrowprops=dict(arrowstyle="->", color=col, lw=1.1, alpha=0.8))
                ax.plot([], [], color=col, label=f"+{axname}")
            ax.scatter(P[:, 0], P[:, 1], c="k", s=12, zorder=3)
            for (x, y), n in zip(P, names):
                ax.annotate(n, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_title(f"{k}, layer {primary_layers[key]}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle("H-B2 group-like axiom lattice (PCA-2D). Parallel same-colour arrows = consistent axiom direction", fontsize=9)
        fig.tight_layout()
        p = FIG / f"lattice_group_like_{key}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p.name)
    return out


# ------------------------------------------------------------------ assemble
def phase_report(cfg: Config) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fams = load_yaml("families.yaml")
    mp, sh, la = _j(cfg, "map_fields.json"), _j(cfg, "shapes.json"), _j(cfg, "lattice.json")
    keys = list(mp)
    prim = {k: mp[k]["primary_layer"] for k in keys}
    figs = fig_layer_curves(mp) + fig_field_maps(cfg, mp) + fig_families(cfg, fams, keys, prim) + fig_lattice(cfg, fams, keys, prim)
    sections = {
        "H-A1 field map (field-centroid RSA vs arXiv cross-listing distance; LOO nearest-centroid field accuracy)": table_field_map(mp),
        "H-A1c clustering specificity: math fields vs non-math control fields": table_clustering(mp),
        "H-B shape families (primary layer; permutation p-values)": table_shapes(sh, fams),
        "H-B negative control: random 11-item lists of math concepts (should reject ≈5%)": table_negative_control(sh),
        "H-B number-coded families: structure beyond the number reference": table_beyond_reference(sh, fams),
        "H-B2 axiom lattices (offset consistency = mean cosine of same-axiom difference vectors)": table_lattice(la),
    }
    md = f"# Results: `{cfg.name}` run\n\nGenerated by `python run.py report --config configs/{cfg.name}.yaml` from `results/{cfg.name}/*.json`. Do not edit by hand.\n\n"
    for title, tab in sections.items():
        md += f"## {title}\n\n{tab}\n\n"
    md += "## Figures\n\n" + "\n\n".join(f"![{f}](figures/{f})" for f in figs) + "\n"
    (ROOT / "reports" / f"{cfg.name}_results.md").write_text(md, encoding="utf-8")
    numbers = {"map_fields": mp, "shapes": sh, "lattice": la}
    (cfg.results_dir / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    # README block: headline tables + key figures
    block = (f"<!-- RESULTS:START -->\n*Auto-generated from `results/{cfg.name}/` by `run.py report`. Full tables: [`reports/{cfg.name}_results.md`](reports/{cfg.name}_results.md).*\n\n"
             f"**H-A1 field map.**\n\n{table_field_map(mp)}\n\n**Clustering specificity.**\n\n{table_clustering(mp)}\n\n**H-B shape families.**\n\n{table_shapes(sh, fams)}\n\n"
             f"**H-B2 axiom lattices.**\n\n{table_lattice(la)}\n\n" + "\n\n".join(f"![{f}](reports/figures/{f})" for f in figs) + "\n<!-- RESULTS:END -->")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda _: block, readme, flags=re.S)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    log.info("report written: %d figures, reports/%s_results.md, README results block", len(figs), cfg.name)
