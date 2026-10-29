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
    rows = ["| Family | Expected | Model | line ρ | circle ρ | closure (d first-last / adjacent) | closure_far† (null 5%) | Robinson (p) | PCA order \\|ρ\\| (p) | LOO order ρ (p) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, f in fams["chains"].items():
        for key, res in sh.items():
            r = res[name]
            rows.append(f"| {name} | {f['expected']} | {key} | {r['line_rho']:.2f} | {r['circle_rho']:.2f} | {r['closure']:.2f} | {r['closure_far']:.2f} ({r['closure_far_null95']:.2f}) | {r['robinson']:.2f} ({_p(r['robinson_p'])}) | {r['pca_order']:.2f} ({_p(r['pca_order_p'])}) | {r['loo_order']:.2f} ({_p(r['loo_order_p'])}) |")
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


def table_posthoc(ph: dict) -> str:
    ks = list(next(iter(ph.values()))["abtt_sweep"])
    rows = ["| Model | " + " | ".join(f"k={k}" for k in ks) + " | top-5 PC explained variance | field η² on PC1–10 |", "|---" * (len(ks) + 3) + "|"]
    for key, r in ph.items():
        rows.append(f"| {key} | " + " | ".join(f"{r['abtt_sweep'][k]:.3f}" for k in ks) + f" | {', '.join(f'{v:.3f}' for v in r['explained_var_top5'])} | {', '.join(f'{v:.2f}' for v in r['eta2_field_pc1_20'][:10])} |")
    return "\n".join(rows)


def table_cycles(ph: dict, layer_of: dict) -> str:
    fams = list(next(iter(ph.values()))["pc_plane_cycle"])
    rows = ["| Family | " + " | ".join(ph) + " |", "|---" * (len(ph) + 1) + "|"]
    for f in fams:
        cells = []
        for key, r in ph.items():
            e = r["pc_plane_cycle"][f][layer_of[key] - 1]
            cells.append(f"{e['score']:.2f} (p={_p(e['p'])})")
        rows.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def holm_rows(mp: dict, sh: dict, la: dict, key: str) -> list[tuple[str, float]]:
    """The 8 pre-registered confirmatory tests of the MVP (docs/prereg.md §4) and their p-values."""
    rows = [("H-A1 Δρ trained − random-init ≤ 0 (bootstrap)", mp[key]["math_primary"]["rsa_boot"]["delta_vs_random_init"]["p_le_0"]),
            ("H-B ring chain Robinson", sh[key]["ring_chain"]["robinson_p"]), ("H-B space chain Robinson", sh[key]["space_chain"]["robinson_p"])]
    rows += [(f"H-B2 {a} offset consistency", o["p"]) for a, o in la[key]["group_like"]["offsets"].items()]
    return rows


def table_holm(mp: dict, sh: dict, la: dict) -> str:
    """Holm over the p-valued confirmatory tests; the angles prediction is a pre-registered decision
    rule (closure < 2 AND circle ρ > line ρ) with no p-value, so it is listed but not adjusted."""
    from statsmodels.stats.multitest import multipletests

    out = ["| Test | Model | p | Holm-adjusted p | reject at α=0.05 |", "|---|---|---|---|---|"]
    for key in mp:
        rows = holm_rows(mp, sh, la, key)
        rej, padj, _, _ = multipletests([max(p, 1e-12) for _, p in rows], alpha=0.05, method="holm")
        out += [f"| {n} | {key} | {_p(p)} | {_p(pa)} | {'yes' if r else 'no'} |" for (n, p), pa, r in zip(rows, padj, rej)]
        a = sh[key]["angles"]
        met = a["closure"] < 2 and a["circle_rho"] > a["line_rho"]
        out.append(f"| H-B angles circle prediction (decision rule: closure {a['closure']:.2f} < 2 and circle ρ {a['circle_rho']:.2f} > line ρ {a['line_rho']:.2f}) | {key} | — | — | {'met' if met else 'not met (prediction falsified)'} |")
    return "\n".join(out)


def table_hierarchy(h: dict) -> str:
    rows = ["| Model | super-tree match (cophenetic ρ): trained / twin / layer-0 | shuffled-tree p (null95) | Δ trained − twin [95% CI] | cophenetic c trained / twin (Gaussian 95th pct) | relative Gromov δ trained / twin (Gaussian 5th pct) |", "|---|---|---|---|---|---|"]
    for key, r in h.items():
        cm, cc, dl, g = r["coph_match"], r["coph_c"], r["delta_rel"], r["gaussian_null"]
        rows.append(f"| {key} | {cm['trained']:.3f} / {cm['random_init']:.3f} / {cm['layer0']:.3f} | {_p(r['coph_match_p_shuffled_tree'])} ({r['coph_match_null95']:.3f}) | "
                    f"{_ci(r['coph_match_delta_vs_random_ci'])} | {cc['trained']:.3f} / {cc['random_init']:.3f} ({g['coph_c_95']:.3f}) | {dl['trained']:.3f} / {dl['random_init']:.3f} ({g['delta_rel_5']:.3f}) |")
    return "\n".join(rows)


def table_baselines(b: dict) -> str:
    rows = ["| Model | concept set | n | field ρ: LLM / Qwen3-Embedding / fastText | Δ LLM − Qwen3-Emb [CI] | Δ LLM − fastText [CI] | LOO acc: LLM / Qwen3-Emb / fastText |", "|---|---|---|---|---|---|---|"]
    for key, res in b.items():
        if key == "infinigram":
            continue
        for sub, r in res.items():
            rsa, bt, la = r["rsa"], r["boot"], r["loo_acc"]
            ft = f"{rsa['fasttext']:.3f}" if "fasttext" in rsa else "—"
            dft = _ci(bt["delta_vs_fasttext"]["ci"]) if "delta_vs_fasttext" in bt else "—"
            lft = f"{la['fasttext']['acc']:.3f}" if "fasttext" in la else "—"
            rows.append(f"| {key} | {sub} | {r['n']} | {rsa['llm_trained']:.3f} / {rsa['qwen3_embedding']:.3f} / {ft} | {_ci(bt['delta_vs_qwen3_embedding']['ci'])} | {dft} | "
                        f"{la['llm_trained']['acc']:.3f} / {la['qwen3_embedding']['acc']:.3f} / {lft} |")
    ig = b.get("infinigram", {})
    if ig:
        rows.append(f"\ninfini-gram ({ig['index']}): Spearman ρ(wordfreq Zipf, log count) = {ig['rho_zipf_vs_log_count']:.3f}; concepts with zero count: {ig['frac_zero']:.1%}.")
    return "\n".join(rows)


def table_smds(s: dict) -> str:
    rows = ["| Family | Model | SMDS line score (null95, p) | SMDS discrete-circle score (null95, p) | preferred (vs null) |", "|---|---|---|---|---|"]
    fams = [k for k in next(iter(s.values())) if k != "primary_layer"]
    for f in fams:
        for key, r in s.items():
            e, c = r[f]["euclidean"], r[f]["discrete_circular"]
            from mathgeom.analyses.smds_fit import preferred

            rows.append(f"| {f} | {key} | {e['score']:.3f} ({e['null95']:.3f}, {_p(e['p'])}) | {c['score']:.3f} ({c['null95']:.3f}, {_p(c['p'])}) | {preferred(r[f])} |")
    return "\n".join(rows)


def table_structures(s: dict) -> str:
    rows = ["| Model | commutativity transfer cos (null95, p) | identity transfer cos (p) | group-object parallelograms cos (null95, p) | parent ranks: topological / Lie / algebraic group |", "|---|---|---|---|---|"]
    for key, r in s.items():
        t, go = r["transfer"], r["group_objects"]
        pr = go["parent_ranks"]
        rows.append(f"| {key} | {t['commutative']['cos']:+.3f} ({t['commutative']['null95']:.3f}, {_p(t['commutative']['p'])}) | {t['identity']['cos']:+.3f} ({_p(t['identity']['p'])}) | "
                    f"{go['parallelograms']['cos']:+.3f} ({go['parallelograms']['null95']:.3f}, {_p(go['parallelograms']['p'])}) | "
                    f"{pr['topological group']} / {pr['Lie group']} / {pr['algebraic group']} (of {go['pool_size']}) |")
    return "\n".join(rows)


def table_judges(j: dict) -> str:
    rows = ["| Judges | n concepts | mean pairwise Spearman | ICC(2,1) | ICC(2,k) |", "|---|---|---|---|---|",
            f"| {', '.join(j['judges'])} | {j['n']} | {j['mean_pairwise_spearman']:.3f} | {j['icc2_1']:.3f} | {j['icc2_k']:.3f} |"]
    return "\n".join(rows)


def table_axes(a: dict) -> str:
    rows = ["| Model | PC | explained var | ρ with applied (p) | ρ with abstraction (p) |", "|---|---|---|---|---|"]
    for key, r in a.items():
        for pc, ev in zip(r["pcs"], r["explained_var"]):
            rows.append(f"| {key} | {pc['pc']} | {ev:.3f} | {pc['rho_applied']:+.3f} ({_p(pc['p_applied'])}) | {pc['rho_abstraction']:+.3f} ({_p(pc['p_abstraction'])}) |")
        rows.append(f"| {key} | 5-fold CV ridge readout of abstraction | | | ρ = {r['cv_ridge_rho_abstraction']:+.3f} |")
        for pc in r.get("pcs_with_infinigram", []):
            rows.append(f"| {key} | {pc['pc']} (+ infini-gram covariate) | | {pc['rho_applied']:+.3f} | {pc['rho_abstraction']:+.3f} |")
    return "\n".join(rows)


def table_causal(c: dict) -> str:
    rows = ["| Model | Chain | baseline acc (gate) | noise scale | acc, chain-subspace noise | acc, random-subspace noise (mean of 5) |", "|---|---|---|---|---|---|"]
    for key, r in c.items():
        for name, v in r.items():
            if not isinstance(v, dict):
                continue
            if not v["scales"]:
                rows.append(f"| {key} | {name} | {v['baseline_acc']:.3f} ({v['gate']}) — gate not passed | — | — | — |")
            for s, sv in v["scales"].items():
                rows.append(f"| {key} | {name} | {v['baseline_acc']:.3f} ({v['gate']}) | {s} | {sv['acc_chain_subspace']:.3f} | {sv['acc_random_subspace_mean']:.3f} |")
    return "\n".join(rows)


def table_sae(s: dict) -> str:
    rows = ["| Model | SAE block | ρ field map (SAE code) | LOO acc math / control | features math-only / shared / control-only |", "|---|---|---|---|---|"]
    for key, r in s.items():
        rows.append(f"| {key} | {r['sae_block']} | {r['math']['rsa_sae']:.3f} | {r['math']['loo_acc_sae']:.3f} / {r['control']['loo_acc_sae']:.3f} | {r['features_math_only']} / {r['features_shared']} / {r['features_control_only']} |")
    return "\n".join(rows)


def table_calibration(cal: dict) -> str:
    desc = {
        "concept_linear_null": "concept level, nuisance correctly specified (null)",
        "concept_itemnoise_null": "concept level, item-level noise / nonlinear link (null)",
        "field_pure_null": "field level, representation unrelated to fields (null)",
        "field_confounded_null": "field level, nuisance = noisy proxy of a confounder (null)",
        "field_alternative": "field level, true field geometry present (power)",
    }
    rows = ["| Scenario | Rejection rate at α=0.05 (± SE) | n simulations |", "|---|---|---|"]
    for k, v in cal.items():
        rows.append(f"| {desc.get(k, k)} | {v['reject_rate_0.05']:.3f} ± {v['se']:.3f} | {v['n_sims']} |")
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
    global FIG
    sub = cfg.raw.get("figure_subdir", "")  # variant configs reuse model keys: keep their figures apart
    FIG = ROOT / "reports" / "figures" / sub if sub else ROOT / "reports" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    fams = load_yaml("families.yaml")
    mp, sh, la = _j(cfg, "map_fields.json"), _j(cfg, "shapes.json"), _j(cfg, "lattice.json")
    keys = list(mp)
    prim = {k: mp[k]["primary_layer"] for k in keys}
    figs = fig_layer_curves(mp) + fig_field_maps(cfg, mp) + fig_families(cfg, fams, keys, prim) + fig_lattice(cfg, fams, keys, prim)
    cal = _j(cfg, "calibration.json")
    sections = {
        "Method validation: calibration of partial RDM regression (Freedman–Lane), simulation study": table_calibration(cal) if cal else "*not run*",
        "H-A1 field map (field-centroid RSA vs arXiv cross-listing distance; LOO nearest-centroid field accuracy)": table_field_map(mp),
        "POST-HOC: field-map ρ after removing the top-k principal components (all-but-the-top sweep)": table_posthoc(_j(cfg, "posthoc.json")) if _j(cfg, "posthoc.json") else "*not run*",
        "POST-HOC: cyclic order in the top-2-PC plane at the pre-registered layer (1 = perfect cycle; positive-control debug)": table_cycles(_j(cfg, "posthoc.json"), {k: v["primary_layer"] for k, v in _j(cfg, "shapes.json").items()}) if _j(cfg, "posthoc.json") else "*not run*",
        "Multiplicity: Holm correction over the 8 pre-registered confirmatory tests (bootstrap p for H-A1 is the fraction of resamples with Δ ≤ 0; 0 means < 1/2000)": table_holm(mp, sh, la),
        "H-A1c clustering specificity: math fields vs non-math control fields": table_clustering(mp),
        "H-B shape families (primary layer; permutation p-values; † = post-hoc statistic, see docs/deviations.md)": table_shapes(sh, fams),
        "H-B negative control: random 11-item lists of math concepts (should reject ≈5%)": table_negative_control(sh),
        "H-B number-coded families: structure beyond the number reference": table_beyond_reference(sh, fams),
        "H-B2 axiom lattices (offset consistency = mean cosine of same-axiom difference vectors)": table_lattice(la),
    }
    for title, name, fn in (("H-A2 hierarchy as pre-registered: cophenetic correlation (UPGMA) vs super-tree; spectrum-matched Gaussian and Gromov δ (plan)", "hierarchy.json", table_hierarchy),
                            ("Baselines: field map from fastText and Qwen3-Embedding vs the LLM (paired bootstrap); infini-gram covariate", "baselines.json", table_baselines),
                            ("Secondary: SMDS line vs discrete-circle fits, each vs label-shuffled refits", "smds.json", table_smds),
                            ("H-B2 tests 4–5: axiom transfer to ring-like pairs; group objects (parent ranks = nearest-neighbour rank of [group, other parent])", "structures.json", table_structures),
                            ("H-A4 judges: agreement of LLM abstraction raters", "judges.json", table_judges),
                            ("H-A4 axes: top PCs vs pure/applied and abstraction (residualized on frequency, token count)", "axes.json", table_axes),
                            ("H-B causal test (gated on baseline accuracy ≥ 0.70)", "causal.json", table_causal),
                            ("Exploratory: Qwen-Scope SAE codes", "sae.json", table_sae)):
        if _j(cfg, name):
            sections[title] = fn(_j(cfg, name))
    md = f"# Results: `{cfg.name}` run\n\nGenerated by `python run.py report --config configs/{cfg.name}.yaml` from `results/{cfg.name}/*.json`. Do not edit by hand.\n\n"
    for title, tab in sections.items():
        md += f"## {title}\n\n{tab}\n\n"
    md += "## Figures\n\n" + "\n\n".join(f"![{f}](figures/{sub + '/' if sub else ''}{f})" for f in figs) + "\n"
    (ROOT / "reports" / f"{cfg.name}_results.md").write_text(md, encoding="utf-8")
    numbers = {"map_fields": mp, "shapes": sh, "lattice": la}
    (cfg.results_dir / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    # README results block is rendered across all configs by `run.py paper` (report/paper.py)
    from mathgeom.report.site import export_site

    if cfg.raw.get("site", True):  # variant configs would overwrite the explorer entry of the same model key
        export_site(cfg, mp, sh, la)
    log.info("report written: %d figures, reports/%s_results.md, site data", len(figs), cfg.name)
