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


PLANE_FAMS = ("days", "months", "angles", "digits_0_330")


def _plane(e: dict) -> str:
    return f"PC{e['plane'][0] + 1}–PC{e['plane'][1] + 1}"


def table_planes_cycle(ph: dict, layer_of: dict) -> str:
    """POST-HOC 4 (reports/prereg_addendum_3d.md): best closed-cycle score over the 10 planes of PC1–5 at
    the pre-registered layer, its plane and max-over-planes permutation p; the twin's score in brackets."""
    keys = [k for k in ph if not k.endswith("__random") and "pc_planes_cycle" in ph[k]]
    if not keys:
        return "*not run*"
    at = lambda k, f: ph[k]["pc_planes_cycle"][f][layer_of[k.replace("__random", "")] - 1]  # noqa: E731
    rows = ["| Family | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for f in PLANE_FAMS:
        rows.append(f"| {f} | " + " | ".join(f"{at(k, f)['score']:.2f} · {_plane(at(k, f))} · p={_p(at(k, f)['p'])} (twin {at(k + '__random', f)['score']:.2f})" for k in keys) + " |")
    return "\n".join(rows)


def planes_decision(ph: dict, layer_of: dict) -> dict:
    """The pre-specified rule for "angles have a circular component beyond PC1–2" (addendum, all four must
    hold at the pre-registered layer): Holm across models on the best-plane p; best plane ≠ PC1–2; trained
    score > twin's; and the bare numerals do not meet the first two conditions themselves."""
    from statsmodels.stats.multitest import multipletests

    keys = [k for k in ph if not k.endswith("__random") and "pc_planes_cycle" in ph[k]]
    at = lambda k, f: ph[k]["pc_planes_cycle"][f][layer_of[k.replace("__random", "")] - 1]  # noqa: E731
    holm = {f: dict(zip(keys, multipletests([at(k, f)["p"] for k in keys], alpha=0.05, method="holm")[0])) for f in ("angles", "digits_0_330")}
    out = {}
    for k in keys:
        a, d, tw = at(k, "angles"), at(k, "digits_0_330"), at(k + "__random", "angles")
        c = {"holm": bool(holm["angles"][k]), "not_pc12": a["plane"] != [0, 1], "beats_twin": a["score"] > tw["score"],
             "numerals_clear": not (bool(holm["digits_0_330"][k]) and d["plane"] != [0, 1])}
        out[k] = {"score": a["score"], "plane": _plane(a), "p": a["p"], **c, "accepted": all(c.values())}
    return out


def table_planes_decision(ph: dict, layer_of: dict) -> str:
    from mathgeom.report.style import MODEL

    d = planes_decision(ph, layer_of)
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    rows = ["| Model | Angles: best closed plane | p (max over planes) | 1. Holm across models | 2. not PC1–PC2 | 3. > twin | 4. numerals do not cycle | Accepted |", "|---" * 8 + "|"]
    rows += [f"| {MODEL.get(k, k)} | {v['score']:.2f} · {v['plane']} | {_p(v['p'])} | {yn(v['holm'])} | {yn(v['not_pc12'])} | {yn(v['beats_twin'])} | {yn(v['numerals_clear'])} | **{yn(v['accepted'])}** |"
             for k, v in d.items()]
    met = [MODEL.get(k, k) for k, v in d.items() if v["accepted"]]
    verdict = (f"**The rule is met in {', '.join(met)}**: there, the post-hoc result is a circular component beyond PC1–2." if met else
               "**No model meets the rule**, so \"angles are not a circle\" stands, now for every plane of the top 5 principal components too.")
    return "\n".join(rows + ["", verdict])


def holm_rows(mp: dict, sh: dict, la: dict, key: str) -> list[tuple[str, float]]:
    """The 8 pre-registered confirmatory tests of the MVP (docs/prereg.md §4) and their p-values."""
    rows = [("H-A1 Δρ trained − random-init ≤ 0 (bootstrap)", mp[key]["math_primary"]["rsa_boot"]["delta_vs_random_init"]["p_le_0"]),
            ("H-B ring chain Robinson", sh[key]["ring_chain"]["robinson_p"]), ("H-B space chain Robinson", sh[key]["space_chain"]["robinson_p"])]
    rows += [(f"H-B2 {a} offset consistency", o["p"]) for a, o in la[key]["group_like"]["offsets"].items()]
    return rows


def holm_decisions(mp: dict, sh: dict, la: dict, key: str) -> list[dict]:
    """Holm (statsmodels multipletests) over one model's 8 p-valued confirmatory tests."""
    from statsmodels.stats.multitest import multipletests

    rows = holm_rows(mp, sh, la, key)
    rej, padj, _, _ = multipletests([max(p, 1e-12) for _, p in rows], alpha=0.05, method="holm")
    return [{"name": n, "p": p, "p_holm": float(pa), "reject": bool(r)} for (n, p), pa, r in zip(rows, padj, rej)]


def table_holm(mp: dict, sh: dict, la: dict) -> str:
    """Holm over the p-valued confirmatory tests; the angles prediction is a pre-registered decision
    rule (closure < 2 AND circle ρ > line ρ) with no p-value, so it is listed but not adjusted."""
    out = ["| Test | Model | p | Holm-adjusted p | reject at α=0.05 |", "|---|---|---|---|---|"]
    for key in mp:
        out += [f"| {d['name']} | {key} | {_p(d['p'])} | {_p(d['p_holm'])} | {'yes' if d['reject'] else 'no'} |" for d in holm_decisions(mp, sh, la, key)]
        a = sh[key]["angles"]
        met = a["closure"] < 2 and a["circle_rho"] > a["line_rho"]
        out.append(f"| H-B angles circle prediction (decision rule: closure {a['closure']:.2f} < 2 and circle ρ {a['circle_rho']:.2f} > line ρ {a['line_rho']:.2f}) | {key} | — | — | {'met' if met else 'not met (prediction falsified)'} |")
    return "\n".join(out)


SCORE_LEGEND = ("● significant (Holm-adjusted for the axiom tests; 95% CI excludes 0 for Δ tests) · ◐ raw p < .05 only · ○ not significant · "
                "✗ pre-registered prediction falsified. Numbers: the statistic for the trained model.")


def table_scorecard(r: dict) -> str:
    """Hypotheses × models at a glance (all configs merged), as a Markdown table."""
    sc = scorecard_rows(r)
    out = ["| Test | Status | " + " | ".join(sc["models"]) + " |", "|---" * (len(sc["models"]) + 2) + "|"]
    out += [f"| {row['test']} | {row['status']} | " + " | ".join(row["cells"]) + " |" for row in sc["rows"]]
    return "\n".join(out + ["\n" + SCORE_LEGEND])


def scorecard_rows(r: dict) -> dict:
    """Hypotheses × models (shared by the Markdown table and the site). ● Holm-significant / CI excludes 0,
    ◐ raw p < .05 only, ○ not significant, ✗ pre-registered prediction falsified."""
    from mathgeom.report.style import MODEL

    mp, sh, la, st, hi, ph, cf = (r[k] for k in ("map_fields", "shapes", "lattice", "structures", "hierarchy", "posthoc", "chain_frequency"))
    keys = [k for k in MODEL if k in mp]
    ex = lambda ci: ci[0] > 0  # noqa: E731  (CI entirely above 0)

    def axiom(key, name):
        d = next(x for x in holm_decisions(mp, sh, la, key) if x["name"] == f"H-B2 {name} offset consistency")
        c = la[key]["group_like"]["offsets"][name]["consistency"]
        return f"{'●' if d['reject'] else '◐' if d['p'] < 0.05 else '○'} {c:.2f}"

    def cyc(key):
        e = ph[key]["pc_plane_cycle"]["days"][mp[key]["primary_layer"] - 1]
        return f"{'●' if e['p'] < 0.05 else '○'} {e['score']:.2f}"

    def chain(key, f):
        s = sh[key][f]
        return f"{'●' if s['robinson_p'] < 0.05 and s['robinson'] > sh[key + '__random'][f]['robinson'] else '○'} {s['robinson']:.2f}"

    def angles(key):
        a = sh[key]["angles"]
        return f"{'●' if a['closure'] < 2 and a['circle_rho'] > a['line_rho'] else '✗'} {a['closure']:.2f}"

    conf_, pre, added = "pre-registered, confirmatory", "pre-registered", "added by plan audit"
    planes = planes_decision(ph, {k: mp[k]["primary_layer"] for k in keys}) if all("pc_planes_cycle" in ph.get(k, {}) for k in keys) else {}
    rows = [
        ("Field map tracks arXiv beyond the twin (H-A1; Δρ CI)", conf_, lambda k: f"{'●' if ex(mp[k]['math_primary']['rsa_boot']['delta_vs_random_init']['ci']) else '○'} {mp[k]['math_primary']['rsa']['trained']:.2f}"),
        ("Field map is a hierarchy beyond the twin (H-A2; cophenetic Δ CI)", pre, lambda k: f"{'●' if ex(hi[k]['coph_match_delta_vs_random_ci']) else '○'} {hi[k]['coph_match']['trained']:.2f}"),
        ("Days form a cycle (positive control; PC plane)", "positive control; post-hoc statistic", cyc),
        ("Ring chain ordered (Robinson; > twin)", conf_, lambda k: chain(k, "ring_chain")),
        ("Space chain ordered (Robinson; > twin)", conf_, lambda k: chain(k, "space_chain")),
        ("Ring / space order beyond frequency", added, lambda k: " / ".join("●" if cf[k][f]["p"] < 0.05 else "○" for f in ("ring_chain", "space_chain"))),
        ("Angles form a circle (closure < 2 and circle ρ > line ρ)", conf_, angles),
        *([("Angles: closed cycle in any top-5 PC plane (addendum rule)", "post-hoc, pre-specified addendum", lambda k: f"{'●' if planes[k]['accepted'] else '○'} {planes[k]['score']:.2f}"),
           ("Days / months: a loop that closes in some top-5 PC plane (p < .05, unadjusted)", "post-hoc, descriptive",
            lambda k: " / ".join(("●" if e["p"] < 0.05 else "○") + f" {e['score']:.2f}" for e in (ph[k]["pc_planes_cycle"][f][mp[k]["primary_layer"] - 1] for f in ("days", "months"))))]
          if planes else []),
        *[(f"Axiom direction: {a} (offset consistency; Holm)", conf_, lambda k, a=a: axiom(k, a)) for a in la[keys[0]]["group_like"]["offsets"]],
        ("Lie / topological / algebraic groups as parallelograms", "pre-registered, exploratory", lambda k: f"{'●' if st[k]['group_objects']['parallelograms']['p'] < 0.05 else '○'} {st[k]['group_objects']['parallelograms']['cos']:.2f}"),
    ]
    return {"models": [MODEL[k] for k in keys], "rows": [{"test": name, "status": status, "cells": [fn(k) for k in keys]} for name, status, fn in rows], "legend": SCORE_LEGEND}


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


def table_mathlib(m: dict) -> str:
    """H-A4 judge validation against Mathlib import depth, and Mathlib coverage per field (second audit)."""
    src = m["n_by_source"]
    rows = ["| Matched concepts (curated YAML / CamelCase) | median import depth | ρ(mean judge rating, depth) [95% CI] (p) | ρ partialling Zipf | ρ per judge |", "|---|---|---|---|---|",
            f"| {m['n_matched']} of {m['n_math_concepts']} ({src.get('mathlib_yaml', 0)} / {src.get('camelcase', 0)}) | {m['depth_median']:.0f} | "
            f"{m['rho_mean_rating_depth']:+.3f} {_ci(m['ci95'])} ({_p(m['p'])}) | {m['rho_partial_zipf']:+.3f} | " + "; ".join(f"{j.split('/')[-1]}: {r:+.2f}" for j, r in m["rho_per_judge"].items()) + " |",
            "", "| Field | Mathlib coverage (fraction of concepts matched) |", "|---|---|"]
    return "\n".join(rows + [f"| {f} | {c:.3f} |" for f, c in m["coverage_by_field"].items()])


def table_chain_frequency(cf: dict) -> str:
    """Plan H-B frequency control: chain order beyond log-frequency and token count (second audit)."""
    rows = ["| Chain | Model | ρ(order, Zipf) | ρ(order, token count) | LOO order beyond covariates (null95, p) |", "|---|---|---|---|---|"]
    for name in next(iter(cf.values())):
        for key, res in cf.items():
            b = res[name]
            rows.append(f"| {name} | {key} | {b['rho_rank_covariates'][0]:+.2f} | {b['rho_rank_covariates'][1]:+.2f} | {b['loo_order_beyond']:+.2f} ({b['null95']:.2f}, {_p(b['p'])}) |")
    return "\n".join(rows)


def table_lattice_probes(la: dict, st: dict) -> str:
    """Flagged lattice items and lattice-like identity transfer (second audit, exploratory)."""
    rows = ["| Model | associative quasigroup: d(·, group) / d(·, quasigroup) (3 nearest) | inverse semigroup: 3 nearest | identity direction → lattice-like pairs: cos (null95, p) |", "|---|---|---|---|"]
    for key, res in la.items():
        a, t = res["audit2"], st[key]["transfer"]["identity_lattice"]
        rows.append(f"| {key} | {a['associative_quasigroup']['ratio']:.2f} ({', '.join(a['associative_quasigroup']['nearest'])}) | {', '.join(a['inverse_semigroup']['nearest'])} | {t['cos']:+.3f} ({t['null95']:.3f}, {_p(t['p'])}) |")
    return "\n".join(rows)


def table_lattice_variants(la: dict) -> str:
    """Group-like offsets on mean-pooled tokens / with the token-count direction removed; partial-operation lattice."""
    rows = ["| Model | Variant | offset consistency per axiom: obs / null95 (p) |", "|---|---|---|"]
    for key, res in la.items():
        a = res["audit2"]
        for name, offs in (("group-like, mean-pooled tokens", a["group_like_offsets"]["pool"]), ("group-like, token-count direction removed", a["group_like_offsets"]["length_projected"]),
                           ("partial-operation lattice", a["partial_operation"]["offsets"])):
            rows.append(f"| {key} | {name} | " + "; ".join(f"{ax}: {o['consistency']:.2f} / {o['null95']:.2f} ({_p(o['p'])})" for ax, o in offs.items()) + " |")
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


# ------------------------------------------------------------------ assemble
def build_model_figures(cfg: Config, outdir) -> list[str]:
    """Per-model static figures (report/figures/*) for every model of this config → file names."""
    from mathgeom.report import style
    from mathgeom.report.figures import axes, families, lattice, maps

    style.use_style()
    mp = _j(cfg, "map_fields.json")
    figs = maps.build(cfg, mp, outdir)
    figs += families.build(cfg, _j(cfg, "shapes.json"), _j(cfg, "posthoc.json"), _j(cfg, "chain_frequency.json"), outdir)
    figs += lattice.build(cfg, _j(cfg, "lattice.json"), outdir)
    if _j(cfg, "axes.json"):
        figs += axes.build(cfg, _j(cfg, "axes.json"), _j(cfg, "mathlib.json") or None, {k: v["primary_layer"] for k, v in mp.items()}, outdir)
    return [f if f.endswith(".gif") else f + ".png" for f in figs]


def phase_report(cfg: Config) -> None:
    sub = cfg.raw.get("figure_subdir", "")  # variant configs reuse model keys: keep their figures apart
    fams = load_yaml("families.yaml")
    mp, sh, la = _j(cfg, "map_fields.json"), _j(cfg, "shapes.json"), _j(cfg, "lattice.json")
    figs = build_model_figures(cfg, ROOT / "reports" / "figures" / sub if sub else ROOT / "reports" / "figures")
    cal = _j(cfg, "calibration.json")
    sections = {
        "Method validation: calibration of partial RDM regression (Freedman–Lane), simulation study": table_calibration(cal) if cal else "*not run*",
        "H-A1 field map (field-centroid RSA vs arXiv cross-listing distance; LOO nearest-centroid field accuracy)": table_field_map(mp),
        "POST-HOC: field-map ρ after removing the top-k principal components (all-but-the-top sweep)": table_posthoc(_j(cfg, "posthoc.json")) if _j(cfg, "posthoc.json") else "*not run*",
        "POST-HOC: cyclic order in the top-2-PC plane at the pre-registered layer (1 = perfect cycle; positive-control debug)": table_cycles(_j(cfg, "posthoc.json"), {k: v["primary_layer"] for k, v in _j(cfg, "shapes.json").items()}) if _j(cfg, "posthoc.json") else "*not run*",
        "POST-HOC 4 (reports/prereg_addendum_3d.md): closed cycle in the best plane of PC1–5 at the pre-registered layer (score · plane · max-over-planes p; twin in brackets)": table_planes_cycle(_j(cfg, "posthoc.json"), {k: v["primary_layer"] for k, v in _j(cfg, "shapes.json").items()}) if _j(cfg, "posthoc.json") else "*not run*",
        "Multiplicity: Holm correction over the 8 pre-registered confirmatory tests (bootstrap p for H-A1 is the fraction of resamples with Δ ≤ 0; 0 means < 1/2000)": table_holm(mp, sh, la),
        "H-A1c clustering specificity: math fields vs non-math control fields": table_clustering(mp),
        "H-B shape families (primary layer; permutation p-values; † = post-hoc statistic, see docs/deviations.md)": table_shapes(sh, fams),
        "H-B negative control: random 11-item lists of math concepts (should reject ≈5%)": table_negative_control(sh),
        "H-B number-coded families: structure beyond the number reference": table_beyond_reference(sh, fams),
        "H-B2 axiom lattices (offset consistency = mean cosine of same-axiom difference vectors)": table_lattice(la),
    }
    st = _j(cfg, "structures.json")
    if _j(cfg, "chain_frequency.json"):
        sections["H-B frequency control (plan; second audit): chain order beyond Zipf frequency and token count"] = table_chain_frequency(_j(cfg, "chain_frequency.json"))
    if all("audit2" in r for r in la.values()) and st:
        sections["H-B2 exploratory (second audit): flagged items and lattice-like identity transfer"] = table_lattice_probes(la, st)
        sections["H-B2 exploratory (second audit): offset variants and the partial-operation lattice"] = table_lattice_variants(la)
    for title, name, fn in (("H-A2 hierarchy as pre-registered: cophenetic correlation (UPGMA) vs super-tree; spectrum-matched Gaussian and Gromov δ (plan)", "hierarchy.json", table_hierarchy),
                            ("Baselines: field map from fastText and Qwen3-Embedding vs the LLM (paired bootstrap); infini-gram covariate", "baselines.json", table_baselines),
                            ("Secondary: SMDS line vs discrete-circle fits, each vs label-shuffled refits", "smds.json", table_smds),
                            ("H-B2 tests 4–5: axiom transfer to ring-like pairs; group objects (parent ranks = nearest-neighbour rank of [group, other parent])", "structures.json", table_structures),
                            ("H-A4 judges: agreement of LLM abstraction raters", "judges.json", table_judges),
                            ("H-A4 judge validation: mean rating vs Mathlib import depth; Mathlib coverage per field (plan; second audit)", "mathlib.json", table_mathlib),
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
        export_site(cfg, mp)
    log.info("report written: %d figures, reports/%s_results.md, site data", len(figs), cfg.name)
