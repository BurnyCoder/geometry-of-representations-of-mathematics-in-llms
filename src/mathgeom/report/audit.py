"""Tables and decision rules of the independent audit (reports/prereg_addendum_lexical.md), from results JSON.

Global context: `lexical.json` (word-form baselines, within-block order, nonce modifiers, bootstrap BCa; every
CROSS config) and `twins.json` (random-init seeds; configs/twin_seeds.yaml) are rendered here, so that every
audit number in the README, summary and paper comes from a script, like the rest of report/render.py. The
decision rules are the addendum's D1–D13; scorecard_rows and fact in render.py call the `*_ok` helpers.
"""

from __future__ import annotations

from mathgeom.report.render import _ci, _p

BASES = ("char3", "word_bag", "layer0_bag")
BASE_NAME = {"char3": "char-3-grams", "word_bag": "word bag", "layer0_bag": "layer-0 token bag"}


# ------------------------------------------------------------------ decision rules (addendum D1–D13)
def field_beyond_ok(lx: dict, k: str) -> bool:
    """D1: the paired bootstrap Δ(trained − baseline) CI excludes 0 for all three baselines."""
    return all(lx[k]["field_map"]["boot"][f"delta_vs_{b}"]["ci"][0] > 0 for b in BASES)


def axiom_beyond_ok(lx: dict, la: dict, k: str, ax: str) -> bool:
    """D2: trained offset consistency above every baseline's AND the lexically disjoint pairs pass their null."""
    c = la[k]["group_like"]["offsets"][ax]["consistency"]
    g = lx[k]["group_like"]
    return c > max(g["offsets"][b][ax]["consistency"] for b in BASES) and g["decomposition"]["trained"][ax]["p"] < 0.05


def abelian_ok(lx: dict, k: str) -> bool:
    """D3: abelian group − group follows the 'commutative' direction (exact p < .05) beyond every baseline."""
    a = lx[k]["group_like"]["abelian_heldout"]
    return a["trained"]["p"] < 0.05 and a["trained"]["cos"] > max(a[b]["cos"] for b in BASES)


def group_objects_ok(lx: dict, st: dict, k: str) -> bool:
    """D4: the trained parallelogram cosine exceeds every baseline's."""
    return st[k]["group_objects"]["parallelograms"]["cos"] > max(lx[k]["group_objects"][b]["cos"] for b in BASES)


def ring_within_ok(lx: dict, k: str) -> bool:
    """D5: the six "… domain" items are ordered (exact Robinson p < .05)."""
    return lx[k]["ring_within_domain"]["trained"]["robinson_p"] < 0.05


def space_blocks_ok(lx: dict, k: str) -> list[bool]:
    """D6: order within each topical block of the space chain (exact Robinson p < .05)."""
    return [v["trained"]["robinson_p"] < 0.05 for v in lx[k]["space_blocks"].values()]


def nonce_above_ok(lx: dict, k: str) -> bool:
    """D13: 'commutative' gives more consistent offsets than both control modifiers in the same design."""
    n = lx[k]["nonce_modifier"]
    return n["commutative"]["trained"]["consistency"] > max(n[m]["trained"]["consistency"] for m in n if m != "commutative")


def scorecard_audit_rows(r: dict, keys: list[str]) -> dict[str, tuple]:
    """Scorecard rows {id: (test, status, cell function)}; render.scorecard_rows places each after its pre-registered row."""
    lx, la, st = r["lexical"], r["lattice"], r["structures"]
    mark = lambda ok: "●" if ok else "○"  # noqa: E731
    status = "post-hoc, audit addendum"
    axioms = list(la[keys[0]]["group_like"]["offsets"])
    return {
        "field": ("Field map beyond word form (Δρ vs char-3-grams, word bag, layer-0 token bag; CIs > 0)", status,
         lambda k: f"{mark(field_beyond_ok(lx, k))} {min(lx[k]['field_map']['boot'][f'delta_vs_{b}']['ci'][0] for b in BASES):+.2f}"),
        "space": ("Space chain ordered within both topical blocks (exact p)", status, lambda k: " / ".join(mark(x) for x in space_blocks_ok(lx, k))),
        "ring": ("Ring chain ordered within the six “… domain” items (exact p)", status,
         lambda k: f"{mark(ring_within_ok(lx, k))} {lx[k]['ring_within_domain']['trained']['robinson']:.2f}"),
        "axioms": ("Axiom offsets beyond word form (assoc / ident / div / comm)", status, lambda k: " / ".join(mark(axiom_beyond_ok(lx, la, k, a)) for a in axioms)),
        "abelian": ("Commutative direction predicts abelian group − group (held out)", status,
         lambda k: f"{mark(abelian_ok(lx, k))} {lx[k]['group_like']['abelian_heldout']['trained']['cos']:.2f}"),
        "nonce": ("“commutative X” more consistent than “wooden X” and “zorbic X”", status,
         lambda k: (lambda n: f"{mark(nonce_above_ok(lx, k))} {n['commutative']['trained']['consistency']:.2f} vs {max(n[m]['trained']['consistency'] for m in n if m != 'commutative'):.2f}")(lx[k]["nonce_modifier"])),
        "group_objects": ("Group objects beyond word form (cos above every lexical baseline)", status,
         lambda k: f"{mark(group_objects_ok(lx, st, k))} {st[k]['group_objects']['parallelograms']['cos']:.2f} vs {max(lx[k]['group_objects'][b]['cos'] for b in BASES):.2f}"),
    }


# ------------------------------------------------------------------ tables
def table_field(lx: dict) -> str:
    rows = ["| Model | ρ trained | ρ char-3-grams (Mantel p) | ρ word bag (p) | ρ layer-0 token bag (p) | Δ − char-3-grams [CI] | Δ − word bag [CI] | Δ − layer-0 bag [CI] | LOO field acc: trained / char-3 / word / layer-0 |",
            "|---" * 9 + "|"]
    for k, r in _models(lx):
        f = r["field_map"]
        rows.append(f"| {k} | {f['rsa']['trained']:.3f} | " + " | ".join(f"{f['rsa'][b]:.3f} ({_p(f['mantel_p'][b])})" for b in BASES) + " | "
                    + " | ".join(f"{f['rsa']['trained'] - f['rsa'][b]:+.3f} {_ci(f['boot'][f'delta_vs_{b}']['ci'])}" for b in BASES) + " | "
                    + " / ".join(f"{f['loo_acc'][x]:.3f}" for x in ("trained", *BASES)) + " |")
    return "\n".join(rows)


def table_offsets(lx: dict, la: dict) -> str:
    rows = ["| Model | Axiom | trained (p) | char-3-grams (p) | word bag (p) | layer-0 token bag (p) | duplicated cosines: n, mean | disjoint cosines: mean (p); twin | beyond word form |", "|---" * 9 + "|"]
    for k, r in _models(lx):
        g = r["group_like"]
        for ax, o in la[k]["group_like"]["offsets"].items():
            d, dt = g["decomposition"]["trained"][ax], g["decomposition"]["twin"][ax]
            dup = f"{d['n_dup']}, {d['dup_mean']:.2f}" if d["n_dup"] else "0, —"
            rows.append(f"| {k} | {ax} | {o['consistency']:.2f} ({_p(o['p'])}) | " + " | ".join(f"{g['offsets'][b][ax]['consistency']:.2f} ({_p(g['offsets'][b][ax]['p'])})" for b in BASES)
                        + f" | {dup} | {d['disjoint_mean']:.2f} ({_p(d['p'])}); {dt['disjoint_mean']:.2f} | {'yes' if axiom_beyond_ok(lx, la, k, ax) else 'no'} |")
    return "\n".join(rows)


def table_sublattice(lx: dict) -> str:
    kinds = ("trained", "twin", *BASES)
    rows = ["| Model | Axiom (3 pairs) | " + " | ".join(("trained", "twin", *[BASE_NAME[b] for b in BASES])) + " |", "|---" * (2 + len(kinds)) + "|"]
    for k, r in _models(lx):
        s = r["group_like"]["sublattice"]
        for ax in s["trained"]:
            rows.append(f"| {k} | {ax} | " + " | ".join(f"{s[x][ax]['consistency']:+.2f} ({_p(s[x][ax]['p'])})" for x in kinds) + " |")
    return "\n".join(rows)


def table_abelian(lx: dict) -> str:
    kinds = ("trained", "twin", *BASES)
    rows = ["| Model | " + " | ".join(f"{('trained', 'twin', *[BASE_NAME[b] for b in BASES])[i]}: cos (p)" for i in range(len(kinds))) + " | semantic transfer |", "|---" * (2 + len(kinds)) + "|"]
    for k, r in _models(lx):
        a = r["group_like"]["abelian_heldout"]
        rows.append(f"| {k} | " + " | ".join(f"{a[x]['cos']:+.2f} ({_p(a[x]['p'])})" for x in kinds) + f" | {'yes' if abelian_ok(lx, k) else 'no'} |")
    return "\n".join(rows)


def table_nonce(lx: dict) -> str:
    rows = ["| Model | modifier m (X → m X, 6 heads) | trained (null95, p) | twin | char-3-grams | layer-0 token bag |", "|---" * 6 + "|"]
    for k, r in _models(lx):
        for mod, v in r.get("nonce_modifier", {}).items():
            rows.append(f"| {k} | {mod} | {v['trained']['consistency']:.2f} ({v['trained']['null95']:.2f}, {_p(v['trained']['p'])}) | {v['twin']['consistency']:.2f} | {v['char3']['consistency']:.2f} | {v['layer0_bag']['consistency']:.2f} |")
    return "\n".join(rows)


def table_group_objects(lx: dict, st: dict) -> str:
    rows = ["| Model | trained cos (p) | " + " | ".join(f"{BASE_NAME[b]} cos (p)" for b in BASES) + " | beyond word form |", "|---" * (3 + len(BASES)) + "|"]
    for k, r in _models(lx):
        t = st[k]["group_objects"]["parallelograms"]
        rows.append(f"| {k} | {t['cos']:+.3f} ({_p(t['p'])}) | " + " | ".join(f"{r['group_objects'][b]['cos']:+.3f} ({_p(r['group_objects'][b]['p'])})" for b in BASES)
                    + f" | {'yes' if group_objects_ok(lx, st, k) else 'no'} |")
    return "\n".join(rows)


def table_blocks(lx: dict) -> str:
    rows = ["| Model | Block (n) | trained: Robinson (exact p); PC1 \\|ρ\\| (p) | twin | char-3-grams | layer-0 token bag |", "|---" * 6 + "|"]
    for k, r in _models(lx):
        for name, v in (("ring chain, “… domain” items", r["ring_within_domain"]), *((f"space chain, {b.replace('_', ' ')}", x) for b, x in r["space_blocks"].items())):
            t = v["trained"]
            rows.append(f"| {k} | {name} ({t['n']}) | {t['robinson']:.2f} ({_p(t['robinson_p'])}); {t['pc1']:.2f} ({_p(t['pc1_p'])}) | "
                        + " | ".join(f"{v[x]['robinson']:.2f} ({_p(v[x]['robinson_p'])})" for x in ("twin", "char3", "layer0_bag")) + " |")
    return "\n".join(rows)


def table_chains(lx: dict, key: str) -> str:
    """Shape statistics of every ordered family on the lexical baselines for one model (same tests as table_shapes)."""
    rows = ["| Family | Baseline | line ρ | circle ρ | closure | Robinson (p) | PCA order \\|ρ\\| (p) | LOO order ρ (p) |", "|---" * 8 + "|"]
    for fam, v in lx[key]["chains"].items():
        for b, s in v.items():
            rows.append(f"| {fam} | {BASE_NAME[b]} | {s['line_rho']:.2f} | {s['circle_rho']:.2f} | {s['closure']:.2f} | {s['robinson']:.2f} ({_p(s['robinson_p'])}) | {s['pca_order']:.2f} ({_p(s['pca_order_p'])}) | {s['loo_order']:.2f} ({_p(s['loo_order_p'])}) |")
    return "\n".join(rows)


def table_block_simulation(lx: dict) -> str:
    rows = ["| head-noun cluster SNR | conjunction rejects (Robinson p < .05 and LOO p < .05) | mean Robinson | mean LOO ρ | simulations |", "|---" * 5 + "|"]
    rows += [f"| {s['snr']} | {s['reject_rate']:.2f} | {s['mean_robinson']:.2f} | {s['mean_loo']:+.2f} | {s['n_sims']} |" for s in lx["block_only_simulation"]]
    return "\n".join(rows)


def table_bootstrap(lx: dict) -> str:
    rows = ["| Model | Statistic | point | bootstrap mean (bias) | percentile 95% CI (pre-registered) | BCa 95% CI |", "|---" * 6 + "|"]
    items = [(k, r["bootstrap"]) for k, r in _models(lx)] + list(lx.get("compare_bootstrap", {}).items())
    for k, b in items:
        for stat, v in b.items():
            rows.append(f"| {k} | {stat} | {v['point']:+.3f} | {v['boot_mean']:+.3f} ({v['bias']:+.3f}) | {_ci(v['percentile'])} | {_ci(v['bca'])} |")
    return "\n".join(rows)


def table_twins(tw: dict) -> str:
    rows = []
    for key, r in tw.items():
        tags = [t for t in r if t not in ("field_boot", "summary")]
        seeds = [t for t in tags if t != "trained"]
        stats = [("field ρ", lambda e: e["field_rho"]), ("LOO field accuracy", lambda e: e["field_loo_acc"]),
                 *[(f"offset consistency: {a}", lambda e, a=a: e["offsets"][a]["consistency"]) for a in r["trained"]["offsets"]],
                 *[(f"{c.replace('_', ' ')} Robinson", lambda e, c=c: e["chains"][c]["robinson"]) for c in ("ring_chain", "space_chain")]]
        rows += [f"| Statistic ({key}) | trained | " + " | ".join(f"twin {s}" for s in seeds) + " | twin range | trained > every twin |", "|---" * (len(seeds) + 4) + "|"]
        for name, f in stats:
            v = [f(r[s]) for s in seeds]
            rows.append(f"| {name} | {f(r['trained']):.3f} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {min(v):.3f} to {max(v):.3f} | {'yes' if f(r['trained']) > max(v) else 'no'} |")
        b = r["field_boot"]
        rows.append("| field Δρ trained − twin [95% CI] | — | " + " | ".join(_ci(b[f"delta_vs_{s}"]["ci"]) for s in seeds) + f" | lowest bound {min(b[f'delta_vs_{s}']['ci'][0] for s in seeds):+.3f} | — |")
    return "\n".join(rows)


def table_loo_alpha(sh: dict, fams: dict) -> str:
    """D9–D10: LOO order with 10,000 permutations, for the pre-registered α = 10 and the scale-relative α."""
    rows = ["| Family | Model | LOO order ρ, α = 10 (p) | LOO order ρ, α = mean ‖x‖² (p) |", "|---" * 4 + "|"]
    for name in fams["chains"]:
        for k, r in sh.items():
            s = r[name]
            if "loo_order_rel" in s:
                rows.append(f"| {name} | {k} | {s['loo_order']:+.2f} ({_p(s['loo_order_p'])}) | {s['loo_order_rel']:+.2f} ({_p(s['loo_order_rel_p'])}) |")
    return "\n".join(rows)


def _models(lx: dict):
    """(model key, entry) pairs of a lexical.json, skipping the non-model blocks."""
    return [(k, v) for k, v in lx.items() if isinstance(v, dict) and "field_map" in v]
