"""The report's decision rules must be the pre-registered ones (docs/prereg.md), and generated docs must
not repeat themselves. Global context: every ●/○ in the README, paper and site scorecard comes from
report/render.py, so a rule bug there silently changes headline claims; these tests pin the rules."""

import json
import re

import pytest

from mathgeom.config import ROOT
from mathgeom.report import render


def _results(cfg):
    return {n: json.loads((ROOT / "results" / cfg / f"{n}.json").read_text()) for n in ("map_fields", "shapes", "lattice")}


def test_chain_p_is_the_conjunction_of_robinson_and_loo():
    """prereg H-B: ordered only if Robinson p < .05 AND LOO order p < .05 → p of the conjunction = max."""
    assert render.chain_p({"robinson_p": 0.001, "loo_order_p": 0.08}) == 0.08
    assert render.chain_p({"robinson_p": 0.03, "loo_order_p": 0.0033}) == 0.03


@pytest.mark.skipif(not (ROOT / "results" / "robustness" / "shapes.json").exists(), reason="results not generated")
def test_holm_family_has_eight_tests_with_the_conjunctive_chain_p():
    r = _results("robustness")
    rows = render.holm_rows(r["map_fields"], r["shapes"], r["lattice"], "qwen3.5-0.8b")
    assert len(rows) == 8  # prereg §4 lists 8 tests; the falsified angles rule enters with p = 1 (audit D8)
    assert dict(rows)[render.ANGLES_ROW] == 1.0
    ring = dict(rows)["H-B ring chain (Robinson and LOO order)"]
    s = r["shapes"]["qwen3.5-0.8b"]["ring_chain"]
    assert ring == max(s["robinson_p"], s["loo_order_p"]) and ring > 0.05  # LOO order p = .074 in this model (10,000 permutations)
    assert 0 < dict(rows)["H-A1 Δρ trained − random-init ≤ 0 (bootstrap)"] <= 1 / 2001 + 1e-12  # +1 convention, never 0


@pytest.mark.skipif(not all((ROOT / "results" / c / "posthoc.json").exists() for c in ("mvp", "robustness", "gemma")), reason="results not generated")
def test_scorecard_ring_chain_fails_where_the_loo_condition_fails():
    from mathgeom.report.paper import _results as merged

    sc = render.scorecard_rows(merged("all"))
    ring = next(row for row in sc["rows"] if row["test"].startswith("Ring chain ordered"))
    cells = dict(zip(sc["models"], ring["cells"]))
    assert not cells["Qwen3.5-0.8B"].startswith("●") and not cells["Gemma-4-E2B"].startswith("●")
    assert cells["Qwen3.5-2B"].startswith("●")


def test_fact_helpers_count_and_name_the_exceptions():
    keys = ["qwen3.5-2b", "qwen3.5-0.8b", "qwen2.5-1.5b", "qwen2.5-math-1.5b", "gemma-4-e2b"]
    assert render._n_of(keys, lambda k: True) == "all 5 models"
    assert render._n_of(keys, lambda k: False, "twins") == "none of the 5 twins"
    assert render._n_of(keys, lambda k: k != "gemma-4-e2b") == "4 of 5 models (all but Gemma-4-E2B)"
    assert render._n_of(keys, lambda k: k == "qwen2.5-1.5b", "twins") == "1 of 5 twins (that of Qwen2.5-1.5B)"
    assert render._n_of(keys, lambda k: k.startswith("qwen2.5"), "twins") == "2 of 5 twins (those of Qwen2.5-1.5B and Qwen2.5-Math-1.5B)"
    assert render._runs([0, 1, 2, 3, 10, 11, 22]) == [(0, 3), (10, 11), (22, 22)]


def test_scorecard_applies_every_preregistered_condition():
    """docs/prereg.md H-A1 needs (a) Mantel p < .05 AND (b) a Δρ CI above 0; H-B2 needs the null (here Holm) AND
    trained > twin. Breaking only (a), or only trained > twin, must remove the ●."""
    import copy

    mp = {"m": {"math_primary": {"rsa_perm_p": 0.01, "rsa_boot": {"delta_vs_random_init": {"ci": [0.1, 0.5]}}}}}
    assert render.ha1_supported(mp, "m")
    bad = copy.deepcopy(mp)
    bad["m"]["math_primary"]["rsa_perm_p"] = 0.2
    assert not render.ha1_supported(bad, "m")
    offs = {a: {"consistency": 0.5, "p": 1e-4} for a in ("associative", "identity", "divisible", "commutative")}
    la = {"m": {"group_like": {"offsets": offs}}, "m__random": {"group_like": {"offsets": {a: {"consistency": 0.1} for a in offs}}}}
    sh = {"m": {s: {"robinson_p": 1e-4, "loo_order_p": 1e-4} for s in ("ring_chain", "space_chain")} | {"angles": {"closure": 3, "circle_rho": 0, "line_rho": 1}}}
    mp["m"]["math_primary"]["rsa_boot"]["delta_vs_random_init"]["p_le_0"] = 0.0
    assert render.axiom_holm_ok(mp, sh, la, "m", "commutative")
    la["m__random"]["group_like"]["offsets"]["commutative"]["consistency"] = 0.9  # the twin is more consistent
    assert not render.axiom_holm_ok(mp, sh, la, "m", "commutative")


def test_readme_has_no_repeated_paragraph():
    paras = [p.strip() for p in re.split(r"\n\s*\n", (ROOT / "README.md").read_text(encoding="utf-8")) if p.strip()]
    repeated = [a for a, b in zip(paras, paras[1:]) if a == b]
    assert not repeated, repeated[:1]
