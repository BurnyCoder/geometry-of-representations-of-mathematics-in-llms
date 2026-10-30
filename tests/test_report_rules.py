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
def test_holm_family_has_seven_p_values_with_the_conjunctive_chain_p():
    r = _results("robustness")
    rows = render.holm_rows(r["map_fields"], r["shapes"], r["lattice"], "qwen3.5-0.8b")
    assert len(rows) == 7  # prereg §4 lists 8 tests; the angles rule has no p
    ring = dict(rows)["H-B ring chain (Robinson and LOO order)"]
    s = r["shapes"]["qwen3.5-0.8b"]["ring_chain"]
    assert ring == max(s["robinson_p"], s["loo_order_p"]) and ring > 0.05  # LOO order p = .076 in this model
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
    assert render._n_of(keys, lambda k: k == "qwen2.5-1.5b", "twins") == "1 of 5 twins (those of Qwen2.5-1.5B)"
    assert render._runs([0, 1, 2, 3, 10, 11, 22]) == [(0, 3), (10, 11), (22, 22)]


def test_readme_has_no_repeated_paragraph():
    paras = [p.strip() for p in re.split(r"\n\s*\n", (ROOT / "README.md").read_text(encoding="utf-8")) if p.strip()]
    repeated = [a for a, b in zip(paras, paras[1:]) if a == b]
    assert not repeated, repeated[:1]
