"""Lint: every number in the generated docs must come from results JSON (no hand edits).

README's results block, reports/summary.md and paper/paper.md are re-rendered from their templates
and the committed results; any hand-typed or stale number makes this test fail.
"""

import re

import pytest

from mathgeom.config import ROOT


def _have_results():
    return all((ROOT / "results" / c / "map_fields.json").exists() for c in ("mvp", "robustness", "gemma"))


@pytest.mark.skipif(not _have_results(), reason="results not generated")
def test_generated_docs_match_results():
    from mathgeom.report.paper import render_template

    summary = render_template((ROOT / "reports" / "summary_template.md").read_text(encoding="utf-8"), fig_prefix="figures/")
    assert summary == (ROOT / "reports" / "summary.md").read_text(encoding="utf-8")
    paper = render_template((ROOT / "paper" / "template.md").read_text(encoding="utf-8"), fig_ext=".pdf")
    assert paper == (ROOT / "paper" / "paper.md").read_text(encoding="utf-8")
    block = render_template((ROOT / "reports" / "readme_results_template.md").read_text(encoding="utf-8"), fig_prefix="reports/figures/")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- RESULTS:START -->\n(.*)\n<!-- RESULTS:END -->", readme, flags=re.S)
    assert m and m.group(1) == block
