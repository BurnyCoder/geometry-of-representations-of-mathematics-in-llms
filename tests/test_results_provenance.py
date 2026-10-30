"""Post-hoc 4 (reports/prereg_addendum_3d.md) was added to phase_posthoc with its own random stream, so every
post-hoc number computed before it must be unchanged. This compares the current results with the files at
the last commit before the addendum (skipped when git history is not available, e.g. a source archive)."""

import json
import subprocess

import pytest

from mathgeom.config import ROOT

BEFORE_ADDENDUM = "249f6f9"  # master before reports/prereg_addendum_3d.md


@pytest.mark.parametrize("cfg_name", ["mvp", "robustness", "gemma"])
def test_posthoc4_left_earlier_posthoc_results_unchanged(cfg_name):
    try:
        old = json.loads(subprocess.run(["git", "show", f"{BEFORE_ADDENDUM}:results/{cfg_name}/posthoc.json"], cwd=ROOT, capture_output=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        pytest.skip("git history before the addendum is not available")
    new = json.loads((ROOT / "results" / cfg_name / "posthoc.json").read_text())
    assert all("pc_planes_cycle" in v for v in new.values())  # the new block is there for every model and twin
    assert {k: {kk: vv for kk, vv in v.items() if kk != "pc_planes_cycle"} for k, v in new.items()} == old
