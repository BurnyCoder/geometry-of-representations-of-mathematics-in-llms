"""Model-free regression checks for the November 26 scientific review."""
from itertools import permutations

import numpy as np
import pytest

from mathgeom.config import load_yaml
from mathgeom.report import render


def test_deterministic_angle_pass_never_generates_a_p_value():
    from mathgeom.report.paper import _results

    r = _results("mvp")
    key = "qwen3.5-2b"
    r["shapes"][key]["angles"].update(closure=1.0, circle_rho=1.0, line_rho=0.1)
    original = render.holm_decisions(r["map_fields"], r["shapes"], r["lattice"], key)
    ring = next(row for row in original if "ring chain" in row["name"])
    angle = next(row for row in original if row["name"] == render.ANGLES_ROW)
    assert angle["p"] == 1 and not angle["reject"]
    assert ring["p_holm"] == pytest.approx(0.0419790105)
    assert ring["reserved_holm"] == pytest.approx(0.0524737631)


def test_exact_order_preserves_ties_and_counts_all_labelings():
    from mathgeom.verification.geometry import exact_robinson
    from mathgeom.geometry.manifold import robinson_score

    d = np.abs(np.subtract.outer([0, 0, 1, 2, 4], [0, 0, 1, 2, 4]))
    orders = list(permutations(range(5)))
    scores = [robinson_score(d, p) for p in orders]
    r = exact_robinson(d)
    assert r["orders"] == len(orders)
    assert r["extreme"] == sum(s >= scores[0] for s in scores)


def test_axiom_profile_counterexample_and_mathlib_dispositions():
    from mathgeom.verification.geometry import minimal_upper_bounds
    from mathgeom.verification.mathlib import reviewed_rows

    bits = load_yaml("families.yaml")["lattices"]["group_like"]["items"]
    assert set(minimal_upper_bounds(bits, "commutative magma", "unital magma")) == {"commutative loop", "commutative monoid"}
    rows = reviewed_rows()
    assert len(rows) == 39 and rows.index.is_unique
    assert rows.loc["inductive limit", "decision"] == "exclude"
    assert rows.loc["normal subgroup", "decision"] == "retain"

def test_module_header_with_comment_keeps_its_imports(tmp_path):
    from mathgeom.data.mathlib_depth import import_depth, module_imports
    from mathgeom.verification.mathlib import module_depths

    root = tmp_path / "Mathlib"
    root.mkdir()
    (root / "Core.lean").write_text("module\n", encoding="utf-8")
    (root / "Base.lean").write_text("module -- shake: keep-all\npublic meta import Mathlib.Core\n", encoding="utf-8")
    (root / "Result.lean").write_text("module\nimport Mathlib.Base\n", encoding="utf-8")
    assert import_depth(module_imports(tmp_path))["Mathlib.Result"] == 2
    assert module_depths(tmp_path, ["Mathlib.Result"]) == {"Mathlib.Result": 2}


def test_calendar_controls_field_units_and_candidate_scores():
    from mathgeom.report.paper import _results

    s = _results("mvp")["science"]
    for r in s["checkpoints"].values():
        families = r["trained"]["families"]
        assert not families["days"]["full_space_control"] and not families["months"]["full_space_control"]
        assert all(families[f]["line_rho"] > families[f]["circle_rho"] for f in ("days","months","angles"))
    p = s["checkpoints"]["qwen3.5-2b"]["applied_labels"]
    assert p["assignments"] == 680 and p["max_five_pc_p"][1] == pytest.approx(2/680)
    assert s["intervention"]["space_chain"]["chain_s4.0"] == {"correct": 87, "presentations": 110}
