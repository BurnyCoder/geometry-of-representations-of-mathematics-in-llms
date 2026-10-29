"""Smoke tests for the static figures: every draw function renders synthetic data without error, and
the shared helpers behave (alignment is a pure rotation/reflection; ranks are within-matrix)."""

import numpy as np
import pytest

from mathgeom.report import figdata, style

style.use_style()


def test_align_is_rotation_and_rank_unit_is_rank():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(10, 2))
    R = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90° rotation + scale + shift
    out = figdata.align(3 * ref @ R + 5, ref)
    assert np.allclose(out, figdata.unit_rms(ref), atol=1e-9)
    D = figdata.rank_unit(np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], float))
    assert D[0, 2] == 0 and D[0, 1] == 1 and np.allclose(D, D.T)


def test_exact_coords_preserve_centred_geometry():
    X = np.random.default_rng(1).normal(size=(9, 300))
    E = figdata.exact_coords(X)
    Xc = X - X.mean(0)
    assert E.shape[1] <= 8 and np.allclose(E @ E.T, Xc @ Xc.T, atol=1e-8)


def test_offset_matrix_blocks_and_shared_vertices():
    bits = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    om = figdata.offset_matrix(np.random.default_rng(2).normal(size=(4, 50)), bits, ["a", "b"])
    assert len(om["pairs"]) == 4 and list(om["group"]) == [0, 0, 1, 1]
    assert not om["shared"][0, 1] and om["shared"][0, 2]  # (0→1) and (0→2) share vertex 0


def test_draw_functions_render(tmp_path):
    from mathgeom.report.figures import lattice, maps

    rng = np.random.default_rng(3)
    names = [f"f{i}" for i in range(6)]
    D = {k: np.abs(rng.normal(size=(6, 6))) for k in ("arxiv", "trained", "random_init")}
    for v in D.values():
        v[:] = (v + v.T) / 2
        np.fill_diagonal(v, 0)
    lay = figdata.field_layouts(D)
    t = {"arxiv": "a", "trained": "b", "random_init": "c"}
    style.save(maps.draw_field_maps(lay, names, ["pure"] * 5 + ["applied"], figdata.neighbour_edges(D["arxiv"]), t, t), tmp_path, "m")
    style.save(maps.draw_rdms(D, names, ["x"] * 3 + ["y"] * 3, list(range(6)), t, t), tmp_path, "r")
    bits = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    X = rng.normal(size=(4, 40))
    off = {"group_like": {"offsets": {a: {"consistency": 0.1, "null_mean": 0.0, "null95": 0.2, "p": 0.3} for a in ("associative", "identity")}}}
    oms = {k: figdata.offset_matrix(X, bits, ["associative", "identity"]) for k in ("trained", "random_init")}
    style.save(lattice.draw_cosines(oms, ["associative", "identity"], off, off, {"trained": "t", "random_init": "r"}), tmp_path, "c")
    assert {p.name for p in tmp_path.iterdir()} >= {"m.png", "m.pdf", "r.png", "c.png"}


def test_cross_model_figures_and_scorecard_render(tmp_path):
    from mathgeom.config import ROOT
    from mathgeom.report.figures import overview
    from mathgeom.report.paper import _results
    from mathgeom.report.render import scorecard_rows

    if not all((ROOT / "results" / c / "map_fields.json").exists() for c in ("mvp", "robustness", "gemma")):
        pytest.skip("results not generated")
    r = _results("all")
    style.save(overview.draw_forest(r), tmp_path, "forest")
    style.save(overview.draw_depth(r), tmp_path, "depth")
    sc = scorecard_rows(r)
    assert len(sc["models"]) == 5 and all(len(row["cells"]) == 5 for row in sc["rows"])
