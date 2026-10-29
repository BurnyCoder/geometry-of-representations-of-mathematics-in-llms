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


def test_remaining_draw_functions_render(tmp_path):
    """Pure draw functions not covered above: lattice facets, a family plane, the abstraction figure."""
    import matplotlib.pyplot as plt

    from mathgeom.report.figures import axes, families, lattice

    rng = np.random.default_rng(4)
    bits = np.array([[a, i, d, c] for a in (0, 1) for i in (0, 1) for d in (0, 1) for c in (0, 1)])[:13]
    names = [f"s{i}" for i in range(13)]
    ax_names = ["associative", "identity", "divisible", "commutative"]
    notes = {k: {a: "n" for a in ax_names} for k in ("trained", "random_init")}
    Ps = {k: rng.normal(size=(13, 2)) for k in ("trained", "random_init")}
    style.save(lattice.draw_facets(Ps, names, bits, ax_names, notes), tmp_path, "facets")
    fig, ax = plt.subplots()
    families.draw_plane(ax, rng.normal(size=(7, 2)), [f"d{i}" for i in range(7)], closed=True, highlight=((6, 0),))
    style.save(fig, tmp_path, "plane")
    d = {"pcs": rng.normal(size=(40, 2)), "abstraction": rng.uniform(1, 7, 40), "concept": np.array([f"c{i}" for i in range(40)]), "pa": np.array(["pure", "applied"] * 20)}
    res = {"pcs": [{"rho_abstraction": 0.4, "rho_applied": 0.1}, {"rho_abstraction": 0.0, "rho_applied": -0.3}]}
    style.save(axes.draw(d, res, res, None, None, "M"), tmp_path, "abstraction")
    assert {"facets.png", "plane.png", "abstraction.png"} <= {p.name for p in tmp_path.iterdir()}


def test_cross_model_extra_figures_render(tmp_path):
    from mathgeom.config import ROOT
    from mathgeom.report import figdata as fd
    from mathgeom.report.figures import overview
    from mathgeom.report.paper import _results

    if not all((ROOT / "results" / c / "map_fields.json").exists() for c in ("mvp", "robustness", "gemma")):
        pytest.skip("results not generated")
    r, conf = _results("all"), "qwen3.5-2b"
    rng = np.random.default_rng(5)
    n = len(r["map_fields"][conf]["math_primary"]["fields"])
    D = np.abs(rng.normal(size=(n, n)))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    fmap = {"lay": {k: rng.normal(size=(n, 2)) for k in ("arxiv", "trained", "random_init")}, "names": [f"f{i}" for i in range(n)], "pa": ["pure"] * n,
            "edges": fd.neighbour_edges(D)}
    planes = {"days": (rng.normal(size=(7, 2)), [f"d{i}" for i in range(7)]), "angles": (rng.normal(size=(12, 2)), [f"a{i}" for i in range(12)])}
    style.save(overview.draw_hero(r, planes, fmap, conf), tmp_path, "hero")
    style.save(overview.draw_group_objects(r), tmp_path, "groups")
    style.save(overview.draw_partial_operation(r, conf), tmp_path, "partial")
    rows = [rw for _, rr in overview.forest_rows(r) for rw in rr]
    assert all({"label", "est", "ci", "status", "n", "holm"} <= set(rw) for rw in rows)
    assert sum(rw["holm"] is not None for rw in rows) == 5  # only the H-A1 rows are in the pre-registered Holm family
