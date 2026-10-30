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


def test_3d_draw_functions_render(tmp_path):
    """3-D panels (mplot3d): a family in its top-3 PCs and the per-axiom 3-D lattice facets."""
    import matplotlib.pyplot as plt

    from mathgeom.report.figures import families, lattice

    rng = np.random.default_rng(8)
    var = np.array([0.3, 0.2, 0.1, 0.05, 0.05])
    fig = plt.figure()
    families.draw_plane3d(fig.add_subplot(projection="3d"), rng.normal(size=(7, 3)), [f"d{i}" for i in range(7)], True, var, highlight=((6, 0),))
    style.save(fig, tmp_path, "plane3d")
    bits = np.array([[a, i, d, c] for a in (0, 1) for i in (0, 1) for d in (0, 1) for c in (0, 1)])[:13]
    ax_names = ["associative", "identity", "divisible", "commutative"]
    Vs = {k: (rng.normal(size=(13, 3)), var) for k in ("trained", "random_init")}
    notes = {k: {a: "n" for a in ax_names} for k in Vs}
    style.save(lattice.draw_facets3d(Vs, [f"s{i}" for i in range(13)], bits, ax_names, notes), tmp_path, "lattice3d")
    assert families.planes_note({}, "m", "days", 3) == ""  # no post-hoc 4 results yet: no note
    assert {"plane3d.png", "lattice3d.png"} <= {p.name for p in tmp_path.iterdir()}


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
    fmap = {"lay3": {k: rng.normal(size=(n, 3)) for k in ("arxiv", "trained", "random_init")}, "names": [f"f{i}" for i in range(n)], "pa": ["pure"] * n,
            "edges": fd.neighbour_edges(D)}
    var = np.array([0.3, 0.2, 0.1, 0.05, 0.05])
    fam3d = {f: (rng.normal(size=(m, 3)), var, [f"{f[0]}{i}" for i in range(m)]) for f, m in (("days", 7), ("angles", 12), ("ring_chain", 11), ("space_chain", 11))}
    lat3d = {"trained": rng.normal(size=(13, 3)), "random_init": rng.normal(size=(13, 3)), "pairs": [(0, 1), (2, 3), (4, 5)]}
    style.save(overview.draw_hero(r, fam3d, lat3d, fmap, conf), tmp_path, "hero")
    style.save(overview.draw_group_objects(r), tmp_path, "groups")
    style.save(overview.draw_partial_operation(r, conf), tmp_path, "partial")
    rows = [rw for _, rr in overview.forest_rows(r) for rw in rr]
    assert all({"label", "est", "ci", "status", "n", "holm"} <= set(rw) for rw in rows)
    assert sum(rw["holm"] is not None for rw in rows) == 5  # only the H-A1 rows are in the pre-registered Holm family


def test_turntable_gif_renders(tmp_path):
    """README turntable (2 frames here): needs the cached activations, skipped without them."""
    from mathgeom.config import load_config
    from mathgeom.report.figures import families

    cfg = load_config("configs/mvp.yaml")
    if not (cfg.artifacts / "acts" / "family_index.json").exists():
        pytest.skip("activations not present")
    style.use_style()
    name = families.save_turntable(cfg, "qwen3.5-2b", 12, tmp_path, frames=2)
    from PIL import Image

    assert Image.open(tmp_path / name).n_frames == 2


def test_arrow3d_draws_a_filled_head_that_fits_the_arrow():
    """3-D arrows (lattice3d) get a filled, camera-facing head of fixed size, clamped to 60% of a short arrow."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    tail, tip = np.zeros(3), np.array([0.05, 0.0, 0.0])
    head = style.arrow3d(ax, tail, tip, style.BLUE, size=0.4)
    assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
    base = head[1:].mean(0)  # midpoint of the head's base
    assert np.linalg.norm(tip - base) <= 0.6 * np.linalg.norm(tip - tail) + 1e-12
    np.testing.assert_allclose(head[0], tip)
    plt.close(fig)


def test_label_placement_is_reproducible():
    """Direct labels must land in the same place on every run (adjustText's default stops on wall-clock
    time, which made reruns differ); two identical calls give identical text positions."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(9)
    P, labels = rng.normal(size=(12, 2)) * 0.1, [f"a long label {i}" for i in range(12)]  # crowded on purpose

    def place():
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.scatter(*P.T)
        style.label_points(ax, P, labels)
        pos = np.array([t.get_position() for t in ax.texts])
        plt.close(fig)
        return pos

    np.testing.assert_array_equal(place(), place())


def test_axiom_explainer_and_gif_render(tmp_path):
    """The 4-step 'how the test works' figure (synthetic data) and the 3-D arrow GIF (2 frames, real
    activations, skipped without them)."""
    from mathgeom.config import load_config
    from mathgeom.report.figures import lattice

    rng = np.random.default_rng(11)
    bits = np.array([[a, i, d, c] for a in (0, 1) for i in (0, 1) for d in (0, 1) for c in (0, 1)])[:13]
    axioms = ["associative", "identity", "divisible", "commutative"]
    names = [f"s{i}" for i in range(13)]
    X = {k: rng.normal(size=(13, 60)) for k in ("trained", "random_init")}
    off = {a: {"consistency": 0.3, "null95": 0.2, "p": 0.01} for a in axioms}
    style.save(lattice.draw_explainer(X, names, bits, axioms, {"trained": off, "random_init": off}, "commutative", "M"), tmp_path, "explainer")
    assert (tmp_path / "explainer.png").exists()
    cfg = load_config("configs/mvp.yaml")
    if not (cfg.artifacts / "acts" / "family_index.json").exists():
        pytest.skip("activations not present")
    import json

    from mathgeom.config import ROOT

    la = json.loads((ROOT / "results" / "mvp" / "lattice.json").read_text())
    name = lattice.save_axiom_gif(cfg, "qwen3.5-2b", la, 12, tmp_path, frames_per_axiom=1, axioms_shown=["commutative", "associative"])
    from PIL import Image

    assert Image.open(tmp_path / name).n_frames == 2
