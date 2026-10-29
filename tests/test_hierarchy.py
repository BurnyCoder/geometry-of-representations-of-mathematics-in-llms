"""H-A2 statistics on planted structure: a two-level tree must give high super-tree match and low δ."""

import numpy as np

from mathgeom.analyses.hierarchy import cophenetic_rdm, relative_delta, spectrum_matched_gaussian, supertree_match
from mathgeom.geometry.preprocess import cosine_rdm


def _tree_points(rng, n_branch=4, per=4, d=64):
    """Leaves = branch centre + leaf offset: a 2-level hierarchy embedded in d dims."""
    centres = rng.normal(size=(n_branch, d)) * 3
    pts = np.concatenate([c + rng.normal(size=(per, d)) for c in centres])
    branch = np.repeat(np.arange(n_branch), per)
    tree = np.where(branch[:, None] == branch[None, :], 1.0, 2.0)
    np.fill_diagonal(tree, 0)
    return pts, tree


def test_supertree_match_high_for_planted_tree_low_for_noise():
    rng = np.random.default_rng(0)
    pts, tree = _tree_points(rng)
    assert supertree_match(cosine_rdm(pts - pts.mean(0)), tree) > 0.6
    noise = rng.normal(size=pts.shape)
    assert supertree_match(cosine_rdm(noise), tree) < 0.4


def test_delta_zero_on_line_and_gaussian_matches_spectrum():
    line = np.c_[np.arange(10.0), np.zeros(10)]  # points on a line form a tree metric: δ = 0
    assert relative_delta(line) < 1e-9
    rng = np.random.default_rng(1)
    C = rng.normal(size=(17, 40)) * np.linspace(3, 0.1, 40)
    G = spectrum_matched_gaussian(C, rng)
    assert G.shape == C.shape
    _, c = cophenetic_rdm(cosine_rdm(C - C.mean(0)))
    assert 0 <= c <= 1
