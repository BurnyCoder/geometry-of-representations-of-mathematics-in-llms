"""Tests for the independent audit's lexical controls and precision fixes (reports/prereg_addendum_lexical.md).

Global context: the random-init twin was used as the "form-only" control, but it scrambles even exact
word-level composition. The lexical baselines measure word form directly; these tests pin their
definitions, the duplicate/disjoint split of the axiom test, the exact block-order test, the kernel-form
ridge that makes 10,000 LOO permutations affordable, BCa intervals, and the 8-test Holm family.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from mathgeom.analyses import lexical
from mathgeom.analyses.lattice import axiom_pairs
from mathgeom.config import load_yaml
from mathgeom.geometry.manifold import loo_order_rho
from mathgeom.geometry.offsets import offset_consistency
from mathgeom.geometry.preprocess import char_ngram_rdm, char_ngrams


def _group_like():
    g = load_yaml("families.yaml")["lattices"]["group_like"]
    names = list(g["items"])
    return names, np.array([g["items"][c] for c in names]), g["axioms"]


def test_char_ngrams_pad_word_boundaries_and_back_the_rdm():
    assert char_ngrams("ab") == {" ab": 1, "ab ": 1}
    assert char_ngrams("Aaa")[" aa"] == 1 and char_ngrams("aaaa")["aaa"] == 2  # lower-cased counts
    D = char_ngram_rdm(["group", "semigroup"])
    a, b = set(char_ngrams("group")), set(char_ngrams("semigroup"))
    assert D[0, 1] == pytest.approx(1 - len(a & b) / len(a | b))


def test_word_bag_makes_a_shared_modifier_a_perfect_direction():
    """X → 'commutative X' is the same word-count vector for every X: consistency exactly 1."""
    heads = ["magma", "semigroup", "monoid", "loop"]
    names = heads + [f"commutative {h}" for h in heads]
    X = lexical.word_bag(names)
    assert offset_consistency(X, [(i, i + 4) for i in range(4)]) == pytest.approx(1.0)
    C = lexical.char3_vectors(names)
    assert C.shape[0] == 8 and offset_consistency(C, [(i, i + 4) for i in range(4)]) > 0.5  # shared n-grams


def test_lexical_duplicates_per_axiom_are_1_2_1_0():
    """Addendum D2: pairs that coincide after deleting 'commutative ' from their names."""
    names, bits, axioms = _group_like()
    counts = {ax: len(lexical.lexical_duplicates(names, axiom_pairs(bits, k))) for k, ax in enumerate(axioms)}
    assert counts == {"associative": 1, "identity": 2, "divisible": 1, "commutative": 0}


def test_offset_consistency_keep_restricts_the_mean():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 32))
    pairs = [(0, 1), (2, 3), (4, 5)]
    full = offset_consistency(X, pairs)
    parts = [offset_consistency(X, [pairs[i], pairs[j]]) for i, j in ((0, 1), (0, 2), (1, 2))]
    assert full == pytest.approx(np.mean(parts))
    assert offset_consistency(X, pairs, keep=[(0, 2)]) == pytest.approx(parts[1])


def test_heldout_direction_excludes_training_axis_from_its_exact_null():
    """Addendum D3: abelian group − group against the direction of the other commutative pairs."""
    names, bits, _ = _group_like()
    rng = np.random.default_rng(1)
    v = rng.normal(size=64)
    X = rng.normal(size=(13, 64)) + np.outer(bits[:, 3], 4 * v)  # commutativity adds 4v everywhere
    r = lexical.heldout_commutative_test(X, names, bits)
    assert r["cos"] > 0.8 and r["p"] < 0.05
    c = int(bits[:, 3].sum())
    assert r["n_null"] == c * (c - 1) + (13 - c) * (12 - c) == 72  # ordered pairs with equal commutativity bit (amendment 1)


def test_exact_block_order_p_for_a_perfect_line():
    """Robinson is reversal-symmetric, so a perfect line ties only with its reversal: p = 2/720."""
    t = np.arange(6.0)
    X = np.c_[t, 0.1 * t**2] @ np.random.default_rng(2).normal(size=(2, 50))
    r = lexical.exact_order_test(X)
    assert r["robinson"] == 1.0 and r["robinson_p"] == pytest.approx(2 / 720)
    assert r["pc1"] == pytest.approx(1.0) and r["pc1_p"] == pytest.approx(2 / 720)


@pytest.mark.parametrize("alpha", [10.0, 0.1, 1000.0])
def test_kernel_loo_equals_sklearn_ridge(alpha):
    """D9: the kernel form is the same estimator as sklearn's Ridge(alpha) with an intercept."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(11, 300)) * 2.0
    for order in (np.arange(11), rng.permutation(11)):
        rank = np.empty(11)
        rank[order] = np.arange(11)
        ref = np.empty(11)
        for i in range(11):
            tr = np.arange(11) != i
            ref[i] = Ridge(alpha=alpha).fit(X[tr], rank[tr]).predict(X[i : i + 1])[0]
        from scipy.stats import spearmanr

        assert loo_order_rho(X, order, alpha=alpha) == pytest.approx(spearmanr(ref, rank)[0], abs=1e-9)


def test_relative_alpha_is_scale_invariant():
    """D10: with α = mean squared norm of the training items, rescaling the data changes nothing."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(11, 200))
    o = rng.permutation(11)
    assert loo_order_rho(0.01 * X, o, alpha="rel") == pytest.approx(loo_order_rho(1000 * X, o, alpha="rel"), abs=1e-9)
    # the fixed α is not: on tiny activations it shrinks every fold to the training mean, whose LOO ranks are
    # exactly anti-correlated with the held-out ranks (ρ = −1, the LOO artifact behind values like −0.93)
    assert loo_order_rho(0.01 * X, o) == pytest.approx(-1.0) and loo_order_rho(1000 * X, o) > -0.9


def test_bca_matches_percentile_for_a_symmetric_mean():
    from mathgeom.stats.bootstrap import bca_interval

    rng = np.random.default_rng(5)
    x = rng.normal(size=400)
    boots = np.array([rng.choice(x, len(x)).mean() for _ in range(4000)])
    jack = np.array([np.delete(x, i).mean() for i in range(len(x))])
    lo, hi = bca_interval(x.mean(), boots, jack)
    assert lo == pytest.approx(np.percentile(boots, 2.5), abs=0.01) and hi == pytest.approx(np.percentile(boots, 97.5), abs=0.01)


def test_bca_shifts_toward_the_point_estimate_under_downward_bias():
    """A statistic whose resamples are biased low (like ρ on duplicated concepts) gets a BCa interval
    shifted up relative to the percentile interval."""
    from mathgeom.stats.bootstrap import bca_interval

    rng = np.random.default_rng(6)
    boots = rng.normal(0.30, 0.04, size=4000)  # resamples centred below the point estimate 0.35
    lo, hi = bca_interval(0.35, boots, rng.normal(0.35, 0.01, size=300))
    assert lo > np.percentile(boots, 2.5) and hi > np.percentile(boots, 97.5)


def test_block_only_simulation_is_deterministic_and_bounded():
    a = lexical.block_only_simulation(snr=1.0, n_sims=3, n_perm_robinson=200, n_perm_loo=50, seed=0)
    b = lexical.block_only_simulation(snr=1.0, n_sims=3, n_perm_robinson=200, n_perm_loo=50, seed=0)
    assert a == b and 0.0 <= a["reject_rate"] <= 1.0 and a["n_sims"] == 3


def test_holm_family_has_eight_tests_with_angles_as_p1_when_falsified():
    from mathgeom.report import render

    rows = render.holm_rows({"k": {"math_primary": {"rsa_boot": {"delta_vs_random_init": {"p_le_0": 0.0}}}}},
                            {"k": {"ring_chain": {"robinson_p": 0.01, "loo_order_p": 0.009}, "space_chain": {"robinson_p": 0.0005, "loo_order_p": 0.001},
                                   "angles": {"closure": 2.1, "circle_rho": 0.3, "line_rho": 0.6}}},
                            {"k": {"group_like": {"offsets": {a: {"p": 0.04} for a in ("associative", "identity", "divisible", "commutative")}}}}, "k")
    assert len(rows) == 8 and rows[-1][1] == 1.0 and rows[-1][0].startswith("H-B angles")


def test_perm_p_of_an_undefined_statistic_is_undefined():
    """A one-hot word bag of single-word items makes every distance equal, so template ρ is NaN; its p must
    not become 1/(n+1)."""
    from mathgeom.geometry.manifold import perm_p

    assert np.isnan(perm_p(float("nan"), np.array([0.1, 0.2])))
    assert perm_p(0.15, np.array([0.1, 0.2])) == pytest.approx(2 / 3)
