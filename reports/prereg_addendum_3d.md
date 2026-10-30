# Addendum to the preregistration: cycles beyond the PC1–2 plane (post-hoc)

**Status.** Written and committed *before* this analysis was run, as an addendum to [`docs/prereg.md`](../docs/prereg.md) (git tag `prereg-v1`). It is **post-hoc**:
- It was added after all preregistered results were known.
- It was prompted by adding 3-D (top-3 principal component) views to the figures.
- Every result it produces is labelled post-hoc.

**What we had seen before writing it:**
- the PC1–2 views and the post-hoc PC1–2 cycle scores already in `results/*/posthoc.json`;
- the share of variance on each of the top 5 PCs for the confirmatory model at its preregistered layer. For angles, PC3 carries 18% of the variance (days 14%).

**What we had not seen:**
- any cycle score in a plane other than PC1–2;
- any 3-D picture of the families.

## Why

**Prior findings:**
- Engels et al. found days and months on circles in principal-component planes that are not always the top two ([arXiv:2405.14860](https://arxiv.org/abs/2405.14860)).
- Numbers are represented on a generalized helix: a linear component plus periodic components ([Kantamneni & Tegmark 2025, arXiv:2502.00873](https://arxiv.org/abs/2502.00873)).

**The gap in our analysis.** Our angles follow the number line in the PC1–2 plane and in the full space, which falsified the preregistered circle prediction. But a periodic, circle-like component could sit in other principal directions. A 3-D view could suggest one, and this test decides it instead of an impression from a rotated plot.

## Question

Do the items of a cyclic family form a cycle in some plane spanned by two of the top-5 principal components?

- **Families:** days, months, angles.
- **Lexical control for angles:** the bare numerals `digits_0_330` ("0", "30", …, "330").

## Statistic, null, data

**Score.** The existing post-hoc cycle score (`mathgeom.geometry.manifold.pc_plane_cycle_score`): the fraction of items whose angular successor in the plane is the true next item, cyclically, taking the better of the two directions.
- It is computed in each of the 10 planes (PCi, PCj), 1 ≤ i < j ≤ 5.
- PCs are those of the mean-centred items at that layer.
- **Statistic:** `best_plane_cycle`, the maximum over the 10 planes. We also record which plane attains it.

**Null.** 1,000 random permutations of the true order.
- Each permutation is scored with the same maximum over the same 10 planes, which corrects for choosing the best plane.
- p = (1 + #{null ≥ observed}) / 1001.
- Seed: `cfg.seed + 7`, a separate stream, so existing results are untouched.

**Where.**
- Every model in mvp, robustness and gemma, and every random-init twin.
- Every layer is recorded; the **headline is the preregistered layer** (relative depth 0.5).
- Per-layer values are descriptive and unadjusted.

### Amendment 1: closed loops only

Made before any real-data run, and committed separately.

**What the unit test found.** A unit test showed that the cycle score alone gives a perfect 1.0 to an *open arc*. Example: the plane of a linear axis and one periodic coordinate, where points trace a U. The angular-successor order is then perfect, and only the "closing" step last → first spans the opening.

**Why it matters.** PCA routinely bends a line into such an arc (the horseshoe effect; [Diaconis, Goel & Holmes 2008, Ann. Appl. Stat. 2(3)](https://arxiv.org/abs/0811.1477)). So the statistic above would call a number line a cycle.

**The change.** A plane counts only if the loop also closes there:
- The *closed-cycle score* of a plane is its cycle score if the in-plane closure ratio is below 2, and 0 otherwise.
- The closure ratio is the preregistered `closure_ratio`: d(first, last) ÷ mean adjacent distance, computed on the plane's Euclidean distances.
- The threshold is the preregistered "closure < 2".

**What uses it.** The maximum over planes, the permutation null (which applies the same rule to every permuted order) and the decision rule below all use the closed-cycle score. The plain cycle score and the closure ratio of every plane are recorded too.

**What it does not change.** The existing PC1–2 scores are unchanged. We report that a high PC1–2 cycle score alone does not show a *closed* loop. For days and months, the full-space closure ratios (1.2–2.0) say whether the loop closes.

### Amendment 2: the test is conservative

Made before any real-data run.

**The finding.** The closure rule gives many null draws a score of exactly 0. That makes the statistic discrete with an atom at 0, and the test conservative. In 1,000 simulations on structureless Gaussian data, the rejection rate at α = 0.05 was:
- n = 12: 1.8% ± 0.8%;
- n = 7: 4.0% ± 1.2%.

**What changes.**
- A conservative test keeps the false-positive rate below α, so the decision rule is unchanged; it only lowers power.
- The unit-test criterion in "Code" (rejection within [0.02, 0.09]) is replaced by a validity check: rejection ≤ 0.08 at n = 7 and n = 12.
- A non-significant result for angles (n = 12) is reported together with this loss of power.

## Decision rule (angles)

"Angles have a circular component beyond PC1–2" is accepted only if **all four** hold at the preregistered layer:

1. Best-plane p < 0.05 after Holm correction across the 5 trained models (one p per model). The claim is made for each model that passes.
2. The best plane is **not** PC1–PC2.
3. The trained model's best-plane score is higher than its random-init twin's.
4. The numeral reference `digits_0_330` does **not** meet conditions 1–2 in the same model. If the bare numbers already cycle, the cycle is lexical (tokens), not a property of angles.

## What changes in the write-up

**If the rule is met for a model:**
- The statement "angles are not a circle" is changed everywhere: README, paper (abstract, §4.3, discussion), summary, site section 8 and the site tile, scorecard.
- New wording: the preregistered circle prediction (full-space closure, and the PC1–2 view) is still falsified, but a **post-hoc** circular component exists in the PCi–PCj plane in that model.
- The 3-D views of angles open on the axes that include that plane.

**If it is not met:** the statement stands and gains "no cycle in any of the top-5 PC planes either".

**Days and months:** reported descriptively the same way (best plane, score, p). They are the positive control, and no claim depends on them here.

## What this cannot change

Every preregistered statistic is computed in the full activation space, and none is affected by projections:
- field-map RSA;
- Robinson scores;
- the closure ratio and circle/line template correlations;
- offset consistency;
- analogy ranks.

## Code

- `pc_plane_cycle_score(X, order, plane=(0, 1))` gains the `plane` argument. Its default reproduces the existing scores exactly, which a test checks.
- `best_plane_cycle` computes the principal components once per layer; permutations only re-score orders.
- The results go to the `pc_planes_cycle` block of `results/<config>/posthoc.json`, produced by `python run.py posthoc --config configs/<config>.yaml`.
- The tests in `tests/test_geometry.py` cover:
  - a cycle planted in the PC3–PC4 plane is found;
  - on structureless data the test rejects at about the nominal 5% rate (200 simulations, rejection rate within [0.02, 0.09]); *note added 2026-10-30 (audit): superseded by amendment 2, which made the test conservative, so the committed test checks validity with 300 simulations and a rejection rate ≤ 0.08 (`tests/test_geometry.py`)*;
  - the default plane equals the old function.
