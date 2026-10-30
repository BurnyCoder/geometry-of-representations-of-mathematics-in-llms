---
title: "Do LLMs Organize Mathematics Geometrically? Field Maps, Ordered Chains and Axiom Lattices in Residual Streams"
author: "Libor \"Burny\" Burian"
date: "October 2026"
abstract: |
  Large language models represent days of the week and months on circles (Engels et al., 2024). We ask whether *abstract mathematics* is organized geometrically too. We test three preregistered hypotheses on five small base models (the Qwen models have at most 1.9B text parameters; Gemma-4-E2B has 2.3B effective, 5.1B with embeddings) and compare every model with a random-init twin (same architecture and tokenizer, no learning). An independent post-hoc audit then reran every test on representations that encode only the spelling of the concept names, because the twin turned out not to control word form: it scrambles even exact word composition.

  **(A) Field map, the primary hypothesis.** Centroids of 17 mathematical fields are computed from concepts drawn from Wikipedia glossaries, excluding concepts whose names leak the field. In Qwen3.5-2B-Base, their geometry correlates with field relatedness measured by arXiv cross-listing over 3.2M papers: Spearman ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.2f}}, against ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.random_init:.2f}} for the random-init twin (paired bootstrap 95% CI of the difference [{{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.0:.2f}}, {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.1:.2f}}]). The map replicates in Qwen3.5-0.8B (ρ = {{num:robustness.map_fields.qwen3.5-0.8b.math_primary.rsa.trained:.2f}}), Qwen2.5-1.5B (ρ = {{num:robustness.map_fields.qwen2.5-1.5b.math_primary.rsa.trained:.2f}}) and Gemma-4-E2B (ρ = {{num:gemma.map_fields.gemma-4-e2b.math_primary.rsa.trained:.2f}}). It does not follow from the names: their character 3-grams give ρ = {{num:mvp.lexical.qwen3.5-2b.field_map.rsa.char3:.2f}} and the model's own input embeddings of the names ρ = {{num:mvp.lexical.qwen3.5-2b.field_map.rsa.layer0_bag:.2f}}, and the trained map exceeds all three word-form baselines in {{fact:field_beyond_lexical}}. It is not a clean hierarchy (our preregistered cophenetic test does not beat the twin), a contrastive text embedder (Qwen3-Embedding-0.6B) matches it, and over one trillion tokens of math continued pretraining (Qwen2.5-Math) do not sharpen it (Δρ CI [{{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.0:.2f}}, {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.1:.2f}}]).

  **(B) Shapes.** The space chain (T0 ⊃ … ⊃ Hilbert space) is ordered under the full preregistered test (Robinson score and leave-one-out order, Holm-corrected, above the twin) in {{fact:chain_ordered.space_chain}}; spelling baselines do not order it, and it stays ordered within its metrizable … Hilbert block in {{fact:space_block.1}}. The ring chain (rng ⊃ ring ⊃ … ⊃ algebraically closed field) passes the same test in {{fact:chain_ordered.ring_chain}}, but its order is confounded with its head nouns (rng | ring | … domain | field): clusters of head nouns with no order at all pass the test in up to {{num:mvp.lexical.block_only_simulation.2.reject_rate:.0%}} of simulations, and the six "… domain" items are ordered among themselves in {{fact:ring_within}}. Angles 0°–330° are encoded like the number line rather than a circle in all five models, which falsifies our preregistered prediction.

  **(B2) Axiom lattices.** For each axiom of the group-like structures (magma … abelian group), difference vectors between structures that differ only in that axiom are more parallel than random pairs: in Qwen3.5-2B, offset consistency is {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.consistency:.2f}} for commutativity and {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.associative.consistency:.2f}} for associativity, against {{num:mvp.lattice.qwen3.5-2b__random.group_like.offsets.commutative.consistency:.2f}} and {{num:mvp.lattice.qwen3.5-2b__random.group_like.offsets.associative.consistency:.2f}} in the twin (Holm-significant for associativity in {{fact:axiom_holm.associative}} and for commutativity in {{fact:axiom_holm.commutative}}). The audit shows that most of this follows from the names. Character 3-grams of the names, with no model, pass the same test for all four axioms. "X → wooden X" and the nonsense "X → zorbic X" are about as consistent as "X → commutative X"; "commutative" is the most consistent of the three in {{fact:nonce_above}}. Two signs of more than word form remain. Associativity exceeds every word-form baseline and its lexically disjoint pairs pass the null in {{fact:axiom_beyond_lexical.associative}}. The "commutative" direction predicts *abelian group − group*, a pair that shares no word with it, in {{fact:abelian_transfer}}. Group-object parallelograms (Lie group − topological group ≈ smooth manifold − topological space) are explained by shared words.

  The days/months positive control met our preregistered criterion only partially; in post-hoc tests, days close into a loop in the top-2 principal-component plane in {{fact:closes_top2.days}} and months in a lower plane in {{fact:closes_lower_plane.months}}. Methodologically, partial regression on distance matrices inflates type-I error when nuisances are proxies of confounders, and a random-init twin is not a control for word form: studies of concept geometry need spelling-only baselines.
---

# 1 Introduction

Interpretability research has found that LLMs encode some concepts as low-dimensional geometric objects rather than isolated directions:
- circles for days and months, which the models causally use for modular arithmetic ([Engels et al., 2024](https://arxiv.org/abs/2405.14860));
- a helix for numbers ([Kantamneni & Tegmark, 2025](https://arxiv.org/abs/2502.00873));
- a spiral for the periodic table ([Lei & Cooper, 2025](https://arxiv.org/abs/2502.10871));
- linear "crystals" of analogies among sparse-autoencoder features ([Li et al., 2024](https://arxiv.org/abs/2410.19750)).

[Karkada et al. (2026)](https://arxiv.org/abs/2602.15029) explain such shapes with symmetries of co-occurrence statistics: cyclic latents give circles, and ordered latents give 1-D curved manifolds.

Mathematics is an unusually good test bed for these ideas, because its concepts carry *exact* external structure. Structures are defined by axiom sets, so they sit on the vertices of a hypercube. Fields relate through shared methods. Algebraic structures form inclusion chains. As far as our literature search found, nobody has tested whether LLM representations of abstract mathematical concepts reflect any of this. The closest result is the "math/code lobe" of [Li et al. (2024)](https://arxiv.org/abs/2410.19750), which does not resolve fields or structures.

**Contributions.**

1. Evidence that field-level geometry tracks how mathematicians cross-list papers (H-A1, the primary preregistered hypothesis), replicated across five models from two families, beyond random-init twins and beyond three word-form baselines, and unchanged by heavy math pretraining.
2. Evidence that the space chain of separation axioms through normed, Banach and Hilbert spaces is an ordered path, also within its topical blocks. The ring chain's order is not established beyond the head nouns of its names.
3. A sober reading of axiom offsets: they are consistent, but mostly because the names encode the axioms. Associativity, and a commutativity direction that reaches "abelian group", are suggestive exceptions.
4. A preregistered test (`docs/prereg.md`, git tag `prereg-v1`) with random-init twins, leakage flags, permutation nulls and paired bootstrap contrasts, followed by an independent post-hoc audit (`reports/prereg_addendum_lexical.md`) with word-form baselines, nonce modifiers, exact within-block tests and extra twin seeds.
5. Negative results: angles are not circular, math pretraining does not sharpen the map, and group-object parallelograms are lexical. Two methodological negatives: partial RDM regression is miscalibrated under realistic confounding, and a random-init twin does not control for word form.

# 2 Related work

**Feature geometry.**
- *Low-dimensional and circular features:* [Engels et al. (2024)](https://arxiv.org/abs/2405.14860), [Kantamneni & Tegmark (2025)](https://arxiv.org/abs/2502.00873), [Gurnee & Tegmark (2023)](https://arxiv.org/abs/2310.02207).
- *Manifolds:* [Modell et al. (2025)](https://arxiv.org/abs/2505.18235), and supervised MDS for discovering them ([Tiblias et al., 2025](https://arxiv.org/abs/2510.01025)).
- *A statistical account:* [Karkada et al. (2026)](https://arxiv.org/abs/2602.15029).

**Hierarchy and categories.** [Park et al. (2024)](https://arxiv.org/abs/2406.01506) proposed orthogonality between hierarchical concepts. Golechha et al., *Intricacies of Feature Geometry in Large Language Models* ([ICLR 2025 blog track](https://iclr.cc/virtual/2025/poster/31349); [long version](https://www.greaterwrong.com/posts/eczwWrmX5XNEo7JsS/intricacies-of-feature-geometry-in-large-language-models)), showed that the same geometry appears for random concepts and random unembeddings after whitening. We therefore avoid whitening and compare every statistic with an untrained twin.

**Science concepts.** [Lei & Cooper (2025)](https://arxiv.org/abs/2502.10871) found a periodic-table spiral. Mathematical "world models" in LLMs have been reported as local and not robust ([Kvinge et al., 2025](https://icml.cc/virtual/2025/49128), *Probing the Limits of Mathematical World Models in LLMs*, ICML 2025 workshop poster).

**Offsets and word form.** Vector offsets of word embeddings ([Mikolov, Yih & Zweig, 2013](https://aclanthology.org/N13-1090/)) mix offset consistency with plain similarity, and simple baselines solve many analogies without the offset ([Linzen, 2016](https://aclanthology.org/W16-2503/)). With additive composition, "modifier + noun" is a fixed offset by construction ([Mitchell & Lapata, 2010](https://doi.org/10.1111/j.1551-6709.2010.01106.x)). Our audit therefore compares axiom offsets with nonce modifiers and with spelling-only baselines.

**Statistics.** Partial Mantel tests are known to be anticonservative ([Guillot & Rousset, 2013](https://doi.org/10.1111/2041-210X.12018)). We test Freedman–Lane permutation ([Freedman & Lane, 1983](https://doi.org/10.1080/07350015.1983.10509354); compared with alternatives by [Anderson & Legendre, 1999](https://doi.org/10.1080/00949659908811936)) in our setting (Section 3.5).

# 3 Methods

## 3.1 Concepts and fields

**Field map (H-A).**
- **Source.** Concepts come from 21 Wikipedia "Glossary of …" pages covering 16 mathematical fields, plus "List of numerical analysis topics" for the 17th, at pinned revisions. Four non-math control fields (biology, chemistry, geology, astronomy) go through the same pipeline, except that they have no field root words, so the `leak_own` flag cannot fire for them.
- **Cleaning.** Terms keep 1–4 alphabetic words. Fragments, prefixes and adverbs are dropped.
- **Flags.** These exclude a concept from the *primary* set:
  - `leak_own`: the name contains its field's root word, e.g. "topological space";
  - `eponym`: a proper name, detected by mid-sentence capitalization on the same page (this missed five eponyms in the primary set, e.g. "tate curve"; `docs/deviations.md`);
  - `polysemous`: a single everyday word with wordfreq Zipf ≥ 4.
- **Sampling.** At most 30 primary concepts per field.
- **Split.** A 20% pilot split was used only for debugging. All statistics below use the 80% confirmatory split: {{num:mvp.map_fields.qwen3.5-2b.math_primary.n}} primary math concepts across {{num:mvp.map_fields.qwen3.5-2b.math_primary.n_fields}} fields.

## 3.2 Ground-truth field distance

The Mathematics Subject Classification ([MSC2020](https://msc2020.org/)) is flat above its two-digit codes, so we measure field relatedness from practice instead. For fields A and B, with S_A the set of arXiv papers listing any of A's categories, the distance is 1 − |S_A∩S_B|/√(|S_A||S_B|), i.e. one minus the Ochiai similarity. This is computed over the CC0 arXiv metadata snapshot (3,224,635 papers, pinned commit).

## 3.3 Concept families

Every family lists its ground truth and source in `configs/families.yaml`.

**Ordered families:**
- days and months, as positive controls;
- a ring chain (11 items);
- a space chain of separation axioms through normed, Banach and Hilbert spaces (11 items);
- number systems (7 items, exploratory);
- n-gons (3–12) and n-dimensional spaces (0–10), each with a number-word reference;
- angles 0°–330°, with a digit reference.

**Group-like axiom lattice.** Thirteen structures with bits (associative, identity, divisible, commutative), from the [Magma group-like table](https://en.wikipedia.org/wiki/Magma_(algebra)). The self-distributive lattice (shelf, rack, spindle, quandle, kei) is exploratory.

## 3.4 Models and representations

| Model | Role |
|---|---|
| [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (Feb 2026) | **Confirmatory**; official Qwen-Scope SAEs ([Deng et al., 2026](https://arxiv.org/abs/2605.11887)) |
| `Qwen3.5-0.8B-Base` | Scale |
| `Qwen2.5-1.5B` vs `Qwen2.5-Math-1.5B` | Same base, + Qwen Math Corpus v2, over 1T tokens ([Qwen2.5-Math report, §2 and Fig. 2](https://arxiv.org/abs/2409.12122)) |
| [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) | Second model family; 2.3B effective parameters, 5.1B with embeddings |

Each model has a random-init twin with weights redrawn using seed 0.

- **Read point.** Each item is embedded in 8 templates ending with the concept, e.g. "In mathematics, the notion of *X*". Field-map templates name the domain ("mathematics" or "science") but never a sub-field. Family templates use one fixed domain per family, which can be a sub-field ("In abstract algebra, consider *X*"), so the domain is constant across the items being compared. The representation is the residual stream at the concept's last token, averaged over templates.
- **Preprocessing.** Mean-centering, with no whitening. Centering removes the shared offset that otherwise dominates cosine similarity in residual streams ([Timkey & van Schijndel, 2021](https://arxiv.org/abs/2109.04404)).
- **Distances.** Cosine distance for the field map. Euclidean distance for families: cosine distance on a small, centered, collinear family is degenerate, which our synthetic tests showed.
- **Layer.** The primary layer is fixed in advance at relative depth 0.5.

## 3.5 Statistics: why paired twin contrasts

Before analysing any LLM data, we simulated regression on distance matrices with Freedman–Lane permutation:

{{table:calibration:mvp}}

The test is calibrated when the nuisance distance matrix *is* the confounder. It rejects far too often when the nuisance only approximates the confounder, which is exactly the situation for lexical form and frequency. We therefore made **paired contrasts against the random-init twin** the primary tests. We intended the twin to measure form-only similarity on the same concepts; the audit (§3.6) shows that it does not, because a random network scrambles even exact word composition. Uncertainty comes from a stratified concept bootstrap with identical resamples for both models: 2,000 resamples for H-A1 and H-A3, and 1,000 for the hierarchy, baseline and tree-distance contrasts.

- **Field map.** We use field-level RSA, i.e. the Spearman ρ between the centroid RDM and the arXiv distance, with a Mantel permutation over field labels (10,000 permutations) and a max-over-layers null for layer curves.
- **Families.** Every statistic is unsupervised or leave-one-out, and is tested against an order-permutation null. With n ≤ 13 items in more than 1,000 dimensions, a supervised fit can impose any order. As a negative control, 100 random 11-item lists of math concepts must be rejected at about 5%; this checks the code, since a permutation test is exact under exchangeability by construction, and it cannot detect confounds such as spelling. The leave-one-out order null has 10,000 permutations at the preregistered layer (300 before the audit).
- **Multiplicity.** Holm correction over the 8 preregistered confirmatory tests (§4 of the preregistration); the angles prediction is a decision rule without a p-value and enters with p = 1, because it was falsified.
- **Figures, including 3-D views.** Every picture is an unsupervised projection: PCA, or metric MDS for field maps. 3-D views show each layer's own top principal components with true proportions, and state the share of variance on each axis (PCA) or the stress-1 of the layout (MDS; [Kruskal, 1964](https://doi.org/10.1007/BF02289565)). Alignment between panels and layers is display-only: rotation, or only sign flips for 3-D PCA axes. No statistic is computed from a picture. The single test that looks beyond PC1–2, the closed-cycle test over the top-5 planes, is post-hoc and was pre-specified in an addendum before it ran (§4.3).

## 3.6 Independent audit: word-form controls (post-hoc)

After all preregistered results were known, an independent audit found that no control measured word form directly. In the Qwen3.5-2B twin, the pairs magma → semigroup and commutative magma → commutative semigroup, which differ only by the literal word "commutative", have difference vectors with cosine {{num:mvp.lexical.qwen3.5-2b.group_like.decomposition.twin.associative.dup_mean:.2f}}; the trained model gives {{num:mvp.lexical.qwen3.5-2b.group_like.decomposition.trained.associative.dup_mean:.2f}}, and a bag of words gives 1. We pre-specified the following in an addendum committed before the analyses ran ([`reports/prereg_addendum_lexical.md`](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/blob/master/reports/prereg_addendum_lexical.md)); all of it is post-hoc.
- **Word-form baselines.** Every test is rerun, with the same statistic, centring, null and seed, on (i) counts of the names' character 3-grams, (ii) a bag of words and (iii) the sum of the model's trained input embeddings over the name's tokens, which carries token meaning but no context.
- **Where the axiom offsets come from.** Same-axiom pairs that coincide after deleting "commutative" are *lexical duplicates*; the remaining, disjoint cosines are tested against the same null. We also test the 8 names without "commutative", and whether the "commutative" direction predicts *abelian group − group*, the one commutativity pair that shares no word with it (exact null over the 72 ordered pairs that do not change commutativity). A nonce-modifier lattice calibrates plain composition: X → wooden X and X → zorbic X for six heads.
- **Chains.** Exact within-block order tests (the six "… domain" items of the ring chain; the separation-axiom and metric blocks of the space chain) and a simulation of how often head-noun clusters without any order pass the preregistered chain test.
- **Precision.** 10,000 leave-one-out permutations (a kernel-form ridge identical to the preregistered one), a scale-relative ridge penalty as sensitivity, the 8-test Holm family, BCa bootstrap intervals ([Efron, 1987](https://doi.org/10.1080/01621459.1987.10478410)) and four extra random-init seeds for the confirmatory model's twin.

An effect counts as *beyond word form* if it exceeds every word-form baseline and, for the axioms, its lexically disjoint pairs also pass the null.

# 4 Results

The results follow how well they survive the audit's word-form controls: the field map (H-A1, the primary preregistered hypothesis, §4.1) and its robustness (§4.2), the shapes of ordered families (H-B, §4.3), the axiom lattices (H-B2, §4.4) and the secondary analyses (§4.5). Every test keeps its preregistered status (confirmatory, secondary or exploratory); analyses added later are labelled as added, post-hoc or audit. Where a 3-D and a 2-D view of the same data exist, the 3-D view comes first.

![**Results at a glance.** Blue: trained model; grey: its random-init twin (same architecture and tokenizer, no learning); × and +: word-form baselines (the names' character 3-grams; the model's own input embeddings summed over the name's tokens). The starred model is the preregistered confirmatory model, Qwen3.5-2B. *Top row, the field map.* Left: field-centroid geometry correlates with arXiv cross-listing in all five models (dots: ρ; lines: 95% bootstrap CI), well above the twins and both word-form baselines. Right: 3-D field maps (metric MDS) for arXiv cross-listing, the confirmatory model and its twin; grey edges join each field to its two nearest arXiv neighbours. *Middle row, axiom offsets.* Left: every pair of structures that differ only in commutativity, drawn as an arrow from one common origin in the top-3 principal components; the model's arrows point roughly one way, and the note gives the consistency of X → wooden X for comparison. Right: offset consistency per axiom for every model and twin, against the confirmatory model's random-pair null (grey bar: mean to 95th percentile); diamonds: the confirmatory model with the token-count direction removed; ×: the names' character 3-grams, which need no model. *Bottom row, shapes.* Days of the week go round in order (PC-plane cycle score; positive control). Angles do not close; the orange edge is the 330° → 0° closure the preregistration predicted. The space chain is an ordered path, also within its blocks; the ring chain is grouped by its head nouns, and its six "… domain" items are not ordered among themselves (exact p under the panel). Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#s-glance>.]({{fig:hero}}){width=100%}

**Every test in every model.** The table below summarizes the confirmatory and secondary tests for all five models.

{{table:scorecard:all}}

## 4.1 H-A1: a map of mathematical fields

![**Effect sizes with 95% bootstrap CIs (unadjusted).** Blue: CI excludes 0. "Added" marks analyses added by the plan audits and "audit" those of the independent audit, rather than the preregistration. The field map beats the twin and the names' character 3-grams in every model; in Qwen3.5-2B it beats fastText, but not the Qwen3-Embedding text embedder. The Judges-vs-Mathlib line is a plain bootstrap of one correlation. Math pretraining (H-A3) and the hierarchy test (H-A2) are nulls. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-forest>.]({{fig:forest}}){width=82%}

{{table:field_map:mvp}}

![**The map is present at every depth in the Qwen models, and at most depths in Gemma.** ρ between the field-centroid RDM and the arXiv distance at each layer (x: relative depth). Blue: trained; grey: twin. The horizontal line is the 95th percentile of the max-over-layers permutation null; the vertical line marks the preregistered layer. {{fact:clears_null_every_layer}} clear the null at every layer; Gemma-4-E2B is at or below it in {{fact:below_null_runs.gemma-4-e2b}}. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-depth>.]({{fig:depth_curves}}){width=100%}

**The trained model's field geometry tracks arXiv relatedness, and the random-init twin's does not.** For primary concepts at the preregistered layer, ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}} with Mantel p = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_perm_p:.4f}}. The twin gives ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.random_init:.3f}} and layer-0 embeddings give ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.layer0:.3f}}. The difference from the twin has 95% CI [{{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.0:.3f}}, {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.1:.3f}}].

**Layer profile.** The map emerges in the first blocks and peaks at layer {{num:mvp.map_fields.qwen3.5-2b.math_primary.layer_curve.best_layer}}. It stays above the 95% max-over-layers null ({{num:mvp.map_fields.qwen3.5-2b.math_primary.layer_curve.max_null95:.2f}}) at every layer.

**Clustering.** Leave-one-out nearest-centroid field accuracy is {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.trained.acc:.3f}}, against {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.random_init.acc:.3f}} for the twin and {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.chance:.3f}} chance.

**Reliability.** Split-half template reliability of the field RDM is {{num:mvp.map_fields.qwen3.5-2b.math_primary.split_half:.3f}}.

![**Related fields stay close in the trained model, and the twin scrambles them.** Metric MDS of three 17 × 17 distance matrices: arXiv cross-listing (ground truth), Qwen3.5-2B field centroids at layer {{num:mvp.map_fields.qwen3.5-2b.primary_layer}}, and the random-init twin. Panels are rotated or reflected onto the arXiv layout for display only. Grey edges join each field to its two nearest arXiv neighbours and are drawn in every panel: short edges mean the neighbourhood is preserved. Blue: pure; orange: applied. ρ is computed on the full matrices, not on the 2-D layout. Interactive version, with a layer slider: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-map>.]({{fig:field_maps_qwen3.5-2b}}){width=100%}

![**The same matrices, ranked.** Colour is the rank of each distance within its matrix (Spearman RSA compares ranks), with rows in one fixed order grouped by the five super-tree branches. The trained matrix partly reproduces the arXiv block structure; the twin's does not. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-rdm>.]({{fig:field_rdms_qwen3.5-2b}}){width=100%}

**Not a rogue-dimension artifact (post-hoc).** The preregistered all-but-the-top variant ([Mu & Viswanath, 2018](https://arxiv.org/abs/1702.01417)) removes the top d/100 = 20 principal components, and it reverses the correlation (ρ = {{num:mvp.posthoc.qwen3.5-2b.abtt_sweep.20:.2f}}). A post-hoc sweep explains why:

{{table:posthoc:mvp}}

- Removing the top component *raises* ρ to {{num:mvp.posthoc.qwen3.5-2b.abtt_sweep.1:.2f}}.
- No component dominates.
- Field identity is spread over the top ~10 components, which carry far more field variance (η²) than the twin's.

So the map lives in the dominant semantic subspace, and removing 20 components removes the map itself.

**H-A2: hierarchy is not supported as preregistered.** The preregistered measure is the cophenetic correlation ([Sokal & Rohlf, 1962](https://doi.org/10.2307/1217208)) between an average-linkage (UPGMA) dendrogram of the field centroids and the five-branch super-tree.

{{table:hierarchy:mvp}}

{{table:hierarchy:robustness}}

{{table:hierarchy:gemma}}

- **Against shuffled trees:** the trained model's dendrogram matches the super-tree better (ρ = {{num:mvp.hierarchy.qwen3.5-2b.coph_match.trained:.3f}}, p = {{num:mvp.hierarchy.qwen3.5-2b.coph_match_p_shuffled_tree:.3f}}).
- **Against the twin:** the paired difference CI includes zero ([{{num:mvp.hierarchy.qwen3.5-2b.coph_match_delta_vs_random_ci.0:.3f}}, {{num:mvp.hierarchy.qwen3.5-2b.coph_match_delta_vs_random_ci.1:.3f}}]), and the same holds in every replication model. The preregistered criterion requires beating both, so it fails.
- **Intrinsic tree-likeness:** the cophenetic correlation of the representation with its own dendrogram is no higher than for a spectrum-matched Gaussian cloud.
- **Gromov δ** (relative δ = 2δ/diameter, as in [Khrulkov et al., 2019](https://arxiv.org/abs/1904.02239)): relative δ is lower than the Gaussian cloud's, but higher than the twin's.

A tree-distance RSA we first reported, the Spearman correlation of the centroid RDM with the super-tree distance, did beat the twin (trained {{num:mvp.map_fields.qwen3.5-2b.math_primary.hierarchy.rho.trained:.3f}} vs twin {{num:mvp.map_fields.qwen3.5-2b.math_primary.hierarchy.rho.random_init:.3f}}). It is *not* the preregistered measure, so it is post-hoc. It largely restates H-A1: fields that mathematicians relate are close.

**Specificity.** The non-math control fields (biology, chemistry, geology, astronomy) are also clustered by the trained model and not by its twin:

{{table:clustering:mvp}}

The pre-registered comparison confounds domain with the number of fields: 17 for math, 4 for the control. We therefore ran a post-hoc matched comparison over 200 random 4-field subsets of the math fields:
- **Math subsets:** mean leave-one-out accuracy {{num:mvp.posthoc.qwen3.5-2b.matched_clustering.math_subsets_mean:.3f}} (95% range {{num:mvp.posthoc.qwen3.5-2b.matched_clustering.math_subsets_ci.0:.3f}}–{{num:mvp.posthoc.qwen3.5-2b.matched_clustering.math_subsets_ci.1:.3f}}).
- **Control fields:** {{num:mvp.posthoc.qwen3.5-2b.matched_clustering.control_acc:.3f}}.
- **Overlap:** {{num:mvp.posthoc.qwen3.5-2b.matched_clustering.frac_math_subsets_ge_control:.2f}} of the math subsets reach or exceed the control value.

The control fields cluster more strongly than most matched math subsets in {{fact:control_beats_most_math_subsets}} (Gemma: {{num:gemma.posthoc.gemma-4-e2b.matched_clustering.frac_math_subsets_ge_control:.2f}} of the subsets reach the control value). This does not show that the map is unspecific to mathematics: the controls are four whole sciences, while the math fields are sub-fields of one discipline, so the comparison confounds domain with granularity (audit). A granularity-matched control, sub-fields of another science, was not run, and specificity remains open.

**Baselines (plan; added after the first report; Qwen3.5-2B only).** We compare against two non-generative representations of the same concepts:
- **fastText** wiki-news-300d-1M vectors, averaged over words ([Mikolov et al., 2018](https://arxiv.org/abs/1712.09405)), and **[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)**, a contrastively trained text embedder built on an LLM.
- **vs fastText:** the LLM's field map is stronger (Δ CI [{{num:mvp.baselines.qwen3.5-2b.fasttext_covered.boot.delta_vs_fasttext.ci.0:.2f}}, {{num:mvp.baselines.qwen3.5-2b.fasttext_covered.boot.delta_vs_fasttext.ci.1:.2f}}] on the {{num:mvp.baselines.qwen3.5-2b.fasttext_covered.n}} concepts that fastText covers).
- **vs Qwen3-Embedding:** the LLM is not stronger; the embedder reaches ρ = {{num:mvp.baselines.qwen3.5-2b.all_primary.rsa.qwen3_embedding:.3f}} against the LLM's {{num:mvp.baselines.qwen3.5-2b.all_primary.rsa.llm_trained:.3f}} (Δ CI [{{num:mvp.baselines.qwen3.5-2b.all_primary.boot.delta_vs_qwen3_embedding.ci.0:.2f}}, {{num:mvp.baselines.qwen3.5-2b.all_primary.boot.delta_vs_qwen3_embedding.ci.1:.2f}}]).

So the field map is a property of contextual text representations in general, not something specific to generative LLMs.

{{table:baselines:mvp}}

**Beyond word form (audit).** Could the map follow from the spelling of the concept names, for example suffixes shared within a glossary? Character 3-grams, a bag of words and the model's own input embeddings of the names give field maps of ρ = {{num:mvp.lexical.qwen3.5-2b.field_map.rsa.char3:.2f}}, {{num:mvp.lexical.qwen3.5-2b.field_map.rsa.word_bag:.2f}} and {{num:mvp.lexical.qwen3.5-2b.field_map.rsa.layer0_bag:.2f}} in Qwen3.5-2B, far below the trained model's {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.2f}}. The paired bootstrap Δ CIs exclude zero against all three baselines in {{fact:field_beyond_lexical}}. The map is built by the model's processing of the names in context, not read off their spelling.

{{table:lexical_field:all}}

**Bootstrap bias (audit).** Resampling concepts with replacement duplicates some of them, which makes resampled centroids noisier and pulls resampled ρ toward zero. The preregistered percentile intervals therefore sit below the point estimates: in Qwen3.5-2B, ρ = {{num:mvp.lexical.qwen3.5-2b.bootstrap.trained.point:.3f}} but the bootstrap mean is {{num:mvp.lexical.qwen3.5-2b.bootstrap.trained.boot_mean:.3f}}. BCa intervals ([Efron, 1987](https://doi.org/10.1080/01621459.1987.10478410)) correct for this bias and shift every interval upward; no conclusion changes, because the bias works against H-A1 and cancels in the H-A3 contrast:

{{table:bootstrap:all}}

## 4.2 Robustness: scale, model family, math training, templates

{{table:field_map:robustness}}

{{table:field_map:gemma}}

**H-A3 (math training): null.** Qwen2.5-Math-1.5B is Qwen2.5-1.5B continued on over 1T tokens of math ([Qwen2.5-Math report](https://arxiv.org/abs/2409.12122)). Its field map is no stronger than its base's: ρ = {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.rho.qwen2.5-math-1.5b:.3f}} vs {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.rho.qwen2.5-1.5b:.3f}}, and the paired CI of the difference is [{{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.0:.3f}}, {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.1:.3f}}].

**Twin seeds (audit).** Every twin contrast above uses one random draw (seed 0). Four more draws of the confirmatory model's twin show how much the twin statistics vary. The trained model exceeds all {{num:twin_seeds.twins.qwen3.5-2b.summary.n_seeds}} draws for the field ρ ({{num:twin_seeds.twins.qwen3.5-2b.summary.trained_exceeds_every_twin.field_rho}}), and the lowest lower bound of the H-A1 Δ CI over the draws is {{num:twin_seeds.twins.qwen3.5-2b.summary.field_delta_ci_lowest_lower:+.3f}}; the last column gives the same comparison for the other statistics:

{{table:twins:twin_seeds}}

**Template variants (exploratory; Qwen3.5-2B).** We reran the field-map, shape, lattice, group-object, post-hoc and chain-frequency analyses with two other template sets (not the hierarchy, baseline, SMDS, axis, causal or SAE analyses).

*Field-revealing concept templates* name the field, e.g. "In group theory, the notion of X":
- The graded field map barely changes (ρ = {{num:mvp_variants.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}}, against {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}} with neutral templates).
- Leave-one-out field accuracy jumps to {{num:mvp_variants.map_fields.qwen3.5-2b.math_primary.loo_acc.trained.acc:.3f}}, and the twin also rises to {{num:mvp_variants.map_fields.qwen3.5-2b.math_primary.loo_acc.random_init.acc:.3f}}. This is exactly the label leakage the preregistration excluded from the primary analysis.

*No-domain family templates* drop the shared context, e.g. "Consider X":
- Axiom offsets (§4.4) weaken but persist for commutativity ({{num:mvp_variants.lattice.qwen3.5-2b.group_like.offsets.commutative.consistency:.2f}}, p = {{num:mvp_variants.lattice.qwen3.5-2b.group_like.offsets.commutative.p:.4f}}) and associativity ({{num:mvp_variants.lattice.qwen3.5-2b.group_like.offsets.associative.consistency:.2f}}, p = {{num:mvp_variants.lattice.qwen3.5-2b.group_like.offsets.associative.p:.4f}}).
- Group-object parallelograms persist ({{num:mvp_variants.structures.qwen3.5-2b.group_objects.parallelograms.cos:.2f}}, p = {{num:mvp_variants.structures.qwen3.5-2b.group_objects.parallelograms.p:.4f}}).
- Of the shapes (§4.3), days remain a perfect PC-plane cycle, and months lose theirs.
- The chains' Robinson scores stay above their nulls before correction (p = {{num:mvp_variants.shapes.qwen3.5-2b.ring_chain.robinson_p:.3f}} and {{num:mvp_variants.shapes.qwen3.5-2b.space_chain.robinson_p:.3f}}), but the ring chain's leave-one-out order does not (p = {{num:mvp_variants.shapes.qwen3.5-2b.ring_chain.loo_order_p:.3f}}), and neither chain survives Holm correction in this run. Angles remain line-like.

- *Audit, same template variants:* the word-form controls give the same verdicts. The field map stays above the names' character 3-grams (Δρ CI [{{num:mvp_variants.lexical.qwen3.5-2b.field_map.boot.delta_vs_char3.ci.0:.2f}}, {{num:mvp_variants.lexical.qwen3.5-2b.field_map.boot.delta_vs_char3.ci.1:.2f}}]), X → commutative X is no more consistent than X → wooden X ({{num:mvp_variants.lexical.qwen3.5-2b.nonce_modifier.commutative.trained.consistency:.2f}} vs {{num:mvp_variants.lexical.qwen3.5-2b.nonce_modifier.wooden.trained.consistency:.2f}}), and the "commutative" direction still predicts abelian group − group (cos {{num:mvp_variants.lexical.qwen3.5-2b.group_like.abelian_heldout.trained.cos:.2f}}, p = {{num:mvp_variants.lexical.qwen3.5-2b.group_like.abelian_heldout.trained.p:.3f}}).

Full tables: `reports/mvp_variants_results.md`.

## 4.3 H-B: shapes of ordered families

{{table:shapes:mvp}}

![**A third principal component does not turn the angles into a circle.** Days, months and angles in their top-3 principal components at the preregistered layer. Each axis is that model's own PC, labelled with its share of variance, with true proportions and one fixed camera for every panel. Top row: Qwen3.5-2B; bottom row: its random-init twin. Below each panel: the share of variance in PC1–3 (for example {{num:mvp.views.qwen3.5-2b.angles.cum3.11:.0%}} for the angles, {{num:mvp.views.qwen3.5-2b__random.angles.cum3.11:.0%}} in the twin), and the best plane among PC1–5 where the loop closes, with its p (the closed-cycle test below): days and months close in some plane, while the angles have no significant closed cycle in any (best score {{num:mvp.posthoc.qwen3.5-2b.pc_planes_cycle.angles.11.score:.2f}}, p = {{num:mvp.posthoc.qwen3.5-2b.pc_planes_cycle.angles.11.p:.2f}}). A single camera angle can shorten or lengthen any edge; the evidence is the test, not the picture. Interactive 3-D version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:families3d_qwen3.5-2b}}){width=82%}

![**Days and months go round in their true order; angles do not.** Top-2 principal-component plane of each family at the preregistered layer (unsupervised; no labels are used to fit it). Colour runs light to dark along the true order; the grey path joins consecutive items. For angles, the orange edge is the 330° → 0° closure the preregistration predicted. Below each panel: the post-hoc PC-plane cycle score (1 = perfect cyclic order) with its order-permutation p. The score also gives 1 to an open arc: here the months' Dec → Jan step is {{num:mvp.posthoc.qwen3.5-2b.pc_planes_cycle.months.11.planes.0.3:.1f}}× a neighbour step, so in this plane they are an open arc. Bottom row: the random-init twin. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:cyclic_qwen3.5-2b}}){width=82%}

![**The cycles are present at almost every depth.** Days (top) and months (bottom) at fixed relative depths, each plane rotated onto the preregistered layer's plane for display, plus the cycle score at every layer for the trained model and the twin (right). Post-hoc and unadjusted. Animated version: `reports/figures/circle_depth_qwen3.5-2b.gif`; interactive: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:circle_depth_qwen3.5-2b}}){width=100%}

**Positive control.** With our preregistered full-space statistics, days and months show significant circle-template correlation and closure ratios near 1. They nevertheless fail "circle ρ > line ρ" at layer {{num:mvp.shapes.qwen3.5-2b.primary_layer}}; across models, line ρ exceeds circle ρ for days in {{fact:line_beats_circle.days}} and for months in {{fact:line_beats_circle.months}}. Following the prereg, we debugged before interpreting further. In the top-2 principal-component plane, where [Engels et al. (2024)](https://arxiv.org/abs/2405.14860) show their raw-activation circles (their Figs. 4 and 9; their SAE-cluster circles are in PCs 2–3, and their circular probe uses the top 5 PCs), days and months are in *perfect* cyclic order in the trained model and in none in the twin:

{{table:cycles:mvp}}

**Beyond the top-2 plane (post-hoc, pre-specified addendum).** The cycle score above counts a point's angular successor, so an open arc scores 1 too, and PCA bends lines into such arcs ([Diaconis, Goel & Holmes 2008](https://arxiv.org/abs/0811.1477)). After adding 3-D views we tested every plane (PCi, PCj), i < j ≤ 5, for a loop that also *closes* in the plane. A plane counts only if its in-plane closure ratio is below 2, the preregistered threshold. The p-value is a permutation p of the maximum over the 10 planes.
- **Pre-specification.** The rule was written down before the run ([`reports/prereg_addendum_3d.md`](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/blob/master/reports/prereg_addendum_3d.md)), with two amendments made from unit tests before any real data was run.
- **Days:** close in the top-2 plane in {{fact:closes_top2.days}}.
- **Months:** close in a lower plane (PC2–PC3 or PC4–PC5; p < .05 each, unadjusted) in {{fact:closes_lower_plane.months}}.
- **Twin:** a closed months loop also appears in {{fact:twin_closes.months}}, so the test is not fully specific.

{{table:planes:all}}

This statistic is post-hoc. We conclude that the pipeline works, and that the full-space template criterion was insensitive because variance outside the circle plane dominates. Spelling contributes to the order statistics of the positive control too: character 3-grams of the month names alone order the months (Robinson p = {{num:mvp.lexical.qwen3.5-2b.chains.months.char3.robinson_p:.4f}}; the "-ber" and "-uary" endings; Appendix A).

**Chains.** In the confirmatory model the ring and space chains are ordered beyond permutation nulls and beyond the twin (Robinson p = {{num:mvp.shapes.qwen3.5-2b.ring_chain.robinson_p:.3f}} and {{num:mvp.shapes.qwen3.5-2b.space_chain.robinson_p:.4f}}; leave-one-out order p = {{num:mvp.shapes.qwen3.5-2b.ring_chain.loo_order_p:.3f}} and {{num:mvp.shapes.qwen3.5-2b.space_chain.loo_order_p:.4f}}). Across models, the full preregistered test holds for the space chain in {{fact:chain_ordered.space_chain}} and for the ring chain in {{fact:chain_ordered.ring_chain}}; the leave-one-out order fails for the ring chain in Qwen3.5-0.8B (p = {{num:robustness.shapes.qwen3.5-0.8b.ring_chain.loo_order_p:.3f}}) and Gemma (p = {{num:gemma.shapes.gemma-4-e2b.ring_chain.loo_order_p:.3f}}). Robinson scores {{num:mvp.shapes.qwen3.5-2b.ring_chain.robinson:.2f}} and {{num:mvp.shapes.qwen3.5-2b.space_chain.robinson:.2f}}, against about 1/3 for chance and for the twin. Random concept lists are rejected at the nominal rate (by construction; §3.5):

{{table:negative_control:mvp}}

![**In 3-D, the ring and space chains run from general to special in the trained model; in the twin they collapse.** The ordered families (ring chain, space chain, number systems, n-gons, dimensions) in their top-3 principal components at the preregistered layer, with true proportions and one fixed camera; the path joins items in their true order, and its grey shadow on the floor is a depth cue. Top row: Qwen3.5-2B; bottom row: its twin. Below each panel: the share of variance in PC1–3 (ring chain {{num:mvp.views.qwen3.5-2b.ring_chain.cum3.11:.0%}}, twin {{num:mvp.views.qwen3.5-2b__random.ring_chain.cum3.11:.0%}}). The order statistics are computed in the full space and printed under the 2-D version (next figure). Interactive 3-D version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:chains3d_qwen3.5-2b}}){width=100%}

![**Class-inclusion chains are ordered paths; the twin's are tangles.** PC planes of the ordered families at the preregistered layer. Below each panel: the Robinson score (fraction of item triples consistent with the order; chance 1/3) with its permutation p, and for the ring and space chains the p of the frequency control. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:chains_qwen3.5-2b}}){width=100%}

**Frequency control (plan; added in the second audit).** Could the chain order just be a frequency or word-length order? We removed wordfreq Zipf frequency and model token count from both the representation and the ranks, fitting that regression inside each leave-one-out fold to avoid leakage ([Snoek et al., 2019](https://doi.org/10.1016/j.neuroimage.2018.09.074)), then predicted the residual order.
- *Result.* Both preregistered chains stay ordered: ring chain ρ = {{num:mvp.chain_frequency.qwen3.5-2b.ring_chain.loo_order_beyond:.2f}} (p = {{num:mvp.chain_frequency.qwen3.5-2b.ring_chain.p:.3f}}), space chain ρ = {{num:mvp.chain_frequency.qwen3.5-2b.space_chain.loo_order_beyond:.2f}} (p = {{num:mvp.chain_frequency.qwen3.5-2b.space_chain.p:.3f}}). The twin gives {{num:mvp.chain_frequency.qwen3.5-2b__random.ring_chain.loo_order_beyond:.2f}} and {{num:mvp.chain_frequency.qwen3.5-2b__random.space_chain.loo_order_beyond:.2f}}.
- *Replication (Appendix A).* Both chains stay ordered in all four Qwen models (p ≤ {{num:robustness.chain_frequency.qwen2.5-1.5b.ring_chain.p:.3f}}), but not in Gemma-4-E2B (ring p = {{num:gemma.chain_frequency.gemma-4-e2b.ring_chain.p:.2f}}, space p = {{num:gemma.chain_frequency.gemma-4-e2b.space_chain.p:.3f}}).
- *Word form carries part of the ring-chain order.* Some twins also order the ring chain beyond frequency: Qwen2.5-1.5B's twin (p = {{num:robustness.chain_frequency.qwen2.5-1.5b__random.ring_chain.p:.3f}}) and Gemma's twin (p = {{num:gemma.chain_frequency.gemma-4-e2b__random.ring_chain.p:.3f}}). A plausible lexical cue is that the middle items all end in "domain". The space chain's twins are weaker: none reaches p < .05, though the two Qwen2.5 twins come close (p = {{num:robustness.chain_frequency.qwen2.5-1.5b__random.space_chain.p:.3f}}).
- *Caveat.* The fold-wise version was adopted post-hoc. A first, whole-sample version leaked the held-out item into the training span and was degenerate in the twin. Both versions are reported in `docs/deviations.md`.

{{table:chain_frequency:mvp}}

**Within-block order and the head-noun confound (audit).** The ring chain's names end in four head nouns that form contiguous blocks of its order: rng | ring, commutative ring | six "… domain" items | field, algebraically closed field. Any representation that groups items by head noun therefore scores above chance on both preregistered order statistics, even with no order between or within the groups. In a simulation with equidistant head-noun clusters and no order, the preregistered conjunction rejected in {{num:mvp.lexical.block_only_simulation.0.reject_rate:.0%}}, {{num:mvp.lexical.block_only_simulation.1.reject_rate:.0%}} and {{num:mvp.lexical.block_only_simulation.2.reject_rate:.0%}} of simulations at three cluster strengths. Within the six "… domain" items, where the head noun is constant, the order is significant in {{fact:ring_within}} (exact test over all 720 orders). The space chain, whose items all end in "space", is ordered within its separation-axiom block (T0 … normal) in {{fact:space_block.0}} and within its metric block (metrizable … Hilbert) in {{fact:space_block.1}}, and its spelling does not order it (Appendix A). We therefore count the space chain as ordered, and the ring chain's order as not established beyond its head nouns.

{{table:blocks:all}}

{{table:block_simulation:mvp}}

**Test precision and the ridge penalty (audit).** The leave-one-out order nulls have 10,000 permutations (300 before the audit). Together with the 8-test Holm family (§3.5), this changes three decisions: Gemma-4-E2B's space chain becomes Holm-significant, and identity and divisibility lose Holm significance in Qwen2.5-Math-1.5B. The preregistered ridge penalty α = 10 is not scale-invariant: the residual streams of the Qwen3.5 models have much smaller norms than those of Qwen2.5 and Gemma, so their leave-one-out fits shrink toward the training mean, whose ranks are anti-correlated with the held-out ranks. This is why leave-one-out ρ is strongly negative for many families in those models and their twins; these values are not effect sizes. A scale-relative penalty is reported in Appendix A as a sensitivity analysis.

**Angles: preregistered prediction falsified.** Angles are not circular: closure ratio {{num:mvp.shapes.qwen3.5-2b.angles.closure:.2f}} (range {{fact:angles_closure_range}} across models), no cycle in the top-2 PC plane, and, under the pre-specified addendum rule, no significant closed cycle in any plane of the top five components in any model (below). Their geometry closely matches that of the bare digits 0–330 (ρ = {{num:mvp.shapes.qwen3.5-2b.angles.beyond_reference.rho_fam_ref:.2f}}). At this scale, "330 degrees" is farther from "0 degrees" than neighbouring angles are from each other.

**The preregistered rule is not specific.** The circle rule is *met* by {{fact:twins_meet_angles_rule}}, for example the Qwen3.5-2B twin: closure {{num:mvp.shapes.qwen3.5-2b__random.angles.closure:.2f}}, circle ρ {{num:mvp.shapes.qwen3.5-2b__random.angles.circle_rho:.2f}} vs line ρ {{num:mvp.shapes.qwen3.5-2b__random.angles.line_rho:.2f}}. A structureless point cloud has a closure ratio near 1 and near-zero template correlations, so it passes by chance. The informative result is therefore not the failed rule itself but the trained models' clear line-like order: line ρ {{num:mvp.shapes.qwen3.5-2b.angles.line_rho:.2f}} in Qwen3.5-2B, against circle ρ {{num:mvp.shapes.qwen3.5-2b.angles.circle_rho:.2f}}. That order tracks the bare numbers. We found this when we added the twin panel to the angles figure; it was not part of the preregistered analysis.

{{table:beyond_reference:mvp}}

The addendum's decision rule for "angles have a circular component beyond PC1–2" requires all four conditions:

{{table:planes_decision:all}}

![**Predicted vs found for angles.** Left: the preregistered prediction, a circle on which 330° neighbours 0° (orange edge). Middle: Qwen3.5-2B's PC plane, where that edge spans the whole layout. Right: the bare numbers 0–330, whose geometry the angles follow. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:angles_qwen3.5-2b}}){width=90%}

**SMDS (plan, secondary).** Supervised MDS fits a projection to a line or a discrete-circle template, and each fit is tested against refits on shuffled labels.

{{table:smds:mvp}}

- **The twin shows why the shuffle null is necessary.** Its line fit scores {{num:mvp.smds.qwen3.5-2b__random.days.euclidean.score:.2f}} on *every* family, a perfect fit to any ordering, yet never beats its shuffled null.
- **Trained model:**
  - Days prefer the circle template: circle p = {{num:mvp.smds.qwen3.5-2b.days.discrete_circular.p:.3f}}, line p = {{num:mvp.smds.qwen3.5-2b.days.euclidean.p:.3f}}.
  - The chains and angles prefer the line template.
  - Months fit both templates beyond the null.

This agrees with the unsupervised statistics, including the conclusion that angles are line-like.

## 4.4 H-B2: axiom lattices

**The test.** The 13 group-like structures are defined by four axioms (associativity, identity, divisibility, commutativity), so each sits on a vertex of a 4-cube (§3.3). For each axiom we take every pair of structures that differ only in that axiom, for example monoid → commutative monoid and group → abelian group for commutativity ({{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.n_pairs}} pairs), and compute their difference vectors at the preregistered layer. *Offset consistency* is the mean pairwise cosine of these difference vectors in the full residual stream: 1 if they are parallel, about 0 if they are unrelated. The null draws the same number of random pairs from the same 13 structures (2,000 draws), and p = (1 + the number of null draws at least as large as the observed value)/(1 + 2,000). The next figure walks through the procedure on real data.

![**Adding an axiom moves a structure in a consistent direction in the trained model, and not in its random-init twin; the word-form controls below show that the names' spelling produces such bundles too.** The test on Qwen3.5-2B at the preregistered layer, in four steps. (1) Each structure is a set of axioms, so the 13 group-like structures form a lattice (ground truth). (2) Every pair of structures that differ only in commutativity, drawn as an arrow in the model's top-3 principal components. (3) The same arrows moved to one common origin: in the model they form a bundle (offset consistency {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.consistency:.2f}}), in the twin they point every way ({{num:mvp.lattice.qwen3.5-2b__random.group_like.offsets.commutative.consistency:.2f}}). (4) A compass for every axiom, measured in the full residual stream: the angle between each same-axiom difference vector and the mean of the others (leave-one-out); the model is drawn above the dashed line, the twin below; 0° means the same direction and 90° unrelated. Steps 2 and 3 are pictures; the statistic and the compass are computed in the full space, and the compass is a visual aid rather than a test. Interactive version, step by step: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-axiom-how>.]({{fig:axioms_explainer_qwen3.5-2b}}){width=100%}

{{table:lattice:mvp}}

![**In 3-D, the trained model's same-axiom arrows stay roughly parallel; the twin's point every way.** The 13 structures in their top-3 principal components ({{num:mvp.views.qwen3.5-2b.group_like.cum3.11:.0%}} of their variance; twin {{num:mvp.views.qwen3.5-2b__random.group_like.cum3.11:.0%}}), with true proportions and one fixed camera. In each facet, the arrows (blue; grey in the twin) are the pairs that differ only in that axiom. Top row: Qwen3.5-2B; bottom row: its twin. A 3-D view can still hide or exaggerate parallelism; the cosine matrices below are the evidence. Interactive 3-D version, with the analogy query b − a + c placed exactly: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-lattice>.]({{fig:lattice3d_qwen3.5-2b}}){width=100%}

![**The same facets in the top-2 plane.** Left: the ideal geometry, in which each axiom is one fixed step. Other panels: the same unsupervised PCA plane of the 13 group-like structures, one panel per axiom, with the pairs that differ only in that axiom drawn as arrows (blue; grey in the twin). Parallel arrows mean a consistent direction. Below each panel: offset consistency and its random-pair p. Bottom row: the twin. Interactive version, with an analogy playground: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-lattice>.]({{fig:lattice_axes_qwen3.5-2b}}){width=100%}

![**The direction evidence, unprojected.** Cosine between every pair of single-axiom difference vectors in the full residual space, grouped by axiom. Red diagonal blocks mean a consistent direction. Hatched cells are cross-axiom pairs that share a structure; isotropic noise alone puts these near ±0.5, so they carry no evidence. The self-cosines on the diagonal are masked. Right: offset consistency per axiom for the trained model, the trained model with the token-count direction removed, the twin and the names' character 3-grams (×), against the random-pair null. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-cosines>.]({{fig:lattice_cosines_qwen3.5-2b}}){width=100%}

For every group-like axiom, difference vectors between structures that differ only in that axiom are more parallel than random pairs (raw p < .05):

| Axiom | Consistency | Null 95th percentile | p |
|---|---|---|---|
| Associativity | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.associative.consistency:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.associative.null95:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.associative.p:.4f}} |
| Identity | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.identity.consistency:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.identity.null95:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.identity.p:.4f}} |
| Divisibility | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.divisible.consistency:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.divisible.null95:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.divisible.p:.4f}} |
| Commutativity | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.consistency:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.null95:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.p:.4f}} |

In this model's random-init twin, all four consistencies are ≈ 0.

**Analogies.** Analogy completion is top-1 for {{num:mvp.lattice.qwen3.5-2b.group_like.analogy.top1:.2f}} of {{num:mvp.lattice.qwen3.5-2b.group_like.analogy.n}} parallelograms (chance {{num:mvp.lattice.qwen3.5-2b.group_like.analogy.chance_top1:.2f}}). The twin reaches {{num:mvp.lattice.qwen3.5-2b__random.group_like.analogy.top1:.2f}} and the names' character 3-grams {{num:mvp.lexical.qwen3.5-2b.group_like.analogy_top1.char3:.2f}}, so much of the analogy accuracy is lexical. So is much of the offset consistency (word-form controls below).

**Caveats.** The gender-offset positive control passes. Multiplicity across the preregistered confirmatory tests is handled by Holm correction:

{{table:holm:mvp}}

- **After correction:** associativity and commutativity remain significant; identity and divisibility do not.
- **Across the replication models (Appendix A):** all four offsets are positive in every model. They are near zero in the Qwen3.5 twins. The other twins are weaker but not always null: raw p < .05 for commutativity in the two Qwen2.5 twins (consistency ≈ {{num:robustness.lattice.qwen2.5-1.5b__random.group_like.offsets.commutative.consistency:.2f}}) and for associativity and divisibility in Gemma's twin, all lexical. Divisibility reaches raw p < .05 in {{fact:axiom_raw.divisible}}, identity in {{fact:axiom_raw.identity}}.
**Word-form controls (audit).** The names encode the axioms. Five of the six commutativity pairs add the word "commutative". Most associativity and divisibility pairs end in "-group" on one side only (magma → semigroup, loop → group; magma → quasigroup, monoid → group). Several pairs of pairs repeat the same head-noun change with and without "commutative". A representation that knew only the spelling would therefore show consistent offsets, and it does: character 3-grams of the names pass the same random-pair test for every axiom (associativity {{num:mvp.lexical.qwen3.5-2b.group_like.offsets.char3.associative.consistency:.2f}}, identity {{num:mvp.lexical.qwen3.5-2b.group_like.offsets.char3.identity.consistency:.2f}}, divisibility {{num:mvp.lexical.qwen3.5-2b.group_like.offsets.char3.divisible.consistency:.2f}}, commutativity {{num:mvp.lexical.qwen3.5-2b.group_like.offsets.char3.commutative.consistency:.2f}}; p ≤ {{num:mvp.lexical.qwen3.5-2b.group_like.offsets.char3.divisible.p:.3f}}).

{{table:lexical_offsets:all}}

- *Commutativity is largely composition with a modifier.* For six heads, X → wooden X and X → zorbic X are about as consistent as X → commutative X; "commutative" is the most consistent of the three modifiers in {{fact:nonce_above}}. Plain composition with any modifier gives offsets of this size.

{{table:nonce:all}}

- *But the commutative direction carries meaning.* The direction of the five pairs that add the word "commutative" predicts *abelian group − group*, which shares no word with it, in {{fact:abelian_transfer}} (cosine {{fact:abelian_range}}). Character 3-grams and a bag of words give about 0, and the token-embedding bag stays below the trained models. This is one held-out pair per model: an existence proof, not a systematic result.

{{table:abelian:all}}

- *Identity is lexical duplication.* Identity's two duplicated pairs (quasigroup → loop vs commutative quasigroup → commutative loop; semigroup → monoid vs commutative semigroup → commutative monoid) have cosines {{fact:dup_range.identity}}; its eight disjoint cosines average {{fact:disjoint_range.identity}}. On the 8 names without "commutative" its consistency is {{fact:sublattice_range.identity}}.
- *Associativity goes beyond word form most clearly.* It exceeds every word-form baseline and its disjoint pairs pass the null in {{fact:axiom_beyond_lexical.associative}} (divisibility: {{fact:axiom_beyond_lexical.divisible}}; identity: {{fact:axiom_beyond_lexical.identity}}; commutativity: {{fact:axiom_beyond_lexical.commutative}}). On the 8-name sub-lattice it is {{fact:sublattice_range.associative}}, above every baseline in {{fact:sublattice_beyond.associative}}, but with three pairs per model no single value is significant.

{{table:lexical_sublattice:all}}

- **Our reading:** most of the offset geometry follows from the names. The evidence for axioms as directions beyond word form is suggestive: an associativity direction, and a commutativity direction that generalizes to "abelian group".

**Lie groups and other group objects (exploratory).** A Lie group is a group that is also a smooth manifold; topological and algebraic groups are defined analogously. We tested three parallelograms:
- topological group − group ≈ topological space − set;
- Lie group − topological group ≈ smooth manifold − topological space;
- algebraic group − group ≈ algebraic variety − set.

{{table:structures:mvp}}

{{table:structures:robustness}}

{{table:structures:gemma}}

![**The group-object parallelograms clear their null, but the names' spelling is more parallel.** Left: mean cosine of the three parallelograms (for example, Lie group − topological group vs smooth manifold − topological space) for each model (blue), its twin (grey) and the names' character 3-grams (×), against the 95th percentile of each model's random-quadruple null (dark tick) and of the twin's own null (small grey tick); p values from those nulls. Right: the nearest-neighbour rank of "group" (circle) and "smooth manifold" (diamond) among the family items around "Lie group", trained vs twin (log scale). Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-groups>.]({{fig:group_objects}}){width=88%}

**Group objects: explained by shared words.**
- The parallelograms clear their null in {{fact:parallelograms}}: mean cosine {{num:mvp.structures.qwen3.5-2b.group_objects.parallelograms.cos:.2f}} in Qwen3.5-2B (p = {{num:mvp.structures.qwen3.5-2b.group_objects.parallelograms.p:.4f}}). The twins are near zero, except the Qwen2.5 twins, whose weak lexical signal also passes their own nulls (p = {{num:robustness.structures.qwen2.5-1.5b__random.group_objects.parallelograms.p:.3f}} and {{num:robustness.structures.qwen2.5-math-1.5b__random.group_objects.parallelograms.p:.3f}}).
- Trained models place "Lie group" close to *both* parents: "group" and "smooth manifold" rank {{num:mvp.structures.qwen3.5-2b.group_objects.parent_ranks.Lie group.0}} and {{num:mvp.structures.qwen3.5-2b.group_objects.parent_ranks.Lie group.1}} among {{num:mvp.structures.qwen3.5-2b.group_objects.pool_size}} items.
- The twins rank "group" first or second (Gemma's twin: second) because of the shared token. "Smooth manifold" ranks {{num:mvp.structures.qwen3.5-2b__random.group_objects.parent_ranks.Lie group.1}} in the Qwen3.5-2B twin; Gemma's twin is the exception, ranking it {{num:gemma.structures.gemma-4-e2b__random.group_objects.parent_ranks.Lie group.1}}.
- *Audit.* Every parallelogram shares "topological" or "algebraic" on both sides, and the names' character 3-grams give a mean cosine of {{num:mvp.lexical.qwen3.5-2b.group_objects.char3.cos:.2f}} and the token-embedding bag {{num:mvp.lexical.qwen3.5-2b.group_objects.layer0_bag.cos:.2f}}, above Qwen3.5-2B's {{num:mvp.structures.qwen3.5-2b.group_objects.parallelograms.cos:.2f}}. Models whose cosine exceeds every word-form baseline: {{fact:group_objects_beyond}}. The parallelograms therefore show composition of the names, not knowledge of group objects.

{{table:lexical_group_objects:all}}

**Axiom transfer: inconclusive.**
- The commutativity direction learned on group-like pairs transfers only weakly to ring → commutative ring and division ring → field ({{num:mvp.structures.qwen3.5-2b.transfer.commutative.cos:.2f}}, p = {{num:mvp.structures.qwen3.5-2b.transfer.commutative.p:.3f}}).
- Some Qwen twins show similar values. "Commutative ring" literally contains the token "commutative", so we consider transfer not established.
- Identity transfer, from the single pair rng → ring, is null.

**Further lattice items from the plan (exploratory; second audit).**

{{table:lattice_probes:mvp}}

{{table:lattice_variants:mvp}}

- *No semantic collapse.* A non-empty associative quasigroup is a group, but the model does not place it there. It is {{num:mvp.lattice.qwen3.5-2b.audit2.associative_quasigroup.ratio:.1f}}× farther from "group" than from "quasigroup" (twin {{num:mvp.lattice.qwen3.5-2b__random.audit2.associative_quasigroup.ratio:.1f}}×), and its nearest neighbours are quasigroup-like. The representation follows the name's parts, not this theorem.
- *Inverse semigroup* sits next to semigroup, commutative semigroup and monoid in the trained model, which is sensible.
- *Not a length artifact.* Removing the token-count direction (fitted on all family items) leaves the group-like offsets intact: commutativity {{num:mvp.lattice.qwen3.5-2b.audit2.group_like_offsets.length_projected.commutative.consistency:.2f}} (p = {{num:mvp.lattice.qwen3.5-2b.audit2.group_like_offsets.length_projected.commutative.p:.4f}}), associativity {{num:mvp.lattice.qwen3.5-2b.audit2.group_like_offsets.length_projected.associative.consistency:.2f}} (p = {{num:mvp.lattice.qwen3.5-2b.audit2.group_like_offsets.length_projected.associative.p:.4f}}). Across the five models (Appendix A), after this projection the offsets stay significant before correction for associativity in {{fact:axiom_length_projected.associative}}, commutativity in {{fact:axiom_length_projected.commutative}}, divisibility in {{fact:axiom_length_projected.divisible}} and identity in {{fact:axiom_length_projected.identity}}. Mean-pooling the concept's tokens gives a similar pattern, with divisibility no longer significant (table above). This is a one-direction analogue of the LDA distractor removal of [Li et al. (2024)](https://arxiv.org/abs/2410.19750).
- *Partial-operation lattice: no evidence.* The "total operation" offset (semigroupoid → semigroup, small category → monoid, groupoid → group) is {{num:mvp.lattice.qwen3.5-2b.audit2.partial_operation.offsets.total.consistency:.2f}}, below its null's 95th percentile ({{num:mvp.lattice.qwen3.5-2b.audit2.partial_operation.offsets.total.null95:.2f}}). With 6 items and 2–3 pairs per axis, the test is weak.
- *Identity transfer to lattice-like pairs* (semilattice → bounded semilattice, lattice → bounded lattice) is null too: cos = {{num:mvp.structures.qwen3.5-2b.transfer.identity_lattice.cos:.2f}}, p = {{num:mvp.structures.qwen3.5-2b.transfer.identity_lattice.p:.2f}}.

![**A negative result: no consistent "make the operation total" direction.** Offset consistency for the partial-operation lattice (semigroupoid, small category, groupoid vs semigroup, monoid, group): the confirmatory model (filled), the other four models (open) and the five twins (grey), against the confirmatory model's random-pair null (grey bar: mean to 95th percentile). With 6 items and 2–3 pairs per axis the test has little power. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-partial>.]({{fig:partial_operation}}){width=70%}

## 4.5 Secondary analyses

### H-A4: concrete ↔ abstract and pure ↔ applied are the main axes

**Ratings.** Three judge LLMs rated every math concept's abstraction from 1 to 7: IBM Granite-4.0-micro, SmolLM3-3B and Phi-4-mini-instruct, none from a model family we probe. We read each rating as the expectation over next-token digit probabilities.
- Rank agreement is moderate (mean pairwise Spearman {{num:mvp.judges.mean_pairwise_spearman:.2f}}).
- Absolute agreement is low ([Shrout & Fleiss, 1979](https://doi.org/10.1037/0033-2909.86.2.420); ICC(2,1) = {{num:mvp.judges.icc2_1:.2f}}; ICC(2,k) = {{num:mvp.judges.icc2_k:.2f}}), because the judges use the scale with different offsets.
- The ratings have face validity. The least abstract concepts include "trivial group", "bar chart" and "dot plot"; the most abstract include étale cohomology, Artin stacks and generalized cohomology theories.

**Validation against Mathlib (plan; specified before computing, `docs/deviations.md`).** As an external, non-LLM measure, we matched concepts to declarations in [Mathlib4](https://github.com/leanprover-community/mathlib4) (commit `{{num:mvp.mathlib.mathlib_rev}}`).
- *Matching.* First through Mathlib's curated English-name maps (`docs/overview.yaml`, `docs/undergrad.yaml`), then through exact CamelCase names for multi-word concepts ("compact space" → `CompactSpace`).
- *Depth.* The longest chain of Mathlib imports below the defining module ({{num:mvp.mathlib.n_modules}} modules).
- *Result.* The mean judge rating correlates with depth: ρ = {{num:mvp.mathlib.rho_mean_rating_depth:.2f}}, 95% CI [{{num:mvp.mathlib.ci95.0:.2f}}, {{num:mvp.mathlib.ci95.1:.2f}}], meeting the pre-specified criterion.
- *Caveat.* Only {{num:mvp.mathlib.n_matched}} of {{num:mvp.mathlib.n_math_concepts}} concepts could be matched, and coverage is uneven across fields (table below). The judges also differ: Phi-4-mini's ratings track depth (ρ = {{num:mvp.mathlib.rho_per_judge.microsoft/Phi-4-mini-instruct:.2f}}), SmolLM3's barely (ρ = {{num:mvp.mathlib.rho_per_judge.HuggingFaceTB/SmolLM3-3B:.2f}}).

{{table:mathlib:mvp}}

![**Concrete ↔ abstract is the top axis of the math-concept cloud.** Left and middle: PC1 × PC2 of the {{num:mvp.map_fields.qwen3.5-2b.math_primary.n}} primary concepts, both residualized on frequency and token count exactly as in the statistic. Left is coloured by judge abstraction, with the most and least abstract concepts labelled; middle is coloured pure/applied. The PC signs are flipped for display so that abstract and applied point right and up. Right: mean judge rating vs Mathlib import depth for the matched concepts. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-abstraction>.]({{fig:abstraction_qwen3.5-2b}}){width=100%}

**Axes.** We take the principal components of the centred math-concept cloud and residualize both the PC scores and the labels on word frequency and token count. The top principal component then correlates with abstraction (ρ = {{num:mvp.axes.qwen3.5-2b.pcs.0.rho_abstraction:.2f}}), and the second with the pure/applied label (ρ = {{num:mvp.axes.qwen3.5-2b.pcs.1.rho_applied:.2f}}). In the twin these correlations are {{num:mvp.axes.qwen3.5-2b__random.pcs.0.rho_abstraction:.2f}} and {{num:mvp.axes.qwen3.5-2b__random.pcs.1.rho_applied:.2f}}. A 5-fold cross-validated linear readout predicts abstraction with ρ = {{num:mvp.axes.qwen3.5-2b.cv_ridge_rho_abstraction:.2f}}, against {{num:mvp.axes.qwen3.5-2b__random.cv_ridge_rho_abstraction:.2f}} in the twin. We also partialled out pretraining-scale frequency: [infini-gram](https://infini-gram.io/api_doc) counts in Dolma v1.7 ([Liu et al., 2024](https://arxiv.org/abs/2401.17377)), which correlate ρ = {{num:mvp.baselines.infinigram.rho_zipf_vs_log_count:.2f}} with wordfreq. The axes are essentially unchanged: PC1 ↔ abstraction ρ = {{num:mvp.axes.qwen3.5-2b.pcs_with_infinigram.0.rho_abstraction:.2f}}, PC2 ↔ applied ρ = {{num:mvp.axes.qwen3.5-2b.pcs_with_infinigram.1.rho_applied:.2f}}.

**Interpretation.** In this model, concrete ↔ abstract is the single largest axis of variation among mathematical concepts. That is a correlational, model-rated finding: abstraction ratings come from LLM judges, not humans.

### Causal test of chain geometry: inconclusive

The preregistered gate requires baseline accuracy ≥ 0.70 on "which is more general, a A or a B?". The noise subspace is fitted on the other chain items, not on the tested pair, and compared with random subspaces of the same size, because a subspace can look causally relevant by accident ([Makelov et al., 2023](https://arxiv.org/abs/2311.17030)).
- **Ring chain:** it failed the gate (accuracy {{num:mvp.causal.qwen3.5-2b.ring_chain.baseline_acc:.3f}}), so no intervention was run.
- **Space chain:** it passed ({{num:mvp.causal.qwen3.5-2b.space_chain.baseline_acc:.3f}}).

At noise scale s = 4, noise injected into the chain's 2-D subspace lowered accuracy to {{num:mvp.causal.qwen3.5-2b.space_chain.scales.4.0.acc_chain_subspace:.3f}}. Noise in random 2-D subspaces gave a mean of {{num:mvp.causal.qwen3.5-2b.space_chain.scales.4.0.acc_random_subspace_mean:.3f}}. At s = 2 the values were {{num:mvp.causal.qwen3.5-2b.space_chain.scales.2.0.acc_chain_subspace:.3f}} and {{num:mvp.causal.qwen3.5-2b.space_chain.scales.2.0.acc_random_subspace_mean:.3f}}; at s = 1, chain-subspace noise hurt *less* than random noise ({{num:mvp.causal.qwen3.5-2b.space_chain.scales.1.0.acc_chain_subspace:.3f}} vs {{num:mvp.causal.qwen3.5-2b.space_chain.scales.1.0.acc_random_subspace_mean:.3f}}). At the two larger scales the direction matches the prediction, but the effect is small and five random draws cannot establish significance. We report it as inconclusive.

### SAE features (exploratory)

At block {{num:mvp.sae.qwen3.5-2b.sae_block}}, we encoded the template-averaged residuals with the official Qwen-Scope TopK SAE (k = 50).
- **Graded map:** the SAE code preserves the field map only weakly (ρ = {{num:mvp.sae.qwen3.5-2b.math.rsa_sae:.2f}}, against {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.2f}} for the dense residual).
- **Clustering:** field clustering is largely preserved (LOO accuracy {{num:mvp.sae.qwen3.5-2b.math.loo_acc_sae:.2f}}).
- **Field-selective features exist.** For example, a game-theory feature fires on most game-theory concepts and almost no others (`results/mvp/sae.json`).
- **Feature sets:** {{num:mvp.sae.qwen3.5-2b.features_math_only}} features fire only on math concepts, {{num:mvp.sae.qwen3.5-2b.features_control_only}} only on control concepts, and {{num:mvp.sae.qwen3.5-2b.features_shared}} on both. The sample sizes are unequal, so this is only suggestive of the "math lobe" of [Li et al. (2024)](https://arxiv.org/abs/2410.19750).


# 5 Discussion

**Axiom offsets.** The offsets of group-like structures behave like the man → woman offset of word embeddings ([Mikolov, Yih & Zweig, 2013](https://aclanthology.org/N13-1090/)), and like those offsets they mix structure with word form ([Linzen, 2016](https://aclanthology.org/W16-2503/)). With additive composition, "modifier + noun" is a fixed vector by construction ([Mitchell & Lapata, 2010](https://doi.org/10.1111/j.1551-6709.2010.01106.x)), and the models compose "wooden" and "zorbic" about as consistently as "commutative". What survives the word-form controls is narrower: an associativity direction above every spelling baseline, and a commutativity direction that also points from "group" to "abelian group". The geometry follows the names more than the definitions: "associative quasigroup" stays with the quasigroups, although a non-empty one is a group. Whether the model uses these directions in reasoning, for example to decide that a monoid with inverses is a group, needs causal tests, which we did not run for the axioms (§6).

**The field map.** Relatedness between fields is encoded at the level of field centroids, in the dominant principal components, from the first few layers onward in the Qwen models (Gemma's map is at or below the max-over-layers null in {{fact:below_null_runs.gemma-4-e2b}}). It is similarly strong (ρ ≈ 0.4) in models with very different pretraining mixtures. Combined with the null effect of math continued pretraining, this suggests the map is learned from general co-occurrence statistics, as [Karkada et al. (2026)](https://arxiv.org/abs/2602.15029) predict. It does not seem to require math-specific training, and it does not follow from the spelling of the names. Whether it is specific to mathematics remains open: our control fields are whole sciences, not sub-fields, so the matched clustering comparison cannot decide it.

**Controls for concept geometry.** A random-init twin controls for architecture and tokenization, but not for word form: it scrambles even exact word composition, so a trained model can "beat the twin" with geometry that any spelling-based representation would show. Spelling-only baselines and nonce controls are cheap and should accompany studies of concept geometry, especially for multi-word names that encode their own structure, as mathematical names often do.

**Angles.** "330 degrees" is not near "0 degrees". This mirrors the finding of [Kantamneni & Tegmark (2025)](https://arxiv.org/abs/2502.00873) that numbers lie on a helix whose dominant axis is linear. A model this size appears to encode degrees by their numerals rather than by their angular meaning.

# 6 Limitations

- **Concept sample.** Glossary terms are a convenience sample, and some junk survives cleaning. Field labels come from one glossary per term. The eponym flag missed five eponyms in the primary set (`docs/deviations.md`).
- **Ground truth.** Ochiai cross-listing is one operationalization of field relatedness. The space chain is a chain only if "regular" and "normal" include T1 (T3 and T4, as in Munkres); the prompts use the plain words.
- **Read point.** A single read-out: last token, averaged over templates. Template wording affects absolute values, though split-half reliability is high.
- **Model size.** The Qwen models have at most 1.9B text parameters and Gemma-4-E2B 2.3B effective parameters, because of the 8 GB GPU.
- **Positive control.** It passed only under post-hoc statistics. In the plane where its loop closes, it holds for days in {{fact:closes_top2.days}} and for months in {{fact:closes_lower_plane.months}}.
- **Family size.** Families are small (n = 5–13), so shape claims rest on permutation tests with limited resolution.
- **Word form.** Our preregistered control, the random-init twin, did not control the spelling of names; the audit's word-form baselines do, post-hoc. Lexical cues beyond character 3-grams, bags of words and context-free token embeddings are not ruled out.
- **Twin seeds.** Five random draws for the confirmatory model's twin; one for each other model.
- **Specificity.** The non-math control fields are whole sciences; a granularity-matched control was not run.
- **Ridge penalty.** The preregistered α = 10 is not scale-invariant, so the leave-one-out order test has different power in different models (§4.3).
- **Correlation, not mechanism.** Every test except the noise-injection test on the chains (§4.5) is correlational, including the axiom offsets (§4.4).
- **Abstraction ground truth.** Abstraction ratings come from LLM judges. Their validation against Mathlib import depth rests on only {{num:mvp.mathlib.n_matched}} of {{num:mvp.mathlib.n_math_concepts}} concepts that could be matched to Mathlib, and import depth also differs systematically between areas of Mathlib.
- **Not run.** Three planned items were not run; reasons are in `docs/deviations.md`:
  - Qwen3.5-4B in 8-bit, whose random-init twin cannot be held in 8 GB;
  - the optional OctoThinker pair;
  - persistent homology, since no family has ≥30 points.
- **Baseline scope.** The fastText and Qwen3-Embedding baselines, SMDS and the template variants were run for the confirmatory model only.

# 7 Conclusion

Field-level geometry in LLM residual streams tracks how mathematicians cross-list their papers, in five models, beyond random-init twins and beyond the spelling of the concept names; math continued pretraining does not sharpen it. The space chain T0 ⊃ … ⊃ Hilbert space is an ordered path, also within its topical blocks, while the ring chain's order is not established beyond its head nouns. Axioms of group-like structures give consistent offsets, but mostly because the names encode them; an associativity direction and a commutativity direction that reaches "abelian group" are suggestive exceptions. Angles are not circles. A random-init twin does not control for word form, so studies of concept geometry need spelling-only baselines.

# References

Every source cited in the text, with the identifier we checked (arXiv API or Crossref, 2026-10-30):

- Anderson, M. J. & Legendre, P. (1999). An empirical comparison of permutation methods for tests of partial regression coefficients in a linear model. *J. Stat. Comput. Simul.* 62. [doi:10.1080/00949659908811936](https://doi.org/10.1080/00949659908811936)
- Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics* 24:295–300 (intersection-union test, used for the chain conjunction). [doi:10.1080/00401706.1982.10487790](https://doi.org/10.1080/00401706.1982.10487790)
- Deng, B. et al. (2026). Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models. [arXiv:2605.11887](https://arxiv.org/abs/2605.11887)
- Diaconis, P., Goel, S. & Holmes, S. (2008). Horseshoes in multidimensional scaling and local kernel methods. *Ann. Appl. Stat.* 2(3). [arXiv:0811.1477](https://arxiv.org/abs/0811.1477)
- Engels, J. et al. (2024). Not All Language Model Features Are (One-Dimensionally) Linear. [arXiv:2405.14860](https://arxiv.org/abs/2405.14860)
- Efron, B. (1987). Better bootstrap confidence intervals. *J. Am. Stat. Assoc.* 82(397):171–185 (BCa intervals). [doi:10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410)
- Freedman, D. & Lane, D. (1983). A nonstochastic interpretation of reported significance levels. *J. Bus. Econ. Stat.* 1:292–298. [doi:10.1080/07350015.1983.10509354](https://doi.org/10.1080/07350015.1983.10509354)
- Golechha, S., Bushnaq, L., Ong, E., Kayal, N. & Schoots, N. (2025). Intricacies of Feature Geometry in Large Language Models. ICLR 2025 blog track. [iclr.cc/virtual/2025/poster/31349](https://iclr.cc/virtual/2025/poster/31349)
- Guillot, G. & Rousset, F. (2013). Dismantling the Mantel tests. *Methods Ecol. Evol.* 4. [doi:10.1111/2041-210X.12018](https://doi.org/10.1111/2041-210X.12018)
- Gurnee, W. & Tegmark, M. (2023). Language Models Represent Space and Time. [arXiv:2310.02207](https://arxiv.org/abs/2310.02207)
- Kantamneni, S. & Tegmark, M. (2025). Language Models Use Trigonometry to Do Addition. [arXiv:2502.00873](https://arxiv.org/abs/2502.00873)
- Karkada, D., Korchinski, D. J., Nava, A., Wyart, M. & Bahri, Y. (2026). Symmetry in language statistics shapes the geometry of model representations. [arXiv:2602.15029](https://arxiv.org/abs/2602.15029)
- Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I. & Lempitsky, V. (2019). Hyperbolic Image Embeddings (relative Gromov δ). [arXiv:1904.02239](https://arxiv.org/abs/1904.02239)
- Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. *Psychometrika* 29:1–27 (stress-1). [doi:10.1007/BF02289565](https://doi.org/10.1007/BF02289565)
- Kvinge, H. et al. (2025). Probing the Limits of Mathematical World Models in LLMs. ICML 2025 workshop *Assessing World Models*. [icml.cc/virtual/2025/49128](https://icml.cc/virtual/2025/49128)
- Lei, G. & Cooper, S. J. (2025). Layerwise Recall and the Geometry of Interwoven Knowledge in LLMs. [arXiv:2502.10871](https://arxiv.org/abs/2502.10871)
- Li, Y., Michaud, E. J., Baek, D. D., Engels, J., Sun, X. & Tegmark, M. (2024). The Geometry of Concepts: Sparse Autoencoder Feature Structure. [arXiv:2410.19750](https://arxiv.org/abs/2410.19750)
- Linzen, T. (2016). Issues in evaluating semantic spaces using word analogies. *Proc. 1st Workshop on Evaluating Vector-Space Representations for NLP (RepEval)*, 13–18. [aclanthology.org/W16-2503](https://aclanthology.org/W16-2503/)
- Liu, J., Min, S., Zettlemoyer, L., Choi, Y. & Hajishirzi, H. (2024). Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens. [arXiv:2401.17377](https://arxiv.org/abs/2401.17377)
- Makelov, A., Lange, G. & Nanda, N. (2023). Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching. [arXiv:2311.17030](https://arxiv.org/abs/2311.17030)
- Mathematics Subject Classification 2020 (MSC2020). <https://msc2020.org/>
- Mikolov, T., Yih, W. & Zweig, G. (2013). Linguistic Regularities in Continuous Space Word Representations. NAACL-HLT 2013. [aclanthology.org/N13-1090](https://aclanthology.org/N13-1090/)
- Mikolov, T., Grave, E., Bojanowski, P., Puhrsch, C. & Joulin, A. (2018). Advances in Pre-Training Distributed Word Representations (fastText vectors). [arXiv:1712.09405](https://arxiv.org/abs/1712.09405)
- Mitchell, J. & Lapata, M. (2010). Composition in distributional models of semantics. *Cognitive Science* 34(8):1388–1429. [doi:10.1111/j.1551-6709.2010.01106.x](https://doi.org/10.1111/j.1551-6709.2010.01106.x)
- Modell, A., Rubin-Delanchy, P. & Whiteley, N. (2025). The Origins of Representation Manifolds in Large Language Models. [arXiv:2505.18235](https://arxiv.org/abs/2505.18235)
- Mu, J. & Viswanath, P. (2018). All-but-the-Top: Simple and Effective Postprocessing for Word Representations. [arXiv:1702.01417](https://arxiv.org/abs/1702.01417)
- Munkres, J. R. (2000). *Topology*, 2nd ed., §31–32 (regular and normal spaces assume one-point sets closed).
- Park, K., Choe, Y. J., Jiang, Y. & Veitch, V. (2024). The Geometry of Categorical and Hierarchical Concepts in Large Language Models. [arXiv:2406.01506](https://arxiv.org/abs/2406.01506)
- Shrout, P. E. & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater reliability. *Psychol. Bull.* 86:420–428. [doi:10.1037/0033-2909.86.2.420](https://doi.org/10.1037/0033-2909.86.2.420)
- Snoek, L., Miletić, S. & Scholte, H. S. (2019). How to control for confounds in decoding analyses of neuroimaging data. *NeuroImage* 184. [doi:10.1016/j.neuroimage.2018.09.074](https://doi.org/10.1016/j.neuroimage.2018.09.074)
- Sokal, R. R. & Rohlf, F. J. (1962). The comparison of dendrograms by objective methods. *Taxon* 11. [doi:10.2307/1217208](https://doi.org/10.2307/1217208)
- Tiblias, F., Bigoulaeva, I., Niu, J., Balloccu, S. & Gurevych, I. (2025). Hypothesis-Driven Feature Manifold Analysis in LLMs via Supervised Multi-Dimensional Scaling (v1: "Shape Happens"). [arXiv:2510.01025](https://arxiv.org/abs/2510.01025)
- Timkey, W. & van Schijndel, M. (2021). All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality. [arXiv:2109.04404](https://arxiv.org/abs/2109.04404)
- Yang, A. et al. (2024). Qwen2.5-Math Technical Report. [arXiv:2409.12122](https://arxiv.org/abs/2409.12122)
- Models and data: [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base), [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B), [Gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B), [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), the judges [granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) and [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct); the arXiv metadata mirror [`librarian-bots/arxiv-metadata-snapshot`](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) (CC0); [Mathlib4](https://github.com/leanprover-community/mathlib4).

Code, data, preregistration and every number: <https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms>. Interactive explorer: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/>. Data and activations: <https://huggingface.co/datasets/BurnyCoder/geometry-of-mathematics-in-llms>.

# Appendix A: Replication of lattices and shapes

**Qwen3.5-0.8B, Qwen2.5-1.5B and Qwen2.5-Math-1.5B**, each with its twin:

{{table:lattice:robustness}}

{{table:lattice_probes:robustness}}

{{table:lattice_variants:robustness}}

{{table:shapes:robustness}}

{{table:cycles:robustness}}

{{table:chain_frequency:robustness}}

**Gemma-4-E2B** with its twin:

{{table:lattice:gemma}}

{{table:lattice_probes:gemma}}

{{table:lattice_variants:gemma}}

{{table:shapes:gemma}}

{{table:cycles:gemma}}

{{table:chain_frequency:gemma}}

**Word-form baselines on the ordered families** (audit; the token-embedding bag uses Qwen3.5-2B's embeddings):

{{table:lexical_chains:mvp}}

**Leave-one-out order with 10,000 permutations: the preregistered α = 10 and a scale-relative α** (audit, sensitivity):

{{table:loo_alpha:mvp}}

{{table:loo_alpha:robustness}}

{{table:loo_alpha:gemma}}

Matched 4-field clustering in Gemma-4-E2B:
- math subsets: mean {{num:gemma.posthoc.gemma-4-e2b.matched_clustering.math_subsets_mean:.3f}};
- control fields: {{num:gemma.posthoc.gemma-4-e2b.matched_clustering.control_acc:.3f}}.

# Appendix B: Deviations from the preregistration

See `docs/deviations.md`. Every post-hoc statistic in this paper is labelled post-hoc. The two post-hoc addenda, each committed before its analyses ran, are `reports/prereg_addendum_3d.md` (cycles beyond PC1–2) and `reports/prereg_addendum_lexical.md` (the independent audit).

# AI statement

This codebase and paper were written by Claude Code (Anthropic's Claude, model Opus 5.5) acting as a coding and research agent. A human, Libor Burian, provided the exact specification that was executed. Human cleanup and rewriting are in progress.
