---
title: "Do LLMs Organize Mathematics Geometrically? Field Maps, Ordered Chains and Axiom Lattices in Residual Streams"
author: "Libor \"Burny\" Burian"
date: "October 2026"
abstract: |
  Large language models represent days of the week and months on circles (Engels et al., 2024). We ask whether *abstract mathematics* is organized geometrically too. We test three preregistered hypotheses on base models up to 2B parameters, comparing every model with a random-init twin (same architecture and tokenizer, no learning).

  **(A) Field map.** Centroids of 17 mathematical fields are computed from concepts drawn from Wikipedia glossaries, excluding concepts whose names leak the field. In Qwen3.5-2B-Base, their geometry correlates with field relatedness measured by arXiv cross-listing over 3.2M papers: Spearman ρ = 0.39, against ρ = -0.12 for the random-init twin. The paired bootstrap 95% CI of the difference is [0.28, 0.58]. The map replicates in Qwen3.5-0.8B (ρ = 0.37), Qwen2.5-1.5B (ρ = 0.41) and Gemma-4-E2B (ρ = 0.44). In every case the random-init twin is near zero or below. It is, however, no stronger than the organization of non-math science topics, and no stronger than in a contrastive text embedder (Qwen3-Embedding). It is also not a clean hierarchy: our preregistered cophenetic test does not beat the random-init twin. One trillion tokens of math continued pretraining (Qwen2.5-Math) do not sharpen it (Δρ CI [-0.06, 0.02]).

  **(B) Shapes.** Class-inclusion chains (rng ⊃ ring ⊃ … ⊃ field; T0 ⊃ … ⊃ Hilbert) are ordered beyond permutation nulls and beyond the twin. Angles 0°–330°, however, are encoded like the number line rather than a circle in all five models, which falsifies our preregistered prediction.

  **(B2) Axiom lattices.** For group-like structures (magma … abelian group), adding an axiom behaves like a consistent direction. Same-axiom difference vectors are more parallel than random pairs, and much less so in the random-init twins. This holds clearly for associativity and commutativity, which survive Holm correction in the confirmatory model and are significant in all five models. The evidence is weaker for divisibility (raw p < .05 in all five models, but not after Holm correction) and for identity (raw p < .05 in three of five models).

  The days/months positive control met our preregistered full-space criterion only partially. A post-hoc test in the top-2 principal-component plane recovers perfect cyclic order, but that score also rewards open arcs. Counting only loops that close in their plane (post-hoc, pre-specified in an addendum), days close in the top-2 plane in three of five models and months in a lower principal-component plane in three of five. The same test finds no closed cycle for angles in any plane of the top five components. Simulations show why we use paired random-init contrasts as primary tests: partial regression on distance matrices inflates type-I error when nuisances are proxies of confounders.
---

# 1 Introduction

Interpretability research has found that LLMs encode some concepts as low-dimensional geometric objects rather than isolated directions:
- circles for days and months, which the models causally use for modular arithmetic ([Engels et al., 2024](https://arxiv.org/abs/2405.14860));
- a helix for numbers ([Kantamneni & Tegmark, 2025](https://arxiv.org/abs/2502.00873));
- a spiral for the periodic table ([Lei & Cooper, 2025](https://arxiv.org/abs/2502.10871));
- linear "crystals" of analogies among sparse-autoencoder features ([Li et al., 2024](https://arxiv.org/abs/2410.19750)).

[Karkada et al. (2026)](https://arxiv.org/abs/2602.15029) explain such shapes with symmetries of co-occurrence statistics: cyclic latents give circles, and ordered latents give 1-D curved manifolds.

Mathematics is an unusually good test bed for these ideas, because its concepts carry *exact* external structure. Fields relate through shared methods. Algebraic structures form inclusion chains. Structures are defined by axiom sets, so they sit on the vertices of a hypercube. As far as our literature search found, nobody has tested whether LLM representations of abstract mathematical concepts reflect any of this. The closest result is the "math/code lobe" of [Li et al. (2024)](https://arxiv.org/abs/2410.19750), which does not resolve fields or structures.

**Contributions.**

1. A preregistered test (`docs/prereg.md`, git tag `prereg-v1`) of three geometric hypotheses about mathematics in LLMs, with random-init twins, leakage flags, permutation nulls and paired bootstrap contrasts.
2. Evidence that field-level geometry tracks how mathematicians cross-list papers, replicated across five models from two families and unchanged by heavy math pretraining.
3. Evidence that axioms of group-like structures act as consistent directions (parallelograms). This is strongest for associativity and commutativity. We also find that class-inclusion chains are ordered.
4. Two negative results: angles are not circular, and math pretraining does not sharpen the map. There is also one methodological negative: partial RDM regression is miscalibrated under realistic confounding.

# 2 Related work

**Feature geometry.**
- *Low-dimensional and circular features:* [Engels et al. (2024)](https://arxiv.org/abs/2405.14860), [Kantamneni & Tegmark (2025)](https://arxiv.org/abs/2502.00873), [Gurnee & Tegmark (2023)](https://arxiv.org/abs/2310.02207).
- *Manifolds:* [Modell et al. (2025)](https://arxiv.org/abs/2505.18235), and supervised MDS for discovering them ([Tiblias et al., 2025](https://arxiv.org/abs/2510.01025)).
- *A statistical account:* [Karkada et al. (2026)](https://arxiv.org/abs/2602.15029).

**Hierarchy and categories.** [Park et al. (2024)](https://arxiv.org/abs/2406.01506) proposed orthogonality between hierarchical concepts. [Golechha et al. (ICLR 2025)](https://www.greaterwrong.com/posts/eczwWrmX5XNEo7JsS/intricacies-of-feature-geometry-in-large-language-models) showed that the same geometry appears for random concepts and random unembeddings after whitening. We therefore avoid whitening and compare every statistic with an untrained twin.

**Science concepts.** [Lei & Cooper (2025)](https://arxiv.org/abs/2502.10871) found a periodic-table spiral. Mathematical "world models" in LLMs have been reported as local and fragile ([ICML 2025](https://icml.cc/virtual/2025/49128)).

**Statistics.** Partial Mantel tests are known to be anticonservative ([Guillot & Rousset, 2013](https://doi.org/10.1111/2041-210X.12018)). We test Freedman–Lane permutation ([Anderson & Legendre, 1999](https://doi.org/10.1080/00949659908811936)) in our setting (Section 3.5).

# 3 Methods

## 3.1 Concepts and fields

**Field map (H-A).**
- **Source.** Concepts come from Wikipedia "Glossary of …" pages for 17 mathematical fields, plus "List of numerical analysis topics", at pinned revisions. Four non-math control fields (biology, chemistry, geology, astronomy) go through the identical pipeline.
- **Cleaning.** Terms keep 1–4 alphabetic words. Fragments, prefixes and adverbs are dropped.
- **Flags.** These exclude a concept from the *primary* set:
  - `leak_own`: the name contains its field's root word, e.g. "topological space";
  - `eponym`: a proper name, detected by mid-sentence capitalization on the same page;
  - `polysemous`: a single everyday word with wordfreq Zipf ≥ 4.
- **Sampling.** At most 30 primary concepts per field.
- **Split.** A 20% pilot split was used only for debugging. All statistics below use the 80% confirmatory split: 398 primary math concepts across 17 fields.

## 3.2 Ground-truth field distance

MSC2020 is flat above its two-digit codes, so we measure field relatedness from practice instead. For fields A and B, with S_A the set of arXiv papers listing any of A's categories, the distance is 1 − |S_A∩S_B|/√(|S_A||S_B|), i.e. one minus the Ochiai similarity. This is computed over the CC0 arXiv metadata snapshot (3,224,635 papers, pinned commit).

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
| `Qwen/Qwen3.5-2B-Base` (Feb 2026) | **Confirmatory**; official Qwen-Scope SAEs |
| `Qwen3.5-0.8B-Base` | Scale |
| `Qwen2.5-1.5B` vs `Qwen2.5-Math-1.5B` | Same base, +1T math tokens ([Qwen2.5-Math report, Fig. 2](https://arxiv.org/abs/2409.12122)) |
| `google/gemma-4-E2B` | Second model family |

Each model has a random-init twin with weights redrawn using seed 0.

- **Read point.** Each item is embedded in 8 templates ending with the concept, e.g. "In mathematics, the notion of *X*". Templates name the domain ("mathematics" or "science") but never a sub-field. The representation is the residual stream at the concept's last token, averaged over templates.
- **Preprocessing.** Mean-centering, with no whitening.
- **Distances.** Cosine distance for the field map. Euclidean distance for families: cosine distance on a small, centered, collinear family is degenerate, which our synthetic tests showed.
- **Layer.** The primary layer is fixed in advance at relative depth 0.5.

## 3.5 Statistics: why paired twin contrasts

Before analysing any LLM data, we simulated regression on distance matrices with Freedman–Lane permutation:

| Scenario | Rejection rate at α=0.05 (± SE) | n simulations |
|---|---|---|
| concept level, nuisance correctly specified (null) | 0.065 ± 0.017 | 200 |
| concept level, item-level noise / nonlinear link (null) | 0.220 ± 0.029 | 200 |
| field level, representation unrelated to fields (null) | 0.020 ± 0.010 | 200 |
| field level, nuisance = noisy proxy of a confounder (null) | 0.525 ± 0.035 | 200 |
| field level, true field geometry present (power) | 0.995 ± 0.005 | 200 |

The test is calibrated when the nuisance distance matrix *is* the confounder. It rejects far too often when the nuisance only approximates the confounder, which is exactly the situation for lexical form and frequency. We therefore made **paired contrasts against the random-init twin** the primary tests. The twin measures form-only similarity on the same concepts. Uncertainty comes from a stratified concept bootstrap (2,000 resamples) with identical resamples for both models.

- **Field map.** We use field-level RSA, i.e. the Spearman ρ between the centroid RDM and the arXiv distance, with a Mantel permutation over field labels (10,000 permutations) and a max-over-layers null for layer curves.
- **Families.** Every statistic is unsupervised or leave-one-out, and is tested against an order-permutation null. With n ≤ 13 items in more than 1,000 dimensions, a supervised fit can impose any order. As a negative control, 100 random 11-item lists of math concepts must be rejected at about 5%.
- **Figures, including 3-D views.** Every picture is an unsupervised projection: PCA, or metric MDS for field maps. 3-D views show each layer's own top principal components with true proportions, and state the share of variance on each axis (PCA) or the stress-1 of the layout (MDS). Alignment between panels and layers is display-only: rotation, or only sign flips for 3-D PCA axes. No statistic is computed from a picture. The single test that looks beyond PC1–2, the closed-cycle test over the top-5 planes, is post-hoc and was pre-specified in an addendum before it ran (§4.3).

# 4 Results

![**Results at a glance.** Blue: trained model; grey: its random-init twin (same architecture and tokenizer, no learning). The starred model is the preregistered confirmatory model, Qwen3.5-2B. (A) Field-centroid geometry correlates with arXiv cross-listing in all five models (dots: ρ; lines: 95% bootstrap CI); the twins sit at or below zero. (B) Field maps (metric MDS) for arXiv cross-listing, the confirmatory model and its twin; grey edges join each field to its two nearest arXiv neighbours. (C) Days of the week close into a cycle in the top-2 principal-component plane (positive control). (D) Angles do not; the orange edge is the 330° → 0° closure the preregistration predicted. (E) Offset consistency per axiom: the confirmatory model, the other four models and the five twins, against the random-pair null (grey bar: mean to 95th percentile). Diamonds show the confirmatory model with the token-count direction removed. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#s-glance>.](../reports/figures/hero.pdf){width=100%}

**Every test in every model.** The table below summarizes the confirmatory and secondary tests for all five models.

| Test | Status | Qwen3.5-2B | Qwen3.5-0.8B | Qwen2.5-1.5B | Qwen2.5-Math-1.5B | Gemma-4-E2B |
|---|---|---|---|---|---|---|
| Field map tracks arXiv beyond the twin (H-A1; Δρ CI) | pre-registered, confirmatory | ● 0.39 | ● 0.37 | ● 0.41 | ● 0.39 | ● 0.44 |
| Field map is a hierarchy beyond the twin (H-A2; cophenetic Δ CI) | pre-registered | ○ 0.24 | ○ 0.25 | ○ 0.24 | ○ 0.11 | ○ 0.23 |
| Days form a cycle (positive control; PC plane) | positive control; post-hoc statistic | ● 1.00 | ● 1.00 | ● 1.00 | ● 1.00 | ● 1.00 |
| Ring chain ordered (Robinson; > twin) | pre-registered, confirmatory | ● 0.52 | ● 0.50 | ● 0.54 | ● 0.56 | ● 0.50 |
| Space chain ordered (Robinson; > twin) | pre-registered, confirmatory | ● 0.65 | ● 0.64 | ● 0.71 | ● 0.68 | ● 0.57 |
| Ring / space order beyond frequency | added by plan audit | ● / ● | ● / ● | ● / ● | ● / ● | ○ / ○ |
| Angles form a circle (closure < 2 and circle ρ > line ρ) | pre-registered, confirmatory | ✗ 2.11 | ✗ 2.59 | ✗ 2.15 | ✗ 1.84 | ✗ 2.03 |
| Angles: closed cycle in any top-5 PC plane (addendum rule) | post-hoc, pre-specified addendum | ○ 0.33 | ○ 0.33 | ○ 0.17 | ○ 0.25 | ○ 0.42 |
| Days / months: a loop that closes in some top-5 PC plane (p < .05, unadjusted) | post-hoc, descriptive | ● 1.00 / ● 0.50 | ○ 0.57 / ○ 0.33 | ● 1.00 / ○ 0.33 | ● 1.00 / ● 0.58 | ○ 0.43 / ● 0.50 |
| Axiom direction: associative (offset consistency; Holm) | pre-registered, confirmatory | ● 0.43 | ● 0.47 | ● 0.38 | ● 0.35 | ● 0.56 |
| Axiom direction: identity (offset consistency; Holm) | pre-registered, confirmatory | ◐ 0.24 | ○ 0.22 | ◐ 0.27 | ● 0.27 | ○ 0.19 |
| Axiom direction: divisible (offset consistency; Holm) | pre-registered, confirmatory | ◐ 0.29 | ◐ 0.26 | ◐ 0.25 | ● 0.30 | ● 0.35 |
| Axiom direction: commutative (offset consistency; Holm) | pre-registered, confirmatory | ● 0.53 | ● 0.57 | ● 0.41 | ● 0.45 | ● 0.74 |
| Lie / topological / algebraic groups as parallelograms | pre-registered, exploratory | ● 0.40 | ● 0.46 | ● 0.41 | ● 0.41 | ● 0.48 |

● significant (Holm-adjusted for the axiom tests; 95% CI excludes 0 for Δ tests) · ◐ raw p < .05 only · ○ not significant · ✗ pre-registered prediction falsified. Numbers: the statistic for the trained model.

![**Effect sizes with 95% bootstrap CIs (unadjusted).** Blue: CI excludes 0. "Added" marks analyses added by the plan audits rather than the preregistration. The field map beats the twin in every model and beats fastText, but not the Qwen3-Embedding text embedder. Math pretraining (H-A3) and the hierarchy test (H-A2) are nulls. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-forest>.](../reports/figures/forest.pdf){width=82%}

## 4.1 H-A1: a map of mathematical fields

| Model | Concept set | n concepts / fields | ρ_field trained (perm p) [95% CI] | ρ random-init | ρ layer-0 | Δ trained − random-init [95% CI] | LOO field acc trained / random / layer-0 (chance) | split-half reliability |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | math_primary | 398 / 17 | 0.387 (p=<0.0001) [0.262, 0.420] | -0.118 | 0.024 | +0.506 [0.277, 0.576] | 0.472 / 0.090 / 0.279 (0.059) | 0.999 |
| qwen3.5-2b | math_all | 526 / 17 | 0.426 (p=<0.0001) [0.303, 0.448] | -0.035 | 0.115 | +0.461 [0.231, 0.548] | 0.485 / 0.108 / 0.308 (0.059) | 0.999 |

![**The map is present at almost every depth, in every model.** ρ between the field-centroid RDM and the arXiv distance at each layer (x: relative depth). Blue: trained; grey: twin. The horizontal line is the 95th percentile of the max-over-layers permutation null; the vertical line marks the preregistered layer. The Qwen models clear the null at every layer; Gemma-4-E2B dips below it in two stretches. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-depth>.](../reports/figures/depth_curves.pdf){width=100%}

**The trained model's field geometry tracks arXiv relatedness, and the random-init twin's does not.** For primary concepts at the preregistered layer, ρ = 0.387 with Mantel p = 0.0001. The twin gives ρ = -0.118 and layer-0 embeddings give ρ = 0.024. The difference from the twin has 95% CI [0.277, 0.576].

**Layer profile.** The map emerges in the first blocks and peaks at layer 6. It stays above the 95% max-over-layers null (0.17) at every layer.

**Clustering.** Leave-one-out nearest-centroid field accuracy is 0.472, against 0.090 for the twin and 0.059 chance.

**Reliability.** Split-half template reliability of the field RDM is 0.999.

![**Related fields stay close in the trained model, and the twin scrambles them.** Metric MDS of three 17 × 17 distance matrices: arXiv cross-listing (ground truth), Qwen3.5-2B field centroids at layer 12, and the random-init twin. Panels are rotated or reflected onto the arXiv layout for display only. Grey edges join each field to its two nearest arXiv neighbours and are drawn in every panel: short edges mean the neighbourhood is preserved. Blue: pure; orange: applied. ρ is computed on the full matrices, not on the 2-D layout. Interactive version, with a layer slider: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-map>.](../reports/figures/field_maps_qwen3.5-2b.pdf){width=100%}

![**The same matrices, ranked.** Colour is the rank of each distance within its matrix (Spearman RSA compares ranks), with rows in one fixed order grouped by the five super-tree branches. The trained matrix partly reproduces the arXiv block structure; the twin's does not. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-rdm>.](../reports/figures/field_rdms_qwen3.5-2b.pdf){width=100%}

**Not a rogue-dimension artifact (post-hoc).** The preregistered all-but-the-top variant removes the top d/100 = 20 principal components, and it reverses the correlation (ρ = -0.22). A post-hoc sweep explains why:

| Model | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 | k=20 | k=40 | top-5 PC explained variance | field η² on PC1–10 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 0.387 | 0.466 | 0.351 | 0.312 | 0.178 | 0.021 | -0.218 | -0.341 | 0.066, 0.050, 0.037, 0.035, 0.028 | 0.20, 0.47, 0.22, 0.10, 0.14, 0.13, 0.13, 0.20, 0.27, 0.34 |
| qwen3.5-2b__random | -0.118 | -0.140 | -0.133 | -0.136 | -0.107 | -0.074 | -0.160 | -0.164 | 0.013, 0.012, 0.012, 0.012, 0.012 | 0.04, 0.02, 0.04, 0.06, 0.05, 0.03, 0.02, 0.03, 0.08, 0.06 |

- Removing the top component *raises* ρ to 0.47.
- No component dominates.
- Field identity is spread over the top ~10 components, which carry far more field variance (η²) than the twin's.

So the map lives in the dominant semantic subspace, and removing 20 components removes the map itself.

**H-A2: hierarchy is not supported as preregistered.** The preregistered measure is the cophenetic correlation between an average-linkage (UPGMA) dendrogram of the field centroids and the five-branch super-tree.

| Model | super-tree match (cophenetic ρ): trained / twin / layer-0 | shuffled-tree p (null95) | Δ trained − twin [95% CI] | cophenetic c trained / twin (Gaussian 95th pct) | relative Gromov δ trained / twin (Gaussian 5th pct) |
|---|---|---|---|---|---|
| qwen3.5-2b | 0.237 / -0.033 / 0.171 | 0.0200 (0.164) | [-0.079, 0.348] | 0.760 / 0.590 (0.774) | 0.243 / 0.059 (0.302) |

| Model | super-tree match (cophenetic ρ): trained / twin / layer-0 | shuffled-tree p (null95) | Δ trained − twin [95% CI] | cophenetic c trained / twin (Gaussian 95th pct) | relative Gromov δ trained / twin (Gaussian 5th pct) |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 0.250 / -0.103 / 0.266 | 0.0185 (0.156) | [-0.021, 0.409] | 0.748 / 0.581 (0.788) | 0.293 / 0.082 (0.298) |
| qwen2.5-1.5b | 0.244 / 0.046 / 0.173 | 0.0165 (0.166) | [-0.065, 0.396] | 0.732 / 0.561 (0.782) | 0.244 / 0.078 (0.309) |
| qwen2.5-math-1.5b | 0.113 / 0.039 / 0.178 | 0.1114 (0.164) | [-0.070, 0.385] | 0.721 / 0.597 (0.750) | 0.233 / 0.091 (0.314) |

| Model | super-tree match (cophenetic ρ): trained / twin / layer-0 | shuffled-tree p (null95) | Δ trained − twin [95% CI] | cophenetic c trained / twin (Gaussian 95th pct) | relative Gromov δ trained / twin (Gaussian 5th pct) |
|---|---|---|---|---|---|
| gemma-4-e2b | 0.228 / 0.072 / 0.125 | 0.0215 (0.163) | [-0.080, 0.406] | 0.779 / 0.754 (0.870) | 0.275 / 0.067 (0.251) |

- **Against shuffled trees:** the trained model's dendrogram matches the super-tree better (ρ = 0.237, p = 0.020).
- **Against the twin:** the paired difference CI includes zero ([-0.079, 0.348]), and the same holds in every replication model. The preregistered criterion requires beating both, so it fails.
- **Intrinsic tree-likeness:** the cophenetic correlation of the representation with its own dendrogram is no higher than for a spectrum-matched Gaussian cloud.
- **Gromov δ:** relative δ is lower than the Gaussian cloud's, but higher than the twin's.

A tree-distance RSA we first reported, the Spearman correlation of the centroid RDM with the super-tree distance, did beat the twin (trained 0.312 vs twin 0.077). It is *not* the preregistered measure, so it is post-hoc. It largely restates H-A1: fields that mathematicians relate are close.

**Specificity.** The non-math control fields (biology, chemistry, geology, astronomy) are also clustered by the trained model and not by its twin:

| Model | Domain | n / fields | LOO acc trained [CI] | random-init | layer-0 | chance | Δ trained − random-init [CI] |
|---|---|---|---|---|---|---|---|
| qwen3.5-2b | math | 398 / 17 | 0.472 [0.427, 0.520] | 0.090 | 0.279 | 0.059 | +0.382 [0.329, 0.435] |
| qwen3.5-2b | control | 97 / 4 | 0.814 [0.732, 0.887] | 0.268 | 0.474 | 0.250 | +0.546 [0.423, 0.660] |

The pre-registered comparison confounds domain with the number of fields: 17 for math, 4 for the control. We therefore ran a post-hoc matched comparison over 200 random 4-field subsets of the math fields:
- **Math subsets:** mean leave-one-out accuracy 0.729 (95% range 0.573–0.854).
- **Control fields:** 0.814.
- **Overlap:** 0.12 of the math subsets reach or exceed the control value.

Mathematical fields are therefore organized about as strongly as general science topics, and not more. The field map is an instance of topical organization, not something specific to mathematics.

**Baselines (plan; added after the first report).** We compare against two non-generative representations of the same concepts:
- **fastText** wiki-news vectors (averaged over words) and **Qwen3-Embedding-0.6B**, a contrastively trained text embedder.
- **vs fastText:** the LLM's field map is stronger (Δ CI [0.07, 0.30] on the 367 concepts that fastText covers).
- **vs Qwen3-Embedding:** the LLM is not stronger; the embedder reaches ρ = 0.454 against the LLM's 0.387 (Δ CI [-0.13, 0.04]).

So the field map is a property of contextual text representations in general, not something specific to generative LLMs.

| Model | concept set | n | field ρ: LLM / Qwen3-Embedding / fastText | Δ LLM − Qwen3-Emb [CI] | Δ LLM − fastText [CI] | LOO acc: LLM / Qwen3-Emb / fastText |
|---|---|---|---|---|---|---|
| qwen3.5-2b | all_primary | 398 | 0.387 / 0.454 / — | [-0.125, 0.045] | — | 0.472 / 0.487 / — |
| qwen3.5-2b | fasttext_covered | 367 | 0.401 / 0.490 / 0.190 | [-0.148, 0.034] | [0.074, 0.302] | 0.485 / 0.477 / 0.322 |

infini-gram (v4_dolma-v1_7_llama): Spearman ρ(wordfreq Zipf, log count) = 0.316; concepts with zero count: 0.2%.

## 4.2 Robustness: scale, model family, math training

| Model | Concept set | n concepts / fields | ρ_field trained (perm p) [95% CI] | ρ random-init | ρ layer-0 | Δ trained − random-init [95% CI] | LOO field acc trained / random / layer-0 (chance) | split-half reliability |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | math_primary | 398 / 17 | 0.369 (p=<0.0001) [0.243, 0.402] | -0.182 | 0.059 | +0.551 [0.300, 0.588] | 0.430 / 0.163 / 0.284 (0.059) | 0.998 |
| qwen3.5-0.8b | math_all | 526 / 17 | 0.402 (p=0.0002) [0.272, 0.429] | -0.035 | 0.160 | +0.436 [0.243, 0.530] | 0.437 / 0.173 / 0.314 (0.059) | 0.998 |
| qwen2.5-1.5b | math_primary | 398 / 17 | 0.406 (p=0.0002) [0.288, 0.436] | -0.093 | 0.044 | +0.499 [0.318, 0.565] | 0.565 / 0.219 / 0.289 (0.059) | 0.998 |
| qwen2.5-1.5b | math_all | 526 / 17 | 0.432 (p=<0.0001) [0.310, 0.453] | 0.033 | 0.118 | +0.399 [0.256, 0.505] | 0.529 / 0.202 / 0.310 (0.059) | 0.999 |
| qwen2.5-math-1.5b | math_primary | 398 / 17 | 0.390 (p=<0.0001) [0.264, 0.416] | -0.087 | 0.021 | +0.477 [0.297, 0.540] | 0.588 / 0.211 / 0.307 (0.059) | 0.999 |
| qwen2.5-math-1.5b | math_all | 526 / 17 | 0.406 (p=0.0002) [0.289, 0.428] | 0.059 | 0.141 | +0.348 [0.226, 0.468] | 0.568 / 0.200 / 0.325 (0.059) | 0.999 |

| Model | Concept set | n concepts / fields | ρ_field trained (perm p) [95% CI] | ρ random-init | ρ layer-0 | Δ trained − random-init [95% CI] | LOO field acc trained / random / layer-0 (chance) | split-half reliability |
|---|---|---|---|---|---|---|---|---|
| gemma-4-e2b | math_primary | 398 / 17 | 0.436 (p=<0.0001) [0.315, 0.477] | -0.021 | 0.091 | +0.457 [0.290, 0.598] | 0.417 / 0.196 / 0.307 (0.059) | 0.993 |
| gemma-4-e2b | math_all | 526 / 17 | 0.464 (p=<0.0001) [0.342, 0.501] | 0.085 | 0.183 | +0.379 [0.252, 0.543] | 0.399 / 0.215 / 0.312 (0.059) | 0.991 |

**H-A3 (math training): null.** Qwen2.5-Math-1.5B is Qwen2.5-1.5B plus 1T tokens of math. Its field map is no stronger than its base's: ρ = 0.390 vs 0.406, and the paired CI of the difference is [-0.059, 0.017].

## 4.3 H-B: shapes of ordered families

| Family | Expected | Model | line ρ | circle ρ | closure (d first-last / adjacent) | closure_far† (null 5%) | Robinson (p) | PCA order \|ρ\| (p) | LOO order ρ (p) |
|---|---|---|---|---|---|---|---|---|---|
| days | circle | qwen3.5-2b | 0.78 | 0.65 | 1.19 | 0.99 (0.76) | 0.77 (0.0010) | 0.54 (0.2284) | -0.14 (0.0731) |
| days | circle | qwen3.5-2b__random | -0.04 | -0.07 | 0.99 | 1.02 (0.96) | 0.34 (0.5002) | 0.18 (0.7106) | -0.93 (0.1993) |
| months | circle | qwen3.5-2b | 0.79 | 0.70 | 1.23 | 0.96 (0.76) | 0.66 (0.0005) | 0.55 (0.0730) | 0.77 (0.0066) |
| months | circle | qwen3.5-2b__random | -0.10 | -0.06 | 0.99 | 0.97 (0.94) | 0.30 (0.6962) | 0.32 (0.3048) | -0.78 (0.6013) |
| ring_chain | line | qwen3.5-2b | 0.48 | 0.38 | 1.63 | 1.04 (0.63) | 0.52 (0.0105) | 0.46 (0.1619) | 0.65 (0.0133) |
| ring_chain | line | qwen3.5-2b__random | -0.03 | -0.01 | 1.17 | 1.84 (0.53) | 0.30 (0.6667) | 0.50 (0.1234) | -0.45 (0.2292) |
| space_chain | line | qwen3.5-2b | 0.67 | 0.40 | 1.79 | 1.42 (0.57) | 0.65 (0.0005) | 0.83 (0.0025) | 0.87 (0.0033) |
| space_chain | line | qwen3.5-2b__random | -0.04 | 0.03 | 0.66 | 0.99 (0.54) | 0.33 (0.4993) | 0.56 (0.0770) | -0.41 (0.2625) |
| number_systems | line | qwen3.5-2b | 0.77 | 0.36 | 1.67 | 1.66 (0.56) | 0.74 (0.0045) | 0.96 (0.0020) | 0.71 (0.0066) |
| number_systems | line | qwen3.5-2b__random | 0.11 | 0.16 | 1.01 | 1.01 (0.93) | 0.26 (0.7856) | 0.39 (0.3773) | -0.82 (0.3887) |
| ngons | number_reference | qwen3.5-2b | 0.56 | 0.28 | 1.94 | 1.06 (0.54) | 0.64 (0.0010) | 0.87 (0.0030) | 0.90 (0.0033) |
| ngons | number_reference | qwen3.5-2b__random | 0.42 | 0.14 | 1.65 | 1.76 (0.57) | 0.42 (0.1159) | 0.19 (0.6007) | -0.89 (0.9236) |
| dimensions | number_reference | qwen3.5-2b | 0.66 | 0.37 | 2.17 | 1.02 (0.31) | 0.75 (0.0005) | 0.95 (0.0010) | 0.72 (0.0033) |
| dimensions | number_reference | qwen3.5-2b__random | -0.17 | -0.15 | 1.01 | 1.01 (0.93) | 0.32 (0.6117) | 0.00 (1.0000) | -0.75 (0.6578) |
| angles | circle | qwen3.5-2b | 0.56 | 0.29 | 2.11 | 1.01 (0.39) | 0.65 (0.0005) | 0.90 (0.0005) | 0.76 (0.0033) |
| angles | circle | qwen3.5-2b__random | -0.01 | 0.22 | 0.80 | 1.01 (0.77) | 0.26 (0.8581) | 0.08 (0.8076) | -0.49 (0.3754) |

![**Days and months go round in their true order; angles do not.** Top-2 principal-component plane of each family at the preregistered layer (unsupervised; no labels are used to fit it). Colour runs light to dark along the true order; the grey path joins consecutive items. For angles, the orange edge is the 330° → 0° closure the preregistration predicted. Below each panel: the post-hoc PC-plane cycle score (1 = perfect cyclic order) with its order-permutation p. The score also gives 1 to an open arc: here the months' Dec → Jan step is 3.0× a neighbour step, so in this plane they are an open arc. Bottom row: the random-init twin. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/cyclic_qwen3.5-2b.pdf){width=82%}

![**The cycles are present at almost every depth.** Days (top) and months (bottom) at fixed relative depths, each plane rotated onto the preregistered layer's plane for display, plus the cycle score at every layer for the trained model and the twin (right). Post-hoc and unadjusted. Animated version: `reports/figures/circle_depth_qwen3.5-2b.gif`; interactive: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/circle_depth_qwen3.5-2b.pdf){width=100%}

**Positive control.** With our preregistered full-space statistics, days and months show significant circle-template correlation and closure ratios near 1. They nevertheless fail "circle ρ > line ρ" at layer 12. Following the prereg, we debugged before interpreting further. In the top-2 principal-component plane, where Engels et al. visualized their circles, days and months are in *perfect* cyclic order in the trained model and in none in the twin:

| Family | qwen3.5-2b | qwen3.5-2b__random |
|---|---|---|
| days | 1.00 (p=0.0080) | 0.14 (p=0.9581) |
| months | 1.00 (p=0.0020) | 0.08 (p=0.9102) |
| ring_chain | 0.36 (p=0.0559) | 0.18 (p=0.5429) |
| space_chain | 0.45 (p=0.0100) | 0.09 (p=0.9182) |
| number_systems | 0.29 (p=0.6188) | 0.43 (p=0.2016) |
| ngons | 0.50 (p=0.0100) | 0.30 (p=0.2116) |
| dimensions | 0.55 (p=0.0040) | 0.18 (p=0.5549) |
| angles | 0.33 (p=0.0559) | 0.17 (p=0.5469) |

**Beyond the top-2 plane (post-hoc, pre-specified addendum).** The cycle score above counts a point's angular successor, so an open arc scores 1 too, and PCA bends lines into such arcs ([Diaconis, Goel & Holmes 2008](https://arxiv.org/abs/0811.1477)). After adding 3-D views we tested every plane (PCi, PCj), i < j ≤ 5, for a loop that also *closes* in the plane. A plane counts only if its in-plane closure ratio is below 2, the preregistered threshold. The p-value is a permutation p of the maximum over the 10 planes.
- **Pre-specification.** The rule was written down before the run ([`reports/prereg_addendum_3d.md`](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/blob/master/reports/prereg_addendum_3d.md)), with two amendments made from unit tests before any real data was run.
- **Days:** close in the top-2 plane in three of five models.
- **Months:** close in a lower plane in three of five (PC2–PC3 or PC4–PC5; p < .05 each, unadjusted).
- **Twin:** one twin also shows a closed months loop, so the test is not fully specific.

| Family | qwen3.5-2b | qwen3.5-0.8b | qwen2.5-1.5b | qwen2.5-math-1.5b | gemma-4-e2b |
|---|---|---|---|---|---|
| days | 1.00 · PC1–PC2 · p=0.0310 (twin 0.29) | 0.57 · PC2–PC4 · p=0.6224 (twin 0.57) | 1.00 · PC1–PC2 · p=0.0310 (twin 0.57) | 1.00 · PC1–PC2 · p=0.0270 (twin 0.57) | 0.43 · PC1–PC4 · p=0.8472 (twin 0.29) |
| months | 0.50 · PC2–PC3 · p=0.0220 (twin 0.25) | 0.33 · PC2–PC3 · p=0.3826 (twin 0.25) | 0.33 · PC2–PC3 · p=0.3976 (twin 0.58) | 0.58 · PC2–PC3 · p=0.0040 (twin 0.25) | 0.50 · PC4–PC5 · p=0.0200 (twin 0.42) |
| angles | 0.33 · PC3–PC5 · p=0.3447 (twin 0.33) | 0.33 · PC3–PC5 · p=0.3327 (twin 0.33) | 0.17 · PC2–PC3 · p=0.9890 (twin 0.17) | 0.25 · PC2–PC3 · p=0.8472 (twin 0.42) | 0.42 · PC2–PC4 · p=0.0839 (twin 0.17) |
| digits_0_330 | 0.33 · PC2–PC3 · p=0.3287 (twin 0.33) | 0.17 · PC2–PC3 · p=0.9950 (twin 0.25) | 0.25 · PC2–PC3 · p=0.7912 (twin 0.25) | 0.17 · PC2–PC4 · p=0.9910 (twin 0.25) | 0.33 · PC2–PC5 · p=0.3287 (twin 0.25) |

![**Days, months and angles in their top-3 principal components.** Each axis is that model's own PC, labelled with its share of variance, with true proportions and one fixed camera for every panel. Top row: Qwen3.5-2B; bottom row: its random-init twin. Below each panel: the share of variance in PC1–3 (for example 85% for the angles, 33% in the twin), and the best plane among PC1–5 where the loop closes, with its p. Interactive 3-D version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/families3d_qwen3.5-2b.pdf){width=82%}

This statistic is post-hoc. We conclude that the pipeline works, and that the full-space template criterion was insensitive because variance outside the circle plane dominates.

**Chains.** The ring and space chains are ordered beyond permutation nulls and beyond the twin: Robinson scores 0.52 and 0.65, against about 1/3 for chance and for the twin. Random concept lists are rejected at the nominal rate:

| Model | Robinson reject rate | PCA-order reject rate | line-ρ reject rate | n lists |
|---|---|---|---|---|
| qwen3.5-2b | 0.04 | 0.04 | 0.04 | 100 |
| qwen3.5-2b__random | 0.04 | 0.04 | 0.05 | 100 |

![**Class-inclusion chains are ordered paths; the twin's are tangles.** PC planes of the ordered families at the preregistered layer. Below each panel: the Robinson score (fraction of item triples consistent with the order; chance 1/3) with its permutation p, and for the ring and space chains the p of the frequency control. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/chains_qwen3.5-2b.pdf){width=100%}

**Frequency control (plan; added in the second audit).** Could the chain order just be a frequency or word-length order? We removed wordfreq Zipf frequency and model token count from both the representation and the ranks, fitting that regression inside each leave-one-out fold to avoid leakage ([Snoek et al., 2019](https://doi.org/10.1016/j.neuroimage.2018.09.074)), then predicted the residual order.
- *Result.* Both preregistered chains stay ordered: ring chain ρ = 0.94 (p = 0.003), space chain ρ = 0.81 (p = 0.017). The twin gives 0.13 and -0.04.
- *Replication (Appendix A).* Both chains stay ordered in all four Qwen models (p ≤ 0.023), but not in Gemma-4-E2B (ring p = 0.17, space p = 0.056).
- *Word form carries part of the ring-chain order.* Some twins also order the ring chain beyond frequency: Qwen2.5-1.5B's twin (p = 0.037) and Gemma's twin (p = 0.027). A plausible lexical cue is that the middle items all end in "domain". The space chain's twins are weaker: none reaches p < .05, though the two Qwen2.5 twins come close (p = 0.053).
- *Caveat.* The fold-wise version was adopted post-hoc. A first, whole-sample version leaked the held-out item into the training span and was degenerate in the twin. Both versions are reported in `docs/deviations.md`.

| Chain | Model | ρ(order, Zipf) | ρ(order, token count) | LOO order beyond covariates (null95, p) |
|---|---|---|---|---|
| ring_chain | qwen3.5-2b | -0.04 | +0.29 | +0.94 (0.65, 0.0033) |
| ring_chain | qwen3.5-2b__random | -0.04 | +0.29 | +0.13 (0.42, 0.2791) |
| space_chain | qwen3.5-2b | -0.10 | -0.05 | +0.81 (0.60, 0.0166) |
| space_chain | qwen3.5-2b__random | -0.10 | -0.05 | -0.04 (0.41, 0.5349) |
| number_systems | qwen3.5-2b | -0.57 | +0.80 | +0.21 (0.86, 0.3920) |
| number_systems | qwen3.5-2b__random | -0.57 | +0.80 | -0.21 (0.89, 0.5681) |

**Angles: preregistered prediction falsified.** Angles are not circular: closure ratio 2.11, no cycle in the top-2 PC plane, and, under the pre-specified addendum rule, no closed cycle in any plane of the top five components in any model (below). Their geometry closely matches that of the bare digits 0–330 (ρ = 0.84). At this scale, "330 degrees" sits next to "300 degrees", not next to "0 degrees".

**The preregistered rule is not specific.** Every random-init twin *meets* the circle rule, for example the Qwen3.5-2B twin: closure 0.80, circle ρ 0.22 vs line ρ -0.01. A structureless point cloud has a closure ratio near 1 and near-zero template correlations, so it passes by chance. The informative result is therefore not the failed rule itself but the trained models' clear line-like order: line ρ 0.56 in Qwen3.5-2B, against circle ρ 0.29. That order tracks the bare numbers. We found this when we added the twin panel to the angles figure; it was not part of the preregistered analysis.

| Family | Model | ρ(family RDM, reference RDM) | template β beyond reference (p) |
|---|---|---|---|
| ngons vs number_words_3_12 | qwen3.5-2b | 0.59 | +0.223 (0.3089) |
| ngons vs number_words_3_12 | qwen3.5-2b__random | 0.11 | +0.411 (0.0066) |
| dimensions vs number_words_0_10 | qwen3.5-2b | 0.76 | +0.260 (0.0204) |
| dimensions vs number_words_0_10 | qwen3.5-2b__random | -0.58 | -0.153 (0.9104) |
| angles vs digits_0_330 | qwen3.5-2b | 0.84 | +0.054 (0.1858) |
| angles vs digits_0_330 | qwen3.5-2b__random | 0.22 | +0.227 (0.0104) |

The addendum's decision rule for "angles have a circular component beyond PC1–2" requires all four conditions:

| Model | Angles: best closed plane | p (max over planes) | 1. Holm across models | 2. not PC1–PC2 | 3. > twin | 4. numerals do not cycle | Accepted |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B | 0.33 · PC3–PC5 | 0.3447 | no | yes | no | yes | **no** |
| Qwen3.5-0.8B | 0.33 · PC3–PC5 | 0.3327 | no | yes | no | yes | **no** |
| Qwen2.5-1.5B | 0.17 · PC2–PC3 | 0.9890 | no | yes | no | yes | **no** |
| Qwen2.5-Math-1.5B | 0.25 · PC2–PC3 | 0.8472 | no | yes | no | yes | **no** |
| Gemma-4-E2B | 0.42 · PC2–PC4 | 0.0839 | no | yes | yes | yes | **no** |

**No model meets the rule**, so "angles are not a circle" stands, now for every plane of the top 5 principal components too.

![**Predicted vs found for angles.** Left: the preregistered prediction, a circle on which 330° neighbours 0° (orange edge). Middle: Qwen3.5-2B's PC plane, where that edge spans the whole layout. Right: the bare numbers 0–330, whose geometry the angles follow. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/angles_qwen3.5-2b.pdf){width=90%}

**SMDS (plan, secondary).** Supervised MDS fits a projection to a line or a discrete-circle template, and each fit is tested against refits on shuffled labels.

| Family | Model | SMDS line score (null95, p) | SMDS discrete-circle score (null95, p) | preferred (vs null) |
|---|---|---|---|---|
| days | qwen3.5-2b | 0.966 (0.970, 0.1569) | 0.891 (0.889, 0.0196) | discrete_circular |
| days | qwen3.5-2b__random | 1.000 (1.000, 0.4902) | 0.895 (0.895, 0.5686) | none |
| months | qwen3.5-2b | 0.961 (0.941, 0.0196) | 0.869 (0.862, 0.0196) | euclidean |
| months | qwen3.5-2b__random | 1.000 (1.000, 0.8431) | 0.870 (0.870, 0.9608) | none |
| ring_chain | qwen3.5-2b | 0.964 (0.945, 0.0196) | 0.877 (0.874, 0.0196) | euclidean |
| ring_chain | qwen3.5-2b__random | 1.000 (1.000, 0.6471) | 0.886 (0.886, 0.2745) | none |
| space_chain | qwen3.5-2b | 0.926 (0.887, 0.0196) | 0.853 (0.851, 0.0588) | euclidean |
| space_chain | qwen3.5-2b__random | 1.000 (1.000, 0.1373) | 0.886 (0.886, 0.5294) | none |
| number_systems | qwen3.5-2b | 0.959 (0.956, 0.0392) | 0.885 (0.887, 0.2157) | euclidean |
| number_systems | qwen3.5-2b__random | 1.000 (1.000, 0.7451) | 0.895 (0.895, 0.9412) | none |
| ngons | qwen3.5-2b | 0.928 (0.914, 0.0588) | 0.843 (0.843, 0.0784) | none |
| ngons | qwen3.5-2b__random | 1.000 (1.000, 0.6078) | 0.867 (0.867, 0.2157) | none |
| dimensions | qwen3.5-2b | 0.800 (0.684, 0.0196) | 0.721 (0.668, 0.0196) | euclidean |
| dimensions | qwen3.5-2b__random | 1.000 (1.000, 0.2941) | 0.886 (0.886, 0.8824) | none |
| angles | qwen3.5-2b | 0.856 (0.659, 0.0196) | 0.744 (0.673, 0.0196) | euclidean |
| angles | qwen3.5-2b__random | 1.000 (1.000, 0.9608) | 0.870 (0.870, 0.7451) | none |

- **The twin shows why the shuffle null is necessary.** Its line fit scores 1.00 on *every* family, a perfect fit to any ordering, yet never beats its shuffled null.
- **Trained model:**
  - Days prefer the circle template: circle p = 0.020, line p = 0.157.
  - The chains and angles prefer the line template.
  - Months fit both templates beyond the null.

This agrees with the unsupervised statistics, including the conclusion that angles are line-like.

## 4.4 H-B2: axioms as directions

| Lattice | Model | axiom-RDM β (p) | lexical β | offset consistency per axiom: obs / null95 (p) | analogy top-1 (chance) | mean rank (chance) |
|---|---|---|---|---|---|---|
| group_like | qwen3.5-2b | +0.492 (0.0002) | +0.236 | associative: 0.43/0.23 (0.0035); identity: 0.24/0.22 (0.0425); divisible: 0.29/0.22 (0.0275); commutative: 0.53/0.20 (0.0010) | 0.56 (0.10) | 2.5 (5.5) |
| self_distributive | qwen3.5-2b | +0.123 (0.3393) | -0.446 | invertible: 0.48/0.78 (0.2039); idempotent: 0.17/0.78 (0.4268) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | qwen3.5-2b | — | — | gender: 0.28/0.09 | king:queen rank 1 | — |
| group_like | qwen3.5-2b__random | +0.302 (0.0010) | -0.403 | associative: 0.02/0.16 (0.3678); identity: 0.04/0.16 (0.3068); divisible: 0.01/0.16 (0.4338); commutative: 0.01/0.14 (0.4048) | 0.31 (0.10) | 3.6 (5.5) |
| self_distributive | qwen3.5-2b__random | +0.316 (0.1284) | -0.275 | invertible: -0.01/0.56 (0.5207); idempotent: 0.01/0.56 (0.4493) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | qwen3.5-2b__random | — | — | gender: 0.00/0.07 | king:queen rank 15 | — |

![**Adding an axiom moves structures in a consistent direction in the trained model only.** Left: the ideal geometry, in which each axiom is one fixed step. Other panels: the same unsupervised PCA plane of the 13 group-like structures, one panel per axiom, with the pairs that differ only in that axiom drawn as blue arrows. Parallel arrows mean a consistent direction. Below each panel: offset consistency and its random-pair p. Bottom row: the twin. Interactive version, with an analogy playground: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-lattice>.](../reports/figures/lattice_axes_qwen3.5-2b.pdf){width=100%}

![**The same facets in 3-D.** The 13 structures in their top-3 principal components (70% of their variance; twin 52%), with true proportions and one fixed camera. In each facet, blue arrows are the pairs that differ only in that axiom. Top row: Qwen3.5-2B; bottom row: its twin. A 3-D view can still hide or exaggerate parallelism; the cosine matrices (next figure) are the evidence. Interactive 3-D version, with the analogy query b − a + c placed exactly: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-lattice>.](../reports/figures/lattice3d_qwen3.5-2b.pdf){width=100%}

![**The direction evidence, unprojected.** Cosine between every pair of single-axiom difference vectors in the full residual space, grouped by axiom. Red diagonal blocks mean a consistent direction. Hatched cells are cross-axiom pairs that share a structure; isotropic noise alone puts these near ±0.5, so they carry no evidence. The self-cosines on the diagonal are masked. Right: offset consistency per axiom for the trained model, the trained model with the token-count direction removed, and the twin, against the random-pair null. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-cosines>.](../reports/figures/lattice_cosines_qwen3.5-2b.pdf){width=100%}

For every group-like axiom, difference vectors between structures that differ only in that axiom are more parallel than random pairs (raw p < .05):

| Axiom | Consistency | Null 95th percentile | p |
|---|---|---|---|
| Associativity | 0.43 | 0.23 | 0.0035 |
| Identity | 0.24 | 0.22 | 0.0425 |
| Divisibility | 0.29 | 0.22 | 0.0275 |
| Commutativity | 0.53 | 0.20 | 0.0010 |

In this model's random-init twin, all four consistencies are ≈ 0.

**Analogies.** Analogy completion is top-1 for 0.56 of 45 parallelograms (chance 0.10). The twin reaches 0.31, so part of the analogy accuracy is lexical (shared "commutative" tokens); offset consistency is not.

**Caveats.** The gender-offset positive control passes. Multiplicity across the preregistered confirmatory tests is handled by Holm correction:

| Test | Model | p | Holm-adjusted p | reject at α=0.05 |
|---|---|---|---|---|
| H-A1 Δρ trained − random-init ≤ 0 (bootstrap) | qwen3.5-2b | <0.0001 | <0.0001 | yes |
| H-B ring chain Robinson | qwen3.5-2b | 0.0105 | 0.0315 | yes |
| H-B space chain Robinson | qwen3.5-2b | 0.0005 | 0.0030 | yes |
| H-B2 associative offset consistency | qwen3.5-2b | 0.0035 | 0.0140 | yes |
| H-B2 identity offset consistency | qwen3.5-2b | 0.0425 | 0.0550 | no |
| H-B2 divisible offset consistency | qwen3.5-2b | 0.0275 | 0.0550 | no |
| H-B2 commutative offset consistency | qwen3.5-2b | 0.0010 | 0.0050 | yes |
| H-B angles circle prediction (decision rule: closure 2.11 < 2 and circle ρ 0.29 > line ρ 0.56) | qwen3.5-2b | — | — | not met (prediction falsified) |

- **After correction:** associativity and commutativity remain significant; identity and divisibility do not.
- **Across the replication models (Appendix A):** all four offsets are positive in every model. They are near zero in the Qwen twins and weaker in Gemma's twin, which has small lexical offsets for associativity and divisibility. Divisibility reaches raw p < .05 in all five models, identity in three.
- **Our reading:** strong evidence for associativity and commutativity directions, and suggestive evidence for divisibility and identity.

**Lie groups and other group objects (exploratory).** A Lie group is a group that is also a smooth manifold; topological and algebraic groups are defined analogously. We tested three parallelograms:
- topological group − group ≈ topological space − set;
- Lie group − topological group ≈ smooth manifold − topological space;
- algebraic group − group ≈ algebraic variety − set.

| Model | commutativity transfer cos (null95, p) | identity transfer cos (p) | group-object parallelograms cos (null95, p) | parent ranks: topological / Lie / algebraic group |
|---|---|---|---|---|
| qwen3.5-2b | +0.198 (0.198, 0.0510) | +0.015 (0.3983) | +0.398 (0.235, 0.0045) | [5, 8] / [3, 7] / [3, 6] (of 113) |
| qwen3.5-2b__random | -0.017 (0.167, 0.7126) | +0.001 (0.4743) | +0.006 (0.031, 0.3768) | [2, 44] / [1, 18] / [3, 71] (of 113) |

| Model | commutativity transfer cos (null95, p) | identity transfer cos (p) | group-object parallelograms cos (null95, p) | parent ranks: topological / Lie / algebraic group |
|---|---|---|---|---|
| qwen3.5-0.8b | +0.188 (0.212, 0.0760) | +0.075 (0.1794) | +0.457 (0.241, 0.0025) | [5, 15] / [4, 6] / [4, 5] (of 113) |
| qwen3.5-0.8b__random | +0.010 (0.144, 0.4038) | -0.014 (0.6552) | -0.028 (0.058, 0.7941) | [1, 72] / [1, 65] / [2, 88] (of 113) |
| qwen2.5-1.5b | +0.183 (0.165, 0.0380) | +0.020 (0.3933) | +0.406 (0.221, 0.0035) | [4, 6] / [3, 6] / [4, 6] (of 113) |
| qwen2.5-1.5b__random | +0.142 (0.135, 0.0435) | +0.038 (0.1424) | +0.156 (0.138, 0.0350) | [1, 2] / [1, 18] / [1, 2] (of 113) |
| qwen2.5-math-1.5b | +0.202 (0.160, 0.0210) | +0.096 (0.1244) | +0.409 (0.208, 0.0015) | [4, 7] / [3, 6] / [3, 6] (of 113) |
| qwen2.5-math-1.5b__random | +0.124 (0.137, 0.0600) | +0.005 (0.4358) | +0.150 (0.139, 0.0410) | [1, 2] / [1, 15] / [1, 2] (of 113) |

| Model | commutativity transfer cos (null95, p) | identity transfer cos (p) | group-object parallelograms cos (null95, p) | parent ranks: topological / Lie / algebraic group |
|---|---|---|---|---|
| gemma-4-e2b | +0.129 (0.271, 0.2324) | -0.180 (0.9320) | +0.484 (0.303, 0.0085) | [41, 6] / [2, 10] / [7, 3] (of 113) |
| gemma-4-e2b__random | +0.003 (0.063, 0.4568) | +0.037 (0.2344) | -0.003 (0.141, 0.5337) | [2, 31] / [2, 6] / [2, 7] (of 113) |

![**Lie, topological and algebraic groups combine their parents in every trained model.** Left: mean cosine of the three parallelograms (for example, Lie group − topological group vs smooth manifold − topological space) for each model (blue) and its twin (grey), against the 95th percentile of a random-quadruple null (grey tick); p values from that null. Right: the nearest-neighbour rank of "group" (circle) and "smooth manifold" (diamond) among the family items around "Lie group", trained vs twin (log scale). Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-groups>.](../reports/figures/group_objects.pdf){width=88%}

**Group objects: supported.**
- The parallelograms hold in all five trained models: mean cosine 0.40 in Qwen3.5-2B (p = 0.0045). The twins are near zero; the Qwen2.5 twins show a weak lexical signal.
- Trained models place "Lie group" close to *both* parents: "group" and "smooth manifold" rank 3 and 7 among 113 items.
- The twins place it nearest to "group" because of the shared token. "Smooth manifold" ranks 18 in the Qwen3.5-2B twin; Gemma's twin is the exception, ranking it 6.

**Axiom transfer: inconclusive.**
- The commutativity direction learned on group-like pairs transfers only weakly to ring → commutative ring and division ring → field (0.20, p = 0.051).
- Some Qwen twins show similar values. "Commutative ring" literally contains the token "commutative", so we consider transfer not established.
- Identity transfer, from the single pair rng → ring, is null.

**Further lattice items from the plan (exploratory; second audit).**

| Model | associative quasigroup: d(·, group) / d(·, quasigroup) (3 nearest) | inverse semigroup: 3 nearest | identity direction → lattice-like pairs: cos (null95, p) |
|---|---|---|---|
| qwen3.5-2b | 2.81 (commutative quasigroup, quasigroup, commutative loop) | semigroup, commutative semigroup, monoid | +0.074 (0.157, 0.1359) |
| qwen3.5-2b__random | 1.11 (quasigroup, semigroup, monoid) | semigroup, quasigroup, loop | +0.054 (0.149, 0.1454) |

| Model | Variant | offset consistency per axiom: obs / null95 (p) |
|---|---|---|
| qwen3.5-2b | group-like, mean-pooled tokens | associative: 0.30 / 0.22 (0.0235); identity: 0.26 / 0.23 (0.0305); divisible: 0.21 / 0.24 (0.0750); commutative: 0.60 / 0.19 (0.0005) |
| qwen3.5-2b | group-like, token-count direction removed | associative: 0.44 / 0.23 (0.0035); identity: 0.26 / 0.22 (0.0360); divisible: 0.31 / 0.22 (0.0235); commutative: 0.46 / 0.20 (0.0010) |
| qwen3.5-2b | partial-operation lattice | total: 0.29 / 0.54 (0.1704); identity: 0.07 / 0.82 (0.4518); divisible: -0.22 / 0.80 (0.5817) |
| qwen3.5-2b__random | group-like, mean-pooled tokens | associative: 0.03 / 0.17 (0.3378); identity: 0.03 / 0.16 (0.3283); divisible: 0.01 / 0.17 (0.4058); commutative: 0.10 / 0.14 (0.1084) |
| qwen3.5-2b__random | group-like, token-count direction removed | associative: 0.02 / 0.16 (0.3633); identity: 0.04 / 0.16 (0.3068); divisible: 0.01 / 0.16 (0.4278); commutative: -0.02 / 0.14 (0.5487) |
| qwen3.5-2b__random | partial-operation lattice | total: -0.03 / 0.38 (0.5272); identity: -0.08 / 0.55 (0.6872); divisible: -0.07 / 0.55 (0.6707) |

- *No semantic collapse.* A non-empty associative quasigroup is a group, but the model does not place it there. It is 2.8× farther from "group" than from "quasigroup" (twin 1.1×), and its nearest neighbours are quasigroup-like. The representation follows the name's parts, not this theorem.
- *Inverse semigroup* sits next to semigroup, commutative semigroup and monoid in the trained model, which is sensible.
- *Not a length artifact.* Removing the token-count direction (fitted on all family items) leaves the group-like offsets intact: commutativity 0.46 (p = 0.0010), associativity 0.44 (p = 0.0035). Across the five models (Appendix A), associativity, commutativity and divisibility stay significant in all five after this projection, and identity in four. Mean-pooling the concept's tokens gives a similar pattern, with divisibility no longer significant (table above). This is a one-direction analogue of the LDA distractor removal of [Li et al. (2024)](https://arxiv.org/abs/2410.19750).
- *Partial-operation lattice: no evidence.* The "total operation" offset (semigroupoid → semigroup, small category → monoid, groupoid → group) is 0.29, below its null's 95th percentile (0.54). With 6 items and 2–3 pairs per axis, the test is weak.
- *Identity transfer to lattice-like pairs* (semilattice → bounded semilattice, lattice → bounded lattice) is null too: cos = 0.07, p = 0.14.

![**A negative result: no consistent "make the operation total" direction.** Offset consistency for the partial-operation lattice (semigroupoid, small category, groupoid vs semigroup, monoid, group): the confirmatory model (filled), the other four models (open) and the five twins (grey), against the confirmatory model's random-pair null (grey bar: mean to 95th percentile). With 6 items and 2–3 pairs per axis the test has little power. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-partial>.](../reports/figures/partial_operation.pdf){width=70%}

**Template variants (exploratory; Qwen3.5-2B).** We reran every analysis with two other template sets.

*Field-revealing concept templates* name the field, e.g. "In group theory, the notion of X":
- The graded field map barely changes (ρ = 0.410, against 0.387 with neutral templates).
- Leave-one-out field accuracy jumps to 0.633, and the twin also rises to 0.332. This is exactly the label leakage the preregistration excluded from the primary analysis.

*No-domain family templates* drop the shared context, e.g. "Consider X":
- Days remain a perfect PC-plane cycle.
- The chains remain ordered (ring and space Robinson p = 0.020 and 0.021), and angles remain line-like.
- Group-object parallelograms persist (0.47, p = 0.0005).
- Axiom offsets weaken but persist for commutativity (0.47, p = 0.0020) and associativity (0.31, p = 0.0215).
- Months lose their PC-plane cycle.

Full tables: `reports/mvp_variants_results.md`.

## 4.5 Secondary analyses

### H-A4: concrete ↔ abstract and pure ↔ applied are the main axes

**Ratings.** Three judge LLMs rated every math concept's abstraction from 1 to 7: IBM Granite-4.0-micro, SmolLM3-3B and Phi-4-mini-instruct, none from a model family we probe. We read each rating as the expectation over next-token digit probabilities.
- Rank agreement is moderate (mean pairwise Spearman 0.63).
- Absolute agreement is low (ICC(2,1) = 0.24; ICC(2,k) = 0.49), because the judges use the scale with different offsets.
- The ratings have face validity. The least abstract concepts include "trivial group", "bar chart" and "dot plot"; the most abstract include étale cohomology, Artin stacks and generalized cohomology theories.

**Validation against Mathlib (plan; specified before computing, `docs/deviations.md`).** As an external, non-LLM measure, we matched concepts to declarations in [Mathlib4](https://github.com/leanprover-community/mathlib4) (commit `4bc4670bf916c8dc2ede840e3ead86301e52a40d`).
- *Matching.* First through Mathlib's curated English-name maps (`docs/overview.yaml`, `docs/undergrad.yaml`), then through exact CamelCase names for multi-word concepts ("compact space" → `CompactSpace`).
- *Depth.* The longest chain of Mathlib imports below the defining module (8672 modules).
- *Result.* The mean judge rating correlates with depth: ρ = 0.37, 95% CI [0.07, 0.62], meeting the pre-specified criterion.
- *Caveat.* Only 39 of 655 concepts could be matched, and coverage is uneven across fields (table below). The judges also differ: Phi-4-mini's ratings track depth (ρ = 0.54), SmolLM3's barely (ρ = 0.05).

| Matched concepts (curated YAML / CamelCase) | median import depth | ρ(mean judge rating, depth) [95% CI] (p) | ρ partialling Zipf | ρ per judge |
|---|---|---|---|---|
| 39 of 655 (22 / 17) | 72 | +0.373 [0.071, 0.616] (0.0194) | +0.312 | granite-4.0-micro: +0.28; SmolLM3-3B: +0.05; Phi-4-mini-instruct: +0.54 |

| Field | Mathlib coverage (fraction of concepts matched) |
|---|---|
| algebraic_geometry | 0.025 |
| algebraic_topology | 0.025 |
| category_theory | 0.075 |
| commutative_algebra | 0.075 |
| differential_geometry | 0.000 |
| functional_analysis | 0.000 |
| game_theory | 0.038 |
| general_topology | 0.100 |
| graph_theory | 0.000 |
| group_theory | 0.139 |
| lie_representation_theory | 0.051 |
| number_theory | 0.050 |
| numerical_analysis | 0.000 |
| probability_statistics | 0.025 |
| real_complex_analysis | 0.150 |
| ring_module_theory | 0.250 |
| set_theory_logic | 0.000 |

![**Concrete ↔ abstract is the top axis of the math-concept cloud.** Left and middle: PC1 × PC2 of the 398 primary concepts, both residualized on frequency and token count exactly as in the statistic. Left is coloured by judge abstraction, with the most and least abstract concepts labelled; middle is coloured pure/applied. The PC signs are flipped for display so that abstract and applied point right and up. Right: mean judge rating vs Mathlib import depth for the matched concepts. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-abstraction>.](../reports/figures/abstraction_qwen3.5-2b.pdf){width=100%}

**Axes.** We first residualized on word frequency and token count. The top principal component of the math-concept cloud then correlates with abstraction (ρ = 0.49), and the second with the pure/applied label (ρ = -0.41). In the twin these correlations are 0.06 and 0.14. A 5-fold cross-validated linear readout predicts abstraction with ρ = 0.74, against -0.06 in the twin. We also partialled out pretraining-scale frequency: infini-gram counts in Dolma v1.7, which correlate ρ = 0.32 with wordfreq. The axes are essentially unchanged: PC1 ↔ abstraction ρ = 0.44, PC2 ↔ applied ρ = -0.41.

**Interpretation.** In this model, concrete ↔ abstract is the single largest axis of variation among mathematical concepts. That is a correlational, model-rated finding: abstraction ratings come from LLM judges, not humans.

### Causal test of chain geometry: inconclusive

The preregistered gate requires baseline accuracy ≥ 0.70 on "which is more general, a A or a B?".
- **Ring chain:** it failed the gate (accuracy 0.691), so no intervention was run.
- **Space chain:** it passed (0.864).

At noise scale s = 4, noise injected into the chain's 2-D subspace lowered accuracy to 0.791. Noise in random 2-D subspaces gave a mean of 0.822. At s = 1 there was no difference. The direction matches the prediction, but the effect is small and five random draws cannot establish significance. We report it as inconclusive.

### SAE features (exploratory)

At block 11, we encoded the template-averaged residuals with the official Qwen-Scope TopK SAE (k = 50).
- **Graded map:** the SAE code preserves the field map only weakly (ρ = 0.15, against 0.39 for the dense residual).
- **Clustering:** field clustering is largely preserved (LOO accuracy 0.43).
- **Field-selective features exist.** For example, a game-theory feature fires on most game-theory concepts and almost no others (`results/mvp/sae.json`).
- **Feature sets:** 2445 features fire only on math concepts, 871 only on control concepts, and 715 on both. The sample sizes are unequal, so this is only suggestive of the "math lobe" of [Li et al. (2024)](https://arxiv.org/abs/2410.19750).


# 5 Discussion

**What the geometry is.** Relatedness between fields is encoded at the level of field centroids, in the dominant principal components, from the first few layers onward. It is equally strong in models with very different pretraining mixtures. Combined with the null effect of math continued pretraining, this suggests the map is learned from general co-occurrence statistics, as [Karkada et al. (2026)](https://arxiv.org/abs/2602.15029) predict. It does not seem to require math-specific training. The control fields cluster about as strongly once the number of fields is matched. That supports this reading: the "map of mathematics" is one instance of a general topical map.

**Axioms as directions.** Axioms of group-like structures behave like the man→woman offset. The geometry follows definitions stated in the names, not consequences that must be derived: "associative quasigroup" stays with the quasigroups, although a non-empty one is a group. Whether the model uses these directions in reasoning, for example to decide that a monoid with inverses is a group, needs causal tests; see §6 and the causal results.

**Angles.** "330 degrees" is not near "0 degrees". This mirrors the finding of [Kantamneni & Tegmark (2025)](https://arxiv.org/abs/2502.00873) that numbers lie on a helix whose dominant axis is linear. A model this size appears to encode degrees by their numerals rather than by their angular meaning.

# 6 Limitations

- **Concept sample.** Glossary terms are a convenience sample, and some junk survives cleaning. Field labels come from one glossary per term.
- **Ground truth.** Ochiai cross-listing is one operationalization of field relatedness.
- **Read point.** A single read-out: last token, averaged over templates. Template wording affects absolute values, though split-half reliability is high.
- **Model size.** Models are ≤ 2.3B active parameters, because of the 8 GB GPU.
- **Positive control.** It passed only under post-hoc statistics. In the plane where its loop closes, it holds for days in three of five models and for months in three of five.
- **Family size.** Families are small (n = 7–13), so shape claims rest on permutation tests with limited resolution.
- **Correlation, not mechanism.** All tests except §4.5 are correlational.
- **Abstraction ground truth.** Abstraction ratings come from LLM judges. Their validation against Mathlib import depth rests on only 39 of 655 concepts that could be matched to Mathlib, and import depth also differs systematically between areas of Mathlib.
- **Not run.** Three planned items were not run; reasons are in `docs/deviations.md`:
  - Qwen3.5-4B in 8-bit, whose random-init twin cannot be held in 8 GB;
  - the optional OctoThinker pair;
  - persistent homology, since no family has ≥30 points.
- **Baseline scope.** SMDS and the template variants were run for the confirmatory model only.

# 7 Conclusion

Mathematical fields form a map in LLM residual streams that tracks how mathematicians cross-list their papers. Axioms of algebraic structures act as consistent directions, most clearly associativity and commutativity, and class-inclusion chains are ordered. All of this is absent, or much weaker, in untrained twins. Angles are not circles, and heavy math pretraining does not sharpen the field map.

# References

See the inline links. Code, data, preregistration and every number: <https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms>. Interactive explorer: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/>. Data and activations: <https://huggingface.co/datasets/BurnyCoder/geometry-of-mathematics-in-llms>.

# Appendix A: Replication of shapes and lattices

**Qwen3.5-0.8B, Qwen2.5-1.5B and Qwen2.5-Math-1.5B**, each with its twin:

| Family | Expected | Model | line ρ | circle ρ | closure (d first-last / adjacent) | closure_far† (null 5%) | Robinson (p) | PCA order \|ρ\| (p) | LOO order ρ (p) |
|---|---|---|---|---|---|---|---|---|---|
| days | circle | qwen3.5-0.8b | 0.85 | 0.63 | 1.29 | 1.06 (0.79) | 0.77 (0.0010) | 0.75 (0.0735) | -1.00 (1.0000) |
| days | circle | qwen3.5-0.8b__random | 0.20 | 0.07 | 1.01 | 1.04 (0.93) | 0.40 (0.3758) | 0.93 (0.0095) | -0.75 (0.1728) |
| days | circle | qwen2.5-1.5b | 0.77 | 0.66 | 1.22 | 0.99 (0.73) | 0.69 (0.0020) | 0.54 (0.2219) | 0.68 (0.0664) |
| days | circle | qwen2.5-1.5b__random | 0.10 | -0.11 | 1.02 | 1.01 (0.93) | 0.43 (0.2259) | 0.54 (0.2324) | -0.82 (0.4950) |
| days | circle | qwen2.5-math-1.5b | 0.73 | 0.58 | 1.18 | 1.01 (0.76) | 0.71 (0.0010) | 0.54 (0.2219) | 0.36 (0.1229) |
| days | circle | qwen2.5-math-1.5b__random | 0.16 | -0.14 | 1.02 | 1.01 (0.94) | 0.37 (0.3973) | 0.46 (0.2889) | -0.93 (0.7708) |
| months | circle | qwen3.5-0.8b | 0.79 | 0.70 | 1.23 | 0.97 (0.77) | 0.67 (0.0005) | 0.64 (0.0345) | -0.20 (0.0033) |
| months | circle | qwen3.5-0.8b__random | 0.06 | -0.03 | 1.02 | 1.02 (0.94) | 0.38 (0.2274) | 0.30 (0.3483) | -0.86 (0.9169) |
| months | circle | qwen2.5-1.5b | 0.71 | 0.56 | 1.25 | 1.07 (0.75) | 0.64 (0.0005) | 0.85 (0.0010) | 0.80 (0.0100) |
| months | circle | qwen2.5-1.5b__random | 0.07 | -0.02 | 1.02 | 0.97 (0.92) | 0.35 (0.4218) | 0.10 (0.7381) | -0.62 (0.7176) |
| months | circle | qwen2.5-math-1.5b | 0.71 | 0.53 | 1.21 | 0.99 (0.75) | 0.70 (0.0005) | 0.92 (0.0005) | 0.82 (0.0066) |
| months | circle | qwen2.5-math-1.5b__random | 0.04 | -0.01 | 1.02 | 0.97 (0.91) | 0.32 (0.5827) | 0.03 (0.9370) | -0.60 (0.7243) |
| ring_chain | line | qwen3.5-0.8b | 0.46 | 0.36 | 1.70 | 1.00 (0.58) | 0.50 (0.0170) | 0.45 (0.1764) | -0.09 (0.0764) |
| ring_chain | line | qwen3.5-0.8b__random | 0.18 | 0.11 | 1.29 | 1.36 (0.74) | 0.42 (0.1134) | 0.04 (0.9295) | 0.36 (0.0565) |
| ring_chain | line | qwen2.5-1.5b | 0.54 | 0.36 | 1.74 | 1.11 (0.56) | 0.54 (0.0025) | 0.21 (0.5517) | 0.87 (0.0033) |
| ring_chain | line | qwen2.5-1.5b__random | 0.30 | 0.31 | 1.06 | 1.03 (0.85) | 0.35 (0.4318) | 0.03 (0.9405) | 0.23 (0.0731) |
| ring_chain | line | qwen2.5-math-1.5b | 0.54 | 0.39 | 1.67 | 1.05 (0.62) | 0.56 (0.0030) | 0.41 (0.2189) | 0.89 (0.0033) |
| ring_chain | line | qwen2.5-math-1.5b__random | 0.34 | 0.30 | 1.09 | 1.05 (0.85) | 0.41 (0.1529) | 0.05 (0.8786) | 0.25 (0.0698) |
| space_chain | line | qwen3.5-0.8b | 0.65 | 0.33 | 1.72 | 1.67 (0.59) | 0.64 (0.0005) | 0.95 (0.0005) | 0.63 (0.0033) |
| space_chain | line | qwen3.5-0.8b__random | -0.04 | 0.04 | 0.91 | 1.06 (0.72) | 0.36 (0.3703) | 0.36 (0.2659) | 0.29 (0.0432) |
| space_chain | line | qwen2.5-1.5b | 0.69 | 0.38 | 1.83 | 1.68 (0.60) | 0.71 (0.0005) | 0.95 (0.0005) | 0.90 (0.0033) |
| space_chain | line | qwen2.5-1.5b__random | -0.12 | -0.06 | 0.95 | 1.10 (0.80) | 0.34 (0.4888) | 0.67 (0.0315) | 0.21 (0.1229) |
| space_chain | line | qwen2.5-math-1.5b | 0.67 | 0.40 | 1.71 | 1.51 (0.61) | 0.68 (0.0005) | 0.85 (0.0010) | 0.91 (0.0033) |
| space_chain | line | qwen2.5-math-1.5b__random | -0.10 | -0.07 | 0.97 | 1.08 (0.80) | 0.36 (0.3988) | 0.73 (0.0185) | 0.18 (0.1329) |
| number_systems | line | qwen3.5-0.8b | 0.81 | 0.40 | 1.73 | 1.72 (0.57) | 0.77 (0.0030) | 0.93 (0.0065) | 0.29 (0.0033) |
| number_systems | line | qwen3.5-0.8b__random | 0.31 | 0.14 | 1.10 | 1.25 (0.77) | 0.29 (0.6892) | 0.29 (0.5542) | -0.25 (0.1960) |
| number_systems | line | qwen2.5-1.5b | 0.66 | 0.33 | 1.42 | 1.57 (0.60) | 0.66 (0.0165) | 0.82 (0.0295) | 0.79 (0.0166) |
| number_systems | line | qwen2.5-1.5b__random | 0.34 | 0.14 | 1.13 | 1.37 (0.73) | 0.34 (0.4718) | 0.54 (0.2219) | -0.21 (0.2359) |
| number_systems | line | qwen2.5-math-1.5b | 0.71 | 0.33 | 1.42 | 1.55 (0.61) | 0.66 (0.0145) | 0.82 (0.0295) | 0.71 (0.0266) |
| number_systems | line | qwen2.5-math-1.5b__random | 0.36 | 0.24 | 1.13 | 1.38 (0.72) | 0.40 (0.2344) | 0.57 (0.1839) | -0.43 (0.3056) |
| ngons | number_reference | qwen3.5-0.8b | 0.54 | 0.30 | 1.88 | 1.05 (0.51) | 0.63 (0.0015) | 0.87 (0.0030) | 0.60 (0.0033) |
| ngons | number_reference | qwen3.5-0.8b__random | 0.36 | 0.07 | 1.36 | 1.17 (0.77) | 0.47 (0.0555) | 0.73 (0.0140) | -0.42 (0.3256) |
| ngons | number_reference | qwen2.5-1.5b | 0.54 | 0.27 | 1.81 | 1.01 (0.55) | 0.67 (0.0010) | 0.88 (0.0015) | 0.96 (0.0033) |
| ngons | number_reference | qwen2.5-1.5b__random | 0.33 | 0.10 | 1.20 | 1.12 (0.80) | 0.44 (0.0940) | 0.50 (0.1454) | -0.39 (0.3522) |
| ngons | number_reference | qwen2.5-math-1.5b | 0.53 | 0.23 | 1.71 | 1.05 (0.58) | 0.66 (0.0010) | 0.88 (0.0015) | 0.90 (0.0066) |
| ngons | number_reference | qwen2.5-math-1.5b__random | 0.35 | 0.11 | 1.22 | 1.12 (0.79) | 0.49 (0.0295) | 0.49 (0.1709) | -0.21 (0.2027) |
| dimensions | number_reference | qwen3.5-0.8b | 0.62 | 0.37 | 2.29 | 0.98 (0.27) | 0.63 (0.0010) | 0.93 (0.0005) | -0.71 (0.0066) |
| dimensions | number_reference | qwen3.5-0.8b__random | -0.17 | -0.03 | 0.97 | 0.93 (0.92) | 0.27 (0.8621) | 0.30 (0.3828) | -0.71 (0.6645) |
| dimensions | number_reference | qwen2.5-1.5b | 0.62 | 0.44 | 1.84 | 0.97 (0.31) | 0.59 (0.0005) | 0.82 (0.0030) | 0.80 (0.0100) |
| dimensions | number_reference | qwen2.5-1.5b__random | -0.21 | -0.07 | 0.81 | 0.82 (0.87) | 0.25 (0.8151) | 0.23 (0.4758) | -0.84 (0.9535) |
| dimensions | number_reference | qwen2.5-math-1.5b | 0.68 | 0.39 | 2.16 | 1.05 (0.32) | 0.83 (0.0005) | 0.96 (0.0005) | 0.87 (0.0033) |
| dimensions | number_reference | qwen2.5-math-1.5b__random | -0.21 | -0.07 | 0.80 | 0.81 (0.86) | 0.25 (0.8416) | 0.21 (0.5317) | -0.90 (0.9934) |
| angles | circle | qwen3.5-0.8b | 0.62 | 0.31 | 2.59 | 1.19 (0.33) | 0.73 (0.0005) | 0.93 (0.0005) | 0.56 (0.0033) |
| angles | circle | qwen3.5-0.8b__random | 0.10 | 0.15 | 1.01 | 1.13 (0.83) | 0.28 (0.7626) | 0.00 (1.0000) | -0.42 (0.4219) |
| angles | circle | qwen2.5-1.5b | 0.60 | 0.29 | 2.15 | 1.06 (0.42) | 0.72 (0.0005) | 0.94 (0.0005) | 0.90 (0.0033) |
| angles | circle | qwen2.5-1.5b__random | 0.15 | 0.31 | 1.16 | 1.17 (0.64) | 0.33 (0.5277) | 0.05 (0.8866) | 0.34 (0.1163) |
| angles | circle | qwen2.5-math-1.5b | 0.62 | 0.31 | 1.84 | 1.01 (0.44) | 0.71 (0.0005) | 0.94 (0.0005) | 0.92 (0.0033) |
| angles | circle | qwen2.5-math-1.5b__random | 0.17 | 0.34 | 1.16 | 1.16 (0.64) | 0.31 (0.6197) | 0.05 (0.8866) | 0.27 (0.1628) |

| Lattice | Model | axiom-RDM β (p) | lexical β | offset consistency per axiom: obs / null95 (p) | analogy top-1 (chance) | mean rank (chance) |
|---|---|---|---|---|---|---|
| group_like | qwen3.5-0.8b | +0.281 (0.0048) | +0.348 | associative: 0.47/0.24 (0.0035); identity: 0.22/0.24 (0.0570); divisible: 0.26/0.24 (0.0430); commutative: 0.57/0.20 (0.0010) | 0.51 (0.10) | 2.8 (5.5) |
| self_distributive | qwen3.5-0.8b | +0.241 (0.2915) | -0.252 | invertible: 0.42/0.74 (0.2639); idempotent: 0.16/0.74 (0.4338) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | qwen3.5-0.8b | — | — | gender: 0.26/0.09 | king:queen rank 1 | — |
| group_like | qwen3.5-0.8b__random | +0.272 (0.0026) | -0.324 | associative: 0.06/0.17 (0.2359); identity: 0.07/0.16 (0.1929); divisible: 0.05/0.17 (0.2739); commutative: 0.02/0.14 (0.3693) | 0.38 (0.10) | 3.5 (5.5) |
| self_distributive | qwen3.5-0.8b__random | -0.037 (0.5965) | +0.534 | invertible: -0.00/0.56 (0.5127); idempotent: -0.02/0.56 (0.5197) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | qwen3.5-0.8b__random | — | — | gender: -0.01/0.07 | king:queen rank 16 | — |
| group_like | qwen2.5-1.5b | +0.210 (0.0144) | +0.457 | associative: 0.38/0.22 (0.0055); identity: 0.27/0.22 (0.0310); divisible: 0.25/0.23 (0.0410); commutative: 0.41/0.19 (0.0025) | 0.47 (0.10) | 3.0 (5.5) |
| self_distributive | qwen2.5-1.5b | +0.388 (0.1418) | -0.414 | invertible: 0.43/0.75 (0.2604); idempotent: 0.15/0.75 (0.4128) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | qwen2.5-1.5b | — | — | gender: 0.24/0.08 | king:queen rank 1 | — |
| group_like | qwen2.5-1.5b__random | +0.255 (0.0026) | +0.475 | associative: 0.12/0.17 (0.1204); identity: 0.10/0.17 (0.1354); divisible: 0.10/0.18 (0.1544); commutative: 0.16/0.15 (0.0470) | 0.47 (0.10) | 3.3 (5.5) |
| self_distributive | qwen2.5-1.5b__random | +0.191 (0.2392) | +0.345 | invertible: -0.02/0.61 (0.5617); idempotent: -0.03/0.61 (0.5902) | 0.00 (0.50) | 2.0 (1.5) |
| gender control | qwen2.5-1.5b__random | — | — | gender: 0.00/0.07 | king:queen rank 14 | — |
| group_like | qwen2.5-math-1.5b | +0.498 (0.0002) | +0.325 | associative: 0.35/0.22 (0.0090); identity: 0.27/0.22 (0.0265); divisible: 0.30/0.22 (0.0195); commutative: 0.45/0.20 (0.0010) | 0.53 (0.10) | 2.6 (5.5) |
| self_distributive | qwen2.5-math-1.5b | +0.433 (0.0812) | -0.661 | invertible: 0.40/0.80 (0.2814); idempotent: 0.03/0.80 (0.4923) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | qwen2.5-math-1.5b | — | — | gender: 0.25/0.08 | king:queen rank 1 | — |
| group_like | qwen2.5-math-1.5b__random | +0.249 (0.0058) | +0.400 | associative: 0.09/0.17 (0.1574); identity: 0.09/0.17 (0.1494); divisible: 0.10/0.18 (0.1664); commutative: 0.15/0.15 (0.0495) | 0.44 (0.10) | 3.4 (5.5) |
| self_distributive | qwen2.5-math-1.5b__random | +0.258 (0.1436) | +0.323 | invertible: -0.01/0.61 (0.5402); idempotent: -0.01/0.61 (0.5212) | 0.00 (0.50) | 2.0 (1.5) |
| gender control | qwen2.5-math-1.5b__random | — | — | gender: -0.00/0.07 | king:queen rank 17 | — |

| Family | qwen3.5-0.8b | qwen3.5-0.8b__random | qwen2.5-1.5b | qwen2.5-1.5b__random | qwen2.5-math-1.5b | qwen2.5-math-1.5b__random |
|---|---|---|---|---|---|---|
| days | 1.00 (p=0.0080) | 0.43 (p=0.1876) | 1.00 (p=0.0040) | 0.29 (p=0.6168) | 1.00 (p=0.0140) | 0.29 (p=0.6048) |
| months | 0.75 (p=0.0020) | 0.17 (p=0.5269) | 1.00 (p=0.0020) | 0.08 (p=0.9242) | 1.00 (p=0.0020) | 0.17 (p=0.5409) |
| ring_chain | 0.36 (p=0.0539) | 0.36 (p=0.0599) | 0.18 (p=0.5509) | 0.18 (p=0.5030) | 0.45 (p=0.0200) | 0.18 (p=0.5828) |
| space_chain | 0.18 (p=0.5150) | 0.36 (p=0.0539) | 0.55 (p=0.0040) | 0.09 (p=0.9122) | 0.64 (p=0.0020) | 0.18 (p=0.5549) |
| number_systems | 0.29 (p=0.6188) | 0.29 (p=0.6367) | 0.29 (p=0.6567) | 0.57 (p=0.1357) | 0.43 (p=0.1776) | 0.29 (p=0.6028) |
| ngons | 0.40 (p=0.0599) | 0.30 (p=0.2116) | 0.40 (p=0.0539) | 0.20 (p=0.5110) | 0.40 (p=0.0599) | 0.10 (p=0.9261) |
| dimensions | 0.55 (p=0.0080) | 0.09 (p=0.8942) | 0.73 (p=0.0020) | 0.09 (p=0.9082) | 1.00 (p=0.0020) | 0.00 (p=1.0000) |
| angles | 0.33 (p=0.0559) | 0.08 (p=0.8922) | 0.33 (p=0.0579) | 0.25 (p=0.2076) | 0.25 (p=0.1836) | 0.25 (p=0.2136) |

| Chain | Model | ρ(order, Zipf) | ρ(order, token count) | LOO order beyond covariates (null95, p) |
|---|---|---|---|---|
| ring_chain | qwen3.5-0.8b | -0.04 | +0.29 | +0.75 (0.51, 0.0066) |
| ring_chain | qwen3.5-0.8b__random | -0.04 | +0.29 | +0.50 (0.59, 0.0963) |
| ring_chain | qwen2.5-1.5b | -0.04 | +0.29 | +0.79 (0.71, 0.0233) |
| ring_chain | qwen2.5-1.5b__random | -0.04 | +0.29 | +0.71 (0.65, 0.0365) |
| ring_chain | qwen2.5-math-1.5b | -0.04 | +0.29 | +0.97 (0.70, 0.0033) |
| ring_chain | qwen2.5-math-1.5b__random | -0.04 | +0.29 | +0.71 (0.71, 0.0565) |
| space_chain | qwen3.5-0.8b | -0.10 | -0.05 | +0.85 (0.61, 0.0033) |
| space_chain | qwen3.5-0.8b__random | -0.10 | -0.05 | +0.22 (0.57, 0.2724) |
| space_chain | qwen2.5-1.5b | -0.10 | -0.08 | +0.89 (0.64, 0.0066) |
| space_chain | qwen2.5-1.5b__random | -0.10 | -0.08 | +0.58 (0.57, 0.0532) |
| space_chain | qwen2.5-math-1.5b | -0.10 | -0.08 | +0.84 (0.65, 0.0133) |
| space_chain | qwen2.5-math-1.5b__random | -0.10 | -0.08 | +0.59 (0.57, 0.0532) |
| number_systems | qwen3.5-0.8b | -0.57 | +0.80 | +0.21 (0.82, 0.3887) |
| number_systems | qwen3.5-0.8b__random | -0.57 | +0.80 | -0.39 (0.86, 0.7043) |
| number_systems | qwen2.5-1.5b | -0.57 | +0.80 | +0.18 (0.93, 0.3920) |
| number_systems | qwen2.5-1.5b__random | -0.57 | +0.80 | +0.43 (0.89, 0.2392) |
| number_systems | qwen2.5-math-1.5b | -0.57 | +0.80 | +0.18 (0.89, 0.3854) |
| number_systems | qwen2.5-math-1.5b__random | -0.57 | +0.80 | +0.04 (0.86, 0.5316) |

| Model | associative quasigroup: d(·, group) / d(·, quasigroup) (3 nearest) | inverse semigroup: 3 nearest | identity direction → lattice-like pairs: cos (null95, p) |
|---|---|---|---|
| qwen3.5-0.8b | 3.36 (commutative quasigroup, quasigroup, commutative semigroup) | commutative semigroup, semigroup, commutative monoid | +0.027 (0.185, 0.3248) |
| qwen3.5-0.8b__random | 1.28 (quasigroup, semigroup, inverse semigroup) | semigroup, quasigroup, associative quasigroup | -0.018 (0.153, 0.6817) |
| qwen2.5-1.5b | 2.49 (commutative quasigroup, quasigroup, commutative semigroup) | semigroup, commutative semigroup, commutative monoid | +0.072 (0.183, 0.1519) |
| qwen2.5-1.5b__random | 1.71 (quasigroup, commutative quasigroup, semigroup) | semigroup, quasigroup, commutative semigroup | -0.001 (0.130, 0.5022) |
| qwen2.5-math-1.5b | 2.48 (commutative quasigroup, quasigroup, commutative loop) | semigroup, commutative semigroup, commutative monoid | +0.133 (0.167, 0.0770) |
| qwen2.5-math-1.5b__random | 1.67 (quasigroup, commutative quasigroup, semigroup) | semigroup, quasigroup, commutative semigroup | -0.012 (0.130, 0.6662) |

| Model | Variant | offset consistency per axiom: obs / null95 (p) |
|---|---|---|
| qwen3.5-0.8b | group-like, mean-pooled tokens | associative: 0.26 / 0.22 (0.0305); identity: 0.29 / 0.22 (0.0220); divisible: 0.17 / 0.24 (0.1000); commutative: 0.53 / 0.19 (0.0005) |
| qwen3.5-0.8b | group-like, token-count direction removed | associative: 0.46 / 0.24 (0.0040); identity: 0.25 / 0.23 (0.0400); divisible: 0.29 / 0.23 (0.0330); commutative: 0.48 / 0.20 (0.0015) |
| qwen3.5-0.8b | partial-operation lattice | total: 0.28 / 0.52 (0.1714); identity: -0.10 / 0.78 (0.5617); divisible: -0.19 / 0.78 (0.5897) |
| qwen3.5-0.8b__random | group-like, mean-pooled tokens | associative: 0.04 / 0.17 (0.3123); identity: 0.05 / 0.16 (0.2549); divisible: 0.05 / 0.18 (0.2819); commutative: 0.13 / 0.15 (0.0675) |
| qwen3.5-0.8b__random | group-like, token-count direction removed | associative: 0.06 / 0.17 (0.2339); identity: 0.07 / 0.16 (0.1914); divisible: 0.05 / 0.17 (0.2764); commutative: -0.01 / 0.14 (0.5067) |
| qwen3.5-0.8b__random | partial-operation lattice | total: -0.08 / 0.42 (0.5427); identity: -0.09 / 0.60 (0.6412); divisible: -0.11 / 0.62 (0.6502) |
| qwen2.5-1.5b | group-like, mean-pooled tokens | associative: 0.23 / 0.21 (0.0355); identity: 0.23 / 0.22 (0.0440); divisible: 0.14 / 0.24 (0.1289); commutative: 0.51 / 0.19 (0.0010) |
| qwen2.5-1.5b | group-like, token-count direction removed | associative: 0.38 / 0.23 (0.0065); identity: 0.29 / 0.22 (0.0245); divisible: 0.27 / 0.23 (0.0335); commutative: 0.36 / 0.19 (0.0055) |
| qwen2.5-1.5b | partial-operation lattice | total: 0.24 / 0.52 (0.2034); identity: -0.08 / 0.77 (0.5557); divisible: -0.22 / 0.77 (0.6062) |
| qwen2.5-1.5b__random | group-like, mean-pooled tokens | associative: 0.08 / 0.19 (0.2129); identity: 0.09 / 0.18 (0.1694); divisible: 0.08 / 0.20 (0.2104); commutative: 0.25 / 0.16 (0.0180) |
| qwen2.5-1.5b__random | group-like, token-count direction removed | associative: 0.12 / 0.17 (0.1179); identity: 0.10 / 0.17 (0.1319); divisible: 0.11 / 0.18 (0.1474); commutative: 0.14 / 0.15 (0.0645) |
| qwen2.5-1.5b__random | partial-operation lattice | total: -0.07 / 0.45 (0.5347); identity: 0.01 / 0.61 (0.4818); divisible: -0.09 / 0.67 (0.5677) |
| qwen2.5-math-1.5b | group-like, mean-pooled tokens | associative: 0.21 / 0.21 (0.0520); identity: 0.21 / 0.21 (0.0580); divisible: 0.18 / 0.24 (0.0895); commutative: 0.54 / 0.20 (0.0005) |
| qwen2.5-math-1.5b | group-like, token-count direction removed | associative: 0.35 / 0.22 (0.0105); identity: 0.29 / 0.22 (0.0200); divisible: 0.31 / 0.22 (0.0175); commutative: 0.42 / 0.20 (0.0020) |
| qwen2.5-math-1.5b | partial-operation lattice | total: 0.23 / 0.51 (0.2004); identity: -0.06 / 0.75 (0.5382); divisible: -0.15 / 0.75 (0.5597) |
| qwen2.5-math-1.5b__random | group-like, mean-pooled tokens | associative: 0.06 / 0.19 (0.2464); identity: 0.08 / 0.18 (0.1929); divisible: 0.07 / 0.20 (0.2229); commutative: 0.24 / 0.16 (0.0190) |
| qwen2.5-math-1.5b__random | group-like, token-count direction removed | associative: 0.10 / 0.17 (0.1529); identity: 0.09 / 0.16 (0.1484); divisible: 0.10 / 0.18 (0.1579); commutative: 0.14 / 0.15 (0.0660) |
| qwen2.5-math-1.5b__random | partial-operation lattice | total: -0.07 / 0.45 (0.5322); identity: 0.00 / 0.62 (0.4898); divisible: -0.09 / 0.67 (0.5672) |

**Gemma-4-E2B** with its twin:

| Family | Expected | Model | line ρ | circle ρ | closure (d first-last / adjacent) | closure_far† (null 5%) | Robinson (p) | PCA order \|ρ\| (p) | LOO order ρ (p) |
|---|---|---|---|---|---|---|---|---|---|
| days | circle | gemma-4-e2b | 0.72 | 0.58 | 1.65 | 1.02 (0.37) | 0.71 (0.0055) | 0.64 (0.1334) | 0.75 (0.0332) |
| days | circle | gemma-4-e2b__random | -0.23 | 0.00 | 0.96 | 0.96 (0.94) | 0.23 (0.8446) | 0.61 (0.1789) | -0.96 (0.9302) |
| months | circle | gemma-4-e2b | 0.61 | 0.36 | 1.99 | 0.97 (0.53) | 0.69 (0.0005) | 0.66 (0.0240) | 0.86 (0.0033) |
| months | circle | gemma-4-e2b__random | 0.27 | 0.05 | 1.02 | 1.05 (0.96) | 0.45 (0.0230) | 0.13 (0.6927) | -0.51 (0.2292) |
| ring_chain | line | gemma-4-e2b | 0.27 | 0.08 | 1.86 | 1.15 (0.53) | 0.50 (0.0100) | 0.42 (0.1999) | 0.43 (0.1030) |
| ring_chain | line | gemma-4-e2b__random | 0.37 | 0.40 | 1.45 | 0.98 (0.45) | 0.28 (0.7816) | 0.05 (0.8821) | 0.50 (0.0332) |
| space_chain | line | gemma-4-e2b | 0.55 | 0.27 | 1.39 | 1.21 (0.64) | 0.57 (0.0005) | 0.95 (0.0005) | 0.82 (0.0133) |
| space_chain | line | gemma-4-e2b__random | -0.06 | -0.00 | 1.00 | 0.97 (0.81) | 0.27 (0.7916) | 0.16 (0.6307) | -0.22 (0.2890) |
| number_systems | line | gemma-4-e2b | 0.92 | 0.54 | 1.94 | 1.56 (0.52) | 0.91 (0.0010) | 1.00 (0.0010) | 0.89 (0.0033) |
| number_systems | line | gemma-4-e2b__random | 0.24 | -0.09 | 1.36 | 2.39 (0.41) | 0.60 (0.0080) | 0.71 (0.0815) | 0.32 (0.1196) |
| ngons | number_reference | gemma-4-e2b | 0.50 | 0.21 | 1.90 | 1.00 (0.49) | 0.64 (0.0015) | 0.87 (0.0025) | 0.89 (0.0033) |
| ngons | number_reference | gemma-4-e2b__random | 0.19 | 0.05 | 1.23 | 1.01 (0.47) | 0.51 (0.0120) | 0.52 (0.1349) | 0.42 (0.0432) |
| dimensions | number_reference | gemma-4-e2b | 0.62 | 0.42 | 2.37 | 1.09 (0.24) | 0.65 (0.0010) | 0.91 (0.0010) | 0.86 (0.0033) |
| dimensions | number_reference | gemma-4-e2b__random | -0.25 | -0.02 | 0.79 | 0.69 (0.74) | 0.23 (0.9040) | 0.12 (0.7451) | -0.57 (0.6013) |
| angles | circle | gemma-4-e2b | 0.43 | 0.23 | 2.03 | 1.03 (0.37) | 0.54 (0.0010) | 0.48 (0.1184) | 0.87 (0.0033) |
| angles | circle | gemma-4-e2b__random | 0.25 | 0.46 | 1.14 | 0.94 (0.77) | 0.39 (0.1914) | 0.36 (0.2599) | 0.73 (0.0066) |

| Lattice | Model | axiom-RDM β (p) | lexical β | offset consistency per axiom: obs / null95 (p) | analogy top-1 (chance) | mean rank (chance) |
|---|---|---|---|---|---|---|
| group_like | gemma-4-e2b | +0.301 (0.0018) | +0.437 | associative: 0.56/0.26 (0.0020); identity: 0.19/0.25 (0.0900); divisible: 0.35/0.26 (0.0230); commutative: 0.74/0.22 (0.0005) | 0.56 (0.10) | 2.6 (5.5) |
| self_distributive | gemma-4-e2b | +0.080 (0.4149) | -0.084 | invertible: -0.11/0.84 (0.6112); idempotent: 0.23/0.84 (0.3468) | 0.00 (0.50) | 2.0 (1.5) |
| gender control | gemma-4-e2b | — | — | gender: 0.11/0.10 | king:queen rank 6 | — |
| group_like | gemma-4-e2b__random | +0.275 (0.0054) | -0.017 | associative: 0.22/0.21 (0.0475); identity: 0.14/0.20 (0.0980); divisible: 0.23/0.21 (0.0425); commutative: 0.02/0.19 (0.3683) | 0.47 (0.10) | 3.4 (5.5) |
| self_distributive | gemma-4-e2b__random | +0.087 (0.4553) | +0.146 | invertible: -0.05/0.55 (0.6337); idempotent: 0.00/0.55 (0.4763) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | gemma-4-e2b__random | — | — | gender: -0.00/0.07 | king:queen rank 1 | — |

| Family | gemma-4-e2b | gemma-4-e2b__random |
|---|---|---|
| days | 1.00 (p=0.0080) | 0.00 (p=1.0000) |
| months | 0.25 (p=0.1796) | 0.17 (p=0.5349) |
| ring_chain | 0.18 (p=0.5729) | 0.18 (p=0.5609) |
| space_chain | 0.55 (p=0.0060) | 0.18 (p=0.5489) |
| number_systems | 0.57 (p=0.0818) | 0.14 (p=0.9301) |
| ngons | 0.30 (p=0.2176) | 0.10 (p=0.9182) |
| dimensions | 0.45 (p=0.0140) | 0.18 (p=0.5509) |
| angles | 0.25 (p=0.2176) | 0.17 (p=0.5549) |

| Chain | Model | ρ(order, Zipf) | ρ(order, token count) | LOO order beyond covariates (null95, p) |
|---|---|---|---|---|
| ring_chain | gemma-4-e2b | -0.04 | +0.38 | +0.44 (0.66, 0.1728) |
| ring_chain | gemma-4-e2b__random | -0.04 | +0.38 | +0.68 (0.58, 0.0266) |
| space_chain | gemma-4-e2b | -0.10 | -0.39 | +0.75 (0.76, 0.0565) |
| space_chain | gemma-4-e2b__random | -0.10 | -0.39 | -0.37 (0.69, 0.8173) |
| number_systems | gemma-4-e2b | -0.57 | +0.67 | +0.96 (0.89, 0.0266) |
| number_systems | gemma-4-e2b__random | -0.57 | +0.67 | -0.61 (0.88, 0.7907) |

| Model | associative quasigroup: d(·, group) / d(·, quasigroup) (3 nearest) | inverse semigroup: 3 nearest | identity direction → lattice-like pairs: cos (null95, p) |
|---|---|---|---|
| gemma-4-e2b | 2.03 (commutative quasigroup, quasigroup, commutative monoid) | commutative semigroup, semigroup, quasigroup | +0.122 (0.160, 0.0870) |
| gemma-4-e2b__random | 2.29 (quasigroup, commutative quasigroup, monoid) | semigroup, commutative semigroup, magma | -0.023 (0.161, 0.7111) |

| Model | Variant | offset consistency per axiom: obs / null95 (p) |
|---|---|---|
| gemma-4-e2b | group-like, mean-pooled tokens | associative: 0.41 / 0.24 (0.0075); identity: 0.17 / 0.24 (0.0915); divisible: 0.16 / 0.25 (0.1139); commutative: 0.59 / 0.21 (0.0005) |
| gemma-4-e2b | group-like, token-count direction removed | associative: 0.56 / 0.26 (0.0020); identity: 0.20 / 0.25 (0.0830); divisible: 0.37 / 0.25 (0.0185); commutative: 0.71 / 0.22 (0.0005) |
| gemma-4-e2b | partial-operation lattice | total: 0.41 / 0.56 (0.1184); identity: 0.18 / 0.84 (0.4068); divisible: 0.03 / 0.84 (0.4688) |
| gemma-4-e2b__random | group-like, mean-pooled tokens | associative: 0.17 / 0.20 (0.0855); identity: 0.15 / 0.20 (0.0925); divisible: 0.22 / 0.21 (0.0410); commutative: 0.25 / 0.17 (0.0190) |
| gemma-4-e2b__random | group-like, token-count direction removed | associative: 0.22 / 0.21 (0.0470); identity: 0.15 / 0.20 (0.0940); divisible: 0.23 / 0.21 (0.0415); commutative: 0.02 / 0.18 (0.3673) |
| gemma-4-e2b__random | partial-operation lattice | total: -0.11 / 0.55 (0.5847); identity: -0.33 / 0.84 (0.7111); divisible: -0.39 / 0.84 (0.7461) |

Matched 4-field clustering in Gemma-4-E2B:
- math subsets: mean 0.663;
- control fields: 0.866.

# Appendix B: Deviations from the preregistration

See `docs/deviations.md`. Every post-hoc statistic in this paper is labelled post-hoc.

# AI statement

This codebase and paper were written by Claude Code (Anthropic's Claude, model Opus 5.5) acting as a coding and research agent. A human, Libor Burian, provided the exact specification that was executed. Human cleanup and rewriting are in progress.
