---
title: "Do LLMs Organize Mathematics Geometrically? Field Maps, Ordered Chains and Axiom Lattices in Residual Streams"
author: "Libor \"Burny\" Burian"
date: "October 2026"
abstract: |
  Large language models represent days of the week and months on circles (Engels et al., 2024). We ask whether *abstract mathematics* is organized geometrically too. We test three preregistered hypotheses on base models up to 2B parameters, comparing every model with a random-init twin (same architecture and tokenizer, no learning).

  **(A) Field map.** Centroids of 17 mathematical fields are computed from concepts drawn from Wikipedia glossaries, excluding concepts whose names leak the field. In Qwen3.5-2B-Base, their geometry correlates with field relatedness measured by arXiv cross-listing over 3.2M papers: Spearman ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.2f}}, against ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.random_init:.2f}} for the random-init twin. The paired bootstrap 95% CI of the difference is [{{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.0:.2f}}, {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.1:.2f}}]. The map replicates in Qwen3.5-0.8B (ρ = {{num:robustness.map_fields.qwen3.5-0.8b.math_primary.rsa.trained:.2f}}), Qwen2.5-1.5B (ρ = {{num:robustness.map_fields.qwen2.5-1.5b.math_primary.rsa.trained:.2f}}) and Gemma-4-E2B (ρ = {{num:gemma.map_fields.gemma-4-e2b.math_primary.rsa.trained:.2f}}). In every case the random-init twin is near zero or below. It is, however, no stronger than the organization of non-math science topics, and no stronger than in a contrastive text embedder (Qwen3-Embedding). It is also not a clean hierarchy: our preregistered cophenetic test does not beat the random-init twin. One trillion tokens of math continued pretraining (Qwen2.5-Math) do not sharpen it (Δρ CI [{{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.0:.2f}}, {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.1:.2f}}]).

  **(B) Shapes.** Class-inclusion chains (rng ⊃ ring ⊃ … ⊃ field; T0 ⊃ … ⊃ Hilbert) are ordered beyond permutation nulls and beyond the twin. Angles 0°–330°, however, are encoded like the number line rather than a circle in all five models, which falsifies our preregistered prediction.

  **(B2) Axiom lattices.** For group-like structures (magma … abelian group), adding an axiom behaves like a consistent direction. Same-axiom difference vectors are more parallel than random pairs, and much less so in the random-init twins. This holds clearly for associativity and commutativity, which survive Holm correction in the confirmatory model and are significant in all five models. The evidence is weaker for divisibility (raw p < .05 in all five models, but not after Holm correction) and for identity (raw p < .05 in three of five models).

  The days/months positive control met our preregistered full-space criterion only partially. A post-hoc test in the top-2 principal-component plane recovers perfect cycles. Simulations show why we use paired random-init contrasts as primary tests: partial regression on distance matrices inflates type-I error when nuisances are proxies of confounders.
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
- **Split.** A 20% pilot split was used only for debugging. All statistics below use the 80% confirmatory split: {{num:mvp.map_fields.qwen3.5-2b.math_primary.n}} primary math concepts across {{num:mvp.map_fields.qwen3.5-2b.math_primary.n_fields}} fields.

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

{{table:calibration:mvp}}

The test is calibrated when the nuisance distance matrix *is* the confounder. It rejects far too often when the nuisance only approximates the confounder, which is exactly the situation for lexical form and frequency. We therefore made **paired contrasts against the random-init twin** the primary tests. The twin measures form-only similarity on the same concepts. Uncertainty comes from a stratified concept bootstrap (2,000 resamples) with identical resamples for both models.

- **Field map.** We use field-level RSA, i.e. the Spearman ρ between the centroid RDM and the arXiv distance, with a Mantel permutation over field labels (10,000 permutations) and a max-over-layers null for layer curves.
- **Families.** Every statistic is unsupervised or leave-one-out, and is tested against an order-permutation null. With n ≤ 13 items in more than 1,000 dimensions, a supervised fit can impose any order. As a negative control, 100 random 11-item lists of math concepts must be rejected at about 5%.

# 4 Results

![**Results at a glance.** Blue: trained model; grey: its random-init twin (same architecture and tokenizer, no learning). The starred model is the preregistered confirmatory model, Qwen3.5-2B. (A) Field-centroid geometry correlates with arXiv cross-listing in all five models (dots: ρ; lines: 95% bootstrap CI); the twins sit at or below zero. (B) The confirmatory model's field map (metric MDS); grey edges join each field to its two nearest arXiv neighbours. (C) Days of the week close into a cycle in the top-2 principal-component plane (positive control). (D) Angles do not; the orange edge is the 330° → 0° closure the preregistration predicted. (E) Offset consistency per axiom: the confirmatory model, the other four models and the five twins, against the random-pair null (grey bar: mean to 95th percentile). Diamonds show the confirmatory model with the token-count direction removed. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#s-glance>.]({{fig:hero}}){width=100%}

**Every test in every model.** The table below summarizes the confirmatory and secondary tests for all five models.

{{table:scorecard:all}}

![**Effect sizes with 95% bootstrap CIs (unadjusted).** Blue: CI excludes 0. "Added" marks analyses added by the plan audits rather than the preregistration. The field map beats the twin in every model and beats fastText, but not the Qwen3-Embedding text embedder. Math pretraining (H-A3) and the hierarchy test (H-A2) are nulls. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-forest>.]({{fig:forest}}){width=82%}

## 4.1 H-A1: a map of mathematical fields

{{table:field_map:mvp}}

![**The map is present at almost every depth, in every model.** ρ between the field-centroid RDM and the arXiv distance at each layer (x: relative depth). Blue: trained; grey: twin. The horizontal line is the 95th percentile of the max-over-layers permutation null; the vertical line marks the preregistered layer. The Qwen models clear the null at every layer; Gemma-4-E2B dips below it in two stretches. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-depth>.]({{fig:depth_curves}}){width=100%}

**The trained model's field geometry tracks arXiv relatedness, and the random-init twin's does not.** For primary concepts at the preregistered layer, ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}} with Mantel p = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_perm_p:.4f}}. The twin gives ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.random_init:.3f}} and layer-0 embeddings give ρ = {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.layer0:.3f}}. The difference from the twin has 95% CI [{{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.0:.3f}}, {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.1:.3f}}].

**Layer profile.** The map emerges in the first blocks and peaks at layer {{num:mvp.map_fields.qwen3.5-2b.math_primary.layer_curve.best_layer}}. It stays above the 95% max-over-layers null ({{num:mvp.map_fields.qwen3.5-2b.math_primary.layer_curve.max_null95:.2f}}) at every layer.

**Clustering.** Leave-one-out nearest-centroid field accuracy is {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.trained.acc:.3f}}, against {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.random_init.acc:.3f}} for the twin and {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.chance:.3f}} chance.

**Reliability.** Split-half template reliability of the field RDM is {{num:mvp.map_fields.qwen3.5-2b.math_primary.split_half:.3f}}.

![**Related fields stay close in the trained model, and the twin scrambles them.** Metric MDS of three 17 × 17 distance matrices: arXiv cross-listing (ground truth), Qwen3.5-2B field centroids at layer {{num:mvp.map_fields.qwen3.5-2b.primary_layer}}, and the random-init twin. Panels are rotated or reflected onto the arXiv layout for display only. Grey edges join each field to its two nearest arXiv neighbours and are drawn in every panel: short edges mean the neighbourhood is preserved. Blue: pure; orange: applied. ρ is computed on the full matrices, not on the 2-D layout. Interactive version, with a layer slider: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-map>.]({{fig:field_maps_qwen3.5-2b}}){width=100%}

![**The same matrices, ranked.** Colour is the rank of each distance within its matrix (Spearman RSA compares ranks), with rows in one fixed order grouped by the five super-tree branches. The trained matrix partly reproduces the arXiv block structure; the twin's does not. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-rdm>.]({{fig:field_rdms_qwen3.5-2b}}){width=100%}

**Not a rogue-dimension artifact (post-hoc).** The preregistered all-but-the-top variant removes the top d/100 = 20 principal components, and it reverses the correlation (ρ = {{num:mvp.posthoc.qwen3.5-2b.abtt_sweep.20:.2f}}). A post-hoc sweep explains why:

{{table:posthoc:mvp}}

- Removing the top component *raises* ρ to {{num:mvp.posthoc.qwen3.5-2b.abtt_sweep.1:.2f}}.
- No component dominates.
- Field identity is spread over the top ~10 components, which carry far more field variance (η²) than the twin's.

So the map lives in the dominant semantic subspace, and removing 20 components removes the map itself.

**H-A2: hierarchy is not supported as preregistered.** The preregistered measure is the cophenetic correlation between an average-linkage (UPGMA) dendrogram of the field centroids and the five-branch super-tree.

{{table:hierarchy:mvp}}

{{table:hierarchy:robustness}}

{{table:hierarchy:gemma}}

- **Against shuffled trees:** the trained model's dendrogram matches the super-tree better (ρ = {{num:mvp.hierarchy.qwen3.5-2b.coph_match.trained:.3f}}, p = {{num:mvp.hierarchy.qwen3.5-2b.coph_match_p_shuffled_tree:.3f}}).
- **Against the twin:** the paired difference CI includes zero ([{{num:mvp.hierarchy.qwen3.5-2b.coph_match_delta_vs_random_ci.0:.3f}}, {{num:mvp.hierarchy.qwen3.5-2b.coph_match_delta_vs_random_ci.1:.3f}}]), and the same holds in every replication model. The preregistered criterion requires beating both, so it fails.
- **Intrinsic tree-likeness:** the cophenetic correlation of the representation with its own dendrogram is no higher than for a spectrum-matched Gaussian cloud.
- **Gromov δ:** relative δ is lower than the Gaussian cloud's, but higher than the twin's.

A tree-distance RSA we first reported, the Spearman correlation of the centroid RDM with the super-tree distance, did beat the twin (trained {{num:mvp.map_fields.qwen3.5-2b.math_primary.hierarchy.rho.trained:.3f}} vs twin {{num:mvp.map_fields.qwen3.5-2b.math_primary.hierarchy.rho.random_init:.3f}}). It is *not* the preregistered measure, so it is post-hoc. It largely restates H-A1: fields that mathematicians relate are close.

**Specificity.** The non-math control fields (biology, chemistry, geology, astronomy) are also clustered by the trained model and not by its twin:

{{table:clustering:mvp}}

The pre-registered comparison confounds domain with the number of fields: 17 for math, 4 for the control. We therefore ran a post-hoc matched comparison over 200 random 4-field subsets of the math fields:
- **Math subsets:** mean leave-one-out accuracy {{num:mvp.posthoc.qwen3.5-2b.matched_clustering.math_subsets_mean:.3f}} (95% range {{num:mvp.posthoc.qwen3.5-2b.matched_clustering.math_subsets_ci.0:.3f}}–{{num:mvp.posthoc.qwen3.5-2b.matched_clustering.math_subsets_ci.1:.3f}}).
- **Control fields:** {{num:mvp.posthoc.qwen3.5-2b.matched_clustering.control_acc:.3f}}.
- **Overlap:** {{num:mvp.posthoc.qwen3.5-2b.matched_clustering.frac_math_subsets_ge_control:.2f}} of the math subsets reach or exceed the control value.

Mathematical fields are therefore organized about as strongly as general science topics, and not more. The field map is an instance of topical organization, not something specific to mathematics.

**Baselines (plan; added after the first report).** We compare against two non-generative representations of the same concepts:
- **fastText** wiki-news vectors (averaged over words) and **Qwen3-Embedding-0.6B**, a contrastively trained text embedder.
- **vs fastText:** the LLM's field map is stronger (Δ CI [{{num:mvp.baselines.qwen3.5-2b.fasttext_covered.boot.delta_vs_fasttext.ci.0:.2f}}, {{num:mvp.baselines.qwen3.5-2b.fasttext_covered.boot.delta_vs_fasttext.ci.1:.2f}}] on the {{num:mvp.baselines.qwen3.5-2b.fasttext_covered.n}} concepts that fastText covers).
- **vs Qwen3-Embedding:** the LLM is not stronger; the embedder reaches ρ = {{num:mvp.baselines.qwen3.5-2b.all_primary.rsa.qwen3_embedding:.3f}} against the LLM's {{num:mvp.baselines.qwen3.5-2b.all_primary.rsa.llm_trained:.3f}} (Δ CI [{{num:mvp.baselines.qwen3.5-2b.all_primary.boot.delta_vs_qwen3_embedding.ci.0:.2f}}, {{num:mvp.baselines.qwen3.5-2b.all_primary.boot.delta_vs_qwen3_embedding.ci.1:.2f}}]).

So the field map is a property of contextual text representations in general, not something specific to generative LLMs.

{{table:baselines:mvp}}

## 4.2 Robustness: scale, model family, math training

{{table:field_map:robustness}}

{{table:field_map:gemma}}

**H-A3 (math training): null.** Qwen2.5-Math-1.5B is Qwen2.5-1.5B plus 1T tokens of math. Its field map is no stronger than its base's: ρ = {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.rho.qwen2.5-math-1.5b:.3f}} vs {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.rho.qwen2.5-1.5b:.3f}}, and the paired CI of the difference is [{{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.0:.3f}}, {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.1:.3f}}].

## 4.3 H-B: shapes of ordered families

{{table:shapes:mvp}}

![**Days and months close into cycles; angles do not.** Top-2 principal-component plane of each family at the preregistered layer (unsupervised; no labels are used to fit it). Colour runs light to dark along the true order; the grey path joins consecutive items. For angles, the orange edge is the 330° → 0° closure the preregistration predicted. Below each panel: the post-hoc PC-plane cycle score (1 = perfect cycle) with its order-permutation p. Bottom row: the random-init twin. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:cyclic_qwen3.5-2b}}){width=82%}

![**The cycles are present at almost every depth.** Days (top) and months (bottom) at fixed relative depths, each plane rotated onto the preregistered layer's plane for display, plus the cycle score at every layer for the trained model and the twin (right). Post-hoc and unadjusted. Animated version: `reports/figures/circle_depth_qwen3.5-2b.gif`; interactive: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:circle_depth_qwen3.5-2b}}){width=100%}

**Positive control.** With our preregistered full-space statistics, days and months show significant circle-template correlation and closure ratios near 1. They nevertheless fail "circle ρ > line ρ" at layer {{num:mvp.shapes.qwen3.5-2b.primary_layer}}. Following the prereg, we debugged before interpreting further. In the top-2 principal-component plane, where Engels et al. visualized their circles, days and months form *perfect* cycles in the trained model and none in the twin:

{{table:cycles:mvp}}

This statistic is post-hoc. We conclude that the pipeline works, and that the full-space template criterion was insensitive because variance outside the circle plane dominates.

**Chains.** The ring and space chains are ordered beyond permutation nulls and beyond the twin: Robinson scores {{num:mvp.shapes.qwen3.5-2b.ring_chain.robinson:.2f}} and {{num:mvp.shapes.qwen3.5-2b.space_chain.robinson:.2f}}, against about 1/3 for chance and for the twin. Random concept lists are rejected at the nominal rate:

{{table:negative_control:mvp}}

![**Class-inclusion chains are ordered paths; the twin's are tangles.** PC planes of the ordered families at the preregistered layer. Below each panel: the Robinson score (fraction of item triples consistent with the order; chance 1/3) with its permutation p, and for the ring and space chains the p of the frequency control. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:chains_qwen3.5-2b}}){width=100%}

**Frequency control (plan; added in the second audit).** Could the chain order just be a frequency or word-length order? We removed wordfreq Zipf frequency and model token count from both the representation and the ranks, fitting that regression inside each leave-one-out fold to avoid leakage ([Snoek et al., 2019](https://doi.org/10.1016/j.neuroimage.2018.09.074)), then predicted the residual order.
- *Result.* Both preregistered chains stay ordered: ring chain ρ = {{num:mvp.chain_frequency.qwen3.5-2b.ring_chain.loo_order_beyond:.2f}} (p = {{num:mvp.chain_frequency.qwen3.5-2b.ring_chain.p:.3f}}), space chain ρ = {{num:mvp.chain_frequency.qwen3.5-2b.space_chain.loo_order_beyond:.2f}} (p = {{num:mvp.chain_frequency.qwen3.5-2b.space_chain.p:.3f}}). The twin gives {{num:mvp.chain_frequency.qwen3.5-2b__random.ring_chain.loo_order_beyond:.2f}} and {{num:mvp.chain_frequency.qwen3.5-2b__random.space_chain.loo_order_beyond:.2f}}.
- *Replication (Appendix A).* Both chains stay ordered in all four Qwen models (p ≤ {{num:robustness.chain_frequency.qwen2.5-1.5b.ring_chain.p:.3f}}), but not in Gemma-4-E2B (ring p = {{num:gemma.chain_frequency.gemma-4-e2b.ring_chain.p:.2f}}, space p = {{num:gemma.chain_frequency.gemma-4-e2b.space_chain.p:.3f}}).
- *Word form carries part of the ring-chain order.* Some twins also order the ring chain beyond frequency: Qwen2.5-1.5B's twin (p = {{num:robustness.chain_frequency.qwen2.5-1.5b__random.ring_chain.p:.3f}}) and Gemma's twin (p = {{num:gemma.chain_frequency.gemma-4-e2b__random.ring_chain.p:.3f}}). A plausible lexical cue is that the middle items all end in "domain". The space chain's twins are weaker: none reaches p < .05, though the two Qwen2.5 twins come close (p = {{num:robustness.chain_frequency.qwen2.5-1.5b__random.space_chain.p:.3f}}).
- *Caveat.* The fold-wise version was adopted post-hoc. A first, whole-sample version leaked the held-out item into the training span and was degenerate in the twin. Both versions are reported in `docs/deviations.md`.

{{table:chain_frequency:mvp}}

**Angles: preregistered prediction falsified.** Angles are not circular: closure ratio {{num:mvp.shapes.qwen3.5-2b.angles.closure:.2f}} and no cycle in the PC plane. Their geometry closely matches that of the bare digits 0–330 (ρ = {{num:mvp.shapes.qwen3.5-2b.angles.beyond_reference.rho_fam_ref:.2f}}). At this scale, "330 degrees" sits next to "300 degrees", not next to "0 degrees".

{{table:beyond_reference:mvp}}

![**Predicted vs found for angles.** Left: the preregistered prediction, a circle on which 330° neighbours 0° (orange edge). Middle: Qwen3.5-2B's PC plane, where that edge spans the whole layout. Right: the bare numbers 0–330, whose geometry the angles follow. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.]({{fig:angles_qwen3.5-2b}}){width=90%}

**SMDS (plan, secondary).** Supervised MDS fits a projection to a line or a discrete-circle template, and each fit is tested against refits on shuffled labels.

{{table:smds:mvp}}

- **The twin shows why the shuffle null is necessary.** Its line fit scores {{num:mvp.smds.qwen3.5-2b__random.days.euclidean.score:.2f}} on *every* family, a perfect fit to any ordering, yet never beats its shuffled null.
- **Trained model:**
  - Days prefer the circle template: circle p = {{num:mvp.smds.qwen3.5-2b.days.discrete_circular.p:.3f}}, line p = {{num:mvp.smds.qwen3.5-2b.days.euclidean.p:.3f}}.
  - The chains and angles prefer the line template.
  - Months fit both templates beyond the null.

This agrees with the unsupervised statistics, including the conclusion that angles are line-like.

## 4.4 H-B2: axioms as directions

{{table:lattice:mvp}}

![**Adding an axiom moves structures in a consistent direction in the trained model only.** Left: the ideal geometry, in which each axiom is one fixed step. Other panels: the same unsupervised PCA plane of the 13 group-like structures, one panel per axiom, with the pairs that differ only in that axiom drawn as blue arrows. Parallel arrows mean a consistent direction. Below each panel: offset consistency and its random-pair p. Bottom row: the twin. Interactive version, with an analogy playground: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-lattice>.]({{fig:lattice_axes_qwen3.5-2b}}){width=100%}

![**The direction evidence, unprojected.** Cosine between every pair of single-axiom difference vectors in the full residual space, grouped by axiom. Red diagonal blocks mean a consistent direction. Hatched cells are cross-axiom pairs that share a structure; isotropic noise alone puts these near ±0.5, so they carry no evidence. The self-cosines on the diagonal are masked. Right: offset consistency per axiom for the trained model, the trained model with the token-count direction removed, and the twin, against the random-pair null. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-cosines>.]({{fig:lattice_cosines_qwen3.5-2b}}){width=100%}

For every group-like axiom, difference vectors between structures that differ only in that axiom are more parallel than random pairs (raw p < .05):

| Axiom | Consistency | Null 95th percentile | p |
|---|---|---|---|
| Associativity | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.associative.consistency:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.associative.null95:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.associative.p:.4f}} |
| Identity | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.identity.consistency:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.identity.null95:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.identity.p:.4f}} |
| Divisibility | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.divisible.consistency:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.divisible.null95:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.divisible.p:.4f}} |
| Commutativity | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.consistency:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.null95:.2f}} | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.p:.4f}} |

In this model's random-init twin, all four consistencies are ≈ 0.

**Analogies.** Analogy completion is top-1 for {{num:mvp.lattice.qwen3.5-2b.group_like.analogy.top1:.2f}} of {{num:mvp.lattice.qwen3.5-2b.group_like.analogy.n}} parallelograms (chance {{num:mvp.lattice.qwen3.5-2b.group_like.analogy.chance_top1:.2f}}). The twin reaches {{num:mvp.lattice.qwen3.5-2b__random.group_like.analogy.top1:.2f}}, so part of the analogy accuracy is lexical (shared "commutative" tokens); offset consistency is not.

**Caveats.** The gender-offset positive control passes. Multiplicity across the preregistered confirmatory tests is handled by Holm correction:

{{table:holm:mvp}}

- **After correction:** associativity and commutativity remain significant; identity and divisibility do not.
- **Across the replication models (Appendix A):** all four offsets are positive in every model. They are near zero in the Qwen twins and weaker in Gemma's twin, which has small lexical offsets for associativity and divisibility. Divisibility reaches raw p < .05 in all five models, identity in three.
- **Our reading:** strong evidence for associativity and commutativity directions, and suggestive evidence for divisibility and identity.

**Lie groups and other group objects (exploratory).** A Lie group is a group that is also a smooth manifold; topological and algebraic groups are defined analogously. We tested three parallelograms:
- topological group − group ≈ topological space − set;
- Lie group − topological group ≈ smooth manifold − topological space;
- algebraic group − group ≈ algebraic variety − set.

{{table:structures:mvp}}

{{table:structures:robustness}}

{{table:structures:gemma}}

![**Lie, topological and algebraic groups combine their parents in every trained model.** Left: mean cosine of the three parallelograms (for example, Lie group − topological group vs smooth manifold − topological space) for each model (blue) and its twin (grey), against the 95th percentile of a random-quadruple null (grey tick); p values from that null. Right: the nearest-neighbour rank of "group" (circle) and "smooth manifold" (diamond) among the family items around "Lie group", trained vs twin (log scale). Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-groups>.]({{fig:group_objects}}){width=88%}

**Group objects: supported.**
- The parallelograms hold in all five trained models: mean cosine {{num:mvp.structures.qwen3.5-2b.group_objects.parallelograms.cos:.2f}} in Qwen3.5-2B (p = {{num:mvp.structures.qwen3.5-2b.group_objects.parallelograms.p:.4f}}). The twins are near zero; the Qwen2.5 twins show a weak lexical signal.
- Trained models place "Lie group" close to *both* parents: "group" and "smooth manifold" rank {{num:mvp.structures.qwen3.5-2b.group_objects.parent_ranks.Lie group.0}} and {{num:mvp.structures.qwen3.5-2b.group_objects.parent_ranks.Lie group.1}} among {{num:mvp.structures.qwen3.5-2b.group_objects.pool_size}} items.
- The twins place it nearest to "group" because of the shared token. "Smooth manifold" ranks {{num:mvp.structures.qwen3.5-2b__random.group_objects.parent_ranks.Lie group.1}} in the Qwen3.5-2B twin; Gemma's twin is the exception, ranking it {{num:gemma.structures.gemma-4-e2b__random.group_objects.parent_ranks.Lie group.1}}.

**Axiom transfer: inconclusive.**
- The commutativity direction learned on group-like pairs transfers only weakly to ring → commutative ring and division ring → field ({{num:mvp.structures.qwen3.5-2b.transfer.commutative.cos:.2f}}, p = {{num:mvp.structures.qwen3.5-2b.transfer.commutative.p:.3f}}).
- Some Qwen twins show similar values. "Commutative ring" literally contains the token "commutative", so we consider transfer not established.
- Identity transfer, from the single pair rng → ring, is null.

**Further lattice items from the plan (exploratory; second audit).**

{{table:lattice_probes:mvp}}

{{table:lattice_variants:mvp}}

- *No semantic collapse.* A non-empty associative quasigroup is a group, but the model does not place it there. It is {{num:mvp.lattice.qwen3.5-2b.audit2.associative_quasigroup.ratio:.1f}}× farther from "group" than from "quasigroup" (twin {{num:mvp.lattice.qwen3.5-2b__random.audit2.associative_quasigroup.ratio:.1f}}×), and its nearest neighbours are quasigroup-like. The representation follows the name's parts, not this theorem.
- *Inverse semigroup* sits next to semigroup, commutative semigroup and monoid in the trained model, which is sensible.
- *Not a length artifact.* Removing the token-count direction (fitted on all family items) leaves the group-like offsets intact: commutativity {{num:mvp.lattice.qwen3.5-2b.audit2.group_like_offsets.length_projected.commutative.consistency:.2f}} (p = {{num:mvp.lattice.qwen3.5-2b.audit2.group_like_offsets.length_projected.commutative.p:.4f}}), associativity {{num:mvp.lattice.qwen3.5-2b.audit2.group_like_offsets.length_projected.associative.consistency:.2f}} (p = {{num:mvp.lattice.qwen3.5-2b.audit2.group_like_offsets.length_projected.associative.p:.4f}}). Across the five models (Appendix A), associativity, commutativity and divisibility stay significant in all five after this projection, and identity in four. Mean-pooling the concept's tokens gives a similar pattern, with divisibility no longer significant (table above). This is a one-direction analogue of the LDA distractor removal of [Li et al. (2024)](https://arxiv.org/abs/2410.19750).
- *Partial-operation lattice: no evidence.* The "total operation" offset (semigroupoid → semigroup, small category → monoid, groupoid → group) is {{num:mvp.lattice.qwen3.5-2b.audit2.partial_operation.offsets.total.consistency:.2f}}, below its null's 95th percentile ({{num:mvp.lattice.qwen3.5-2b.audit2.partial_operation.offsets.total.null95:.2f}}). With 6 items and 2–3 pairs per axis, the test is weak.
- *Identity transfer to lattice-like pairs* (semilattice → bounded semilattice, lattice → bounded lattice) is null too: cos = {{num:mvp.structures.qwen3.5-2b.transfer.identity_lattice.cos:.2f}}, p = {{num:mvp.structures.qwen3.5-2b.transfer.identity_lattice.p:.2f}}.

**Template variants (exploratory; Qwen3.5-2B).** We reran every analysis with two other template sets.

*Field-revealing concept templates* name the field, e.g. "In group theory, the notion of X":
- The graded field map barely changes (ρ = {{num:mvp_variants.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}}, against {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}} with neutral templates).
- Leave-one-out field accuracy jumps to {{num:mvp_variants.map_fields.qwen3.5-2b.math_primary.loo_acc.trained.acc:.3f}}, and the twin also rises to {{num:mvp_variants.map_fields.qwen3.5-2b.math_primary.loo_acc.random_init.acc:.3f}}. This is exactly the label leakage the preregistration excluded from the primary analysis.

*No-domain family templates* drop the shared context, e.g. "Consider X":
- Days remain a perfect PC-plane cycle.
- The chains remain ordered (ring and space Robinson p = {{num:mvp_variants.shapes.qwen3.5-2b.ring_chain.robinson_p:.3f}} and {{num:mvp_variants.shapes.qwen3.5-2b.space_chain.robinson_p:.3f}}), and angles remain line-like.
- Group-object parallelograms persist ({{num:mvp_variants.structures.qwen3.5-2b.group_objects.parallelograms.cos:.2f}}, p = {{num:mvp_variants.structures.qwen3.5-2b.group_objects.parallelograms.p:.4f}}).
- Axiom offsets weaken but persist for commutativity ({{num:mvp_variants.lattice.qwen3.5-2b.group_like.offsets.commutative.consistency:.2f}}, p = {{num:mvp_variants.lattice.qwen3.5-2b.group_like.offsets.commutative.p:.4f}}) and associativity ({{num:mvp_variants.lattice.qwen3.5-2b.group_like.offsets.associative.consistency:.2f}}, p = {{num:mvp_variants.lattice.qwen3.5-2b.group_like.offsets.associative.p:.4f}}).
- Months lose their PC-plane cycle.

Full tables: `reports/mvp_variants_results.md`.

## 4.5 Secondary analyses

### H-A4: concrete ↔ abstract and pure ↔ applied are the main axes

**Ratings.** Three judge LLMs rated every math concept's abstraction from 1 to 7: IBM Granite-4.0-micro, SmolLM3-3B and Phi-4-mini-instruct, none from a model family we probe. We read each rating as the expectation over next-token digit probabilities.
- Rank agreement is moderate (mean pairwise Spearman {{num:mvp.judges.mean_pairwise_spearman:.2f}}).
- Absolute agreement is low (ICC(2,1) = {{num:mvp.judges.icc2_1:.2f}}; ICC(2,k) = {{num:mvp.judges.icc2_k:.2f}}), because the judges use the scale with different offsets.
- The ratings have face validity. The least abstract concepts include "trivial group", "bar chart" and "dot plot"; the most abstract include étale cohomology, Artin stacks and generalized cohomology theories.

**Validation against Mathlib (plan; specified before computing, `docs/deviations.md`).** As an external, non-LLM measure, we matched concepts to declarations in [Mathlib4](https://github.com/leanprover-community/mathlib4) (commit `{{num:mvp.mathlib.mathlib_rev}}`).
- *Matching.* First through Mathlib's curated English-name maps (`docs/overview.yaml`, `docs/undergrad.yaml`), then through exact CamelCase names for multi-word concepts ("compact space" → `CompactSpace`).
- *Depth.* The longest chain of Mathlib imports below the defining module ({{num:mvp.mathlib.n_modules}} modules).
- *Result.* The mean judge rating correlates with depth: ρ = {{num:mvp.mathlib.rho_mean_rating_depth:.2f}}, 95% CI [{{num:mvp.mathlib.ci95.0:.2f}}, {{num:mvp.mathlib.ci95.1:.2f}}], meeting the pre-specified criterion.
- *Caveat.* Only {{num:mvp.mathlib.n_matched}} of {{num:mvp.mathlib.n_math_concepts}} concepts could be matched, and coverage is uneven across fields (table below). The judges also differ: Phi-4-mini's ratings track depth (ρ = {{num:mvp.mathlib.rho_per_judge.microsoft/Phi-4-mini-instruct:.2f}}), SmolLM3's barely (ρ = {{num:mvp.mathlib.rho_per_judge.HuggingFaceTB/SmolLM3-3B:.2f}}).

{{table:mathlib:mvp}}

![**Concrete ↔ abstract is the top axis of the math-concept cloud.** Left and middle: PC1 × PC2 of the {{num:mvp.map_fields.qwen3.5-2b.math_primary.n}} primary concepts, both residualized on frequency and token count exactly as in the statistic. Left is coloured by judge abstraction, with the most and least abstract concepts labelled; middle is coloured pure/applied. The PC signs are flipped for display so that abstract and applied point right and up. Right: mean judge rating vs Mathlib import depth for the matched concepts. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-abstraction>.]({{fig:abstraction_qwen3.5-2b}}){width=100%}

**Axes.** We first residualized on word frequency and token count. The top principal component of the math-concept cloud then correlates with abstraction (ρ = {{num:mvp.axes.qwen3.5-2b.pcs.0.rho_abstraction:.2f}}), and the second with the pure/applied label (ρ = {{num:mvp.axes.qwen3.5-2b.pcs.1.rho_applied:.2f}}). In the twin these correlations are {{num:mvp.axes.qwen3.5-2b__random.pcs.0.rho_abstraction:.2f}} and {{num:mvp.axes.qwen3.5-2b__random.pcs.1.rho_applied:.2f}}. A 5-fold cross-validated linear readout predicts abstraction with ρ = {{num:mvp.axes.qwen3.5-2b.cv_ridge_rho_abstraction:.2f}}, against {{num:mvp.axes.qwen3.5-2b__random.cv_ridge_rho_abstraction:.2f}} in the twin. We also partialled out pretraining-scale frequency: infini-gram counts in Dolma v1.7, which correlate ρ = {{num:mvp.baselines.infinigram.rho_zipf_vs_log_count:.2f}} with wordfreq. The axes are essentially unchanged: PC1 ↔ abstraction ρ = {{num:mvp.axes.qwen3.5-2b.pcs_with_infinigram.0.rho_abstraction:.2f}}, PC2 ↔ applied ρ = {{num:mvp.axes.qwen3.5-2b.pcs_with_infinigram.1.rho_applied:.2f}}.

**Interpretation.** In this model, concrete ↔ abstract is the single largest axis of variation among mathematical concepts. That is a correlational, model-rated finding: abstraction ratings come from LLM judges, not humans.

### Causal test of chain geometry: inconclusive

The preregistered gate requires baseline accuracy ≥ 0.70 on "which is more general, a A or a B?".
- **Ring chain:** it failed the gate (accuracy {{num:mvp.causal.qwen3.5-2b.ring_chain.baseline_acc:.3f}}), so no intervention was run.
- **Space chain:** it passed ({{num:mvp.causal.qwen3.5-2b.space_chain.baseline_acc:.3f}}).

At noise scale s = 4, noise injected into the chain's 2-D subspace lowered accuracy to {{num:mvp.causal.qwen3.5-2b.space_chain.scales.4.0.acc_chain_subspace:.3f}}. Noise in random 2-D subspaces gave a mean of {{num:mvp.causal.qwen3.5-2b.space_chain.scales.4.0.acc_random_subspace_mean:.3f}}. At s = 1 there was no difference. The direction matches the prediction, but the effect is small and five random draws cannot establish significance. We report it as inconclusive.

### SAE features (exploratory)

At block {{num:mvp.sae.qwen3.5-2b.sae_block}}, we encoded the template-averaged residuals with the official Qwen-Scope TopK SAE (k = 50).
- **Graded map:** the SAE code preserves the field map only weakly (ρ = {{num:mvp.sae.qwen3.5-2b.math.rsa_sae:.2f}}, against {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.2f}} for the dense residual).
- **Clustering:** field clustering is largely preserved (LOO accuracy {{num:mvp.sae.qwen3.5-2b.math.loo_acc_sae:.2f}}).
- **Field-selective features exist.** For example, a game-theory feature fires on most game-theory concepts and almost no others (`results/mvp/sae.json`).
- **Feature sets:** {{num:mvp.sae.qwen3.5-2b.features_math_only}} features fire only on math concepts, {{num:mvp.sae.qwen3.5-2b.features_control_only}} only on control concepts, and {{num:mvp.sae.qwen3.5-2b.features_shared}} on both. The sample sizes are unequal, so this is only suggestive of the "math lobe" of [Li et al. (2024)](https://arxiv.org/abs/2410.19750).


# 5 Discussion

**What the geometry is.** Relatedness between fields is encoded at the level of field centroids, in the dominant principal components, from the first few layers onward. It is equally strong in models with very different pretraining mixtures. Combined with the null effect of math continued pretraining, this suggests the map is learned from general co-occurrence statistics, as [Karkada et al. (2026)](https://arxiv.org/abs/2602.15029) predict. It does not seem to require math-specific training. The control fields cluster about as strongly once the number of fields is matched. That supports this reading: the "map of mathematics" is one instance of a general topical map.

**Axioms as directions.** Axioms of group-like structures behave like the man→woman offset. The geometry follows definitions stated in the names, not consequences that must be derived: "associative quasigroup" stays with the quasigroups, although a non-empty one is a group. Whether the model uses these directions in reasoning, for example to decide that a monoid with inverses is a group, needs causal tests; see §6 and the causal results.

**Angles.** "330 degrees" is not near "0 degrees". This mirrors the finding of [Kantamneni & Tegmark (2025)](https://arxiv.org/abs/2502.00873) that numbers lie on a helix whose dominant axis is linear. A model this size appears to encode degrees by their numerals rather than by their angular meaning.

# 6 Limitations

- **Concept sample.** Glossary terms are a convenience sample, and some junk survives cleaning. Field labels come from one glossary per term.
- **Ground truth.** Ochiai cross-listing is one operationalization of field relatedness.
- **Read point.** A single read-out: last token, averaged over templates. Template wording affects absolute values, though split-half reliability is high.
- **Model size.** Models are ≤ 2.3B active parameters, because of the 8 GB GPU.
- **Positive control.** It passed only under a post-hoc statistic.
- **Family size.** Families are small (n = 7–13), so shape claims rest on permutation tests with limited resolution.
- **Correlation, not mechanism.** All tests except §4.5 are correlational.
- **Abstraction ground truth.** Abstraction ratings come from LLM judges. Their validation against Mathlib import depth rests on only {{num:mvp.mathlib.n_matched}} of {{num:mvp.mathlib.n_math_concepts}} concepts that could be matched to Mathlib, and import depth also differs systematically between areas of Mathlib.
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

{{table:shapes:robustness}}

{{table:lattice:robustness}}

{{table:cycles:robustness}}

{{table:chain_frequency:robustness}}

{{table:lattice_probes:robustness}}

{{table:lattice_variants:robustness}}

**Gemma-4-E2B** with its twin:

{{table:shapes:gemma}}

{{table:lattice:gemma}}

{{table:cycles:gemma}}

{{table:chain_frequency:gemma}}

{{table:lattice_probes:gemma}}

{{table:lattice_variants:gemma}}

Matched 4-field clustering in Gemma-4-E2B:
- math subsets: mean {{num:gemma.posthoc.gemma-4-e2b.matched_clustering.math_subsets_mean:.3f}};
- control fields: {{num:gemma.posthoc.gemma-4-e2b.matched_clustering.control_acc:.3f}}.

# Appendix B: Deviations from the preregistration

See `docs/deviations.md`. Every post-hoc statistic in this paper is labelled post-hoc.

# AI statement

This codebase and paper were written by Claude Code (Anthropic's Claude, model Opus 5.5) acting as a coding and research agent. A human, Libor Burian, provided the exact specification that was executed. Human cleanup and rewriting are in progress.
