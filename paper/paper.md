---
title: "Do LLMs Organize Mathematics Geometrically? Field Maps, Ordered Chains and Axiom Lattices in Residual Streams"
author: "Libor \"Burny\" Burian"
date: "October 2026"
abstract: |
  Large language models represent days of the week and months on circles (Engels et al., 2024). We ask whether *abstract mathematics* is organized geometrically too. We test three preregistered hypotheses on five small base models (the Qwen models have at most 1.9B text parameters; Gemma-4-E2B has 2.3B effective, 5.1B with embeddings) and compare every model with a random-init twin (same architecture and tokenizer, no learning). An independent post-hoc audit then reran every test on baselines built from the concept names alone, without context (their spelling, and the model's own context-free token embeddings), because the twin turned out not to control word form: the Qwen3.5 twins scramble even exact word composition.

  **(A) Field map, the primary hypothesis.** Centroids of 17 mathematical fields are computed from concepts drawn from Wikipedia glossaries, excluding concepts whose names leak the field. In Qwen3.5-2B-Base, their geometry correlates with field relatedness measured by arXiv cross-listing over 3.2M papers: Spearman ρ = 0.39, against ρ = -0.12 for the random-init twin (paired bootstrap 95% CI of the difference [0.28, 0.58]). The map replicates in Qwen3.5-0.8B (ρ = 0.37), Qwen2.5-1.5B (ρ = 0.41) and Gemma-4-E2B (ρ = 0.44). It does not follow from the names alone: their character 3-grams give ρ = 0.10, the model's own context-free input embeddings of the names a weaker but significant ρ = 0.16, and the trained map exceeds all three name-only baselines in all 5 models. It is not a clean hierarchy (our preregistered cophenetic test does not beat the twin), a contrastive text embedder (Qwen3-Embedding-0.6B) matches it, and over one trillion tokens of math continued pretraining (Qwen2.5-Math) do not sharpen it (Δρ CI [-0.06, 0.02]).

  **(B) Shapes.** The space chain (T0 ⊃ … ⊃ Hilbert space) is ordered under the full preregistered test (Robinson score and leave-one-out order, Holm-corrected, above the twin) in all 5 models; spelling baselines do not order it, and it stays ordered within its metrizable … Hilbert block in all 5 models. The ring chain (rng ⊃ ring ⊃ … ⊃ algebraically closed field) passes the same test in 3 of 5 models (all but Qwen3.5-0.8B and Gemma-4-E2B), but its order is confounded with its head nouns (rng | ring | … domain | field): clusters of head nouns with no order at all pass the test in up to 55% of simulations, and the six "… domain" items are ordered among themselves in 1 of 5 models (Qwen2.5-Math-1.5B). Angles 0°–330° are ordered like the number line rather than a circle (line ρ > circle ρ in all 5 models), which falsifies our preregistered prediction.

  **(B2) Axiom lattices.** For each axiom of the group-like structures (magma … abelian group), difference vectors between structures that differ only in that axiom are more parallel than random pairs: in Qwen3.5-2B, offset consistency is 0.53 for commutativity and 0.43 for associativity, against 0.01 and 0.02 in the twin (Holm-significant for associativity in all 5 models and for commutativity in all 5 models). The audit shows that most of this follows from the names. Character 3-grams of the names, with no model, pass the same test for all four axioms. "X → wooden X" and the nonsense "X → zorbic X" are about as consistent as "X → commutative X"; "commutative" is the most consistent of the three in 1 of 5 models (Gemma-4-E2B). Two signs of more than word form remain. Associativity exceeds every word-form baseline and its lexically disjoint pairs pass the null in all 5 models. The "commutative" direction predicts *abelian group − group*, a pair that shares no word with it, in all 5 models. Group-object parallelograms (Lie group − topological group ≈ smooth manifold − topological space) are explained by shared words.

  The days/months positive control met our preregistered criterion only partially; in post-hoc tests, days close into a loop in the top-2 principal-component plane in 3 of 5 models (all but Qwen3.5-0.8B and Gemma-4-E2B) and months in a lower plane in 3 of 5 models (all but Qwen3.5-0.8B and Qwen2.5-1.5B). Methodologically, partial regression on distance matrices inflates type-I error when nuisances are proxies of confounders, and a random-init twin is not a control for word form: studies of concept geometry need name-only baselines.
---

# 1 Introduction

Interpretability research has found that LLMs encode some concepts as low-dimensional geometric objects rather than isolated directions:
- circles for days and months, which the models causally use for modular arithmetic ([Engels et al., 2024](https://arxiv.org/abs/2405.14860));
- a helix for numbers ([Kantamneni & Tegmark, 2025](https://arxiv.org/abs/2502.00873));
- a spiral for the periodic table ([Lei & Cooper, 2025](https://arxiv.org/abs/2502.10871));
- linear "crystals" of analogies among sparse-autoencoder features ([Li et al., 2024](https://arxiv.org/abs/2410.19750)).

[Karkada et al. (2026)](https://arxiv.org/abs/2602.15029) explain such shapes with symmetries of co-occurrence statistics: cyclic latents give circles, and ordered latents give 1-D curved manifolds.

Mathematics is an unusually good test bed for these ideas, because its concepts carry *exact* external structure. Structures are defined by axiom sets, so they sit on the vertices of a hypercube. Fields relate through shared methods. Algebraic structures form inclusion chains. As far as our literature search found, nobody has tested whether LLM residual streams reflect any of this for abstract mathematical concepts. The closest results are the "math/code lobe" of [Li et al. (2024)](https://arxiv.org/abs/2410.19750), which does not resolve fields or structures, and [Berlioz (2021)](https://arxiv.org/abs/2109.02801), who embedded arXiv mathematical terms with static (standard and hyperbolic) word embeddings and asked whether their dependency structure shows in the vectors.

**Contributions.**

1. Evidence that field-level geometry tracks how mathematicians cross-list papers (H-A1, the primary preregistered hypothesis), replicated across five models from two families, beyond random-init twins and beyond three word-form baselines, and unchanged by heavy math pretraining.
2. Evidence that the space chain of separation axioms through normed, Banach and Hilbert spaces is an ordered path, also within its metric block (metrizable … Hilbert) in all 5 models; within its separation-axiom block the order holds in 1 of 5 models (Qwen3.5-2B). The ring chain's order is not established beyond the head nouns of its names.
3. A sober reading of axiom offsets: they are consistent, but mostly because the names encode the axioms. Associativity, and a commutativity direction that reaches "abelian group", are suggestive exceptions.
4. A preregistered test (`docs/prereg.md`, git tag `prereg-v1`) with random-init twins, leakage flags, permutation nulls and paired bootstrap contrasts, followed by an independent post-hoc audit (`reports/prereg_addendum_lexical.md`) with word-form baselines, nonce modifiers, exact within-block tests and extra twin seeds.
5. Negative results: angles are not circular, math pretraining does not sharpen the map, and group-object parallelograms are lexical. Two methodological negatives: partial RDM regression is miscalibrated under realistic confounding, and a random-init twin does not control for word form.

# 2 Related work

**Feature geometry.**
- *Low-dimensional and circular features:* [Engels et al. (2024)](https://arxiv.org/abs/2405.14860), [Kantamneni & Tegmark (2025)](https://arxiv.org/abs/2502.00873); linear representations of space and time: [Gurnee & Tegmark (2023)](https://arxiv.org/abs/2310.02207).
- *Manifolds:* [Modell et al. (2025)](https://arxiv.org/abs/2505.18235), and supervised MDS for discovering them ([Tiblias et al., 2025](https://arxiv.org/abs/2510.01025)).
- *A statistical account:* [Karkada et al. (2026)](https://arxiv.org/abs/2602.15029).

**Hierarchy and categories.** [Park et al. (2024)](https://arxiv.org/abs/2406.01506) proposed orthogonality between hierarchical concepts. Golechha et al., *Intricacies of Feature Geometry in Large Language Models* ([ICLR 2025 blog track](https://iclr.cc/virtual/2025/poster/31349); [long version](https://www.greaterwrong.com/posts/eczwWrmX5XNEo7JsS/intricacies-of-feature-geometry-in-large-language-models)), showed that the same geometry appears for random concepts and random unembeddings after whitening. We therefore avoid whitening and compare every statistic with an untrained twin.

**Science and mathematics concepts.** [Lei & Cooper (2025)](https://arxiv.org/abs/2502.10871) found a periodic-table spiral. Before LLMs, [Berlioz (2021)](https://arxiv.org/abs/2109.02801) embedded mathematical terms from arXiv with standard and hyperbolic word embeddings and tested whether their dependency and entailment structure shows in the vectors. Mathematical "world models" in LLMs have been reported as local and not robust ([Kvinge et al., 2025](https://icml.cc/virtual/2025/49128), *Probing the Limits of Mathematical World Models in LLMs*, ICML 2025 workshop poster).

**Offsets and word form.** Analogies with vector offsets of word embeddings ([Mikolov, Yih & Zweig, 2013](https://aclanthology.org/N13-1090/)) are scored with a cosine criterion that mixes offset consistency with plain neighbourhood similarity, and baselines that ignore the offset solve analogies in some categories, such as plurals ([Linzen, 2016](https://aclanthology.org/W16-2503/)). With additive composition, "modifier + noun" is a fixed offset by construction ([Mitchell & Lapata, 2010](https://doi.org/10.1111/j.1551-6709.2010.01106.x)). Our audit therefore compares axiom offsets with nonce modifiers and with spelling-only baselines.

**Statistics.** Partial Mantel tests ([Mantel, 1967](https://pubmed.ncbi.nlm.nih.gov/6018555/)) are known to be invalid, with inflated false positives, when the data are autocorrelated ([Guillot & Rousset, 2013](https://doi.org/10.1111/2041-210X.12018)). We test Freedman–Lane permutation ([Freedman & Lane, 1983](https://doi.org/10.1080/07350015.1983.10509354); compared with alternatives by [Anderson & Legendre, 1999](https://doi.org/10.1080/00949659908811936)) in our setting (Section 3.5).

# 3 Methods

## 3.1 Concepts and fields

**Field map (H-A).**
- **Source.** Concepts come from 20 Wikipedia "Glossary of …" pages covering 16 mathematical fields, plus "List of numerical analysis topics" for the 17th, at pinned revisions. A 21st configured page, "Glossary of differential geometry and topology", uses list markup that the parser does not read and contributes no terms. Four non-math control fields (biology, chemistry, geology, astronomy) go through the same pipeline, except that they have no field root words, so the `leak_own` flag cannot fire for them.
- **Cleaning.** Terms keep 1–4 alphabetic words. Fragments, prefixes and adverbs are dropped.
- **Flags.** These exclude a concept from the *primary* set:
  - `leak_own`: the name contains its field's root word, e.g. "topological space";
  - `eponym`: a proper name, detected by mid-sentence capitalization on the same page (this missed five eponyms in the primary set, e.g. "tate curve"; `docs/deviations.md`);
  - `polysemous`: a single everyday word with wordfreq Zipf ≥ 4.
- **Sampling.** At most 30 primary concepts per field.
- **Split.** A random pilot split (each concept drawn with probability 0.2) was used only for debugging. The field-map, shape and lattice statistics below use the remaining confirmatory split: 398 primary math concepts across 17 fields. The judge ratings and their Mathlib validation (§4.5) cover all 655 math concepts.

## 3.2 Ground-truth field distance

The Mathematics Subject Classification ([MSC2020](https://msc2020.org/)) is flat above its two-digit codes, so we measure field relatedness from practice instead. For fields A and B, with S_A the set of arXiv papers listing any of A's categories, the distance is 1 − |S_A∩S_B|/√(|S_A||S_B|), i.e. one minus the Ochiai similarity ([Ochiai, 1957](https://doi.org/10.2331/suisan.22.526)). This is computed over the CC0 arXiv metadata snapshot (3,224,635 papers, pinned commit).

## 3.3 Concept families

Every family lists its ground truth and source in `configs/families.yaml`.

**Ordered families:**
- days and months, as positive controls;
- a ring chain (11 items);
- a space chain of separation axioms through normed, Banach and Hilbert spaces (11 items);
- number systems (7 items, exploratory);
- n-gons (3–12) and n-dimensional spaces (0–10), each with a number-word reference;
- angles 0°–330°, with a digit reference.

**Group-like axiom lattice.** Thirteen structures with bits (associative, identity, divisible, commutative): the seven total structures of the [Magma group-like table](https://en.wikipedia.org/wiki/Magma_(algebra)) (magma, quasigroup, unital magma, loop, semigroup, monoid, group), which has no commutativity column, plus our commutative versions of six of them. The self-distributive lattice (shelf, rack, spindle, quandle, kei) is exploratory.

## 3.4 Models and representations

| Model | Role |
|---|---|
| [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (Feb 2026; [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5)) | **Confirmatory**; official Qwen-Scope SAEs ([Deng et al., 2026](https://arxiv.org/abs/2605.11887)) |
| `Qwen3.5-0.8B-Base` | Scale |
| `Qwen2.5-1.5B` ([Qwen2.5 report](https://arxiv.org/abs/2412.15115)) vs `Qwen2.5-Math-1.5B` | Same base, + Qwen Math Corpus v2, over 1T tokens ([Qwen2.5-Math report, §1–2 and Fig. 2](https://arxiv.org/abs/2409.12122)) |
| [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) ([Gemma 4 report](https://arxiv.org/abs/2607.02770)) | Second model family; 2.3B effective parameters, 5.1B with embeddings |

Each model has a random-init twin with weights redrawn using seed 0.

- **Read point.** Each item is embedded in 8 templates ending with the concept, e.g. "In mathematics, the notion of *X*". Field-map templates name the domain ("mathematics" or "science") but never a sub-field. Family templates use one fixed domain per family, which can be a sub-field ("In abstract algebra, consider *X*"), so the domain is constant across the items being compared. The representation is the residual stream at the concept's last token, averaged over templates.
- **Preprocessing.** Mean-centering, with no whitening. A few "rogue" dimensions with large mean and variance dominate cosine similarity in transformer representations ([Timkey & van Schijndel, 2021](https://arxiv.org/abs/2109.04404)); centering removes their shared mean, though not their variance (the authors recommend standardization, which we did not use).
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

The test is calibrated when the nuisance distance matrix *is* the confounder. It rejects far too often when the nuisance only approximates the confounder, which is exactly the situation for lexical form and frequency. We therefore made **paired contrasts against the random-init twin** the primary tests. We intended the twin to measure form-only similarity on the same concepts; the audit (§3.6) shows that it does not, because a random network can scramble even exact word composition. Uncertainty comes from a stratified concept bootstrap with identical resamples for both models: 2,000 resamples for H-A1 and H-A3, and 1,000 for the hierarchy, baseline and tree-distance contrasts.

- **Field map.** We use field-level representational similarity analysis (RSA; [Kriegeskorte, Mur & Bandettini, 2008](https://doi.org/10.3389/neuro.06.004.2008)), i.e. the Spearman ρ between the centroid RDM and the arXiv distance, with a Mantel permutation over field labels ([Mantel, 1967](https://pubmed.ncbi.nlm.nih.gov/6018555/); 10,000 permutations) and a max-over-layers null for layer curves.
- **Families.** Every statistic is unsupervised or leave-one-out, and every one except the closure ratio, which the preregistered rules use as a fixed threshold (< 2), is tested against an order-permutation null. The Robinson score is the fraction of item triples whose distances agree with a 1-D order, after [Robinson (1951)](https://doi.org/10.2307/276978). With n ≤ 13 items in more than 1,000 dimensions, a supervised fit can impose any order. As a negative control, 100 random 11-item lists of math concepts must be rejected at about 5%; this checks the code, since a permutation test is exact under exchangeability by construction, and it cannot detect confounds such as spelling. The leave-one-out order null has 10,000 permutations at the preregistered layer (300 before the audit).
- **Multiplicity.** Holm correction ([Holm, 1979](https://www.jstor.org/stable/4615733)) over the 8 preregistered confirmatory tests (§4 of the preregistration); the angles prediction is a decision rule without a p-value and enters with p = 1, because it was falsified.
- **Figures, including 3-D views.** Every picture is an unsupervised projection: PCA, or metric MDS for field maps. 3-D views show each layer's own top principal components with true proportions, and state the share of variance on each axis (PCA) or the stress-1 of the layout (MDS; [Kruskal, 1964](https://doi.org/10.1007/BF02289565)). Alignment between panels and layers is display-only: rotation, or only sign flips for 3-D PCA axes. No statistic is computed from a picture. The single test that looks beyond PC1–2, the closed-cycle test over the top-5 planes, is post-hoc and was pre-specified in an addendum before it ran (§4.3).

## 3.6 Independent audit: word-form controls (post-hoc)

After all preregistered results were known, an independent audit found that no control measured word form directly. In the Qwen3.5-2B twin, the pairs magma → semigroup and commutative magma → commutative semigroup, which differ only by the literal word "commutative", have difference vectors with cosine 0.04; the trained model gives 0.86, and a bag of words gives 1. How much a twin keeps such exact compositions depends on the architecture (cosines of these duplicated pairs, range over three axioms: Qwen3.5-2B 0.04 to 0.07; Qwen3.5-0.8B 0.12 to 0.20; Qwen2.5-1.5B 0.31 to 0.41; Qwen2.5-Math-1.5B 0.30 to 0.39; Gemma-4-E2B 0.77 to 0.81). We pre-specified the following in an addendum committed before the analyses ran ([`reports/prereg_addendum_lexical.md`](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/blob/master/reports/prereg_addendum_lexical.md)); all of it is post-hoc.
- **Word-form baselines.** Every test is rerun, with the same statistic, centring, type of null and seed, on (i) counts of the names' character 3-grams, (ii) a bag of words and (iii) the sum of the model's trained input embeddings over the name's tokens, which carries token meaning but no context. Two nulls are smaller for the baselines: their leave-one-out order null has 300 permutations and their field-map Δ bootstrap 1,000 resamples. The chain, within-block and nonce tests skip the bag of words, whose one-word items give a constant RDM.
- **Where the axiom offsets come from.** Same-axiom pairs that coincide after deleting "commutative" are *lexical duplicates*; the remaining, disjoint cosines are tested against the same null. We also test the 8 names without "commutative", and whether the "commutative" direction predicts *abelian group − group*, the one commutativity pair that shares no word with it (exact null over the 72 ordered pairs that do not change commutativity). A nonce-modifier lattice calibrates plain composition: X → wooden X and X → zorbic X for six heads.
- **Chains.** Exact within-block order tests (the six "… domain" items of the ring chain; the separation-axiom and metric blocks of the space chain) and a simulation of how often head-noun clusters without any order pass the preregistered chain test.
- **Precision.** 10,000 leave-one-out permutations for the trained models and twins (a kernel-form ridge identical to the preregistered one), a scale-relative ridge penalty as sensitivity, the 8-test Holm family, BCa bootstrap intervals ([Efron, 1987](https://doi.org/10.1080/01621459.1987.10478410)) and four extra random-init seeds for the confirmatory model's twin.

An effect counts as *beyond word form* if it exceeds every word-form baseline and, for the axioms, its lexically disjoint pairs also pass the null.

# 4 Results

The results follow how well they survive the audit's word-form controls: the field map (H-A1, the primary preregistered hypothesis, §4.1) and its robustness (§4.2), the shapes of ordered families (H-B, §4.3), the axiom lattices (H-B2, §4.4) and the secondary analyses (§4.5). Every test keeps its preregistered status (confirmatory, secondary or exploratory); analyses added later are labelled as added, post-hoc or audit. Where a 3-D and a 2-D view of the same data exist, the 3-D view comes first.

![**Results at a glance.** Blue: trained model; grey: its random-init twin (same architecture and tokenizer, no learning); × and +: word-form baselines (the names' character 3-grams; the model's own input embeddings summed over the name's tokens). The starred model is the preregistered confirmatory model, Qwen3.5-2B. *Top row, the field map.* Left: field-centroid geometry correlates with arXiv cross-listing in every model (dots: ρ; lines: 95% bootstrap CI), well above the twins and both word-form baselines; the token-embedding baseline (+) shows a weak map of its own. Right: 3-D field maps (metric MDS) for arXiv cross-listing, the confirmatory model and its twin; grey edges join each field to its two nearest arXiv neighbours. *Middle row, axiom offsets.* Left: every pair of structures that differ only in commutativity, drawn as an arrow from one common origin in the top-3 principal components; the model's arrows point roughly one way, and the note gives the consistency of X → wooden X for comparison. Right: offset consistency per axiom for every model and twin, against the confirmatory model's random-pair null (grey bar: mean to 95th percentile); diamonds: the confirmatory model with the token-count direction removed; ×: the names' character 3-grams, which need no model. *Bottom row, shapes.* Days of the week go round in order (PC-plane cycle score; positive control). Angles do not close; the orange edge is the 330° → 0° closure the preregistration predicted. The space chain is an ordered path, also within its blocks; the ring chain is grouped by its head nouns, and its six "… domain" items are not ordered among themselves (exact p under the panel). Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#s-glance>.](../reports/figures/hero.pdf){width=100%}

**Every test in every model.** The table below summarizes the confirmatory and secondary tests for all five models.

| Test | Status | Qwen3.5-2B | Qwen3.5-0.8B | Qwen2.5-1.5B | Qwen2.5-Math-1.5B | Gemma-4-E2B |
|---|---|---|---|---|---|---|
| Field map tracks arXiv beyond the twin (H-A1; Mantel p and Δρ CI) | pre-registered, confirmatory | ● 0.39 | ● 0.37 | ● 0.41 | ● 0.39 | ● 0.44 |
| Field map beyond word form (Δρ vs char-3-grams, word bag, layer-0 token bag; CIs > 0) | post-hoc, audit addendum | ● +0.14 | ● +0.07 | ● +0.15 | ● +0.12 | ● +0.14 |
| Field map is a hierarchy beyond the twin (H-A2; cophenetic Δ CI) | pre-registered | ○ 0.24 | ○ 0.25 | ○ 0.24 | ○ 0.11 | ○ 0.23 |
| Space chain ordered (Robinson and LOO order; Holm; > twin) | pre-registered, confirmatory | ● 0.65 | ● 0.64 | ● 0.71 | ● 0.68 | ● 0.57 |
| Space chain ordered within both topical blocks (exact p) | post-hoc, audit addendum | ● / ● | ○ / ● | ○ / ● | ○ / ● | ○ / ● |
| Ring chain ordered (Robinson and LOO order; Holm; > twin) | pre-registered, confirmatory | ● 0.52 | ○ 0.50 | ● 0.54 | ● 0.56 | ○ 0.50 |
| Ring chain ordered within the six “… domain” items (exact p) | post-hoc, audit addendum | ○ 0.40 | ○ 0.40 | ○ 0.45 | ● 0.65 | ○ 0.10 |
| Ring / space order beyond frequency | added by plan audit | ● / ● | ● / ● | ● / ● | ● / ● | ○ / ○ |
| Axiom direction: associative (offset consistency; Holm) | pre-registered, confirmatory | ● 0.43 | ● 0.47 | ● 0.38 | ● 0.35 | ● 0.56 |
| Axiom direction: identity (offset consistency; Holm) | pre-registered, confirmatory | ◐ 0.24 | ○ 0.22 | ◐ 0.27 | ◐ 0.27 | ○ 0.19 |
| Axiom direction: divisible (offset consistency; Holm) | pre-registered, confirmatory | ◐ 0.29 | ◐ 0.26 | ◐ 0.25 | ◐ 0.30 | ◐ 0.35 |
| Axiom direction: commutative (offset consistency; Holm) | pre-registered, confirmatory | ● 0.53 | ● 0.57 | ● 0.41 | ● 0.45 | ● 0.74 |
| Axiom offsets beyond word form (assoc / ident / div / comm) | post-hoc, audit addendum | ● / ○ / ○ / ○ | ● / ○ / ○ / ○ | ● / ○ / ○ / ○ | ● / ○ / ○ / ○ | ● / ○ / ● / ○ |
| Commutative direction predicts abelian group − group (held out) | post-hoc, audit addendum | ● 0.60 | ● 0.54 | ● 0.51 | ● 0.55 | ● 0.73 |
| “commutative X” more consistent than “wooden X” and “zorbic X” | post-hoc, audit addendum | ○ 0.56 vs 0.68 | ○ 0.62 vs 0.65 | ○ 0.45 vs 0.57 | ○ 0.49 vs 0.61 | ● 0.79 vs 0.72 |
| Lie / topological / algebraic groups as parallelograms | pre-registered, exploratory | ● 0.40 | ● 0.46 | ● 0.41 | ● 0.41 | ● 0.48 |
| Group objects beyond word form (cos above every lexical baseline) | post-hoc, audit addendum | ○ 0.40 vs 0.68 | ○ 0.46 vs 0.69 | ○ 0.41 vs 0.68 | ○ 0.41 vs 0.67 | ○ 0.48 vs 0.61 |
| Days form a cycle (positive control; PC plane) | positive control; post-hoc statistic | ● 1.00 | ● 1.00 | ● 1.00 | ● 1.00 | ● 1.00 |
| Angles form a circle (closure < 2 and circle ρ > line ρ) | pre-registered, confirmatory | ✗ 2.11 | ✗ 2.59 | ✗ 2.15 | ✗ 1.84 | ✗ 2.03 |
| Angles: closed cycle in any top-5 PC plane (addendum rule) | post-hoc, pre-specified addendum | ○ 0.33 | ○ 0.33 | ○ 0.17 | ○ 0.25 | ○ 0.42 |
| Days / months: a loop that closes in some top-5 PC plane (p < .05, unadjusted) | post-hoc, descriptive | ● 1.00 / ● 0.50 | ○ 0.57 / ○ 0.33 | ● 1.00 / ○ 0.33 | ● 1.00 / ● 0.58 | ○ 0.43 / ● 0.50 |

● significant: Holm-adjusted over the 8 pre-registered confirmatory tests for the axiom and chain rows (the falsified angles rule counted with p = 1), with the trained value above the twin's; for Δ tests the 95% CI lies entirely above 0 (H-A1 also needs Mantel p < .05); for every other row p < .05, unadjusted · ◐ raw p < .05 but not Holm-significant (axiom and chain rows) · ○ not significant · ✗ pre-registered prediction falsified (the angles rule is met by all 5 twins, so it is not specific). Chains: p = max(Robinson p, LOO order p), the pre-registered conjunction. Numbers: the statistic for the trained model. Audit rows (post-hoc, reports/prereg_addendum_lexical.md): field map = lowest lower CI bound of Δρ against the three lexical baselines; ring chain = Robinson score within the “… domain” items; held-out pair = cosine of abelian group − group with the “commutative” direction; modifier and group-object rows = trained value vs the highest control; ● = the addendum's decision rule is met.

## 4.1 H-A1: a map of mathematical fields

![**Effect sizes with 95% bootstrap CIs (unadjusted).** Blue: CI excludes 0. "Added" marks analyses added by the plan audits and "audit" those of the independent audit, rather than the preregistration. The field map beats the twin in all 5 models and the names' character 3-grams in all 5 models; in Qwen3.5-2B it beats fastText, but not the Qwen3-Embedding text embedder. The Judges-vs-Mathlib line is a plain bootstrap of one correlation. Math pretraining (H-A3) and the hierarchy test (H-A2) are nulls. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-forest>.](../reports/figures/forest.pdf){width=82%}

| Model | Concept set | n concepts / fields | ρ_field trained (perm p) [95% CI] | ρ random-init | ρ layer-0 | Δ trained − random-init [95% CI] | LOO field acc trained / random / layer-0 (chance) | split-half reliability |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | math_primary | 398 / 17 | 0.387 (p=0.0001) [0.262, 0.420] | -0.118 | 0.024 | +0.506 [0.277, 0.576] | 0.472 / 0.090 / 0.279 (0.059) | 0.999 |
| qwen3.5-2b | math_all | 526 / 17 | 0.426 (p=0.0001) [0.303, 0.448] | -0.035 | 0.115 | +0.461 [0.231, 0.548] | 0.485 / 0.108 / 0.308 (0.059) | 0.999 |

![**The map is present at every depth in the Qwen models, and at most depths in Gemma.** ρ between the field-centroid RDM and the arXiv distance at each layer (x: relative depth). Blue: trained; grey: twin. The horizontal line is the 95th percentile of the max-over-layers permutation null; the vertical line marks the preregistered layer. 4 of 5 models (all but Gemma-4-E2B) clear the null at every layer; Gemma-4-E2B is at or below it in 3 stretches (layers 1–4, 11–15, 23–25 of 35). Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-depth>.](../reports/figures/depth_curves.pdf){width=100%}

**The trained model's field geometry tracks arXiv relatedness, and the random-init twin's does not.** For primary concepts at the preregistered layer, ρ = 0.387 with Mantel p = 0.0001. The twin gives ρ = -0.118 and layer-0 embeddings give ρ = 0.024. The difference from the twin has 95% CI [0.277, 0.576].

**Layer profile.** The map emerges in the first blocks and peaks at layer 6. It stays above the 95% max-over-layers null (0.17) at every layer.

**Clustering.** Leave-one-out nearest-centroid field accuracy is 0.472, against 0.090 for the twin and 0.059 chance.

**Reliability.** Split-half template reliability of the field RDM is 0.999.

![**Related fields stay close in the trained model, and the twin scrambles them.** Metric MDS of three 17 × 17 distance matrices: arXiv cross-listing (ground truth), Qwen3.5-2B field centroids at layer 12, and the random-init twin. Panels are rotated or reflected onto the arXiv layout for display only. Grey edges join each field to its two nearest arXiv neighbours and are drawn in every panel: short edges mean the neighbourhood is preserved. Blue: pure; orange: applied. ρ is computed on the full matrices, not on the 2-D layout. Interactive version, with a layer slider: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-map>.](../reports/figures/field_maps_qwen3.5-2b.pdf){width=100%}

![**The same matrices, ranked.** Colour is the rank of each distance within its matrix (Spearman RSA compares ranks), with rows in one fixed order grouped by the five super-tree branches. The trained matrix partly reproduces the arXiv block structure; the twin's does not. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-rdm>.](../reports/figures/field_rdms_qwen3.5-2b.pdf){width=100%}

**Not a rogue-dimension artifact (post-hoc).** The preregistered all-but-the-top variant ([Mu, Bhat & Viswanath, 2018](https://arxiv.org/abs/1702.01417)) removes the top d/100 = 20 principal components, and it reverses the correlation (ρ = -0.22). A post-hoc sweep explains why:

| Model | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 | k=20 | k=40 | top-5 PC explained variance | field η² on PC1–10 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 0.387 | 0.466 | 0.351 | 0.312 | 0.178 | 0.021 | -0.218 | -0.341 | 0.066, 0.050, 0.037, 0.035, 0.028 | 0.20, 0.47, 0.22, 0.10, 0.14, 0.13, 0.13, 0.20, 0.27, 0.34 |
| qwen3.5-2b__random | -0.118 | -0.140 | -0.133 | -0.136 | -0.107 | -0.074 | -0.160 | -0.164 | 0.013, 0.012, 0.012, 0.012, 0.012 | 0.04, 0.02, 0.04, 0.06, 0.05, 0.03, 0.02, 0.03, 0.08, 0.06 |

- Removing the top component *raises* ρ to 0.47.
- No component dominates.
- Field identity is spread over the top ~10 components, which carry far more field variance (η²) than the twin's.

So the map lives in the dominant semantic subspace, and removing 20 components removes the map itself.

**H-A2: hierarchy is not supported as preregistered.** The preregistered measure is the cophenetic correlation ([Sokal & Rohlf, 1962](https://doi.org/10.2307/1217208)) between an average-linkage (UPGMA; Sokal & Michener, 1958) dendrogram of the field centroids and the five-branch super-tree.

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
- **Against the twin:** the paired difference CI includes zero ([-0.079, 0.348]); models whose CI excludes zero: none of the 5 models. The preregistered criterion requires beating both, so it fails.
- **Intrinsic tree-likeness:** the cophenetic correlation of the representation with its own dendrogram (0.760) is not significantly higher than for a spectrum-matched Gaussian cloud (95th percentile 0.774, p = 0.096); models where it is: none of the 5 models.
- **Gromov δ** (relative δ = 2δ/diameter, as in [Khrulkov et al., 2019](https://arxiv.org/abs/1904.02239)): relative δ is lower (more tree-like) than the Gaussian cloud's in 4 of 5 models (all but Gemma-4-E2B), but δ is lower still in all 5 twins (0.06 to 0.09, against 0.23 to 0.29 in the trained models), so δ gives no support for a hierarchy beyond the twin.

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

The control fields cluster more strongly than most matched math subsets in all 5 models (Gemma: 0.00 of the subsets reach the control value). This does not show that the map is unspecific to mathematics: the controls are four whole sciences, while the math fields are sub-fields of one discipline, so the comparison confounds domain with granularity (audit). A granularity-matched control, sub-fields of another science, was not run, and specificity remains open.

**Baselines (plan; added after the first report; Qwen3.5-2B only).** We compare against two non-generative representations of the same concepts:
- **fastText** wiki-news-300d-1M vectors, averaged over words ([Mikolov et al., 2018](https://arxiv.org/abs/1712.09405)), and **[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** ([Zhang et al., 2025](https://arxiv.org/abs/2506.05176)), a contrastively trained text embedder built on an LLM.
- **vs fastText:** the LLM's field map is stronger (Δ CI [0.07, 0.30] on the 367 concepts that fastText covers).
- **vs Qwen3-Embedding:** the LLM is not stronger; the embedder reaches ρ = 0.454 against the LLM's 0.387 (Δ CI [-0.13, 0.04]).

So one contrastive embedder built on an LLM matches the generative model's field map: the map does not need the generative objective. This is a single comparison, in one model.

| Model | concept set | n | field ρ: LLM / Qwen3-Embedding / fastText | Δ LLM − Qwen3-Emb [CI] | Δ LLM − fastText [CI] | LOO acc: LLM / Qwen3-Emb / fastText |
|---|---|---|---|---|---|---|
| qwen3.5-2b | all_primary | 398 | 0.387 / 0.454 / — | [-0.125, 0.045] | — | 0.472 / 0.487 / — |
| qwen3.5-2b | fasttext_covered | 367 | 0.401 / 0.490 / 0.190 | [-0.148, 0.034] | [0.074, 0.302] | 0.485 / 0.477 / 0.322 |

infini-gram (v4_dolma-v1_7_llama): Spearman ρ(wordfreq Zipf, log count) = 0.316; concepts with zero count: 0.2%.

**Beyond word form (audit).** Could the map follow from the spelling of the concept names, for example suffixes shared within a glossary? Character 3-grams, a bag of words and the model's own input embeddings of the names give field maps of ρ = 0.10, -0.02 and 0.16 in Qwen3.5-2B, far below the trained model's 0.39. The paired bootstrap Δ CIs exclude zero against all three baselines in all 5 models. Spelling alone gives no significant map (character 3-grams: Mantel p = 0.14), but the context-free token embeddings already carry a weak one: ρ = 0.16 to 0.22, Mantel p = 0.0071 to 0.0303 (unadjusted), significant in all 5 models. Processing the names in context roughly doubles that ρ; the map is not read off their spelling.

| Model | ρ trained | ρ char-3-grams (Mantel p) | ρ word bag (p) | ρ layer-0 token bag (p) | Δ − char-3-grams [CI] | Δ − word bag [CI] | Δ − layer-0 bag [CI] | LOO field acc: trained / char-3 / word / layer-0 |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 0.387 | 0.105 (0.1353) | -0.017 (0.5205) | 0.157 (0.0303) | +0.283 [0.135, 0.398] | +0.404 [0.220, 0.518] | +0.230 [0.139, 0.344] | 0.472 / 0.261 / 0.249 / 0.349 |
| qwen3.5-0.8b | 0.369 | 0.105 (0.1353) | -0.017 (0.5205) | 0.218 (0.0071) | +0.264 [0.117, 0.376] | +0.385 [0.198, 0.490] | +0.151 [0.068, 0.270] | 0.430 / 0.261 / 0.249 / 0.312 |
| qwen2.5-1.5b | 0.406 | 0.105 (0.1353) | -0.017 (0.5205) | 0.178 (0.0276) | +0.302 [0.150, 0.416] | +0.423 [0.238, 0.531] | +0.228 [0.147, 0.363] | 0.565 / 0.261 / 0.249 / 0.337 |
| qwen2.5-math-1.5b | 0.390 | 0.105 (0.1353) | -0.017 (0.5205) | 0.201 (0.0112) | +0.285 [0.140, 0.396] | +0.407 [0.223, 0.506] | +0.189 [0.117, 0.317] | 0.588 / 0.261 / 0.249 / 0.372 |
| gemma-4-e2b | 0.436 | 0.105 (0.1353) | -0.017 (0.5205) | 0.209 (0.0136) | +0.332 [0.190, 0.441] | +0.453 [0.272, 0.566] | +0.227 [0.137, 0.355] | 0.417 / 0.261 / 0.249 / 0.322 |

**Bootstrap bias (audit).** Resampling concepts with replacement duplicates some of them, which makes resampled centroids noisier and pulls resampled ρ toward zero. The preregistered percentile intervals therefore sit below the point estimates: in Qwen3.5-2B, ρ = 0.387 but the bootstrap mean is 0.341. BCa intervals ([Efron, 1987](https://doi.org/10.1080/01621459.1987.10478410)) correct for this bias: the trained and Δ intervals move up, and the twins' intervals (whose ρ is negative) move down; the twin's BCa interval lies entirely below 0 in 2 of 5 twins (those of Qwen3.5-2B and Qwen3.5-0.8B). No hypothesis decision changes, because the bias works against H-A1 and cancels in the H-A3 contrast:

| Model | Statistic | point | bootstrap mean (bias) | percentile 95% CI (pre-registered) | BCa 95% CI |
|---|---|---|---|---|---|
| qwen3.5-2b | trained | +0.387 | +0.341 (-0.046) | [0.262, 0.420] | [0.357, 0.468] |
| qwen3.5-2b | random_init | -0.118 | -0.087 (+0.032) | [-0.215, 0.058] | [-0.255, -0.009] |
| qwen3.5-2b | delta_vs_random_init | +0.506 | +0.427 (-0.078) | [0.277, 0.576] | [0.436, 0.672] |
| qwen3.5-0.8b | trained | +0.369 | +0.319 (-0.049) | [0.243, 0.402] | [0.336, 0.450] |
| qwen3.5-0.8b | random_init | -0.182 | -0.129 (+0.053) | [-0.252, 0.002] | [-0.333, -0.111] |
| qwen3.5-0.8b | delta_vs_random_init | +0.551 | +0.448 (-0.102) | [0.300, 0.588] | [0.514, 0.672] |
| qwen2.5-1.5b | trained | +0.406 | +0.361 (-0.046) | [0.288, 0.436] | [0.375, 0.476] |
| qwen2.5-1.5b | random_init | -0.093 | -0.085 (+0.008) | [-0.195, 0.024] | [-0.210, 0.013] |
| qwen2.5-1.5b | delta_vs_random_init | +0.499 | +0.445 (-0.054) | [0.318, 0.565] | [0.429, 0.661] |
| qwen2.5-math-1.5b | trained | +0.390 | +0.340 (-0.050) | [0.264, 0.416] | [0.364, 0.460] |
| qwen2.5-math-1.5b | random_init | -0.087 | -0.079 (+0.008) | [-0.186, 0.026] | [-0.198, 0.013] |
| qwen2.5-math-1.5b | delta_vs_random_init | +0.477 | +0.419 (-0.059) | [0.297, 0.540] | [0.414, 0.668] |
| gemma-4-e2b | trained | +0.436 | +0.393 (-0.043) | [0.315, 0.477] | [0.394, 0.555] |
| gemma-4-e2b | random_init | -0.021 | -0.055 (-0.034) | [-0.185, 0.085] | [-0.114, 0.158] |
| gemma-4-e2b | delta_vs_random_init | +0.457 | +0.449 (-0.009) | [0.290, 0.598] | [0.301, 0.607] |
| qwen2.5-math-1.5b_vs_qwen2.5-1.5b | qwen2.5-math-1.5b | +0.390 | +0.340 (-0.050) | [0.264, 0.416] | [0.364, 0.460] |
| qwen2.5-math-1.5b_vs_qwen2.5-1.5b | qwen2.5-1.5b | +0.406 | +0.361 (-0.046) | [0.288, 0.436] | [0.375, 0.476] |
| qwen2.5-math-1.5b_vs_qwen2.5-1.5b | delta_vs_qwen2.5-1.5b | -0.017 | -0.021 (-0.004) | [-0.059, 0.017] | [-0.050, 0.026] |

## 4.2 Robustness: scale, model family, math training, templates

| Model | Concept set | n concepts / fields | ρ_field trained (perm p) [95% CI] | ρ random-init | ρ layer-0 | Δ trained − random-init [95% CI] | LOO field acc trained / random / layer-0 (chance) | split-half reliability |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | math_primary | 398 / 17 | 0.369 (p=0.0001) [0.243, 0.402] | -0.182 | 0.059 | +0.551 [0.300, 0.588] | 0.430 / 0.163 / 0.284 (0.059) | 0.998 |
| qwen3.5-0.8b | math_all | 526 / 17 | 0.402 (p=0.0002) [0.272, 0.429] | -0.035 | 0.160 | +0.436 [0.243, 0.530] | 0.437 / 0.173 / 0.314 (0.059) | 0.998 |
| qwen2.5-1.5b | math_primary | 398 / 17 | 0.406 (p=0.0002) [0.288, 0.436] | -0.093 | 0.044 | +0.499 [0.318, 0.565] | 0.565 / 0.219 / 0.289 (0.059) | 0.998 |
| qwen2.5-1.5b | math_all | 526 / 17 | 0.432 (p=0.0001) [0.310, 0.453] | 0.033 | 0.118 | +0.399 [0.256, 0.505] | 0.529 / 0.202 / 0.310 (0.059) | 0.999 |
| qwen2.5-math-1.5b | math_primary | 398 / 17 | 0.390 (p=0.0001) [0.264, 0.416] | -0.087 | 0.021 | +0.477 [0.297, 0.540] | 0.588 / 0.211 / 0.307 (0.059) | 0.999 |
| qwen2.5-math-1.5b | math_all | 526 / 17 | 0.406 (p=0.0002) [0.289, 0.428] | 0.059 | 0.141 | +0.348 [0.226, 0.468] | 0.568 / 0.200 / 0.325 (0.059) | 0.999 |

| Model | Concept set | n concepts / fields | ρ_field trained (perm p) [95% CI] | ρ random-init | ρ layer-0 | Δ trained − random-init [95% CI] | LOO field acc trained / random / layer-0 (chance) | split-half reliability |
|---|---|---|---|---|---|---|---|---|
| gemma-4-e2b | math_primary | 398 / 17 | 0.436 (p=0.0001) [0.315, 0.477] | -0.021 | 0.091 | +0.457 [0.290, 0.598] | 0.417 / 0.196 / 0.307 (0.059) | 0.993 |
| gemma-4-e2b | math_all | 526 / 17 | 0.464 (p=0.0001) [0.342, 0.501] | 0.085 | 0.183 | +0.379 [0.252, 0.543] | 0.399 / 0.215 / 0.312 (0.059) | 0.991 |

**H-A3 (math training): null.** Qwen2.5-Math-1.5B is Qwen2.5-1.5B continued on over 1T tokens of math ([Qwen2.5-Math report](https://arxiv.org/abs/2409.12122)). Its field map is no stronger than its base's: ρ = 0.390 vs 0.406, and the paired CI of the difference is [-0.059, 0.017].

**Twin seeds (audit).** Every twin contrast above uses one random draw (seed 0). Four more draws of the confirmatory model's twin show how much the twin statistics vary. The trained model exceeds all 5 draws for the field ρ, and the lowest lower bound of the H-A1 Δ CI over the draws is +0.239; the last column gives the same comparison for the other statistics:

| Statistic (qwen3.5-2b) | trained | twin seed0 | twin seed1 | twin seed2 | twin seed3 | twin seed4 | twin range | trained > every twin |
|---|---|---|---|---|---|---|---|---|
| field ρ | 0.387 | -0.118 | -0.162 | -0.156 | -0.106 | -0.161 | -0.162 to -0.106 | yes |
| LOO field accuracy | 0.472 | 0.090 | 0.101 | 0.121 | 0.098 | 0.116 | 0.090 to 0.121 | yes |
| offset consistency: associative | 0.432 | 0.021 | 0.010 | 0.015 | 0.023 | 0.019 | 0.010 to 0.023 | yes |
| offset consistency: identity | 0.237 | 0.040 | 0.031 | 0.020 | 0.036 | 0.024 | 0.020 to 0.040 | yes |
| offset consistency: divisible | 0.289 | 0.012 | 0.015 | 0.007 | 0.020 | 0.029 | 0.007 to 0.029 | yes |
| offset consistency: commutative | 0.534 | 0.013 | 0.011 | 0.012 | 0.000 | 0.004 | 0.000 to 0.013 | yes |
| ring chain Robinson | 0.515 | 0.297 | 0.242 | 0.382 | 0.412 | 0.345 | 0.242 to 0.412 | yes |
| space chain Robinson | 0.655 | 0.333 | 0.388 | 0.333 | 0.382 | 0.376 | 0.333 to 0.388 | yes |
| field Δρ trained − twin [95% CI] | — | [0.279, 0.571] | [0.289, 0.611] | [0.289, 0.599] | [0.239, 0.584] | [0.289, 0.599] | lowest bound +0.239 | — |

**Template variants (exploratory; Qwen3.5-2B).** We reran the field-map, shape, lattice, group-object, post-hoc and chain-frequency analyses with two other template sets (not the hierarchy, baseline, SMDS, axis, causal or SAE analyses).

*Field-revealing concept templates* name the field, e.g. "In group theory, the notion of X":
- The graded field map barely changes (ρ = 0.410, against 0.387 with neutral templates).
- Leave-one-out field accuracy jumps to 0.633, and the twin also rises to 0.332. This is exactly the label leakage the preregistration excluded from the primary analysis.

*No-domain family templates* drop the shared context, e.g. "Consider X":
- Axiom offsets (§4.4) weaken but persist for commutativity (0.47, p = 0.0020) and associativity (0.31, p = 0.0215).
- Group-object parallelograms persist (0.47, p = 0.0005).
- Of the shapes (§4.3), days remain a perfect PC-plane cycle, and months lose theirs.
- The chains' Robinson scores stay above their nulls before correction (p = 0.020 and 0.021), but the ring chain's leave-one-out order does not (p = 0.111), and neither chain survives Holm correction in this run. Angles remain line-like.

- *Audit, same template variants:* the word-form controls give the same verdicts. The field map stays above the names' character 3-grams (Δρ CI [0.17, 0.43]), X → commutative X is no more consistent than X → wooden X (0.49 vs 0.52), and the "commutative" direction still predicts abelian group − group (cos 0.57, p = 0.041).

Full tables: `reports/mvp_variants_results.md`.

## 4.3 H-B: shapes of ordered families

| Family | Expected | Model | line ρ | circle ρ | closure (d first-last / adjacent) | closure_far† (null 5%) | Robinson (p) | PCA order \|ρ\| (p) | LOO order ρ (p) |
|---|---|---|---|---|---|---|---|---|---|
| days | circle | qwen3.5-2b | 0.78 | 0.65 | 1.19 | 0.99 (0.76) | 0.77 (0.0010) | 0.54 (0.2284) | -0.14 (0.0812) |
| days | circle | qwen3.5-2b__random | -0.04 | -0.07 | 0.99 | 1.02 (0.96) | 0.34 (0.5002) | 0.18 (0.7106) | -0.93 (0.2202) |
| months | circle | qwen3.5-2b | 0.79 | 0.70 | 1.23 | 0.96 (0.76) | 0.66 (0.0005) | 0.55 (0.0730) | 0.77 (0.0015) |
| months | circle | qwen3.5-2b__random | -0.10 | -0.06 | 0.99 | 0.97 (0.94) | 0.30 (0.6962) | 0.32 (0.3048) | -0.78 (0.6252) |
| ring_chain | line | qwen3.5-2b | 0.48 | 0.38 | 1.63 | 1.04 (0.63) | 0.52 (0.0105) | 0.46 (0.1619) | 0.65 (0.0085) |
| ring_chain | line | qwen3.5-2b__random | -0.03 | -0.01 | 1.17 | 1.84 (0.53) | 0.30 (0.6667) | 0.50 (0.1234) | -0.45 (0.2381) |
| space_chain | line | qwen3.5-2b | 0.67 | 0.40 | 1.79 | 1.42 (0.57) | 0.65 (0.0005) | 0.83 (0.0025) | 0.87 (0.0001) |
| space_chain | line | qwen3.5-2b__random | -0.04 | 0.03 | 0.66 | 0.99 (0.54) | 0.33 (0.4993) | 0.56 (0.0770) | -0.41 (0.2562) |
| number_systems | line | qwen3.5-2b | 0.77 | 0.36 | 1.67 | 1.66 (0.56) | 0.74 (0.0045) | 0.96 (0.0020) | 0.71 (0.0069) |
| number_systems | line | qwen3.5-2b__random | 0.11 | 0.16 | 1.01 | 1.01 (0.93) | 0.26 (0.7856) | 0.39 (0.3773) | -0.82 (0.3742) |
| ngons | number_reference | qwen3.5-2b | 0.56 | 0.28 | 1.94 | 1.06 (0.54) | 0.64 (0.0010) | 0.87 (0.0030) | 0.90 (0.0001) |
| ngons | number_reference | qwen3.5-2b__random | 0.42 | 0.14 | 1.65 | 1.76 (0.57) | 0.42 (0.1159) | 0.19 (0.6007) | -0.89 (0.9457) |
| dimensions | number_reference | qwen3.5-2b | 0.66 | 0.37 | 2.17 | 1.02 (0.31) | 0.75 (0.0005) | 0.95 (0.0010) | 0.72 (0.0002) |
| dimensions | number_reference | qwen3.5-2b__random | -0.17 | -0.15 | 1.01 | 1.01 (0.93) | 0.32 (0.6117) | 0.00 (1.0000) | -0.75 (0.6614) |
| angles | circle | qwen3.5-2b | 0.56 | 0.29 | 2.11 | 1.01 (0.39) | 0.65 (0.0005) | 0.90 (0.0005) | 0.76 (0.0010) |
| angles | circle | qwen3.5-2b__random | -0.01 | 0.22 | 0.80 | 1.01 (0.77) | 0.26 (0.8581) | 0.08 (0.8076) | -0.49 (0.4139) |

![**A third principal component does not turn the angles into a circle.** Days, months and angles in their top-3 principal components at the preregistered layer. Each axis is that model's own PC, labelled with its share of variance, with true proportions and one fixed camera for every panel. Top row: Qwen3.5-2B; bottom row: its random-init twin. Below each panel: the share of variance in PC1–3 (for example 85% for the angles, 33% in the twin), and the best plane among PC1–5 where the loop closes, with its p (the closed-cycle test below): days and months close in some plane, while the angles have no significant closed cycle in any (best score 0.33, p = 0.34). A single camera angle can shorten or lengthen any edge; the evidence is the test, not the picture. Interactive 3-D version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/families3d_qwen3.5-2b.pdf){width=82%}

![**Days and months go round in their true order; angles do not.** Top-2 principal-component plane of each family at the preregistered layer (unsupervised; no labels are used to fit it). Colour runs light to dark along the true order; the grey path joins consecutive items. For angles, the orange edge is the 330° → 0° closure the preregistration predicted. Below each panel: the post-hoc PC-plane cycle score (1 = perfect cyclic order) with its order-permutation p. The score also gives 1 to an open arc: here the months' Dec → Jan step is 3.0× a neighbour step, so in this plane they are an open arc. Bottom row: the random-init twin. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/cyclic_qwen3.5-2b.pdf){width=82%}

![**The cycles are present at almost every depth.** Days (top) and months (bottom) at fixed relative depths, each plane rotated onto the preregistered layer's plane for display, plus the cycle score at every layer for the trained model and the twin (right). Post-hoc and unadjusted. Animated version: `reports/figures/circle_depth_qwen3.5-2b.gif`; interactive: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/circle_depth_qwen3.5-2b.pdf){width=100%}

**Positive control.** With our preregistered full-space statistics, days and months show significant circle-template correlation and closure ratios near 1. They nevertheless fail "circle ρ > line ρ" at layer 12; across models, line ρ exceeds circle ρ for days in all 5 models and for months in all 5 models. Following the prereg, we debugged before interpreting further. In the top-2 principal-component plane, where [Engels et al. (2024)](https://arxiv.org/abs/2405.14860) show their raw-activation circles (arXiv v3, Figs. 4 and 9; their SAE-cluster circles are usually in PCs 2–3, §4, and their circular probe uses the top 5 PCs, §5.1), days and months are in *perfect* cyclic order in the trained model and in none in the twin:

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
- **Days:** close in the top-2 plane in 3 of 5 models (all but Qwen3.5-0.8B and Gemma-4-E2B).
- **Months:** close in a lower plane (PC2–PC3 or PC4–PC5; p < .05 each, unadjusted) in 3 of 5 models (all but Qwen3.5-0.8B and Qwen2.5-1.5B).
- **Twin:** a closed months loop also appears in 1 of 5 twins (that of Qwen2.5-1.5B), so the test is not fully specific.

| Family | qwen3.5-2b | qwen3.5-0.8b | qwen2.5-1.5b | qwen2.5-math-1.5b | gemma-4-e2b |
|---|---|---|---|---|---|
| days | 1.00 · PC1–PC2 · p=0.0310 (twin 0.29) | 0.57 · PC2–PC4 · p=0.6224 (twin 0.57) | 1.00 · PC1–PC2 · p=0.0310 (twin 0.57) | 1.00 · PC1–PC2 · p=0.0270 (twin 0.57) | 0.43 · PC1–PC4 · p=0.8472 (twin 0.29) |
| months | 0.50 · PC2–PC3 · p=0.0220 (twin 0.25) | 0.33 · PC2–PC3 · p=0.3826 (twin 0.25) | 0.33 · PC2–PC3 · p=0.3976 (twin 0.58) | 0.58 · PC2–PC3 · p=0.0040 (twin 0.25) | 0.50 · PC4–PC5 · p=0.0200 (twin 0.42) |
| angles | 0.33 · PC3–PC5 · p=0.3447 (twin 0.33) | 0.33 · PC3–PC5 · p=0.3327 (twin 0.33) | 0.17 · PC2–PC3 · p=0.9890 (twin 0.17) | 0.25 · PC2–PC3 · p=0.8472 (twin 0.42) | 0.42 · PC2–PC4 · p=0.0839 (twin 0.17) |
| digits_0_330 | 0.33 · PC2–PC3 · p=0.3287 (twin 0.33) | 0.17 · PC2–PC3 · p=0.9950 (twin 0.25) | 0.25 · PC2–PC3 · p=0.7912 (twin 0.25) | 0.17 · PC2–PC4 · p=0.9910 (twin 0.25) | 0.33 · PC2–PC5 · p=0.3287 (twin 0.25) |

This statistic is post-hoc. We conclude that the pipeline works, and that the full-space template criterion was insensitive because variance outside the circle plane dominates. Spelling contributes to the order statistics of the positive control too: character 3-grams of the month names alone order the months (Robinson p = 0.0010; the "-ber" and "-uary" endings; Appendix A).

**Chains.** In the confirmatory model the ring and space chains are ordered beyond permutation nulls and beyond the twin (Robinson p = 0.010 and 0.0005; leave-one-out order p = 0.008 and 0.0001). Across models, the full preregistered test holds for the space chain in all 5 models and for the ring chain in 3 of 5 models (all but Qwen3.5-0.8B and Gemma-4-E2B); the leave-one-out order fails (p ≥ .05) for the ring chain in 2 of 5 models (Qwen3.5-0.8B and Gemma-4-E2B), with p = 0.074 and 0.105. Robinson scores 0.52 and 0.65, against 1/3 for chance and 0.30 and 0.33 for the twin. Random concept lists are rejected at the nominal rate (by construction; §3.5):

| Model | Robinson reject rate | PCA-order reject rate | line-ρ reject rate | n lists |
|---|---|---|---|---|
| qwen3.5-2b | 0.04 | 0.04 | 0.04 | 100 |
| qwen3.5-2b__random | 0.04 | 0.04 | 0.05 | 100 |

![**In 3-D, the ring and space chains run from general to special in the trained model; in the twin they collapse.** The ordered families (ring chain, space chain, number systems, n-gons, dimensions) in their top-3 principal components at the preregistered layer, with true proportions and one fixed camera; the path joins items in their true order, and its grey shadow on the floor is a depth cue. Top row: Qwen3.5-2B; bottom row: its twin. Below each panel: the share of variance in PC1–3 (ring chain 65%, twin 63%). The order statistics are computed in the full space and printed under the 2-D version (next figure). Interactive 3-D version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/chains3d_qwen3.5-2b.pdf){width=100%}

![**Class-inclusion chains are ordered paths; the twin's are tangles.** PC planes of the ordered families at the preregistered layer. Below each panel: the Robinson score (fraction of item triples consistent with the order; chance 1/3) with its permutation p, and for the line-type chains (ring chain, space chain, number systems) the p of the frequency control. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-families>.](../reports/figures/chains_qwen3.5-2b.pdf){width=100%}

**Frequency control (plan; added in the second audit).** Could the chain order just be a frequency or word-length order? We removed wordfreq Zipf frequency and model token count from both the representation and the ranks, fitting that regression inside each leave-one-out fold to avoid leakage ([Snoek et al., 2019](https://doi.org/10.1016/j.neuroimage.2018.09.074)), then predicted the residual order.
- *Result.* Both preregistered chains stay ordered: ring chain ρ = 0.94 (p = 0.003), space chain ρ = 0.81 (p = 0.017). The twin gives 0.13 and -0.04.
- *Replication (Appendix A).* The ring chain stays ordered beyond frequency in 4 of 5 models (all but Gemma-4-E2B) and the space chain in 4 of 5 models (all but Gemma-4-E2B) (Gemma-4-E2B: ring p = 0.17, space p = 0.056).
- *Word form carries part of the ring-chain order.* The ring chain is also ordered beyond frequency in 2 of 5 twins (those of Qwen2.5-1.5B and Gemma-4-E2B), with p = 0.037 and 0.027. A plausible lexical cue is that the middle items all end in "domain". The space chain is ordered beyond frequency in none of the 5 twins; the closest are the two Qwen2.5 twins (p = 0.053 and 0.053).
- *Caveat.* The fold-wise version was adopted post-hoc. A first, whole-sample version leaked the held-out item into the training span and was degenerate in the twin. Both versions are reported in `docs/deviations.md`.

| Chain | Model | ρ(order, Zipf) | ρ(order, token count) | LOO order beyond covariates (null95, p) |
|---|---|---|---|---|
| ring_chain | qwen3.5-2b | -0.04 | +0.29 | +0.94 (0.65, 0.0033) |
| ring_chain | qwen3.5-2b__random | -0.04 | +0.29 | +0.13 (0.42, 0.2791) |
| space_chain | qwen3.5-2b | -0.10 | -0.05 | +0.81 (0.60, 0.0166) |
| space_chain | qwen3.5-2b__random | -0.10 | -0.05 | -0.04 (0.41, 0.5349) |
| number_systems | qwen3.5-2b | -0.57 | +0.80 | +0.21 (0.86, 0.3920) |
| number_systems | qwen3.5-2b__random | -0.57 | +0.80 | -0.21 (0.89, 0.5681) |

**Within-block order and the head-noun confound (audit).** The ring chain's names end in four head nouns that form contiguous blocks of its order: rng | ring, commutative ring | six "… domain" items | field, algebraically closed field. A representation that groups items by head noun therefore tends to score above chance on both preregistered order statistics, even with no order between or within the groups, once the clusters are distinct. In a simulation with equidistant head-noun clusters and no order, the preregistered conjunction rejected in 2.5%, 40% and 55% of simulations at three cluster strengths (signal-to-noise 0.5, 1.0 and 2.0; at the weakest the mean leave-one-out ρ is -0.34). Within the six "… domain" items, where the head noun is constant, the order is significant in 1 of 5 models (Qwen2.5-Math-1.5B) (exact test over all 720 orders). The space chain, whose items all end in "space", is ordered within its separation-axiom block (T0 … normal) in 1 of 5 models (Qwen3.5-2B) and within its metric block (metrizable … Hilbert) in all 5 models, and its spelling does not order it (Appendix A). We therefore count the space chain as ordered, and the ring chain's order as not established beyond its head nouns.

| Model | Block (n) | trained: Robinson (exact p); PC1 \|ρ\| (p) | twin | char-3-grams | layer-0 token bag |
|---|---|---|---|---|---|
| qwen3.5-2b | ring chain, “… domain” items (6) | 0.40 (0.3194); 0.77 (0.1028) | 0.15 (0.8889) | 0.25 (0.7944) | 0.05 (0.9556) |
| qwen3.5-2b | space chain, T0 to normal (6) | 0.70 (0.0167); 0.94 (0.0167) | 0.25 (0.7667) | 0.25 (0.9333) | 0.05 (0.9944) |
| qwen3.5-2b | space chain, metrizable to Hilbert (5) | 1.00 (0.0167); 1.00 (0.0167) | 0.40 (0.4000) | 0.60 (0.2833) | 0.30 (0.6667) |
| qwen3.5-0.8b | ring chain, “… domain” items (6) | 0.40 (0.3528); 0.77 (0.1028) | 0.20 (0.8389) | 0.25 (0.7944) | 0.05 (0.9556) |
| qwen3.5-0.8b | space chain, T0 to normal (6) | 0.30 (0.6694); 0.83 (0.0583) | 0.20 (0.8861) | 0.25 (0.9333) | 0.10 (0.9944) |
| qwen3.5-0.8b | space chain, metrizable to Hilbert (5) | 0.90 (0.0167); 1.00 (0.0167) | 0.50 (0.2500) | 0.60 (0.2833) | 0.40 (0.5667) |
| qwen2.5-1.5b | ring chain, “… domain” items (6) | 0.45 (0.2306); 0.83 (0.0583) | 0.10 (0.9556) | 0.25 (0.7944) | 0.20 (0.8222) |
| qwen2.5-1.5b | space chain, T0 to normal (6) | 0.55 (0.0944); 0.60 (0.2417) | 0.20 (0.9028) | 0.25 (0.9333) | 0.05 (0.9944) |
| qwen2.5-1.5b | space chain, metrizable to Hilbert (5) | 1.00 (0.0167); 1.00 (0.0167) | 0.30 (0.6667) | 0.60 (0.2833) | 0.40 (0.5667) |
| qwen2.5-math-1.5b | ring chain, “… domain” items (6) | 0.65 (0.0056); 0.77 (0.1028) | 0.10 (0.9611) | 0.25 (0.7944) | 0.20 (0.8556) |
| qwen2.5-math-1.5b | space chain, T0 to normal (6) | 0.55 (0.0861); 0.94 (0.0167) | 0.25 (0.7667) | 0.25 (0.9333) | 0.15 (0.9722) |
| qwen2.5-math-1.5b | space chain, metrizable to Hilbert (5) | 1.00 (0.0167); 1.00 (0.0167) | 0.30 (0.6667) | 0.60 (0.2833) | 0.40 (0.5667) |
| gemma-4-e2b | ring chain, “… domain” items (6) | 0.10 (1.0000); 0.09 (0.9194) | 0.20 (0.8389) | 0.25 (0.7944) | 0.20 (0.8417) |
| gemma-4-e2b | space chain, T0 to normal (6) | 0.35 (0.4694); 0.26 (0.6583) | 0.25 (0.7500) | 0.25 (0.9333) | 0.15 (0.9889) |
| gemma-4-e2b | space chain, metrizable to Hilbert (5) | 0.90 (0.0167); 1.00 (0.0167) | 0.10 (0.9333) | 0.60 (0.2833) | 0.30 (0.6833) |

| head-noun cluster SNR | conjunction rejects (Robinson p < .05 and LOO p < .05) | mean Robinson | mean LOO ρ | simulations |
|---|---|---|---|---|
| 0.5 | 0.03 | 0.44 | -0.34 | 40 |
| 1.0 | 0.40 | 0.44 | +0.30 | 40 |
| 2.0 | 0.55 | 0.45 | +0.42 | 40 |

**Test precision and the ridge penalty (audit).** The trained models' and twins' leave-one-out order nulls have 10,000 permutations (300 before the audit; the word-form baselines keep 300). Together with the 8-test Holm family (§3.5), this changes three decisions: Gemma-4-E2B's space chain becomes Holm-significant, and identity and divisibility lose Holm significance in Qwen2.5-Math-1.5B. The preregistered ridge penalty α = 10 is not scale-invariant: the residual streams of the trained Qwen3.5 models have much smaller norms than those of Qwen2.5, Gemma and every twin, so their leave-one-out fits shrink toward the training mean, whose ranks are anti-correlated with the held-out ranks. This is why leave-one-out ρ is strongly negative for some families in those models (Qwen3.5-0.8B days: -1.00 with α = 10, 0.68 with the scale-relative α). The twins' strongly negative values have another cause: with no order to recover, the held-out prediction stays near the training mean whatever the penalty (Qwen3.5-2B twin, days: -0.93 and -1.00). Neither kind of value is an effect size. The scale-relative penalty is reported in Appendix A as a sensitivity analysis.

**Angles: preregistered prediction falsified.** Angles are not circular: closure ratio 2.11 (range 1.84–2.59 across models), no cycle in the top-2 PC plane, and, under the pre-specified addendum rule, no significant closed cycle in any plane of the top five components (models that meet the rule: none of the 5 models; below). Their geometry closely matches that of the bare digits 0–330 (ρ = 0.84; 0.72 to 0.84 across models), but so does the twins' (0.22 to 0.85): the prompts contain the numeral, so this match alone is not evidence. What separates the trained models is order: line ρ 0.43 to 0.62, against -0.01 to 0.25 in the twins. At this scale, "330 degrees" is farther from "0 degrees" than neighbouring angles are from each other.

**The preregistered rule is not specific.** The circle rule is *met* by all 5 twins, for example the Qwen3.5-2B twin: closure 0.80, circle ρ 0.22 vs line ρ -0.01. The twins have no line order (line ρ -0.01 to 0.25), so their closure ratio is near 1, and their circle-template ρ (0.15 to 0.46) is significant in all 5 twins (p ≤ 0.0085), possibly from shared digit tokens; that is enough to meet the rule. The informative result is therefore not the failed rule itself but the trained models' clear line-like order: line ρ 0.56 in Qwen3.5-2B, against circle ρ 0.29. That order tracks the bare numbers. We found this when we added the twin panel to the angles figure; it was not part of the preregistered analysis.

| Family | Model | ρ(family RDM, reference RDM) | template β beyond reference (p) |
|---|---|---|---|
| ngons vs number_words_3_12 | qwen3.5-2b | 0.59 | +0.223 (0.3089) |
| ngons vs number_words_3_12 | qwen3.5-2b__random | 0.11 | +0.411 (0.0066) |
| dimensions vs number_words_0_10 | qwen3.5-2b | 0.76 | +0.260 (0.0204) |
| dimensions vs number_words_0_10 | qwen3.5-2b__random | -0.58 | -0.153 (0.9104) |
| angles vs digits_0_330 | qwen3.5-2b | 0.84 | +0.054 (0.1858) |
| angles vs digits_0_330 | qwen3.5-2b__random | 0.22 | +0.227 (0.0104) |

A circle template beyond the digit reference (partial regression, table above) reaches p < .05 in some trained models but also in several twins (Appendix A); §3.5 shows that this regression rejects too often when the nuisance is only a proxy, so we do not read it as evidence of a circle.

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

## 4.4 H-B2: axiom lattices

**The test.** The 13 group-like structures are defined by four axioms (associativity, identity, divisibility, commutativity), so each sits on a vertex of a 4-cube (§3.3). For each axiom we take every pair of structures that differ only in that axiom, for example monoid → commutative monoid and group → abelian group for commutativity (6 pairs), and compute their difference vectors at the preregistered layer. *Offset consistency* is the mean pairwise cosine of these difference vectors in the full residual stream: 1 if they are parallel, about 0 if they are unrelated. The null draws the same number of random pairs from the same 13 structures (2,000 draws), and p = (1 + the number of null draws at least as large as the observed value)/(1 + 2,000). The next figure walks through the procedure on real data.

![**Adding an axiom moves a structure in a consistent direction in the trained model, and not in its random-init twin; the word-form controls below show that the names' spelling produces such bundles too.** The test on Qwen3.5-2B at the preregistered layer, in four steps. (1) Each structure is a set of axioms, so the 13 group-like structures form a lattice (ground truth). (2) Every pair of structures that differ only in commutativity, drawn as an arrow in the model's top-3 principal components. (3) The same arrows moved to one common origin: in the model they form a bundle (offset consistency 0.53), in the twin they point every way (0.01). (4) A compass for every axiom, measured in the full residual stream: the angle between each same-axiom difference vector and the mean of the others (leave-one-out); the model is drawn above the dashed line, the twin below; 0° means the same direction and 90° unrelated. Steps 2 and 3 are pictures; the statistic and the compass are computed in the full space, and the compass is a visual aid rather than a test. Interactive version, step by step: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-axiom-how>.](../reports/figures/axioms_explainer_qwen3.5-2b.pdf){width=100%}

| Lattice | Model | axiom-RDM β (p) | lexical β | offset consistency per axiom: obs / null95 (p) | analogy top-1 (chance) | mean rank (chance) |
|---|---|---|---|---|---|---|
| group_like | qwen3.5-2b | +0.492 (0.0002) | +0.236 | associative: 0.43/0.23 (0.0035); identity: 0.24/0.22 (0.0425); divisible: 0.29/0.22 (0.0275); commutative: 0.53/0.20 (0.0010) | 0.56 (0.10) | 2.5 (5.5) |
| self_distributive | qwen3.5-2b | +0.123 (0.3393) | -0.446 | invertible: 0.48/0.78 (0.2039); idempotent: 0.17/0.78 (0.4268) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | qwen3.5-2b | — | — | gender: 0.28/0.09 | king:queen rank 1 | — |
| group_like | qwen3.5-2b__random | +0.302 (0.0010) | -0.403 | associative: 0.02/0.16 (0.3678); identity: 0.04/0.16 (0.3068); divisible: 0.01/0.16 (0.4338); commutative: 0.01/0.14 (0.4048) | 0.31 (0.10) | 3.6 (5.5) |
| self_distributive | qwen3.5-2b__random | +0.316 (0.1284) | -0.275 | invertible: -0.01/0.56 (0.5207); idempotent: 0.01/0.56 (0.4493) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | qwen3.5-2b__random | — | — | gender: 0.00/0.07 | king:queen rank 15 | — |

![**In 3-D, the trained model's same-axiom arrows stay roughly parallel; the twin's point every way.** The 13 structures in their top-3 principal components (70% of their variance; twin 52%), with true proportions and one fixed camera. In each facet, the arrows (blue; grey in the twin) are the pairs that differ only in that axiom. Top row: Qwen3.5-2B; bottom row: its twin. A 3-D view can still hide or exaggerate parallelism; the cosine matrices below are the evidence. Interactive 3-D version, with the analogy query b − a + c placed exactly: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-lattice>.](../reports/figures/lattice3d_qwen3.5-2b.pdf){width=100%}

![**The same facets in the top-2 plane.** Left: the ideal geometry, in which each axiom is one fixed step. Other panels: the same unsupervised PCA plane of the 13 group-like structures, one panel per axiom, with the pairs that differ only in that axiom drawn as arrows (blue; grey in the twin). Parallel arrows mean a consistent direction. Below each panel: offset consistency and its random-pair p. Bottom row: the twin. Interactive version, with an analogy playground: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-lattice>.](../reports/figures/lattice_axes_qwen3.5-2b.pdf){width=100%}

![**The direction evidence, unprojected.** Cosine between every pair of single-axiom difference vectors in the full residual space, grouped by axiom. Red diagonal blocks mean a consistent direction. Hatched cells are cross-axiom pairs that share a structure; isotropic noise alone puts these near ±0.5, so they carry no evidence. The self-cosines on the diagonal are masked. Right: offset consistency per axiom for the trained model, the trained model with the token-count direction removed, the twin and the names' character 3-grams (×), against the random-pair null. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-cosines>.](../reports/figures/lattice_cosines_qwen3.5-2b.pdf){width=100%}

For every group-like axiom, difference vectors between structures that differ only in that axiom are more parallel than random pairs (raw p < .05):

| Axiom | Consistency | Null 95th percentile | p |
|---|---|---|---|
| Associativity | 0.43 | 0.23 | 0.0035 |
| Identity | 0.24 | 0.22 | 0.0425 |
| Divisibility | 0.29 | 0.22 | 0.0275 |
| Commutativity | 0.53 | 0.20 | 0.0010 |

In this model's random-init twin, the four consistencies are 0.01 to 0.04.

**Analogies.** Analogy completion is top-1 for 0.56 of 45 parallelograms (chance 0.10). The twin reaches 0.31 and the names' character 3-grams 0.38, so much of the analogy accuracy is lexical. So is much of the offset consistency (word-form controls below).

**Caveats.** The gender-offset positive control passes. Multiplicity across the preregistered confirmatory tests is handled by Holm correction:

| Test | Model | p | Holm-adjusted p | reject at α=0.05 |
|---|---|---|---|---|
| H-A1 Δρ trained − random-init ≤ 0 (bootstrap) | qwen3.5-2b | 0.0005 | 0.0040 | yes |
| H-B ring chain (Robinson and LOO order) | qwen3.5-2b | 0.0105 | 0.0420 | yes |
| H-B space chain (Robinson and LOO order) | qwen3.5-2b | 0.0005 | 0.0040 | yes |
| H-B2 associative offset consistency | qwen3.5-2b | 0.0035 | 0.0175 | yes |
| H-B2 identity offset consistency | qwen3.5-2b | 0.0425 | 0.0850 | no |
| H-B2 divisible offset consistency | qwen3.5-2b | 0.0275 | 0.0825 | no |
| H-B2 commutative offset consistency | qwen3.5-2b | 0.0010 | 0.0060 | yes |
| H-B angles circle prediction (decision rule: closure < 2 and circle ρ > line ρ; observed closure 2.11, circle ρ 0.29, line ρ 0.56; counted with p = 1) | qwen3.5-2b | — | — | not met (prediction falsified) |

- **After correction:** associativity and commutativity remain significant; identity and divisibility do not.
- **Across the replication models (Appendix A):** the offsets are positive for associativity in all 5 models, identity in all 5 models, divisibility in all 5 models and commutativity in all 5 models. The twins are weaker but not always null: raw p < .05 for commutativity in 2 of 5 twins (those of Qwen2.5-1.5B and Qwen2.5-Math-1.5B), with consistency ≈ 0.16 in Qwen2.5-1.5B's twin; for associativity in 1 of 5 twins (that of Gemma-4-E2B) and for divisibility in 1 of 5 twins (that of Gemma-4-E2B); these twins keep more of the names' word composition (§3.6). Divisibility reaches raw p < .05 in all 5 models, identity in 3 of 5 models (all but Qwen3.5-0.8B and Gemma-4-E2B).
**Word-form controls (audit).** The names encode the axioms. Five of the six commutativity pairs add the word "commutative". Most associativity and divisibility pairs end in "-group" on one side only (magma → semigroup, loop → group; magma → quasigroup, monoid → group). Several pairs of pairs repeat the same head-noun change with and without "commutative". A representation that knew only the spelling would therefore show consistent offsets, and it does: character 3-grams of the names pass the same random-pair test for every axiom (associativity 0.32, identity 0.32, divisibility 0.32, commutativity 0.62; p ≤ 0.032).

| Model | Axiom | trained (p) | char-3-grams (p) | word bag (p) | layer-0 token bag (p) | duplicated cosines: n, mean | disjoint cosines: mean (p); twin | beyond word form |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | associative | 0.43 (0.0035) | 0.32 (0.0215) | 0.25 (0.0380) | 0.23 (0.0680) | 1, 0.86 | 0.38 (0.0080); 0.02 | yes |
| qwen3.5-2b | identity | 0.24 (0.0425) | 0.32 (0.0215) | 0.20 (0.0675) | 0.34 (0.0190) | 2, 0.83 | 0.09 (0.2264); 0.04 | no |
| qwen3.5-2b | divisible | 0.29 (0.0275) | 0.32 (0.0320) | 0.25 (0.0460) | 0.22 (0.0780) | 1, 0.92 | 0.22 (0.0600); 0.01 | no |
| qwen3.5-2b | commutative | 0.53 (0.0010) | 0.62 (0.0005) | 0.67 (0.0005) | 0.71 (0.0005) | 0, — | 0.53 (0.0010); 0.01 | no |
| qwen3.5-0.8b | associative | 0.47 (0.0035) | 0.32 (0.0215) | 0.25 (0.0380) | 0.24 (0.0585) | 1, 0.93 | 0.41 (0.0065); 0.05 | yes |
| qwen3.5-0.8b | identity | 0.22 (0.0570) | 0.32 (0.0215) | 0.20 (0.0675) | 0.33 (0.0200) | 2, 0.91 | 0.05 (0.3163); 0.05 | no |
| qwen3.5-0.8b | divisible | 0.26 (0.0430) | 0.32 (0.0320) | 0.25 (0.0460) | 0.21 (0.0840) | 1, 0.94 | 0.18 (0.0925); 0.04 | no |
| qwen3.5-0.8b | commutative | 0.57 (0.0010) | 0.62 (0.0005) | 0.67 (0.0005) | 0.72 (0.0005) | 0, — | 0.57 (0.0010); 0.02 | no |
| qwen2.5-1.5b | associative | 0.38 (0.0055) | 0.32 (0.0215) | 0.25 (0.0380) | 0.24 (0.0560) | 1, 0.79 | 0.33 (0.0145); 0.10 | yes |
| qwen2.5-1.5b | identity | 0.27 (0.0310) | 0.32 (0.0215) | 0.20 (0.0675) | 0.29 (0.0300) | 2, 0.86 | 0.12 (0.1654); 0.03 | no |
| qwen2.5-1.5b | divisible | 0.25 (0.0410) | 0.32 (0.0320) | 0.25 (0.0460) | 0.23 (0.0685) | 1, 0.82 | 0.19 (0.0790); 0.07 | no |
| qwen2.5-1.5b | commutative | 0.41 (0.0025) | 0.62 (0.0005) | 0.67 (0.0005) | 0.69 (0.0005) | 0, — | 0.41 (0.0025); 0.16 | no |
| qwen2.5-math-1.5b | associative | 0.35 (0.0090) | 0.32 (0.0215) | 0.25 (0.0380) | 0.24 (0.0530) | 1, 0.80 | 0.30 (0.0230); 0.07 | yes |
| qwen2.5-math-1.5b | identity | 0.27 (0.0265) | 0.32 (0.0215) | 0.20 (0.0675) | 0.27 (0.0370) | 2, 0.84 | 0.13 (0.1509); 0.02 | no |
| qwen2.5-math-1.5b | divisible | 0.30 (0.0195) | 0.32 (0.0320) | 0.25 (0.0460) | 0.22 (0.0725) | 1, 0.86 | 0.23 (0.0430); 0.07 | no |
| qwen2.5-math-1.5b | commutative | 0.45 (0.0010) | 0.62 (0.0005) | 0.67 (0.0005) | 0.72 (0.0005) | 0, — | 0.45 (0.0010); 0.15 | no |
| gemma-4-e2b | associative | 0.56 (0.0020) | 0.32 (0.0215) | 0.25 (0.0380) | 0.26 (0.0390) | 1, 0.87 | 0.53 (0.0035); 0.16 | yes |
| gemma-4-e2b | identity | 0.19 (0.0900) | 0.32 (0.0215) | 0.20 (0.0675) | 0.20 (0.0675) | 2, 0.84 | 0.03 (0.3673); -0.02 | no |
| gemma-4-e2b | divisible | 0.35 (0.0230) | 0.32 (0.0320) | 0.25 (0.0460) | 0.28 (0.0270) | 1, 0.86 | 0.30 (0.0355); 0.17 | yes |
| gemma-4-e2b | commutative | 0.74 (0.0005) | 0.62 (0.0005) | 0.67 (0.0005) | 0.77 (0.0005) | 0, — | 0.74 (0.0005); 0.02 | no |

- *Commutativity is largely composition with a modifier.* For six heads, X → wooden X and X → zorbic X are about as consistent as X → commutative X; "commutative" is the most consistent of the three modifiers in 1 of 5 models (Gemma-4-E2B). Plain composition with any modifier gives offsets of this size.

| Model | modifier m (X → m X, 6 heads) | trained (null95, p) | twin | char-3-grams | layer-0 token bag |
|---|---|---|---|---|---|
| qwen3.5-2b | commutative | 0.56 (0.19, 0.0005) | 0.01 | 0.92 | 1.00 |
| qwen3.5-2b | wooden | 0.59 (0.19, 0.0005) | 0.01 | 0.87 | 1.00 |
| qwen3.5-2b | zorbic | 0.68 (0.20, 0.0005) | 0.01 | 0.87 | 1.00 |
| qwen3.5-0.8b | commutative | 0.62 (0.20, 0.0005) | 0.02 | 0.92 | 1.00 |
| qwen3.5-0.8b | wooden | 0.62 (0.20, 0.0005) | 0.01 | 0.87 | 1.00 |
| qwen3.5-0.8b | zorbic | 0.65 (0.20, 0.0005) | 0.03 | 0.87 | 1.00 |
| qwen2.5-1.5b | commutative | 0.45 (0.18, 0.0005) | 0.22 | 0.92 | 1.00 |
| qwen2.5-1.5b | wooden | 0.49 (0.18, 0.0005) | 0.18 | 0.87 | 1.00 |
| qwen2.5-1.5b | zorbic | 0.57 (0.19, 0.0005) | 0.23 | 0.87 | 1.00 |
| qwen2.5-math-1.5b | commutative | 0.49 (0.19, 0.0010) | 0.22 | 0.92 | 1.00 |
| qwen2.5-math-1.5b | wooden | 0.54 (0.18, 0.0005) | 0.17 | 0.87 | 1.00 |
| qwen2.5-math-1.5b | zorbic | 0.61 (0.19, 0.0005) | 0.23 | 0.87 | 1.00 |
| gemma-4-e2b | commutative | 0.79 (0.22, 0.0005) | 0.03 | 0.92 | 1.00 |
| gemma-4-e2b | wooden | 0.69 (0.21, 0.0005) | 0.02 | 0.87 | 1.00 |
| gemma-4-e2b | zorbic | 0.72 (0.22, 0.0005) | 0.02 | 0.87 | 1.00 |

- *But the commutative direction carries meaning.* The direction of the five pairs that add the word "commutative" predicts *abelian group − group*, which shares no word with it, in all 5 models (cosine 0.51 to 0.73). Character 3-grams and a bag of words give about 0, and the token-embedding bag stays below the trained models. This is one held-out pair per model: an existence proof, not a systematic result.

| Model | trained: cos (p) | twin: cos (p) | char-3-grams: cos (p) | word bag: cos (p) | layer-0 token bag: cos (p) | semantic transfer |
|---|---|---|---|---|---|---|
| qwen3.5-2b | +0.60 (0.0274) | +0.03 (0.3562) | +0.00 (0.8493) | +0.00 (0.9315) | +0.12 (0.0822) | yes |
| qwen3.5-0.8b | +0.54 (0.0274) | -0.00 (0.5205) | +0.00 (0.8493) | +0.00 (0.9315) | +0.15 (0.0959) | yes |
| qwen2.5-1.5b | +0.51 (0.0274) | +0.02 (0.3288) | +0.00 (0.8493) | +0.00 (0.9315) | +0.08 (0.0959) | yes |
| qwen2.5-math-1.5b | +0.55 (0.0274) | +0.03 (0.3288) | +0.00 (0.8493) | +0.00 (0.9315) | +0.15 (0.0822) | yes |
| gemma-4-e2b | +0.73 (0.0274) | +0.01 (0.4384) | +0.00 (0.8493) | +0.00 (0.9315) | +0.31 (0.0822) | yes |

- *Identity is lexical duplication.* Identity's two duplicated pairs (quasigroup → loop vs commutative quasigroup → commutative loop; semigroup → monoid vs commutative semigroup → commutative monoid) have cosines 0.83 to 0.91; its eight disjoint cosines average 0.03 to 0.13. On the 8 names without "commutative" its consistency is 0.00 to 0.09.
- *Associativity goes beyond word form most clearly.* It exceeds every word-form baseline and its disjoint pairs pass the null in all 5 models (divisibility: 1 of 5 models (Gemma-4-E2B); identity: none of the 5 models; commutativity: none of the 5 models). On the 8-name sub-lattice it is 0.28 to 0.50, above every baseline in all 5 models, but with three pairs per model no single value is significant.

| Model | Axiom (3 pairs) | trained | twin | char-3-grams | word bag | layer-0 token bag |
|---|---|---|---|---|---|---|
| qwen3.5-2b | associative | +0.35 (0.0930) | +0.02 (0.4105) | +0.22 (0.1844) | +0.14 (0.3263) | +0.11 (0.3189) |
| qwen3.5-2b | identity | +0.02 (0.4183) | +0.02 (0.4179) | +0.12 (0.2981) | +0.00 (0.5187) | +0.12 (0.2873) |
| qwen3.5-2b | divisible | +0.19 (0.2174) | -0.00 (0.4825) | +0.21 (0.2026) | +0.14 (0.3219) | +0.10 (0.3365) |
| qwen3.5-0.8b | associative | +0.39 (0.0834) | +0.02 (0.4333) | +0.22 (0.1844) | +0.14 (0.3263) | +0.13 (0.2885) |
| qwen3.5-0.8b | identity | -0.00 (0.4577) | +0.07 (0.3523) | +0.12 (0.2981) | +0.00 (0.5187) | +0.12 (0.2973) |
| qwen3.5-0.8b | divisible | +0.15 (0.2505) | +0.02 (0.4319) | +0.21 (0.2026) | +0.14 (0.3219) | +0.09 (0.3407) |
| qwen2.5-1.5b | associative | +0.29 (0.1260) | +0.11 (0.3119) | +0.22 (0.1844) | +0.14 (0.3263) | +0.12 (0.3085) |
| qwen2.5-1.5b | identity | +0.08 (0.3483) | +0.00 (0.4637) | +0.12 (0.2981) | +0.00 (0.5187) | +0.07 (0.3659) |
| qwen2.5-1.5b | divisible | +0.13 (0.2803) | +0.07 (0.3577) | +0.21 (0.2026) | +0.14 (0.3219) | +0.10 (0.3339) |
| qwen2.5-math-1.5b | associative | +0.28 (0.1340) | +0.09 (0.3475) | +0.22 (0.1844) | +0.14 (0.3263) | +0.13 (0.2863) |
| qwen2.5-math-1.5b | identity | +0.09 (0.3267) | -0.01 (0.4883) | +0.12 (0.2981) | +0.00 (0.5187) | +0.04 (0.4071) |
| qwen2.5-math-1.5b | divisible | +0.21 (0.1992) | +0.05 (0.3815) | +0.21 (0.2026) | +0.14 (0.3219) | +0.10 (0.3343) |
| gemma-4-e2b | associative | +0.50 (0.0522) | +0.12 (0.3177) | +0.22 (0.1844) | +0.14 (0.3263) | +0.13 (0.2899) |
| gemma-4-e2b | identity | +0.03 (0.3859) | -0.01 (0.4881) | +0.12 (0.2981) | +0.00 (0.5187) | -0.03 (0.5071) |
| gemma-4-e2b | divisible | +0.32 (0.1336) | +0.13 (0.2857) | +0.21 (0.2026) | +0.14 (0.3219) | +0.14 (0.2785) |

- **Our reading:** most of the offset geometry follows from the names. The evidence for axioms as directions beyond word form is suggestive: an associativity direction, and a commutativity direction that generalizes to "abelian group".

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

![**The group-object parallelograms clear their null, but the names' spelling is more parallel.** Left: mean cosine of the three parallelograms (for example, Lie group − topological group vs smooth manifold − topological space) for each model (blue), its twin (grey) and the names' character 3-grams (×), against the 95th percentile of each model's random-quadruple null (dark tick) and of the twin's own null (small grey tick); p values from those nulls. Right: the nearest-neighbour rank of "group" (circle) and "smooth manifold" (diamond) among the family items around "Lie group", trained vs twin (log scale). Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-groups>.](../reports/figures/group_objects.pdf){width=88%}

**Group objects: explained by shared words.**
- The parallelograms clear their null in all 5 models: mean cosine 0.40 in Qwen3.5-2B (p = 0.0045). The twins are weaker; the parallelograms pass the twin's own null in 2 of 5 twins (those of Qwen2.5-1.5B and Qwen2.5-Math-1.5B), with p = 0.035 and 0.041.
- Trained models place "Lie group" close to *both* parents: "group" and "smooth manifold" rank 3 and 7 among the 113 other family items.
- The twins rank "group" 1–2, likely because of the shared token. They rank "smooth manifold" 6–65 (trained models: 6–10); Gemma is the exception, whose twin ranks it 6 and whose trained model 10.
- *Audit.* Every parallelogram shares "topological" or "algebraic" on both sides, and the names' character 3-grams give a mean cosine of 0.61 and the token-embedding bag 0.68, above Qwen3.5-2B's 0.40. Models whose cosine exceeds every word-form baseline: none of the 5 models. The parallelograms therefore show composition of the names, not knowledge of group objects.

| Model | trained cos (p) | char-3-grams cos (p) | word bag cos (p) | layer-0 token bag cos (p) | beyond word form |
|---|---|---|---|---|---|
| qwen3.5-2b | +0.398 (0.0045) | +0.611 (0.0005) | +0.503 (0.0005) | +0.678 (0.0005) | no |
| qwen3.5-0.8b | +0.457 (0.0025) | +0.611 (0.0005) | +0.503 (0.0005) | +0.688 (0.0005) | no |
| qwen2.5-1.5b | +0.406 (0.0035) | +0.611 (0.0005) | +0.503 (0.0005) | +0.680 (0.0005) | no |
| qwen2.5-math-1.5b | +0.409 (0.0015) | +0.611 (0.0005) | +0.503 (0.0005) | +0.674 (0.0005) | no |
| gemma-4-e2b | +0.484 (0.0085) | +0.611 (0.0005) | +0.503 (0.0005) | +0.522 (0.0005) | no |

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

- *No semantic collapse.* A non-empty associative quasigroup is a group, but the model does not place it there (models where it sits closer to "group" than to "quasigroup": none of the 5 models). It is 2.8× farther from "group" than from "quasigroup" (twin 1.1×), and its nearest neighbours are quasigroup-like. The representation follows the name's parts, not this theorem.
- *Inverse semigroup* sits next to semigroup, commutative semigroup and monoid in the trained model, which is sensible.
- *Not a length artifact.* Removing the token-count direction (fitted on all family items) leaves the group-like offsets intact: commutativity 0.46 (p = 0.0010), associativity 0.44 (p = 0.0035). Across the five models (Appendix A), after this projection the offsets stay significant before correction for associativity in all 5 models, commutativity in all 5 models, divisibility in all 5 models and identity in 4 of 5 models (all but Gemma-4-E2B). In Qwen3.5-2B, mean-pooling the concept's tokens gives a similar pattern, with divisibility no longer significant (table above). This is a one-direction analogue of the LDA distractor removal of [Li et al. (2024)](https://arxiv.org/abs/2410.19750).
- *Partial-operation lattice: no evidence* (models with any axis below p = .05: none of the 5 models). The "total operation" offset (semigroupoid → semigroup, small category → monoid, groupoid → group) is 0.29, below its null's 95th percentile (0.54). With 6 items and 2–3 pairs per axis, the test is weak.
- *Identity transfer to lattice-like pairs* (semilattice → bounded semilattice, lattice → bounded lattice) is null too: cos = 0.07, p = 0.14 (models with p < .05: none of the 5 models).

![**A negative result: no consistent "make the operation total" direction.** Offset consistency for the partial-operation lattice (semigroupoid, small category, groupoid vs semigroup, monoid, group): the confirmatory model (filled), the other four models (open) and the five twins (grey), against the confirmatory model's random-pair null (grey bar: mean to 95th percentile). With 6 items and 2–3 pairs per axis the test has little power. Interactive version: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/#fig-partial>.](../reports/figures/partial_operation.pdf){width=70%}

## 4.5 Secondary analyses

### H-A4: concrete ↔ abstract and pure ↔ applied are the main axes

**Ratings.** Three judge LLMs rated every math concept's abstraction from 1 to 7: IBM [Granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) and Phi-4-mini-instruct ([Abouelenin et al., 2025](https://arxiv.org/abs/2503.01743)), none from a model family we probe. We read each rating as the expectation over next-token digit probabilities.
- Rank agreement is moderate (mean pairwise Spearman 0.63).
- Absolute agreement is low ([Shrout & Fleiss, 1979](https://doi.org/10.1037/0033-2909.86.2.420); ICC(2,1) = 0.24; ICC(2,k) = 0.49), because the judges use the scale with different offsets.
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

**Axes.** We take the principal components of the centred math-concept cloud and residualize both the PC scores and the labels on word frequency and token count. The top principal component then correlates with abstraction (ρ = 0.49), and the second with the pure/applied label (ρ = -0.41). In the twin these correlations are 0.06 and 0.14. A 5-fold cross-validated linear readout predicts abstraction with ρ = 0.74, against -0.06 in the twin. We also partialled out pretraining-scale frequency: [infini-gram](https://infini-gram.readthedocs.io/en/latest/api.html) counts in Dolma v1.7 ([Liu et al., 2024](https://arxiv.org/abs/2401.17377); index `v4_dolma-v1_7_llama`; corpus: [Soldaini et al., 2024](https://arxiv.org/abs/2402.00159)), which correlate ρ = 0.32 with wordfreq. The axes are essentially unchanged: PC1 ↔ abstraction ρ = 0.44, PC2 ↔ applied ρ = -0.41.

**Interpretation.** In this model, concrete ↔ abstract is the single largest axis of variation among mathematical concepts. That is a correlational, model-rated finding: abstraction ratings come from LLM judges, not humans.

### Causal test of chain geometry: inconclusive

The causal test is a plan item, not part of `docs/prereg.md`; its gate, fixed before the test ran (`docs/deviations.md`), requires baseline accuracy ≥ 0.70 on "which is more general, a A or a B?". The noise subspace is fitted on the other chain items, not on the tested pair, and compared with random subspaces of the same size, because a subspace can look causally relevant by accident ([Makelov et al., 2023](https://arxiv.org/abs/2311.17030)).
- **Ring chain:** it failed the gate (accuracy 0.691), so no intervention was run.
- **Space chain:** it passed (0.864).

At noise scale s = 4, noise injected into the chain's 2-D subspace lowered accuracy to 0.791. Noise in random 2-D subspaces gave a mean of 0.822. At s = 2 the values were 0.836 and 0.856; at s = 1, chain-subspace noise hurt *less* than random noise (0.864 vs 0.847). At the two larger scales the direction matches the prediction, but the effect is small and five random draws cannot establish significance. We report it as inconclusive.

### SAE features (exploratory)

At block 11, we encoded the template-averaged residuals with the official Qwen-Scope TopK SAE (k = 50; k-sparse autoencoders as in [Gao et al., 2024](https://arxiv.org/abs/2406.04093)).
- **Graded map:** the SAE code preserves the field map only weakly (ρ = 0.15, against 0.39 for the dense residual).
- **Clustering:** field clustering is largely preserved (LOO accuracy 0.43).
- **Field-selective features exist.** For example, a game-theory feature fires on most game-theory concepts and almost no others (`results/mvp/sae.json`).
- **Feature sets:** 2445 features fire only on math concepts, 871 only on control concepts, and 715 on both. The sample sizes are unequal, so this is only suggestive of the "math lobe" of [Li et al. (2024)](https://arxiv.org/abs/2410.19750).


# 5 Discussion

**Axiom offsets.** The offsets of group-like structures behave like the man → woman offset of word embeddings ([Mikolov, Yih & Zweig, 2013](https://aclanthology.org/N13-1090/)), and like those offsets they mix structure with word form ([Linzen, 2016](https://aclanthology.org/W16-2503/)). With additive composition, "modifier + noun" is a fixed vector by construction ([Mitchell & Lapata, 2010](https://doi.org/10.1111/j.1551-6709.2010.01106.x)), and the models compose "wooden" and "zorbic" about as consistently as "commutative". What survives the word-form controls is narrower: an associativity direction above every spelling baseline, and a commutativity direction that also points from "group" to "abelian group". The geometry follows the names more than the definitions: "associative quasigroup" stays with the quasigroups, although a non-empty one is a group. Whether the model uses these directions in reasoning, for example to decide that a monoid with inverses is a group, needs causal tests, which we did not run for the axioms (§6).

**The field map.** Relatedness between fields is encoded at the level of field centroids, in the dominant principal components, from the first few layers onward in the Qwen models (Gemma's map is at or below the max-over-layers null in 3 stretches (layers 1–4, 11–15, 23–25 of 35)). It is similarly strong (ρ ≈ 0.4) in models with very different pretraining mixtures. Combined with the null effect of math continued pretraining, this suggests the map is learned from general co-occurrence statistics, consistent with the co-occurrence account of [Karkada et al. (2026)](https://arxiv.org/abs/2602.15029), whose theory covers other kinds of latents. It does not seem to require math-specific training, and it does not follow from the spelling of the names; context-free token embeddings carry a weaker version of it. Whether it is specific to mathematics remains open: our control fields are whole sciences, not sub-fields, so the matched clustering comparison cannot decide it.

**Controls for concept geometry.** A random-init twin controls for architecture and tokenization, but not for word form: it can scramble even exact word composition (strongly in the Qwen3.5 twins, hardly in Gemma's), so a trained model can "beat the twin" with geometry that any spelling-based representation would show. Spelling-only baselines and nonce controls are cheap and should accompany studies of concept geometry, especially for multi-word names that encode their own structure, as mathematical names often do.

**Angles.** "330 degrees" is not near "0 degrees". This mirrors the finding of [Kantamneni & Tegmark (2025)](https://arxiv.org/abs/2502.00873) that numbers lie on a helix whose dominant axis is linear. A model this size appears to order degrees by their numerals rather than by their angular meaning; the similarity to the bare numerals alone does not show this, since the twins share it.

# 6 Limitations

- **Concept sample.** Glossary terms are a convenience sample, and some junk survives cleaning. Field labels come from one glossary per term. The eponym flag missed five eponyms in the primary set (`docs/deviations.md`).
- **Ground truth.** Ochiai cross-listing is one operationalization of field relatedness. The space chain is a chain only if "regular" and "normal" include T1 (T3 and T4, as in Munkres); the prompts use the plain words.
- **Read point.** A single read-out: last token, averaged over templates. Template wording affects absolute values, though split-half reliability is high.
- **Model size.** The Qwen models have at most 1.9B text parameters and Gemma-4-E2B 2.3B effective parameters, because of the 8 GB GPU.
- **Positive control.** It passed only under post-hoc statistics. In the plane where its loop closes, it holds for days in 3 of 5 models (all but Qwen3.5-0.8B and Gemma-4-E2B) and for months in 3 of 5 models (all but Qwen3.5-0.8B and Qwen2.5-1.5B).
- **Family size.** Families are small (n = 5–13), so shape claims rest on permutation tests with limited resolution.
- **Word form.** Our preregistered control, the random-init twin, did not control the spelling of names; the audit's word-form baselines do, post-hoc. Lexical cues beyond character 3-grams, bags of words and context-free token embeddings are not ruled out.
- **Twin seeds.** Five random draws for the confirmatory model's twin; one for each other model.
- **Specificity.** The non-math control fields are whole sciences; a granularity-matched control was not run.
- **Bootstrap intervals.** Concept resampling duplicates concepts and pulls resampled field ρ toward zero, so the preregistered percentile intervals sit below the point estimates; BCa intervals are reported next to them (§4.1). The bias works against H-A1 and cancels in the H-A3 contrast.
- **Ridge penalty.** The preregistered α = 10 is not scale-invariant, so the leave-one-out order test has different power in different models (§4.3).
- **Correlation, not mechanism.** Every test except the noise-injection test on the chains (§4.5) is correlational, including the axiom offsets (§4.4).
- **Abstraction ground truth.** Abstraction ratings come from LLM judges. Their validation against Mathlib import depth rests on only 39 of 655 concepts that could be matched to Mathlib, and import depth also differs systematically between areas of Mathlib.
- **Not run.** Three planned items were not run; reasons are in `docs/deviations.md`:
  - Qwen3.5-4B in 8-bit, whose random-init twin cannot be held in 8 GB;
  - the optional OctoThinker pair;
  - persistent homology, since no family has ≥30 points.
- **Baseline scope.** The fastText and Qwen3-Embedding baselines, SMDS and the template variants were run for the confirmatory model only; the audit's word-form baselines cover all five models.

# 7 Conclusion

Field-level geometry in LLM residual streams tracks how mathematicians cross-list their papers, in five models, beyond random-init twins and beyond the spelling of the concept names; math continued pretraining does not sharpen it. The space chain T0 ⊃ … ⊃ Hilbert space is an ordered path, also within its metric block, while the ring chain's order is not established beyond its head nouns. Axioms of group-like structures give consistent offsets, but mostly because the names encode them; an associativity direction and a commutativity direction that reaches "abelian group" are suggestive exceptions. Angles are not circles. A random-init twin does not control for word form, so studies of concept geometry need name-only baselines.

# References

Every source cited in the text, with the identifier we checked (arXiv, Crossref, PubMed or JSTOR, 2026-10-31):

- Anderson, M. J. & Legendre, P. (1999). An empirical comparison of permutation methods for tests of partial regression coefficients in a linear model. *J. Stat. Comput. Simul.* 62. [doi:10.1080/00949659908811936](https://doi.org/10.1080/00949659908811936)
- Berlioz, L. (2021). ArGoT: A Glossary of Terms extracted from the arXiv. *EPTCS* 342:14–21. [arXiv:2109.02801](https://arxiv.org/abs/2109.02801)
- Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics* 24:295–300 (intersection-union test, used for the chain conjunction). [doi:10.1080/00401706.1982.10487790](https://doi.org/10.1080/00401706.1982.10487790)
- Deng, B. et al. (2026). Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models. [arXiv:2605.11887](https://arxiv.org/abs/2605.11887)
- Diaconis, P., Goel, S. & Holmes, S. (2008). Horseshoes in multidimensional scaling and local kernel methods. *Ann. Appl. Stat.* 2(3). [arXiv:0811.1477](https://arxiv.org/abs/0811.1477)
- Engels, J., Michaud, E. J., Liao, I., Gurnee, W. & Tegmark, M. (2024). Not All Language Model Features Are One-Dimensionally Linear. ICLR 2025. [arXiv:2405.14860](https://arxiv.org/abs/2405.14860) (v3)
- Efron, B. (1987). Better bootstrap confidence intervals. *J. Am. Stat. Assoc.* 82(397):171–185 (BCa intervals). [doi:10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410)
- Freedman, D. & Lane, D. (1983). A nonstochastic interpretation of reported significance levels. *J. Bus. Econ. Stat.* 1:292–298. [doi:10.1080/07350015.1983.10509354](https://doi.org/10.1080/07350015.1983.10509354)
- Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders. [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
- Gemma Team (2026). Gemma 4 Technical Report. [arXiv:2607.02770](https://arxiv.org/abs/2607.02770)
- Golechha, S., Bushnaq, L., Ong, E., Kayal, N. & Schoots, N. (2025). Intricacies of Feature Geometry in Large Language Models. ICLR 2025 blog track. [iclr.cc/virtual/2025/poster/31349](https://iclr.cc/virtual/2025/poster/31349)
- Guillot, G. & Rousset, F. (2013). Dismantling the Mantel tests. *Methods Ecol. Evol.* 4. [doi:10.1111/2041-210X.12018](https://doi.org/10.1111/2041-210X.12018)
- Gurnee, W. & Tegmark, M. (2023). Language Models Represent Space and Time. [arXiv:2310.02207](https://arxiv.org/abs/2310.02207)
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Stat.* 6(2):65–70. [JSTOR 4615733](https://www.jstor.org/stable/4615733)
- Kantamneni, S. & Tegmark, M. (2025). Language Models Use Trigonometry to Do Addition. [arXiv:2502.00873](https://arxiv.org/abs/2502.00873)
- Karkada, D., Korchinski, D. J., Nava, A., Wyart, M. & Bahri, Y. (2026). Symmetry in language statistics shapes the geometry of model representations. [arXiv:2602.15029](https://arxiv.org/abs/2602.15029)
- Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I. & Lempitsky, V. (2019). Hyperbolic Image Embeddings (relative Gromov δ). [arXiv:1904.02239](https://arxiv.org/abs/1904.02239)
- Kriegeskorte, N., Mur, M. & Bandettini, P. (2008). Representational similarity analysis: connecting the branches of systems neuroscience. *Front. Syst. Neurosci.* 2:4. [doi:10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008)
- Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. *Psychometrika* 29:1–27 (stress-1). [doi:10.1007/BF02289565](https://doi.org/10.1007/BF02289565)
- Kvinge, H. et al. (2025). Probing the Limits of Mathematical World Models in LLMs. ICML 2025 workshop *Assessing World Models*. [icml.cc/virtual/2025/49128](https://icml.cc/virtual/2025/49128)
- Lei, G. & Cooper, S. J. (2025). Layerwise Recall and the Geometry of Interwoven Knowledge in LLMs. [arXiv:2502.10871](https://arxiv.org/abs/2502.10871)
- Li, Y., Michaud, E. J., Baek, D. D., Engels, J., Sun, X. & Tegmark, M. (2024). The Geometry of Concepts: Sparse Autoencoder Feature Structure. [arXiv:2410.19750](https://arxiv.org/abs/2410.19750)
- Linzen, T. (2016). Issues in evaluating semantic spaces using word analogies. *Proc. 1st Workshop on Evaluating Vector-Space Representations for NLP (RepEval)*, 13–18. [aclanthology.org/W16-2503](https://aclanthology.org/W16-2503/)
- Liu, J., Min, S., Zettlemoyer, L., Choi, Y. & Hajishirzi, H. (2024). Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens. [arXiv:2401.17377](https://arxiv.org/abs/2401.17377)
- Makelov, A., Lange, G. & Nanda, N. (2023). Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching. [arXiv:2311.17030](https://arxiv.org/abs/2311.17030)
- Mantel, N. (1967). The detection of disease clustering and a generalized regression approach. *Cancer Res.* 27(2):209–220. [PMID 6018555](https://pubmed.ncbi.nlm.nih.gov/6018555/)
- Mathematics Subject Classification 2020 (MSC2020). <https://msc2020.org/>
- Mikolov, T., Yih, W. & Zweig, G. (2013). Linguistic Regularities in Continuous Space Word Representations. NAACL-HLT 2013. [aclanthology.org/N13-1090](https://aclanthology.org/N13-1090/)
- Mikolov, T., Grave, E., Bojanowski, P., Puhrsch, C. & Joulin, A. (2018). Advances in Pre-Training Distributed Word Representations (fastText vectors). [arXiv:1712.09405](https://arxiv.org/abs/1712.09405)
- Mitchell, J. & Lapata, M. (2010). Composition in distributional models of semantics. *Cognitive Science* 34(8):1388–1429. [doi:10.1111/j.1551-6709.2010.01106.x](https://doi.org/10.1111/j.1551-6709.2010.01106.x)
- Modell, A., Rubin-Delanchy, P. & Whiteley, N. (2025). The Origins of Representation Manifolds in Large Language Models. [arXiv:2505.18235](https://arxiv.org/abs/2505.18235)
- Mu, J., Bhat, S. & Viswanath, P. (2018). All-but-the-Top: Simple and Effective Postprocessing for Word Representations. ICLR 2018. [arXiv:1702.01417](https://arxiv.org/abs/1702.01417)
- Munkres, J. R. (2000). *Topology*, 2nd ed., §31–32 (regular and normal spaces assume one-point sets closed).
- Ochiai, A. (1957). Zoogeographical studies on the soleoid fishes found in Japan and its neighbouring regions-II. *Nippon Suisan Gakkaishi* 22(9):526–530. [doi:10.2331/suisan.22.526](https://doi.org/10.2331/suisan.22.526)
- Park, K., Choe, Y. J., Jiang, Y. & Veitch, V. (2024). The Geometry of Categorical and Hierarchical Concepts in Large Language Models. [arXiv:2406.01506](https://arxiv.org/abs/2406.01506)
- Abouelenin, A. et al. (2025). Phi-4-Mini Technical Report. [arXiv:2503.01743](https://arxiv.org/abs/2503.01743)
- Qwen Team (2024). Qwen2.5 Technical Report. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- Qwen Team (2026). Qwen3.5: Towards Native Multimodal Agents (blog). <https://qwen.ai/blog?id=qwen3.5>
- Robinson, W. S. (1951). A method for chronologically ordering archaeological deposits. *American Antiquity* 16(4):293–301. [doi:10.2307/276978](https://doi.org/10.2307/276978)
- Shrout, P. E. & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater reliability. *Psychol. Bull.* 86:420–428. [doi:10.1037/0033-2909.86.2.420](https://doi.org/10.1037/0033-2909.86.2.420)
- Snoek, L., Miletić, S. & Scholte, H. S. (2019). How to control for confounds in decoding analyses of neuroimaging data. *NeuroImage* 184. [doi:10.1016/j.neuroimage.2018.09.074](https://doi.org/10.1016/j.neuroimage.2018.09.074)
- Soldaini, L. et al. (2024). Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. [arXiv:2402.00159](https://arxiv.org/abs/2402.00159)
- Sokal, R. R. & Michener, C. D. (1958). A statistical method for evaluating systematic relationships. *Univ. Kansas Sci. Bull.* 38:1409–1438 (UPGMA).
- Sokal, R. R. & Rohlf, F. J. (1962). The comparison of dendrograms by objective methods. *Taxon* 11. [doi:10.2307/1217208](https://doi.org/10.2307/1217208)
- Tiblias, F., Bigoulaeva, I., Niu, J., Balloccu, S. & Gurevych, I. (2025). Hypothesis-Driven Feature Manifold Analysis in LLMs via Supervised Multi-Dimensional Scaling (v1: "Shape Happens"). [arXiv:2510.01025](https://arxiv.org/abs/2510.01025)
- Timkey, W. & van Schijndel, M. (2021). All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality. [arXiv:2109.04404](https://arxiv.org/abs/2109.04404)
- Yang, A. et al. (2024). Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement. [arXiv:2409.12122](https://arxiv.org/abs/2409.12122)
- Zhang, Y. et al. (2025). Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. [arXiv:2506.05176](https://arxiv.org/abs/2506.05176)
- Models and data: [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base), [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B), [Gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B), [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), the judges [granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro), [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) and [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct); the arXiv metadata mirror [`librarian-bots/arxiv-metadata-snapshot`](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) (CC0); [Mathlib4](https://github.com/leanprover-community/mathlib4).

Code, data, preregistration and every number: <https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms>. Interactive explorer: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/>. Data and activations: <https://huggingface.co/datasets/BurnyCoder/geometry-of-mathematics-in-llms>.

# Appendix A: Replication of lattices and shapes

**Qwen3.5-0.8B, Qwen2.5-1.5B and Qwen2.5-Math-1.5B**, each with its twin:

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

| Family | Expected | Model | line ρ | circle ρ | closure (d first-last / adjacent) | closure_far† (null 5%) | Robinson (p) | PCA order \|ρ\| (p) | LOO order ρ (p) |
|---|---|---|---|---|---|---|---|---|---|
| days | circle | qwen3.5-0.8b | 0.85 | 0.63 | 1.29 | 1.06 (0.79) | 0.77 (0.0010) | 0.75 (0.0735) | -1.00 (1.0000) |
| days | circle | qwen3.5-0.8b__random | 0.20 | 0.07 | 1.01 | 1.04 (0.93) | 0.40 (0.3758) | 0.93 (0.0095) | -0.75 (0.1349) |
| days | circle | qwen2.5-1.5b | 0.77 | 0.66 | 1.22 | 0.99 (0.73) | 0.69 (0.0020) | 0.54 (0.2219) | 0.68 (0.0566) |
| days | circle | qwen2.5-1.5b__random | 0.10 | -0.11 | 1.02 | 1.01 (0.93) | 0.43 (0.2259) | 0.54 (0.2324) | -0.82 (0.5377) |
| days | circle | qwen2.5-math-1.5b | 0.73 | 0.58 | 1.18 | 1.01 (0.76) | 0.71 (0.0010) | 0.54 (0.2219) | 0.36 (0.1158) |
| days | circle | qwen2.5-math-1.5b__random | 0.16 | -0.14 | 1.02 | 1.01 (0.94) | 0.37 (0.3973) | 0.46 (0.2889) | -0.93 (0.7668) |
| months | circle | qwen3.5-0.8b | 0.79 | 0.70 | 1.23 | 0.97 (0.77) | 0.67 (0.0005) | 0.64 (0.0345) | -0.20 (0.0003) |
| months | circle | qwen3.5-0.8b__random | 0.06 | -0.03 | 1.02 | 1.02 (0.94) | 0.38 (0.2274) | 0.30 (0.3483) | -0.86 (0.9039) |
| months | circle | qwen2.5-1.5b | 0.71 | 0.56 | 1.25 | 1.07 (0.75) | 0.64 (0.0005) | 0.85 (0.0010) | 0.80 (0.0033) |
| months | circle | qwen2.5-1.5b__random | 0.07 | -0.02 | 1.02 | 0.97 (0.92) | 0.35 (0.4218) | 0.10 (0.7381) | -0.62 (0.7031) |
| months | circle | qwen2.5-math-1.5b | 0.71 | 0.53 | 1.21 | 0.99 (0.75) | 0.70 (0.0005) | 0.92 (0.0005) | 0.82 (0.0013) |
| months | circle | qwen2.5-math-1.5b__random | 0.04 | -0.01 | 1.02 | 0.97 (0.91) | 0.32 (0.5827) | 0.03 (0.9370) | -0.60 (0.7005) |
| ring_chain | line | qwen3.5-0.8b | 0.46 | 0.36 | 1.70 | 1.00 (0.58) | 0.50 (0.0170) | 0.45 (0.1764) | -0.09 (0.0739) |
| ring_chain | line | qwen3.5-0.8b__random | 0.18 | 0.11 | 1.29 | 1.36 (0.74) | 0.42 (0.1134) | 0.04 (0.9295) | 0.36 (0.0465) |
| ring_chain | line | qwen2.5-1.5b | 0.54 | 0.36 | 1.74 | 1.11 (0.56) | 0.54 (0.0025) | 0.21 (0.5517) | 0.87 (0.0020) |
| ring_chain | line | qwen2.5-1.5b__random | 0.30 | 0.31 | 1.06 | 1.03 (0.85) | 0.35 (0.4318) | 0.03 (0.9405) | 0.23 (0.0815) |
| ring_chain | line | qwen2.5-math-1.5b | 0.54 | 0.39 | 1.67 | 1.05 (0.62) | 0.56 (0.0030) | 0.41 (0.2189) | 0.89 (0.0012) |
| ring_chain | line | qwen2.5-math-1.5b__random | 0.34 | 0.30 | 1.09 | 1.05 (0.85) | 0.41 (0.1529) | 0.05 (0.8786) | 0.25 (0.0760) |
| space_chain | line | qwen3.5-0.8b | 0.65 | 0.33 | 1.72 | 1.67 (0.59) | 0.64 (0.0005) | 0.95 (0.0005) | 0.63 (0.0001) |
| space_chain | line | qwen3.5-0.8b__random | -0.04 | 0.04 | 0.91 | 1.06 (0.72) | 0.36 (0.3703) | 0.36 (0.2659) | 0.29 (0.0353) |
| space_chain | line | qwen2.5-1.5b | 0.69 | 0.38 | 1.83 | 1.68 (0.60) | 0.71 (0.0005) | 0.95 (0.0005) | 0.90 (0.0005) |
| space_chain | line | qwen2.5-1.5b__random | -0.12 | -0.06 | 0.95 | 1.10 (0.80) | 0.34 (0.4888) | 0.67 (0.0315) | 0.21 (0.0969) |
| space_chain | line | qwen2.5-math-1.5b | 0.67 | 0.40 | 1.71 | 1.51 (0.61) | 0.68 (0.0005) | 0.85 (0.0010) | 0.91 (0.0003) |
| space_chain | line | qwen2.5-math-1.5b__random | -0.10 | -0.07 | 0.97 | 1.08 (0.80) | 0.36 (0.3988) | 0.73 (0.0185) | 0.18 (0.1163) |
| number_systems | line | qwen3.5-0.8b | 0.81 | 0.40 | 1.73 | 1.72 (0.57) | 0.77 (0.0030) | 0.93 (0.0065) | 0.29 (0.0022) |
| number_systems | line | qwen3.5-0.8b__random | 0.31 | 0.14 | 1.10 | 1.25 (0.77) | 0.29 (0.6892) | 0.29 (0.5542) | -0.25 (0.2158) |
| number_systems | line | qwen2.5-1.5b | 0.66 | 0.33 | 1.42 | 1.57 (0.60) | 0.66 (0.0165) | 0.82 (0.0295) | 0.79 (0.0129) |
| number_systems | line | qwen2.5-1.5b__random | 0.34 | 0.14 | 1.13 | 1.37 (0.73) | 0.34 (0.4718) | 0.54 (0.2219) | -0.21 (0.2270) |
| number_systems | line | qwen2.5-math-1.5b | 0.71 | 0.33 | 1.42 | 1.55 (0.61) | 0.66 (0.0145) | 0.82 (0.0295) | 0.71 (0.0163) |
| number_systems | line | qwen2.5-math-1.5b__random | 0.36 | 0.24 | 1.13 | 1.38 (0.72) | 0.40 (0.2344) | 0.57 (0.1839) | -0.43 (0.3102) |
| ngons | number_reference | qwen3.5-0.8b | 0.54 | 0.30 | 1.88 | 1.05 (0.51) | 0.63 (0.0015) | 0.87 (0.0030) | 0.60 (0.0006) |
| ngons | number_reference | qwen3.5-0.8b__random | 0.36 | 0.07 | 1.36 | 1.17 (0.77) | 0.47 (0.0555) | 0.73 (0.0140) | -0.42 (0.3267) |
| ngons | number_reference | qwen2.5-1.5b | 0.54 | 0.27 | 1.81 | 1.01 (0.55) | 0.67 (0.0010) | 0.88 (0.0015) | 0.96 (0.0001) |
| ngons | number_reference | qwen2.5-1.5b__random | 0.33 | 0.10 | 1.20 | 1.12 (0.80) | 0.44 (0.0940) | 0.50 (0.1454) | -0.39 (0.3516) |
| ngons | number_reference | qwen2.5-math-1.5b | 0.53 | 0.23 | 1.71 | 1.05 (0.58) | 0.66 (0.0010) | 0.88 (0.0015) | 0.90 (0.0016) |
| ngons | number_reference | qwen2.5-math-1.5b__random | 0.35 | 0.11 | 1.22 | 1.12 (0.79) | 0.49 (0.0295) | 0.49 (0.1709) | -0.21 (0.2316) |
| dimensions | number_reference | qwen3.5-0.8b | 0.62 | 0.37 | 2.29 | 0.98 (0.27) | 0.63 (0.0010) | 0.93 (0.0005) | -0.71 (0.0030) |
| dimensions | number_reference | qwen3.5-0.8b__random | -0.17 | -0.03 | 0.97 | 0.93 (0.92) | 0.27 (0.8621) | 0.30 (0.3828) | -0.71 (0.6733) |
| dimensions | number_reference | qwen2.5-1.5b | 0.62 | 0.44 | 1.84 | 0.97 (0.31) | 0.59 (0.0005) | 0.82 (0.0030) | 0.80 (0.0038) |
| dimensions | number_reference | qwen2.5-1.5b__random | -0.21 | -0.07 | 0.81 | 0.82 (0.87) | 0.25 (0.8151) | 0.23 (0.4758) | -0.84 (0.9356) |
| dimensions | number_reference | qwen2.5-math-1.5b | 0.68 | 0.39 | 2.16 | 1.05 (0.32) | 0.83 (0.0005) | 0.96 (0.0005) | 0.87 (0.0012) |
| dimensions | number_reference | qwen2.5-math-1.5b__random | -0.21 | -0.07 | 0.80 | 0.81 (0.86) | 0.25 (0.8416) | 0.21 (0.5317) | -0.90 (0.9837) |
| angles | circle | qwen3.5-0.8b | 0.62 | 0.31 | 2.59 | 1.19 (0.33) | 0.73 (0.0005) | 0.93 (0.0005) | 0.56 (0.0002) |
| angles | circle | qwen3.5-0.8b__random | 0.10 | 0.15 | 1.01 | 1.13 (0.83) | 0.28 (0.7626) | 0.00 (1.0000) | -0.42 (0.4584) |
| angles | circle | qwen2.5-1.5b | 0.60 | 0.29 | 2.15 | 1.06 (0.42) | 0.72 (0.0005) | 0.94 (0.0005) | 0.90 (0.0002) |
| angles | circle | qwen2.5-1.5b__random | 0.15 | 0.31 | 1.16 | 1.17 (0.64) | 0.33 (0.5277) | 0.05 (0.8866) | 0.34 (0.1309) |
| angles | circle | qwen2.5-math-1.5b | 0.62 | 0.31 | 1.84 | 1.01 (0.44) | 0.71 (0.0005) | 0.94 (0.0005) | 0.92 (0.0004) |
| angles | circle | qwen2.5-math-1.5b__random | 0.17 | 0.34 | 1.16 | 1.16 (0.64) | 0.31 (0.6197) | 0.05 (0.8866) | 0.27 (0.1645) |

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

**Gemma-4-E2B** with its twin:

| Lattice | Model | axiom-RDM β (p) | lexical β | offset consistency per axiom: obs / null95 (p) | analogy top-1 (chance) | mean rank (chance) |
|---|---|---|---|---|---|---|
| group_like | gemma-4-e2b | +0.301 (0.0018) | +0.437 | associative: 0.56/0.26 (0.0020); identity: 0.19/0.25 (0.0900); divisible: 0.35/0.26 (0.0230); commutative: 0.74/0.22 (0.0005) | 0.56 (0.10) | 2.6 (5.5) |
| self_distributive | gemma-4-e2b | +0.080 (0.4149) | -0.084 | invertible: -0.11/0.84 (0.6112); idempotent: 0.23/0.84 (0.3468) | 0.00 (0.50) | 2.0 (1.5) |
| gender control | gemma-4-e2b | — | — | gender: 0.11/0.10 | king:queen rank 6 | — |
| group_like | gemma-4-e2b__random | +0.275 (0.0054) | -0.017 | associative: 0.22/0.21 (0.0475); identity: 0.14/0.20 (0.0980); divisible: 0.23/0.21 (0.0425); commutative: 0.02/0.19 (0.3683) | 0.47 (0.10) | 3.4 (5.5) |
| self_distributive | gemma-4-e2b__random | +0.087 (0.4553) | +0.146 | invertible: -0.05/0.55 (0.6337); idempotent: 0.00/0.55 (0.4763) | 1.00 (0.50) | 1.0 (1.5) |
| gender control | gemma-4-e2b__random | — | — | gender: -0.00/0.07 | king:queen rank 1 | — |

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

| Family | Expected | Model | line ρ | circle ρ | closure (d first-last / adjacent) | closure_far† (null 5%) | Robinson (p) | PCA order \|ρ\| (p) | LOO order ρ (p) |
|---|---|---|---|---|---|---|---|---|---|
| days | circle | gemma-4-e2b | 0.72 | 0.58 | 1.65 | 1.02 (0.37) | 0.71 (0.0055) | 0.64 (0.1334) | 0.75 (0.0311) |
| days | circle | gemma-4-e2b__random | -0.23 | 0.00 | 0.96 | 0.96 (0.94) | 0.23 (0.8446) | 0.61 (0.1789) | -0.96 (0.9336) |
| months | circle | gemma-4-e2b | 0.61 | 0.36 | 1.99 | 0.97 (0.53) | 0.69 (0.0005) | 0.66 (0.0240) | 0.86 (0.0011) |
| months | circle | gemma-4-e2b__random | 0.27 | 0.05 | 1.02 | 1.05 (0.96) | 0.45 (0.0230) | 0.13 (0.6927) | -0.51 (0.2315) |
| ring_chain | line | gemma-4-e2b | 0.27 | 0.08 | 1.86 | 1.15 (0.53) | 0.50 (0.0100) | 0.42 (0.1999) | 0.43 (0.1051) |
| ring_chain | line | gemma-4-e2b__random | 0.37 | 0.40 | 1.45 | 0.98 (0.45) | 0.28 (0.7816) | 0.05 (0.8821) | 0.50 (0.0427) |
| space_chain | line | gemma-4-e2b | 0.55 | 0.27 | 1.39 | 1.21 (0.64) | 0.57 (0.0005) | 0.95 (0.0005) | 0.82 (0.0050) |
| space_chain | line | gemma-4-e2b__random | -0.06 | -0.00 | 1.00 | 0.97 (0.81) | 0.27 (0.7916) | 0.16 (0.6307) | -0.22 (0.2665) |
| number_systems | line | gemma-4-e2b | 0.92 | 0.54 | 1.94 | 1.56 (0.52) | 0.91 (0.0010) | 1.00 (0.0010) | 0.89 (0.0054) |
| number_systems | line | gemma-4-e2b__random | 0.24 | -0.09 | 1.36 | 2.39 (0.41) | 0.60 (0.0080) | 0.71 (0.0815) | 0.32 (0.1197) |
| ngons | number_reference | gemma-4-e2b | 0.50 | 0.21 | 1.90 | 1.00 (0.49) | 0.64 (0.0015) | 0.87 (0.0025) | 0.89 (0.0010) |
| ngons | number_reference | gemma-4-e2b__random | 0.19 | 0.05 | 1.23 | 1.01 (0.47) | 0.51 (0.0120) | 0.52 (0.1349) | 0.42 (0.0195) |
| dimensions | number_reference | gemma-4-e2b | 0.62 | 0.42 | 2.37 | 1.09 (0.24) | 0.65 (0.0010) | 0.91 (0.0010) | 0.86 (0.0005) |
| dimensions | number_reference | gemma-4-e2b__random | -0.25 | -0.02 | 0.79 | 0.69 (0.74) | 0.23 (0.9040) | 0.12 (0.7451) | -0.57 (0.5497) |
| angles | circle | gemma-4-e2b | 0.43 | 0.23 | 2.03 | 1.03 (0.37) | 0.54 (0.0010) | 0.48 (0.1184) | 0.87 (0.0008) |
| angles | circle | gemma-4-e2b__random | 0.25 | 0.46 | 1.14 | 0.94 (0.77) | 0.39 (0.1914) | 0.36 (0.2599) | 0.73 (0.0042) |

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

**Structure beyond the number reference, every model** (partial RDM regression; anticonservative when the nuisance is only a proxy, §3.5, so the twins' small p-values are expected):

| Family | Model | ρ(family RDM, reference RDM) | template β beyond reference (p) |
|---|---|---|---|
| ngons vs number_words_3_12 | qwen3.5-0.8b | 0.63 | -0.146 (0.6597) |
| ngons vs number_words_3_12 | qwen3.5-0.8b__random | -0.54 | +0.276 (0.0132) |
| ngons vs number_words_3_12 | qwen2.5-1.5b | 0.38 | +0.754 (0.0128) |
| ngons vs number_words_3_12 | qwen2.5-1.5b__random | 0.38 | +0.259 (0.0366) |
| ngons vs number_words_3_12 | qwen2.5-math-1.5b | 0.26 | +0.981 (0.0038) |
| ngons vs number_words_3_12 | qwen2.5-math-1.5b__random | 0.41 | +0.275 (0.0270) |
| dimensions vs number_words_0_10 | qwen3.5-0.8b | 0.87 | +0.075 (0.2509) |
| dimensions vs number_words_0_10 | qwen3.5-0.8b__random | -0.07 | -0.158 (0.8796) |
| dimensions vs number_words_0_10 | qwen2.5-1.5b | 0.80 | +0.176 (0.0606) |
| dimensions vs number_words_0_10 | qwen2.5-1.5b__random | 0.37 | -0.232 (0.9498) |
| dimensions vs number_words_0_10 | qwen2.5-math-1.5b | 0.83 | +0.201 (0.0416) |
| dimensions vs number_words_0_10 | qwen2.5-math-1.5b__random | 0.37 | -0.225 (0.9460) |
| angles vs digits_0_330 | qwen3.5-0.8b | 0.75 | +0.136 (0.0402) |
| angles vs digits_0_330 | qwen3.5-0.8b__random | 0.45 | +0.163 (0.0056) |
| angles vs digits_0_330 | qwen2.5-1.5b | 0.79 | +0.104 (0.0694) |
| angles vs digits_0_330 | qwen2.5-1.5b__random | 0.85 | -0.003 (0.5281) |
| angles vs digits_0_330 | qwen2.5-math-1.5b | 0.72 | +0.200 (0.0050) |
| angles vs digits_0_330 | qwen2.5-math-1.5b__random | 0.85 | +0.029 (0.3051) |

| Family | Model | ρ(family RDM, reference RDM) | template β beyond reference (p) |
|---|---|---|---|
| ngons vs number_words_3_12 | gemma-4-e2b | 0.38 | +0.483 (0.0484) |
| ngons vs number_words_3_12 | gemma-4-e2b__random | -0.06 | +0.189 (0.1364) |
| dimensions vs number_words_0_10 | gemma-4-e2b | 0.64 | +0.330 (0.0358) |
| dimensions vs number_words_0_10 | gemma-4-e2b__random | 0.57 | -0.106 (0.8146) |
| angles vs digits_0_330 | gemma-4-e2b | 0.74 | +0.135 (0.0286) |
| angles vs digits_0_330 | gemma-4-e2b__random | 0.66 | +0.242 (0.0020) |

| Chain | Model | ρ(order, Zipf) | ρ(order, token count) | LOO order beyond covariates (null95, p) |
|---|---|---|---|---|
| ring_chain | gemma-4-e2b | -0.04 | +0.38 | +0.44 (0.66, 0.1728) |
| ring_chain | gemma-4-e2b__random | -0.04 | +0.38 | +0.68 (0.58, 0.0266) |
| space_chain | gemma-4-e2b | -0.10 | -0.39 | +0.75 (0.76, 0.0565) |
| space_chain | gemma-4-e2b__random | -0.10 | -0.39 | -0.37 (0.69, 0.8173) |
| number_systems | gemma-4-e2b | -0.57 | +0.67 | +0.96 (0.89, 0.0266) |
| number_systems | gemma-4-e2b__random | -0.57 | +0.67 | -0.61 (0.88, 0.7907) |

**Word-form baselines on the ordered families** (audit; the token-embedding bag uses Qwen3.5-2B's embeddings):

| Family | Baseline | line ρ | circle ρ | closure | Robinson (p) | PCA order \|ρ\| (p) | LOO order ρ (p) |
|---|---|---|---|---|---|---|---|
| days | char-3-grams | -0.10 | 0.25 | 0.79 | 0.34 (0.8816) | 0.36 (0.4363) | -0.89 (0.4219) |
| days | layer-0 token bag | 0.82 | 0.59 | 1.20 | 0.83 (0.0010) | 0.75 (0.0735) | -1.00 (1.0000) |
| months | char-3-grams | 0.60 | 0.26 | 1.24 | 0.63 (0.0010) | 0.78 (0.0070) | 0.71 (0.0033) |
| months | layer-0 token bag | 0.09 | 0.35 | 0.84 | 0.43 (0.0585) | 0.47 (0.1264) | -1.00 (1.0000) |
| ring_chain | char-3-grams | 0.09 | 0.18 | 1.21 | 0.32 (0.6432) | 0.01 (0.9880) | 0.28 (0.0831) |
| ring_chain | layer-0 token bag | -0.01 | 0.13 | 1.10 | 0.26 (0.8371) | 0.27 (0.4118) | -0.90 (0.4485) |
| space_chain | char-3-grams | -0.18 | 0.01 | 0.85 | 0.33 (0.8076) | 0.10 (0.7756) | -0.78 (0.6777) |
| space_chain | layer-0 token bag | 0.15 | 0.04 | 1.16 | 0.37 (0.3358) | 0.43 (0.1954) | -0.25 (0.0299) |
| number_systems | char-3-grams | 0.30 | 0.06 | 1.22 | 0.37 (0.5122) | 0.75 (0.0580) | 0.00 (0.1262) |
| number_systems | layer-0 token bag | 0.42 | 0.12 | 1.42 | 0.46 (0.1264) | 0.54 (0.2359) | -0.82 (0.0399) |
| ngons | char-3-grams | 0.50 | 0.24 | 1.33 | 0.68 (0.0020) | 0.78 (0.0110) | 0.59 (0.0100) |
| ngons | layer-0 token bag | 0.41 | 0.09 | 1.52 | 0.55 (0.0065) | 0.73 (0.0190) | -0.89 (0.0199) |
| dimensions | char-3-grams | -0.12 | -0.02 | 0.96 | 0.38 (0.8036) | 0.28 (0.4063) | -0.73 (0.4053) |
| dimensions | layer-0 token bag | 0.63 | 0.30 | 1.60 | 0.79 (0.0005) | 0.76 (0.0075) | -1.00 (1.0000) |
| angles | char-3-grams | -0.36 | -0.04 | 0.88 | 0.36 (0.9905) | 0.17 (0.5777) | -0.99 (0.9867) |
| angles | layer-0 token bag | 0.60 | 0.46 | 2.38 | 0.46 (0.0450) | 0.45 (0.1424) | 0.20 (0.0066) |

**Leave-one-out order with 10,000 permutations: the preregistered α = 10 and a scale-relative α** (audit, sensitivity):

| Family | Model | LOO order ρ, α = 10 (p) | LOO order ρ, α = mean ‖x‖² (p) |
|---|---|---|---|
| days | qwen3.5-2b | -0.14 (0.0812) | +0.29 (0.0632) |
| days | qwen3.5-2b__random | -0.93 (0.2202) | -1.00 (1.0000) |
| months | qwen3.5-2b | +0.77 (0.0015) | +0.80 (0.0011) |
| months | qwen3.5-2b__random | -0.78 (0.6252) | -0.93 (0.7179) |
| ring_chain | qwen3.5-2b | +0.65 (0.0085) | +0.66 (0.0072) |
| ring_chain | qwen3.5-2b__random | -0.45 (0.2381) | -0.78 (0.1675) |
| space_chain | qwen3.5-2b | +0.87 (0.0001) | +0.87 (0.0002) |
| space_chain | qwen3.5-2b__random | -0.41 (0.2562) | -0.81 (0.3194) |
| number_systems | qwen3.5-2b | +0.71 (0.0069) | +0.79 (0.0031) |
| number_systems | qwen3.5-2b__random | -0.82 (0.3742) | -0.89 (0.2922) |
| ngons | qwen3.5-2b | +0.90 (0.0001) | +0.90 (0.0001) |
| ngons | qwen3.5-2b__random | -0.89 (0.9457) | -0.93 (0.7830) |
| dimensions | qwen3.5-2b | +0.72 (0.0002) | +0.83 (0.0004) |
| dimensions | qwen3.5-2b__random | -0.75 (0.6614) | -0.95 (0.9105) |
| angles | qwen3.5-2b | +0.76 (0.0010) | +0.82 (0.0009) |
| angles | qwen3.5-2b__random | -0.49 (0.4139) | -0.90 (0.6928) |

| Family | Model | LOO order ρ, α = 10 (p) | LOO order ρ, α = mean ‖x‖² (p) |
|---|---|---|---|
| days | qwen3.5-0.8b | -1.00 (1.0000) | +0.68 (0.0143) |
| days | qwen3.5-0.8b__random | -0.75 (0.1349) | -0.96 (0.6903) |
| days | qwen2.5-1.5b | +0.68 (0.0566) | +0.46 (0.0510) |
| days | qwen2.5-1.5b__random | -0.82 (0.5377) | -0.93 (0.4886) |
| days | qwen2.5-math-1.5b | +0.36 (0.1158) | +0.18 (0.0758) |
| days | qwen2.5-math-1.5b__random | -0.93 (0.7668) | -0.96 (0.6962) |
| months | qwen3.5-0.8b | -0.20 (0.0003) | +0.83 (0.0008) |
| months | qwen3.5-0.8b__random | -0.86 (0.9039) | -0.93 (0.9028) |
| months | qwen2.5-1.5b | +0.80 (0.0033) | +0.77 (0.0022) |
| months | qwen2.5-1.5b__random | -0.62 (0.7031) | -0.84 (0.8083) |
| months | qwen2.5-math-1.5b | +0.82 (0.0013) | +0.80 (0.0012) |
| months | qwen2.5-math-1.5b__random | -0.60 (0.7005) | -0.86 (0.8652) |
| ring_chain | qwen3.5-0.8b | -0.09 (0.0739) | +0.69 (0.0041) |
| ring_chain | qwen3.5-0.8b__random | +0.36 (0.0465) | -0.05 (0.0987) |
| ring_chain | qwen2.5-1.5b | +0.87 (0.0020) | +0.75 (0.0021) |
| ring_chain | qwen2.5-1.5b__random | +0.23 (0.0815) | -0.10 (0.1282) |
| ring_chain | qwen2.5-math-1.5b | +0.89 (0.0012) | +0.83 (0.0003) |
| ring_chain | qwen2.5-math-1.5b__random | +0.25 (0.0760) | +0.07 (0.0714) |
| space_chain | qwen3.5-0.8b | +0.63 (0.0001) | +0.86 (0.0005) |
| space_chain | qwen3.5-0.8b__random | +0.29 (0.0353) | +0.14 (0.0090) |
| space_chain | qwen2.5-1.5b | +0.90 (0.0005) | +0.90 (0.0002) |
| space_chain | qwen2.5-1.5b__random | +0.21 (0.0969) | -0.30 (0.1682) |
| space_chain | qwen2.5-math-1.5b | +0.91 (0.0003) | +0.91 (0.0001) |
| space_chain | qwen2.5-math-1.5b__random | +0.18 (0.1163) | -0.32 (0.1906) |
| number_systems | qwen3.5-0.8b | +0.29 (0.0022) | +0.79 (0.0053) |
| number_systems | qwen3.5-0.8b__random | -0.25 (0.2158) | -0.57 (0.2497) |
| number_systems | qwen2.5-1.5b | +0.79 (0.0129) | +0.64 (0.0129) |
| number_systems | qwen2.5-1.5b__random | -0.21 (0.2270) | -0.43 (0.1861) |
| number_systems | qwen2.5-math-1.5b | +0.71 (0.0163) | +0.61 (0.0105) |
| number_systems | qwen2.5-math-1.5b__random | -0.43 (0.3102) | -0.57 (0.2698) |
| ngons | qwen3.5-0.8b | +0.60 (0.0006) | +0.85 (0.0007) |
| ngons | qwen3.5-0.8b__random | -0.42 (0.3267) | -0.71 (0.3523) |
| ngons | qwen2.5-1.5b | +0.96 (0.0001) | +0.92 (0.0001) |
| ngons | qwen2.5-1.5b__random | -0.39 (0.3516) | -0.59 (0.2853) |
| ngons | qwen2.5-math-1.5b | +0.90 (0.0016) | +0.88 (0.0001) |
| ngons | qwen2.5-math-1.5b__random | -0.21 (0.2316) | -0.45 (0.1983) |
| dimensions | qwen3.5-0.8b | -0.71 (0.0030) | +0.75 (0.0007) |
| dimensions | qwen3.5-0.8b__random | -0.71 (0.6733) | -0.84 (0.6018) |
| dimensions | qwen2.5-1.5b | +0.80 (0.0038) | +0.71 (0.0022) |
| dimensions | qwen2.5-1.5b__random | -0.84 (0.9356) | -0.94 (0.9623) |
| dimensions | qwen2.5-math-1.5b | +0.87 (0.0012) | +0.87 (0.0001) |
| dimensions | qwen2.5-math-1.5b__random | -0.90 (0.9837) | -0.95 (0.9716) |
| angles | qwen3.5-0.8b | +0.56 (0.0002) | +0.90 (0.0001) |
| angles | qwen3.5-0.8b__random | -0.42 (0.4584) | -0.66 (0.4261) |
| angles | qwen2.5-1.5b | +0.90 (0.0002) | +0.94 (0.0001) |
| angles | qwen2.5-1.5b__random | +0.34 (0.1309) | +0.27 (0.1037) |
| angles | qwen2.5-math-1.5b | +0.92 (0.0004) | +0.95 (0.0001) |
| angles | qwen2.5-math-1.5b__random | +0.27 (0.1645) | +0.25 (0.1078) |

| Family | Model | LOO order ρ, α = 10 (p) | LOO order ρ, α = mean ‖x‖² (p) |
|---|---|---|---|
| days | gemma-4-e2b | +0.75 (0.0311) | +0.61 (0.0642) |
| days | gemma-4-e2b__random | -0.96 (0.9336) | -0.96 (0.6713) |
| months | gemma-4-e2b | +0.86 (0.0011) | +0.86 (0.0007) |
| months | gemma-4-e2b__random | -0.51 (0.2315) | -0.92 (0.7344) |
| ring_chain | gemma-4-e2b | +0.43 (0.1051) | +0.01 (0.2564) |
| ring_chain | gemma-4-e2b__random | +0.50 (0.0427) | +0.45 (0.0179) |
| space_chain | gemma-4-e2b | +0.82 (0.0050) | +0.82 (0.0015) |
| space_chain | gemma-4-e2b__random | -0.22 (0.2665) | -0.49 (0.2072) |
| number_systems | gemma-4-e2b | +0.89 (0.0054) | +0.89 (0.0019) |
| number_systems | gemma-4-e2b__random | +0.32 (0.1197) | +0.11 (0.1740) |
| ngons | gemma-4-e2b | +0.89 (0.0010) | +0.78 (0.0015) |
| ngons | gemma-4-e2b__random | +0.42 (0.0195) | +0.15 (0.0708) |
| dimensions | gemma-4-e2b | +0.86 (0.0005) | +0.77 (0.0003) |
| dimensions | gemma-4-e2b__random | -0.57 (0.5497) | -0.76 (0.4272) |
| angles | gemma-4-e2b | +0.87 (0.0008) | +0.80 (0.0012) |
| angles | gemma-4-e2b__random | +0.73 (0.0042) | +0.63 (0.0050) |

Matched 4-field clustering in Gemma-4-E2B:
- math subsets: mean 0.663;
- control fields: 0.866.

# Appendix B: Deviations from the preregistration

See `docs/deviations.md`. Every post-hoc statistic in this paper is labelled post-hoc. The two post-hoc addenda, each committed before its analyses ran, are `reports/prereg_addendum_3d.md` (cycles beyond PC1–2) and `reports/prereg_addendum_lexical.md` (the independent audit).

# AI statement

This codebase and paper were written by Claude Code (Anthropic's Claude, model Opus 5.5) acting as a coding and research agent. A human, Libor Burian, provided the exact specification that was executed. Human cleanup and rewriting are in progress.
