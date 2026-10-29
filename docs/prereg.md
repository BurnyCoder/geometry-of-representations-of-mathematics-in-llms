# Preregistration: geometry of representations of mathematics in LLMs

**Author:** Libor "Burny" Burian. **Date frozen:** 2026-10-29, before any LLM activation was extracted.
**Freeze mechanism:** git tag `prereg-v1`, which points at the commit that contains this file. Any later change to the analysis is logged in [`deviations.md`](deviations.md) with a date and a reason.

## 1. Domain and question

Engels et al. 2024 ([arXiv:2405.14860](https://arxiv.org/abs/2405.14860)) showed that LLMs represent days of the week and months of the year on circles, and that the models use these circles causally for modular arithmetic. We ask two questions:

- **(A)** Do concepts from different mathematical fields form a *map*, meaning representation distances that track how related the fields are?
- **(B)** Do structured concept families have predictable *shapes*? The families are ordered chains, cyclic quantities, and lattices of algebraic structures defined by axioms.

## 2. Materials

### Concepts for (A)

- **Source.** Wikipedia "Glossary of …" pages (CC BY-SA 4.0), plus "List of numerical analysis topics", fetched at the pinned revisions in `data/wiki_revisions.json`.
- **Fields.** 17 math fields and 4 non-math control fields (biology, chemistry, geology, astronomy). The mapping is in `configs/fields.yaml`.
- **Cleaning rules.** All rules are in `src/mathgeom/data/wiki_glossary.py` and `src/mathgeom/data/concepts.py`. They were fixed before extraction, from inspection of term lists only:
  - Keep 1–4-word alphabetic names.
  - Drop single-letter tokens, prefixes, adverbs, and single words of 4 characters or fewer.
  - Drop concepts that appear in more than one field's pages.
- **Flags.** A flagged concept is excluded from the **primary set** but kept for sensitivity analyses. There are three flags:
  - `leak_own`: the name contains its own field's root word, e.g. "topological space".
  - `eponym`: the name contains a proper name. Capitalization is checked against mid-sentence usage on the same page.
  - `polysemous`: a single everyday word with wordfreq Zipf ≥ 4, e.g. "mean", "walk".
- **Sampling.** At most 30 primary concepts per field, sampled with seed 0.
- **Split.** A random 20% of concepts is the **pilot** split, used only for debugging. **All confirmatory statistics use the remaining 80%.**

### Field distance (target for A)

For fields A and B, let S_A and S_B be the sets of arXiv papers listing any of each field's categories. The similarity is the Ochiai coefficient |S_A ∩ S_B| / √(|S_A||S_B|), and the distance is 1 − similarity.

- **Data.** `librarian-bots/arxiv-metadata-snapshot` (CC0), pinned by its commit sha in `data/field_distance.json`.
- **Why not MSC2020.** It is flat above the 2-digit codes, so a tree distance would just be same/different.

### Families for (B)

`configs/families.yaml` lists every item with its ground-truth order or axiom bits and the source of that ground truth. The families:

- **Positive controls:** days and months.
- **Chains:**
  - the ring chain, rng ⊃ … ⊃ algebraically closed field (11 items);
  - the space chain, T0 ⊃ … ⊃ Hilbert space (11 items);
  - number systems (7 items, exploratory).
- **Number-coded families:** n-gons (3–12) and dimensions (0–10), each with a number-word reference.
- **Angles:** 0°–330° in 30° steps, compared against a reference built from the bare numbers "0"–"330", digits only.
- **Lattices:**
  - group-like structures, with bits for associative, identity, divisible and commutative (13 items);
  - self-distributive structures: shelf, rack, spindle, quandle, kei (exploratory);
  - group objects: topological, Lie and algebraic groups (exploratory);
  - a transfer set of ring-like pairs.
- **Offset-method positive control:** gender pairs (man/woman, king/queen, …).

### Models and controls

| Model | Role |
|---|---|
| `Qwen/Qwen3.5-2B-Base` | **Confirmatory MVP** |
| `Qwen/Qwen3.5-0.8B-Base` | Robustness: scale |
| `google/gemma-4-E2B` | Robustness: second model family |
| `Qwen/Qwen2.5-1.5B` vs `Qwen/Qwen2.5-Math-1.5B` | Robustness: math-training ablation |

All models are base (pretrained) checkpoints. **Every model is paired with a random-init twin**: the same architecture and tokenizer, with weights redrawn using seed 0. This follows the control Golechha et al. (ICLR 2025) showed is necessary.

### Representations

- **Read point.** The residual stream (HF `output_hidden_states`) at the concept's last token. Mean-pooling over the concept's tokens is a variant.
- **Template averaging.** Each representation is averaged over 8 templates that end with the concept.
  - Math concepts use the domain word "mathematics"; control concepts use "science".
  - Templates never name a sub-field. A separate "field-revealing" template set is exploratory only.
- **Families** use family-domain templates, e.g. "In abstract algebra, …". The domain is constant within a family.
- **Preprocessing.** Mean-centering across the analysed item set. **No whitening.**
  - Field map: cosine distance.
  - Families and lattices: Euclidean distance. Cosine distance on centered collinear points is degenerate; our synthetic tests showed this.

## 3. Hypotheses, primary metrics, falsification criteria

The primary layer for every hypothesis is fixed in advance at **relative depth 0.5**, i.e. hidden-state index round(0.5·L). Full layer curves are reported. Any "best layer" claim is tested against a max-over-layers permutation null.

**Why the primary tests are paired contrasts against the random-init twin.** Before we touched real data, we simulated partial regression on distance matrices with Freedman–Lane permutation. When the nuisance RDM only noisily measures a confounder (lexical form, frequency), the test rejected far too often:

- 15–22% false positives at the concept level;
- 53% at the field level.

The documented test is in `tests/test_rdm_regression.py`. So every primary test is a **paired contrast**: Δ = statistic(trained) − statistic(control). The random-init twin directly measures form-only similarity on the same concepts. Uncertainty comes from a stratified concept bootstrap with 2,000 resamples, in which both models see the same resampled concepts. Partial-regression results are reported as **secondary/descriptive**.

### H-A1: the field map (primary)

- **Metric.** ρ_field: Spearman correlation between (i) the cosine RDM of field centroids, computed from mean-centered concept representations of primary confirmatory concepts, and (ii) the arXiv field distance.
- **Supported if both hold:**
  - (a) the Mantel test over field labels (10,000 permutations) gives p < 0.05;
  - (b) the 95% bootstrap CI of Δ = ρ_field(trained) − ρ_field(random-init) excludes 0, with Δ > 0.
- **Falsified if** (b) fails.
- **Also reported:** Δ vs layer-0 embeddings; the all-concepts set including flagged concepts; the all-but-the-top variant; mean-pooling; split-half reliability, as a noise ceiling.

### H-A1c: field clustering

- **Metric.** Leave-one-out nearest-centroid field accuracy.
- **Contrast.** Paired Δ vs the random-init twin, with a bootstrap CI.
- **Specificity.** The same quantities for the non-math control fields. This tells us whether math concepts are organized by field more or less than generic topical concepts, i.e. whether any clustering is specific to math.

### H-A2: hierarchy (robustness phase)

The representation should be more tree-like with respect to a hand-made super-tree (algebra / geometry–topology / analysis / foundations / applied) than the random-init model and shuffled trees. Measured by cophenetic correlation.

### H-A3: math training (robustness phase)

ρ_field(Qwen2.5-Math-1.5B) > ρ_field(Qwen2.5-1.5B), tested with a paired bootstrap CI over concepts. A null result will be reported as a null result.

### H-A4: abstraction covariates (secondary)

The top principal axes should correlate with the pure/applied label and with LLM-judge abstraction ratings. Judges come from families we don't probe, and correlations are computed after partialling frequency and token count.

### H-B: shape families

**Per-family statistics**, all unsupervised or leave-one-out:

| Statistic | What it measures |
|---|---|
| Line-template ρ, circle-template ρ | Spearman correlation of the Euclidean RDM with \|i−j\| or with the cyclic distance |
| Robinson score | The fraction of ordered triples consistent with a 1D order |
| Closure ratio | d(first, last) / mean adjacent distance |
| PCA order \|ρ\| | Unsupervised order recovery along the first principal component |
| LOO ridge order ρ | Leave-one-out order recovery |

Each statistic has an order-permutation null computed by the identical procedure.

**Predictions:**

| Family | Prediction | Falsified if |
|---|---|---|
| Days, months | **Circle**: circle ρ > line ρ, closure ratio < 2, Robinson p < .05 | It fails. The pipeline is then declared broken and we debug before interpreting anything else. |
| Ring chain, space chain | **Ordered 1D manifold**: Robinson p < .05 and LOO order p < .05; trained Robinson score > random-init | Either p ≥ .05, or trained ≤ random-init |
| n-gons, dimensions | Follow the number reference | Exploratory: is there template fit beyond the number-word RDM (regression p < .05)? |
| Angles | **Circle**: closure ratio < 2 and circle ρ > line ρ, while the digit reference is line-like | Closure ≥ 2, or circle ρ ≤ line ρ |
| Random 11-item lists of math concepts (100 lists) | **Negative control**: rejection rate ≈ 5% | Rejection rate > 10%, which means the statistics are miscalibrated |

### H-B2: axiom lattices

**Primary metric: offset consistency.** For each axiom, take every pair of structures that differ only in that axiom's bit, and compute the mean cosine between their difference vectors, at the primary layer.

- **Supported for an axiom if both hold:**
  - (a) it is above the 95th percentile of a null built from random pairs in the same family;
  - (b) it is greater in the trained model than in the random-init twin.

**Secondary metrics:**
- the axiom-RDM regression coefficient, with Hamming distance as target and a character-trigram lexical nuisance;
- analogy completion, i.e. top-1 rate and mean rank against chance, over every parallelogram in the lattice.

**Positive control:** gender offsets (man→woman, king→queen, …) must clear the same null. If they don't, the offset method is considered insensitive at this layer.

**Exploratory:** the self-distributive lattice and group objects (n < 8), and transfer of the commutativity/identity directions to ring-like pairs.

## 4. Multiplicity

Holm correction is applied across the confirmatory tests of the MVP:

- H-A1 (b);
- H-B for the ring chain, the space chain and angles;
- H-B2, for each of the 4 group-like axioms.

Everything else is labelled exploratory or secondary.

## 5. Sample size rationale

- **H-A1.** About 24 primary confirmatory concepts per field across 17 fields gives 136 field pairs. That is enough to detect a field-level ρ of about 0.3 with a Mantel test; the simulated power at the field level is in `tests/`.
- **Families** are small by nature (n = 7–13). That is why we use permutation nulls, random-init contrasts, and no supervised fits.

## 6. What would change our minds

**(A)** Suppose the trained model's field-centroid geometry tracks arXiv relatedness no better than its random-init twin. Then any apparent "map of mathematics" is explained by the form of the words (tokens, morphology), not by learned knowledge.

**(B)** Suppose chains are not ordered beyond random-init and beyond random lists, or group-like axiom offsets are no more parallel than random pairs. Then the days-of-the-week style geometry does not extend to abstract algebraic structure at this scale.
