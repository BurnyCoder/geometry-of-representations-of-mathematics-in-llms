# Second independent audit: reproduction, hypothesis evaluation, methods, data and sources

Historical report. Current decisions and corrections appear in the [26 November 2026 review](2026-11-26_scientific_audit.md). The archived numerical observations remain available at the [reviewed revision](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/tree/43dcda70a7f6093e946fbc975c88cd65a7d12e5f).

*Date: 2026-11-22. Generated from `reports/reviews/2026-11-22_independent_audit_template.md` by `python run.py paper`; every number below is resolved from `results/*/*.json`, `configs/` or `data/`. No model was run for this audit (no inference, no training): it reads the cached activations (`artifacts/acts`) and the results files. It was done without relying on the conclusions of the earlier audits; where it reaches the same finding, it says so. Its one new analysis phase, `recheck`, was specified in [`docs/deviations.md`](../../docs/deviations.md) (row 2026-11-22) before its full run.*

## Verdict in brief

| Area | Finding | Action |
|---|---|---|
| Numbers | Every headline statistic recomputes exactly from the activations with independent numpy/scipy code (field ρ, axiom offsets, chain Robinson scores, angles), for all five models and twins | added `tests/test_independent_recompute.py` |
| Hypothesis evaluation | The decision rules in `report/render.py` and `report/audit.py` implement the preregistered criteria (H-A1 (a) and (b); the chain conjunction; H-B2 null and trained > twin; the 8-test Holm family; the angles rule), and every outcome is reported, including the falsified and failed ones | none needed |
| Methods | RSA with a Mantel test, Freedman–Lane permutation, Holm, the intersection-union test, BCa, Robinson scores, cophenetic correlation, relative Gromov δ, ICC(2,·) and cross-validated confound regression are used as their sources define them | none needed; caveats in §3 |
| Concept flags | The frozen rules let through 24 eponyms and non-concepts (not five, as earlier documented) and 27 field roots inside compounds ("subgroup"), against the preregistration's own definitions | H-A1 recomputed without them: holds in all 5 models |
| Abelian transfer (addendum D3) | The "commutative" direction reaches *abelian group − group* mostly as a generic modifier direction: *zorbic/wooden group − group* reach 0.40 to 0.51, abelian group 0.51 to 0.73 | claim weakened in README, paper, summary, site |
| Wording | "Concrete ↔ abstract **is** the top principal axis" overstates ρ = 0.49 on a PC with 6.6% of the variance; "very different pretraining mixtures" was unsupported | reworded |
| Sources | Every cited arXiv identifier and DOI resolves to the cited work (arXiv API, Crossref); content claims about Engels et al., Golechha et al., Kantamneni & Tegmark, Karkada et al., Lei & Cooper, Li et al., Berlioz, Kvinge et al., Timkey & van Schijndel, Mu & Viswanath, Khrulkov et al., Linzen, the Qwen2.5-Math report, the model and SAE cards, the licences, the group-like table revision, MSC2020 and Crans 2004 match the primary sources | none needed |

## 1 Reproduction

`tests/test_independent_recompute.py` re-implements, from the definitions in `docs/prereg.md` and without importing the project's analysis modules, the field-map ρ (H-A1), the offset consistency of the four group-like axioms (H-B2), the ring- and space-chain Robinson scores (H-B) and the angles' closure ratio and line/circle template ρ, at relative depth 0.5 (hidden-state index round(0.5·L)) for all five models and their twins, and compares them with the results files. All agree to within 1e-9. The Mantel p of H-A1 is at most 2/10,001 in every model (Qwen3.5-2B: 0.0001); a separate permutation stream in a scratch check gave the same.

Extraction was read line by line: the concept span comes from the tokenizer's offset mapping with the concept joined by one space, the read point is the span's last token, special tokens are handled consistently, and the twin is `AutoModelForCausalLM.from_config` with the model's own initializer under a fixed seed.

## 2 Hypothesis evaluation against the preregistration

| Hypothesis (status) | Preregistered rule (`docs/prereg.md`) | As evaluated | Reported outcome | Audit verdict |
|---|---|---|---|---|
| H-A1 field map (primary) | (a) Mantel p < .05 and (b) Δρ CI vs twin above 0 | `render.ha1_supported`; (b) also in the Holm family | supported in all 5 models | correct |
| H-A1c clustering | LOO nearest-centroid accuracy, Δ vs twin, control fields | `map_fields.loo_centroid_correct` | reported; specificity claim correctly withdrawn (whole sciences vs sub-fields) | correct |
| H-A2 hierarchy | cophenetic correlation beyond twin and shuffled trees | `phase_hierarchy` | not supported (beats twin in none of the 5 models) | correct |
| H-A3 math training | paired bootstrap Δρ (Math − base) | `phase_compare` | null, CI [-0.059, 0.017] | correct |
| H-A4 axes (secondary) | top PCs correlate with abstraction and pure/applied after partialling frequency and token count | `phase_axes` | PC1 ↔ abstraction ρ = 0.49; PC2 ↔ applied ρ = -0.41 | supported as preregistered (a correlation); "is the top axis" was an overstatement, now reworded |
| H-B days/months (positive control) | circle ρ > line ρ, closure < 2, Robinson p < .05; else debug before interpreting | `phase_shapes`, then post-hoc PC-plane tests | failed as preregistered; rescued only by post-hoc statistics, and said so | correct; the rescue is post-hoc |
| H-B ring/space chains | Robinson p < .05 and LOO order p < .05 (Holm) and trained > twin | `render.chain_p` = max of the two p (intersection-union test) | space chain in all 5 models; ring chain in 3 of 5 models (all but Qwen3.5-0.8B and Gemma-4-E2B) | correct |
| H-B angles | closure < 2 and circle ρ > line ρ | `render.angles_met`; enters Holm with p = 1 when falsified | falsified in every model; rule met by all 5 twins | correct |
| H-B negative control | random 11-item lists rejected at ≈ 5% | `_negative_control` | about 5% | correct (checks the code, not confounds, as stated) |
| H-B2 offsets | above the random-pair null's 95th percentile and trained > twin; Holm over 8 tests | `render.axiom_holm_ok` | associativity Holm-significant in all 5 models, commutativity in all 5 models | correct |
| H-B2 positive control | gender offsets must clear the same null | `phase_lattice` | passes in every trained model (Qwen3.5-2B 0.28 vs null95 0.09; narrowest, Gemma-4-E2B 0.112 vs 0.098) | correct |

The Holm family has the 8 tests of `prereg.md` §4 per model, the H-A1 bootstrap p uses the +1 convention, and the angles rule enters with p = 1. Analyses added after the tag are labelled post-hoc, added, or audit throughout, and the frozen files (`prereg.md`, the two addenda) are unchanged.

## 3 Methods: are they used as their fields use them?

- **RSA and the Mantel test.** Field-level RSA (Kriegeskorte, Mur & Bandettini 2008) with a Spearman correlation and a permutation of field labels (Mantel 1967). The hypothesis concerns the distance matrices themselves, which is the case where the simple Mantel test is valid; Guillot & Rousset (2013) criticize Mantel tests for inferences about the underlying variables under autocorrelation, which this study does not make.
- **Concept bootstrap.** Stratified by field and paired across models, as preregistered. It measures concept-sampling variability with the 17 fields fixed; field-level generalization rests on the Mantel test. Its downward bias on ρ is reported, with BCa intervals (Efron 1987; the jackknife acceleration and the tie-split z0 match scipy's implementation).
- **Freedman–Lane** (Freedman & Lane 1983; Anderson & Legendre 1999) permutes the reduced model's residual matrix by relabelling rows and columns together, the exchangeable unit for distance matrices; it is used only for secondary analyses, as preregistered after the calibration simulation.
- **Multiplicity and conjunctions.** Holm via statsmodels; the chain conjunction takes the larger p (Berger 1982).
- **Shape statistics.** The Robinson score counts triples consistent with a 1-D seriation (chance 1/3); the leave-one-out ridge is the kernel form of `sklearn.linear_model.Ridge` (unit-tested); every statistic has an order-permutation null computed by the same procedure. The preregistered ridge penalty α = 10 is not scale-invariant, which is reported with a scale-relative sensitivity analysis.
- **Tree measures.** Cophenetic correlation of a UPGMA dendrogram (Sokal & Rohlf 1962) and relative Gromov δ = 2δ/diam with δ from the four-point condition (Khrulkov et al. 2019, §3.1), exact over all quadruples.
- **Frequency control.** Cross-validated confound regression (Snoek, Miletić & Scholte 2019), fitted inside each leave-one-out fold, after a whole-sample version was found to leak.
- **Judge agreement.** ICC(2,1) and ICC(2,k) follow Shrout & Fleiss (1979).
- **Caveats, not misuses.** (i) The random-pair null of offset consistency resamples structures with random orientation and possible shared endpoints rather than relabelling the lattice; for unstructured points this widens the null (conservative). (ii) The audit's rule "above every word-form baseline" compares raw cosines across different representation spaces; for associativity, the axiom it supports, the null percentiles of the model and the baselines are similar (Qwen3.5-2B: 0.23 vs 0.26 for character 3-grams), so the comparison is fair there. (iii) The abstraction PCs have arbitrary signs, so only |ρ| is meaningful.

## 4 Ground truth and data

**Families.** The ring chain is the class-inclusion chain of the cited Wikipedia article; the space chain holds with regular = T3 and normal = T4 (Munkres §31), as already documented; the group-like bits, the self-distributive bits (Crans 2004, Def. 55), the partial-operation lattice, the group objects and the transfer pairs are correct by the definitions.

**Field distance.** The Otsuka–Ochiai distance is computed as stated over 17 fields. Two fields map to broader arXiv categories: graph theory to math.CO (all of combinatorics) and game theory to cs.GT. Now stated in the paper's limitations.

**Concept flags.** Reading `data/concepts.csv` against each term's pinned glossary entry (`data/raw/wiki/`) shows two gaps between the frozen code and the preregistration's own definitions:
- `eponym` ("the name contains a proper name") let through at least 15 names of people or places and 6 adjectives formed from names. The common-word rule lowercases surnames such as Hardy, Bass or Tate, and the capital-letter check does not look past a hyphen ("quasi-Frobenius"). 3 further entries are not concept names: "other glossaries" (a navigation heading), "quasi" (a prefix entry) and "coCoA" (the software CoCoA). The earlier documentation counted five eponyms.
- `leak_own` ("the name contains its own field's root word") matches a root only at the start of a word, so 27 primary concepts with the root inside a compound are not flagged, among them 12 of the 22 group-theory concepts ("normal subgroup", "commutator subgroup", …), as well as "submodule", "subcategory", "subgraph" and "subrepresentation".

The hand-checked list, with the glossary evidence, is `configs/flag_audit.yaml`. The primary set is frozen and unchanged; the `recheck` phase recomputes H-A1 without these concepts, with the preregistered statistics, nulls (10,000 Mantel permutations) and seeds (2,000 paired bootstrap resamples):

| Model | Concept set | n concepts (smallest field) | ρ trained (Mantel p) | ρ twin | ρ char-3-grams | ρ layer-0 token bag | Δ − twin [CI] | Δ − char-3-grams [CI] | Δ − layer-0 bag [CI] | LOO acc trained / twin |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | primary (pre-registered) | 398 (11) | 0.387 (0.0001) | -0.118 | +0.105 | +0.157 | +0.506 [0.277, 0.576] | +0.283 [0.135, 0.402] | +0.230 [0.138, 0.342] | 0.472 / 0.090 |
| qwen3.5-2b | without missed eponyms and non-concepts | 374 (11) | 0.367 (0.0001) | -0.084 | +0.109 | +0.150 | +0.451 [0.232, 0.549] | +0.258 [0.124, 0.385] | +0.217 [0.133, 0.333] | 0.473 / 0.088 |
| qwen3.5-2b | without own-field roots inside compounds | 371 (10) | 0.402 (0.0001) | -0.041 | +0.194 | +0.194 | +0.443 [0.218, 0.518] | +0.208 [0.074, 0.340] | +0.208 [0.104, 0.321] | 0.469 / 0.078 |
| qwen3.5-2b | without both | 347 (10) | 0.381 (0.0001) | -0.009 | +0.200 | +0.205 | +0.390 [0.181, 0.480] | +0.181 [0.066, 0.326] | +0.175 [0.086, 0.297] | 0.467 / 0.075 |
| qwen3.5-0.8b | primary (pre-registered) | 398 (11) | 0.369 (0.0001) | -0.182 | +0.105 | +0.218 | +0.551 [0.300, 0.588] | +0.264 [0.116, 0.380] | +0.151 [0.066, 0.271] | 0.430 / 0.163 |
| qwen3.5-0.8b | without missed eponyms and non-concepts | 374 (11) | 0.355 (0.0001) | -0.133 | +0.109 | +0.200 | +0.487 [0.257, 0.559] | +0.245 [0.108, 0.365] | +0.154 [0.072, 0.270] | 0.441 / 0.179 |
| qwen3.5-0.8b | without own-field roots inside compounds | 371 (10) | 0.375 (0.0001) | -0.137 | +0.194 | +0.252 | +0.511 [0.260, 0.551] | +0.181 [0.055, 0.313] | +0.123 [0.043, 0.252] | 0.418 / 0.146 |
| qwen3.5-0.8b | without both | 347 (10) | 0.356 (0.0001) | -0.105 | +0.200 | +0.235 | +0.461 [0.230, 0.521] | +0.156 [0.048, 0.295] | +0.121 [0.028, 0.232] | 0.427 / 0.167 |
| qwen2.5-1.5b | primary (pre-registered) | 398 (11) | 0.406 (0.0001) | -0.093 | +0.105 | +0.178 | +0.499 [0.318, 0.565] | +0.302 [0.152, 0.419] | +0.228 [0.145, 0.364] | 0.565 / 0.219 |
| qwen2.5-1.5b | without missed eponyms and non-concepts | 374 (11) | 0.395 (0.0001) | -0.052 | +0.109 | +0.173 | +0.447 [0.285, 0.536] | +0.286 [0.146, 0.404] | +0.222 [0.141, 0.356] | 0.567 / 0.209 |
| qwen2.5-1.5b | without own-field roots inside compounds | 371 (10) | 0.415 (0.0001) | -0.024 | +0.194 | +0.252 | +0.439 [0.259, 0.516] | +0.221 [0.089, 0.349] | +0.162 [0.092, 0.309] | 0.563 / 0.197 |
| qwen2.5-1.5b | without both | 347 (10) | 0.397 (0.0001) | +0.012 | +0.200 | +0.253 | +0.385 [0.218, 0.483] | +0.197 [0.083, 0.333] | +0.144 [0.075, 0.286] | 0.576 / 0.196 |
| qwen2.5-math-1.5b | primary (pre-registered) | 398 (11) | 0.390 (0.0001) | -0.087 | +0.105 | +0.201 | +0.477 [0.297, 0.540] | +0.285 [0.139, 0.398] | +0.189 [0.117, 0.321] | 0.588 / 0.211 |
| qwen2.5-math-1.5b | without missed eponyms and non-concepts | 374 (11) | 0.378 (0.0001) | -0.060 | +0.109 | +0.185 | +0.438 [0.273, 0.521] | +0.269 [0.136, 0.383] | +0.193 [0.118, 0.321] | 0.594 / 0.195 |
| qwen2.5-math-1.5b | without own-field roots inside compounds | 371 (10) | 0.407 (0.0001) | -0.008 | +0.194 | +0.258 | +0.415 [0.234, 0.502] | +0.213 [0.084, 0.340] | +0.149 [0.082, 0.285] | 0.569 / 0.197 |
| qwen2.5-math-1.5b | without both | 347 (10) | 0.401 (0.0001) | +0.013 | +0.200 | +0.264 | +0.388 [0.214, 0.476] | +0.201 [0.087, 0.328] | +0.137 [0.073, 0.272] | 0.573 / 0.179 |
| gemma-4-e2b | primary (pre-registered) | 398 (11) | 0.436 (0.0001) | -0.021 | +0.105 | +0.209 | +0.457 [0.290, 0.598] | +0.332 [0.192, 0.447] | +0.227 [0.135, 0.355] | 0.417 / 0.196 |
| gemma-4-e2b | without missed eponyms and non-concepts | 374 (11) | 0.412 (0.0001) | -0.022 | +0.109 | +0.214 | +0.434 [0.272, 0.589] | +0.303 [0.173, 0.437] | +0.198 [0.113, 0.342] | 0.420 / 0.203 |
| gemma-4-e2b | without own-field roots inside compounds | 371 (10) | 0.442 (0.0001) | +0.026 | +0.194 | +0.266 | +0.416 [0.240, 0.553] | +0.248 [0.118, 0.379] | +0.176 [0.093, 0.320] | 0.404 / 0.181 |
| gemma-4-e2b | without both | 347 (10) | 0.425 (0.0001) | +0.034 | +0.200 | +0.262 | +0.391 [0.228, 0.541] | +0.225 [0.109, 0.363] | +0.163 [0.073, 0.300] | 0.418 / 0.176 |

Without both groups, ρ = 0.36 to 0.43 (change from the primary set -0.01 to 0.01); H-A1 holds as preregistered in all 5 models, and the map stays above the character-3-gram and token-embedding baselines in all 5 models. Removing the compound leaks raises the word-form baselines' ρ (character 3-grams in Qwen3.5-2B: 0.105 on the primary set, 0.200 on the strict set), so the margin of the trained map over word form shrinks on the strict set.

## 5 The held-out abelian test, with a matched-modifier control

The audit addendum's D3 test asks whether the direction of the five pairs X → commutative X predicts *abelian group − group*, which shares no word with them, against the 72 ordered pairs that do not change commutativity. In all 5 models, exactly one null pair lies closer to the direction than the abelian pair: magma → unital magma (cosine 0.56 to 0.76), which is itself "X → modifier X". The nonce lattice (same templates and domain) supplies the matched control on the same head:

| Model | abelian group − group | commutative group − group | wooden group − group | zorbic group − group | abelian − max(nonce) | D3 null pair closest to the direction | D3 null pairs ≥ abelian |
|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 0.603 | 0.698 | 0.357 | 0.466 | +0.138 | magma → unital magma (0.731) | 1 of 72 |
| qwen3.5-0.8b | 0.544 | 0.727 | 0.462 | 0.508 | +0.036 | magma → unital magma (0.711) | 1 of 72 |
| qwen2.5-1.5b | 0.513 | 0.694 | 0.390 | 0.439 | +0.074 | magma → unital magma (0.571) | 1 of 72 |
| qwen2.5-math-1.5b | 0.548 | 0.694 | 0.329 | 0.396 | +0.152 | magma → unital magma (0.562) | 1 of 72 |
| gemma-4-e2b | 0.734 | 0.915 | 0.496 | 0.499 | +0.235 | magma → unital magma (0.763) | 1 of 72 |

*Abelian group − group* exceeds *wooden group − group* and *zorbic group − group* in all 5 models, by 0.04 to 0.24. Most of its alignment with the "commutative" direction is therefore what composing any modifier with "group" produces; the remaining excess comes from one pair per model and has no uncertainty estimate. The addendum's decision rule D3 is unchanged and still reported; the README, paper, summary, audit report and site now describe the result as weak evidence.

## 6 Sources

Every arXiv identifier and DOI cited in the README, paper, documentation, configuration and code comments was resolved through the arXiv API and Crossref, and each resolves to the work it is cited as (title, authors, year, venue). The content-level claims listed in the verdict table were checked against the primary sources (paper HTML/PDF, model and SAE cards, `config.json` files, licence files, the pinned Wikipedia revisions, msc2020.org); none is wrong. Two location details are finer than the text needs: Khrulkov et al. define δ_rel right after their Eq. 4 (the four-point condition is Eq. 3), and Engels et al.'s Fig. 9 is mainly a continuity plot whose text names the top-2 PCA projection.

## 7 Changes made by this audit

- New phase `recheck` (`pipeline.phase_recheck`, `results/<cfg>/recheck.json`), `configs/flag_audit.yaml`, `flags.leaks_field_anywhere`; tables and facts in `report/audit.py` and `report/render.py`; this review is generated like the other reports (`paper.build_reviews`).
- Tests: `tests/test_independent_recompute.py` (independent recomputation), a test documenting the word-start behaviour of `leak_own`, and the generated-docs test now covers `reports/reviews/`.
- Text: the eponym count, the `leak_own` description, the stricter-set results, the abelian control, the H-A4 wording, the pretraining-mixture sentence and the broader arXiv categories, in the README, paper, summary, audit report, data card, site and `docs/deviations.md`.

## 8 Remaining limitations (not fixed here)

- The primary concept set still contains the concepts in `configs/flag_audit.yaml`; it is frozen by the preregistration, and the stricter sets are a sensitivity analysis.
- The days/months positive control failed its preregistered criterion; the pipeline's validity rests on post-hoc statistics (PC-plane cycles, closed cycles), as the paper states.
- The space chain's separation-axiom block is ordered in 1 of 5 models (Qwen3.5-2B); the evidence for order within a block comes from its five-item metric block, where the exact test is at its floor in every model.
- Twin seeds vary only for the confirmatory model; abstraction ratings come from LLM judges; family sizes limit every shape test.
