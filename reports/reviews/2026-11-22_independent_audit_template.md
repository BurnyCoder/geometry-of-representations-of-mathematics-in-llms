# Second independent audit: reproduction, hypothesis evaluation, methods, data and sources

Historical report. Current decisions and corrections appear in the [26 November 2026 review](2026-11-26_scientific_audit.md). The archived numerical observations remain available at the [reviewed revision](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/tree/43dcda70a7f6093e946fbc975c88cd65a7d12e5f).

*Date: 2026-11-22. Generated from `reports/reviews/2026-11-22_independent_audit_template.md` by `python run.py paper`; every number below is resolved from `results/*/*.json`, `configs/` or `data/`. No model was run for this audit (no inference, no training): it reads the cached activations (`artifacts/acts`) and the results files. It was done without relying on the conclusions of the earlier audits; where it reaches the same finding, it says so. Its one new analysis phase, `recheck`, was specified in [`docs/deviations.md`](../../docs/deviations.md) (row 2026-11-22) before its full run.*

## Verdict in brief

| Area | Finding | Action |
|---|---|---|
| Numbers | Every headline statistic recomputes exactly from the activations with independent numpy/scipy code (field ρ, axiom offsets, chain Robinson scores, angles), for all five models and twins | added `tests/test_independent_recompute.py` |
| Hypothesis evaluation | The decision rules in `report/render.py` and `report/audit.py` implement the preregistered criteria (H-A1 (a) and (b); the chain conjunction; H-B2 null and trained > twin; the 8-test Holm family; the angles rule), and every outcome is reported, including the falsified and failed ones | none needed |
| Methods | RSA with a Mantel test, Freedman–Lane permutation, Holm, the intersection-union test, BCa, Robinson scores, cophenetic correlation, relative Gromov δ, ICC(2,·) and cross-validated confound regression are used as their sources define them | none needed; caveats in §3 |
| Concept flags | The frozen rules let through {{num:mvp.recheck.n_named}} eponyms and non-concepts (not five, as earlier documented) and {{num:mvp.recheck.n_compound}} field roots inside compounds ("subgroup"), against the preregistration's own definitions | H-A1 recomputed without them: holds in {{fact:strict_ha1.strict}} |
| Abelian transfer (addendum D3) | The "commutative" direction reaches *abelian group − group* mostly as a generic modifier direction: *zorbic/wooden group − group* reach {{fact:nonce_group_range}}, abelian group {{fact:abelian_range}} | claim weakened in README, paper, summary, site |
| Wording | "Concrete ↔ abstract **is** the top principal axis" overstates ρ = {{num:mvp.axes.qwen3.5-2b.pcs.0.rho_abstraction:.2f}} on a PC with {{num:mvp.axes.qwen3.5-2b.explained_var.0:.1%}} of the variance; "very different pretraining mixtures" was unsupported | reworded |
| Sources | Every cited arXiv identifier and DOI resolves to the cited work (arXiv API, Crossref); content claims about Engels et al., Golechha et al., Kantamneni & Tegmark, Karkada et al., Lei & Cooper, Li et al., Berlioz, Kvinge et al., Timkey & van Schijndel, Mu & Viswanath, Khrulkov et al., Linzen, the Qwen2.5-Math report, the model and SAE cards, the licences, the group-like table revision, MSC2020 and Crans 2004 match the primary sources | none needed |

## 1 Reproduction

`tests/test_independent_recompute.py` re-implements, from the definitions in `docs/prereg.md` and without importing the project's analysis modules, the field-map ρ (H-A1), the offset consistency of the four group-like axioms (H-B2), the ring- and space-chain Robinson scores (H-B) and the angles' closure ratio and line/circle template ρ, at relative depth 0.5 (hidden-state index round(0.5·L)) for all five models and their twins, and compares them with the results files. All agree to within 1e-9. The Mantel p of H-A1 is at most 2/10,001 in every model (Qwen3.5-2B: {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_perm_p:.4f}}); a separate permutation stream in a scratch check gave the same.

Extraction was read line by line: the concept span comes from the tokenizer's offset mapping with the concept joined by one space, the read point is the span's last token, special tokens are handled consistently, and the twin is `AutoModelForCausalLM.from_config` with the model's own initializer under a fixed seed.

## 2 Hypothesis evaluation against the preregistration

| Hypothesis (status) | Preregistered rule (`docs/prereg.md`) | As evaluated | Reported outcome | Audit verdict |
|---|---|---|---|---|
| H-A1 field map (primary) | (a) Mantel p < .05 and (b) Δρ CI vs twin above 0 | `render.ha1_supported`; (b) also in the Holm family | supported in {{fact:field_map_beats_twin}} | correct |
| H-A1c clustering | LOO nearest-centroid accuracy, Δ vs twin, control fields | `map_fields.loo_centroid_correct` | reported; specificity claim correctly withdrawn (whole sciences vs sub-fields) | correct |
| H-A2 hierarchy | cophenetic correlation beyond twin and shuffled trees | `phase_hierarchy` | not supported (beats twin in {{fact:hierarchy_beats_twin}}) | correct |
| H-A3 math training | paired bootstrap Δρ (Math − base) | `phase_compare` | null, CI [{{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.0:.3f}}, {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.1:.3f}}] | correct |
| H-A4 axes (secondary) | top PCs correlate with abstraction and pure/applied after partialling frequency and token count | `phase_axes` | PC1 ↔ abstraction ρ = {{num:mvp.axes.qwen3.5-2b.pcs.0.rho_abstraction:.2f}}; PC2 ↔ applied ρ = {{num:mvp.axes.qwen3.5-2b.pcs.1.rho_applied:.2f}} | supported as preregistered (a correlation); "is the top axis" was an overstatement, now reworded |
| H-B days/months (positive control) | circle ρ > line ρ, closure < 2, Robinson p < .05; else debug before interpreting | `phase_shapes`, then post-hoc PC-plane tests | failed as preregistered; rescued only by post-hoc statistics, and said so | correct; the rescue is post-hoc |
| H-B ring/space chains | Robinson p < .05 and LOO order p < .05 (Holm) and trained > twin | `render.chain_p` = max of the two p (intersection-union test) | space chain in {{fact:chain_ordered.space_chain}}; ring chain in {{fact:chain_ordered.ring_chain}} | correct |
| H-B angles | closure < 2 and circle ρ > line ρ | `render.angles_met`; enters Holm with p = 1 when falsified | falsified in every model; rule met by {{fact:twins_meet_angles_rule}} | correct |
| H-B negative control | random 11-item lists rejected at ≈ 5% | `_negative_control` | about 5% | correct (checks the code, not confounds, as stated) |
| H-B2 offsets | above the random-pair null's 95th percentile and trained > twin; Holm over 8 tests | `render.axiom_holm_ok` | associativity Holm-significant in {{fact:axiom_holm.associative}}, commutativity in {{fact:axiom_holm.commutative}} | correct |
| H-B2 positive control | gender offsets must clear the same null | `phase_lattice` | passes in every trained model (Qwen3.5-2B {{num:mvp.lattice.qwen3.5-2b.analogy_control.consistency:.2f}} vs null95 {{num:mvp.lattice.qwen3.5-2b.analogy_control.null95:.2f}}; narrowest, Gemma-4-E2B {{num:gemma.lattice.gemma-4-e2b.analogy_control.consistency:.3f}} vs {{num:gemma.lattice.gemma-4-e2b.analogy_control.null95:.3f}}) | correct |

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
- **Caveats, not misuses.** (i) The random-pair null of offset consistency resamples structures with random orientation and possible shared endpoints rather than relabelling the lattice; for unstructured points this widens the null (conservative). (ii) The audit's rule "above every word-form baseline" compares raw cosines across different representation spaces; for associativity, the axiom it supports, the null percentiles of the model and the baselines are similar (Qwen3.5-2B: {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.associative.null95:.2f}} vs {{num:mvp.lexical.qwen3.5-2b.group_like.offsets.char3.associative.null95:.2f}} for character 3-grams), so the comparison is fair there. (iii) The abstraction PCs have arbitrary signs, so only |ρ| is meaningful.

## 4 Ground truth and data

**Families.** The ring chain is the class-inclusion chain of the cited Wikipedia article; the space chain holds with regular = T3 and normal = T4 (Munkres §31), as already documented; the group-like bits, the self-distributive bits (Crans 2004, Def. 55), the partial-operation lattice, the group objects and the transfer pairs are correct by the definitions.

**Field distance.** The Otsuka–Ochiai distance is computed as stated over {{num:mvp.map_fields.qwen3.5-2b.math_primary.n_fields}} fields. Two fields map to broader arXiv categories: graph theory to math.CO (all of combinatorics) and game theory to cs.GT. Now stated in the paper's limitations.

**Concept flags.** Reading `data/concepts.csv` against each term's pinned glossary entry (`data/raw/wiki/`) shows two gaps between the frozen code and the preregistration's own definitions:
- `eponym` ("the name contains a proper name") let through at least {{fact:flag_list_n.people_and_places}} names of people or places and {{fact:flag_list_n.adjectives}} adjectives formed from names. The common-word rule lowercases surnames such as Hardy, Bass or Tate, and the capital-letter check does not look past a hyphen ("quasi-Frobenius"). {{fact:flag_list_n.non_concepts}} further entries are not concept names: "other glossaries" (a navigation heading), "quasi" (a prefix entry) and "coCoA" (the software CoCoA). The earlier documentation counted five eponyms.
- `leak_own` ("the name contains its own field's root word") matches a root only at the start of a word, so {{num:mvp.recheck.n_compound}} primary concepts with the root inside a compound are not flagged, among them {{fact:compound_share.group_theory}} group-theory concepts ("normal subgroup", "commutator subgroup", …), as well as "submodule", "subcategory", "subgraph" and "subrepresentation".

The hand-checked list, with the glossary evidence, is `configs/flag_audit.yaml`. The primary set is frozen and unchanged; the `recheck` phase recomputes H-A1 without these concepts, with the preregistered statistics, nulls (10,000 Mantel permutations) and seeds (2,000 paired bootstrap resamples):

{{table:flag_sensitivity:all}}

Without both groups, ρ = {{fact:strict_rho_range.strict}} (change from the primary set {{fact:strict_rho_change_range.strict}}); H-A1 holds as preregistered in {{fact:strict_ha1.strict}}, and the map stays above the character-3-gram and token-embedding baselines in {{fact:strict_beyond.strict}}. Removing the compound leaks raises the word-form baselines' ρ (character 3-grams in Qwen3.5-2B: {{num:mvp.recheck.qwen3.5-2b.flag_sensitivity.primary.rsa.char3:.3f}} on the primary set, {{num:mvp.recheck.qwen3.5-2b.flag_sensitivity.strict.rsa.char3:.3f}} on the strict set), so the margin of the trained map over word form shrinks on the strict set.

## 5 The held-out abelian test, with a matched-modifier control

The audit addendum's D3 test asks whether the direction of the five pairs X → commutative X predicts *abelian group − group*, which shares no word with them, against the 72 ordered pairs that do not change commutativity. In {{fact:abelian_beaten_by_null}}, exactly one null pair lies closer to the direction than the abelian pair: {{fact:abelian_null_top_pairs}} (cosine {{fact:abelian_null_top_range}}), which is itself "X → modifier X". The nonce lattice (same templates and domain) supplies the matched control on the same head:

{{table:abelian_modifiers:all}}

*Abelian group − group* exceeds *wooden group − group* and *zorbic group − group* in {{fact:abelian_beats_nonce}}, by {{fact:abelian_margin_range}}. Most of its alignment with the "commutative" direction is therefore what composing any modifier with "group" produces; the remaining excess comes from one pair per model and has no uncertainty estimate. The addendum's decision rule D3 is unchanged and still reported; the README, paper, summary, audit report and site now describe the result as weak evidence.

## 6 Sources

Every arXiv identifier and DOI cited in the README, paper, documentation, configuration and code comments was resolved through the arXiv API and Crossref, and each resolves to the work it is cited as (title, authors, year, venue). The content-level claims listed in the verdict table were checked against the primary sources (paper HTML/PDF, model and SAE cards, `config.json` files, licence files, the pinned Wikipedia revisions, msc2020.org); none is wrong. Two location details are finer than the text needs: Khrulkov et al. define δ_rel right after their Eq. 4 (the four-point condition is Eq. 3), and Engels et al.'s Fig. 9 is mainly a continuity plot whose text names the top-2 PCA projection.

## 7 Changes made by this audit

- New phase `recheck` (`pipeline.phase_recheck`, `results/<cfg>/recheck.json`), `configs/flag_audit.yaml`, `flags.leaks_field_anywhere`; tables and facts in `report/audit.py` and `report/render.py`; this review is generated like the other reports (`paper.build_reviews`).
- Tests: `tests/test_independent_recompute.py` (independent recomputation), a test documenting the word-start behaviour of `leak_own`, and the generated-docs test now covers `reports/reviews/`.
- Text: the eponym count, the `leak_own` description, the stricter-set results, the abelian control, the H-A4 wording, the pretraining-mixture sentence and the broader arXiv categories, in the README, paper, summary, audit report, data card, site and `docs/deviations.md`.

## 8 Remaining limitations (not fixed here)

- The primary concept set still contains the concepts in `configs/flag_audit.yaml`; it is frozen by the preregistration, and the stricter sets are a sensitivity analysis.
- The days/months positive control failed its preregistered criterion; the pipeline's validity rests on post-hoc statistics (PC-plane cycles, closed cycles), as the paper states.
- The space chain's separation-axiom block is ordered in {{fact:space_block.0}}; the evidence for order within a block comes from its five-item metric block, where the exact test is at its floor in every model.
- Twin seeds vary only for the confirmatory model; abstraction ratings come from LLM judges; family sizes limit every shape test.
