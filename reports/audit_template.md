# Independent audit: word-form controls, twin seeds and test precision

*Generated from `reports/audit_template.md` by `python run.py paper`; every number comes from `results/*/*.json`. Everything here is **post-hoc**: the audit ran after all preregistered results were known. Its tests were defined in [`prereg_addendum_lexical.md`](prereg_addendum_lexical.md), committed before the `lexical` and `twins` phases ran (one amendment, made from a unit test before any real-data run, is recorded there). The log of changes is in [`docs/deviations.md`](../docs/deviations.md).*

## Why an audit was needed

The preregistration uses a random-init twin (same architecture and tokenizer, random weights) as the measure of "form-only similarity". It is not one. In the Qwen3.5-2B twin, the pairs magma → semigroup and commutative magma → commutative semigroup differ only by the literal word "commutative", yet their difference vectors have cosine {{num:mvp.lexical.qwen3.5-2b.group_like.decomposition.twin.associative.dup_mean:.2f}}. The trained model gives {{num:mvp.lexical.qwen3.5-2b.group_like.decomposition.trained.associative.dup_mean:.2f}}, and a bag of words gives 1. A random network can scramble even exact word composition (cosine of such duplicated pairs, range over three axioms: {{fact:twin_dup_by_model}}), so "beats the twin" never ruled out that a result follows from the names alone. Mathematical names often spell out their own structure ("commutative monoid", "topological group", "… domain"), so this matters here more than for days or months.

## Method

Every test is rerun, with the same statistic, centring, type of null and seed as for the trained model, on three **word-form baselines** built from the names alone, without context. Two nulls are smaller for the baselines: their leave-one-out order null has 300 permutations (trained models and twins: 10,000), and their field-map Δ bootstrap 1,000 resamples (2,000). The chain, within-block and nonce tests use only the character 3-grams and the layer-0 token bag, because one-word items give the word bag a constant RDM.

| Baseline | What it encodes |
|---|---|
| character 3-grams | counts of the lower-cased name's letter triples, with word boundaries |
| word bag | counts of the name's words |
| layer-0 token bag | the sum of the model's *trained* input embeddings over the name's tokens: token meaning, no context |

The audit also adds:
- tests that locate where the axiom offsets come from (lexically duplicated pairs, the names without "commutative", a held-out pair, nonce modifiers);
- exact within-block order tests for the chains, and a simulation of how diagnostic the chain test is;
- 4 extra random-init seeds for the confirmatory model's twin;
- 10,000 leave-one-out permutations, a scale-relative ridge penalty, BCa bootstrap intervals, and an 8-test Holm family.

## Verdicts

| Result (pre-registered status) | Pre-registered outcome | Word-form controls | Verdict |
|---|---|---|---|
| H-A1 field map (primary, confirmatory) | beats the twin in {{fact:field_map_beats_twin}} | beats all three baselines in {{fact:field_beyond_lexical}} | **holds** |
| H-B space chain (confirmatory) | ordered in {{fact:chain_ordered.space_chain}} | spelling does not order it; ordered within metrizable … Hilbert in {{fact:space_block.1}} and within T0 … normal in {{fact:space_block.0}} | **holds** |
| H-B ring chain (confirmatory) | ordered in {{fact:chain_ordered.ring_chain}} | head-noun clusters without order pass the test in up to {{num:mvp.lexical.block_only_simulation.2.reject_rate:.0%}} of simulations; the six "… domain" items are ordered in {{fact:ring_within}} | **not established beyond head nouns** |
| H-B2 associativity offset (confirmatory) | Holm-significant in {{fact:axiom_holm.associative}} | above every baseline, disjoint pairs pass, in {{fact:axiom_beyond_lexical.associative}} | **suggestive beyond word form** |
| H-B2 commutativity offset (confirmatory) | Holm-significant in {{fact:axiom_holm.commutative}} | beyond word form in {{fact:axiom_beyond_lexical.commutative}}; "commutative" beats "wooden" and "zorbic" in {{fact:nonce_above}}; the direction predicts abelian group − group in {{fact:abelian_transfer}} | **mostly composition with a word; one held-out pair carries meaning** |
| H-B2 identity offset (confirmatory) | Holm-significant in {{fact:axiom_holm.identity}} | beyond word form in {{fact:axiom_beyond_lexical.identity}}; disjoint cosines {{fact:disjoint_range.identity}} | **lexical duplication** |
| H-B2 divisibility offset (confirmatory) | Holm-significant in {{fact:axiom_holm.divisible}} | beyond word form in {{fact:axiom_beyond_lexical.divisible}} | **mostly not beyond word form** |
| Group objects (exploratory) | parallelograms clear their null in {{fact:parallelograms}} | above every baseline in {{fact:group_objects_beyond}} | **word form** |

## D1: field map

{{table:lexical_field:all}}

Spelling alone gives no significant field map (character 3-grams: Mantel p = {{num:mvp.lexical.qwen3.5-2b.field_map.mantel_p.char3:.2f}}; the same in every model, since it needs no model). The layer-0 token bag, which carries the model's learned token meanings without context, gives a weak map: ρ = {{fact:lexical_field_range.layer0_bag}}, Mantel p = {{fact:lexical_field_p_range.layer0_bag}} (unadjusted), significant in {{fact:lexical_field_significant.layer0_bag}}.

## D11: bootstrap bias and BCa intervals

Concept resampling duplicates concepts, which pulls resampled ρ toward 0, so the preregistered percentile intervals of positive ρ sit below the point estimates. BCa corrects this: the trained and Δ intervals move up, and the twins' intervals (negative ρ) move down; the twin's BCa interval lies entirely below 0 in {{fact:bca_twin_below_zero}}. No hypothesis decision changes.

{{table:bootstrap:all}}

## D2–D3, D13: axiom offsets

Offsets per axiom: trained (from `lattice.json`), each word-form baseline, and where the cosines come from. *Duplicated* cosines compare two pairs that coincide after deleting "commutative"; *disjoint* cosines are all the others.

{{table:lexical_offsets:all}}

The 8 names without "commutative" (3 pairs per axiom):

{{table:lexical_sublattice:all}}

Does the direction of the five pairs that add "commutative" predict *abelian group − group*, which shares no word with it? The exact null uses the 72 ordered pairs that do not change commutativity:

{{table:abelian:all}}

How consistent is adding *any* modifier? X → m X for the same six heads:

{{table:nonce:all}}

## D4: group objects

{{table:lexical_group_objects:all}}

## D5–D7: chains

Exact order tests within blocks of constant head noun or topic, over all n! orders:

{{table:blocks:all}}

How often do clusters of the ring chain's head nouns, with no order at all, pass the preregistered chain test (Robinson p < .05 and leave-one-out p < .05)?

{{table:block_simulation:mvp}}

Ordered families on word-form baselines (Qwen3.5-2B):

{{table:lexical_chains:mvp}}

## D8: Holm over all 8 preregistered tests

Together with the 10,000-permutation LOO p-values (D9), this changes three decisions relative to the pre-audit rules (7-test Holm, 300 permutations): Gemma-4-E2B's space chain becomes Holm-significant, and identity and divisibility lose Holm significance in Qwen2.5-Math-1.5B. The confirmatory model:

{{table:holm:mvp}}

## D9–D10: leave-one-out order, 10,000 permutations, two ridge penalties

{{table:loo_alpha:mvp}}

{{table:loo_alpha:robustness}}

{{table:loo_alpha:gemma}}

## D12: twin seeds

{{table:twins:twin_seeds}}

## What changed in the claims

- The field map and the space chain are now the lead results; both survive every word-form control.
- "Axioms act as directions" is replaced by "axiom offsets are consistent, but mostly because the names encode the axioms". An associativity direction, and a commutativity direction that reaches "abelian group", are suggestive exceptions.
- "Group objects combine their parents" is withdrawn as evidence of mathematical knowledge: the parallelograms follow shared words.
- The ring chain's order is reported as not established beyond its head nouns.
- "The field map is not specific to mathematics" is withdrawn. The control fields are whole sciences, so the comparison confounds domain with granularity, and specificity is untested.
- "The twin measures form-only similarity" is corrected everywhere except in the frozen preregistration. How much a twin keeps exact word composition depends on the architecture: little in the Qwen3.5 twins, most in Gemma's.

## Limitations of the audit

- **Post-hoc.** It was prompted by the results. Its tests were fixed in writing before they ran, but the audit's read-only exploration had already seen several of their values (listed in the addendum).
- **Baselines are lower bounds on word form.** Character 3-grams, bags of words and context-free token embeddings do not exhaust what the names alone can carry. Learned morphology, for example, could still contribute to the field map and the space chain.
- **Granularity.** A granularity-matched specificity control, sub-fields of another science, was not run.
