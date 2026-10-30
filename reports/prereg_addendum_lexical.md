# Addendum to the preregistration: lexical controls, twin seeds and test precision (post-hoc)

**Status.** Written and committed *before* the `lexical` and `twins` phases were run, as an addendum to [`docs/prereg.md`](../docs/prereg.md) (git tag `prereg-v1`). Everything it defines is **post-hoc**. An independent audit (2026-10-30) prompted it, after all preregistered results were known.

**What we had already seen.** The audit recomputed the following read-only from the cached activations, with the repo's own functions. This addendum fixes their exact definitions, so that the new phase reproduces them from code:
- Offset consistency of the group-like axioms on character-3-gram count vectors and on layer-0 token embeddings (Qwen3.5-2B, Qwen2.5-1.5B and Gemma-4-E2B).
- The split of each axiom's cosines into lexically duplicated and disjoint pairs, the 8-item sub-lattice without "commutative" names, and the held-out *abelian group − group* cosine (all 5 models and twins).
- Group-object parallelograms on character 3-grams and the layer-0 bag, with an 8-item pool (not the 113-item pool of `structures.json`).
- Shape statistics of the chains on lexical baselines (Qwen3.5-2B only).
- The within-"domain" ring test, the space-chain sub-block tests and a block-only simulation (all 5 models).
- The ring chain's LOO-order p with 10,000 permutations, and Holm with the angles test counted.
- Field-map ρ for character 3-grams, a word bag and the layer-0 bag (Qwen3.5-2B only).
- The bootstrap bias of the Qwen3.5-2B field ρ (400 resamples).

**What we had not seen:** extra random-init seeds, the nonce-modifier lattice, the scale-relative ridge penalty, BCa intervals, and the lexical field-map baselines for the four replication models.

## Why

The preregistration treats the random-init twin as the measure of "form-only similarity" (`prereg.md` §3). The audit found that the twin does not play that role. In the Qwen3.5-2B twin, two pairs that differ only by the literal prefix "commutative" (magma → semigroup vs commutative magma → commutative semigroup) have a difference-vector cosine of 0.04–0.07. The trained model gives 0.83–0.92, and a bag of words gives about 1. A random network scrambles even exact word-level composition, so "beats the twin" does not rule out word form. The controls below measure word form directly.

## Lexical baselines (no learned context)

For a list of names, each baseline gives one vector per name:

| Baseline | Definition |
|---|---|
| `char3` | Counts of character 3-grams of the lower-cased name padded with one space on each side. These are the n-grams of `preprocess.char_ngram_rdm`. |
| `word_bag` | Counts of lower-cased whitespace-separated words. |
| `layer0_bag` | The sum of the model's *trained* input embeddings over the name's tokens: `pool[:, 0] · n_tokens` from the cached activations. It is template-independent. |

Each baseline is analysed with exactly the statistic, centring, null and seed of the trained model's test.

## Tests and decision rules

All tests run at the preregistered layer, for every model in `mvp`, `robustness` and `gemma`, and for every twin where stated.

**D1. Field map beyond word form.**
- ρ_field and its Mantel p (10,000 permutations) for each baseline.
- A paired stratified concept bootstrap (`map_fields.paired_bootstrap`, 1,000 resamples, seed 0) of ρ(trained) − ρ(baseline).
- *Beyond word form* if all three Δ CIs exclude 0.

**D2. Axiom offsets beyond word form**, per axiom.
- **Lexical duplicates.** Two same-axiom pairs are lexical duplicates if they are identical after deleting "commutative " from all four names. Example: magma → semigroup and commutative magma → commutative semigroup. There are 1, 2, 1 and 0 such cosine pairs for associativity, identity, divisibility and commutativity.
- **Disjoint statistic.** The mean cosine over the remaining (disjoint) pairs.
- **Null.** The same index pairs evaluated on random structure pairs drawn from the 13 items: 2,000 draws, seed `seed + axis`.
- *Beyond word form* if (a) the trained offset consistency exceeds that of every lexical baseline, and (b) the disjoint p < .05.
- **Descriptive.** The same statistics on the 8 names without "commutative" (magma, quasigroup, unital magma, loop, semigroup, monoid, group, abelian group), with 5,000 null draws.

**D3. Held-out commutativity.**
- **Statistic.** The direction is the normalized mean of the 5 unit difference vectors X → commutative X. The statistic is the cosine of *abelian group − group* with that direction.
- **Null.** Exact enumeration of every ordered pair (a, b), a ≠ b, of the 13 items, excluding every commutativity pair in either orientation. p = (1 + #{null ≥ observed}) / (1 + #null).
- *Semantic transfer* if p < .05 and the trained cosine exceeds every baseline's.

**D4. Group objects beyond word form.**
- The parallelogram test of `structures.json`, with the same 113-item pool and seed, run on each baseline.
- *Beyond word form* if the trained cosine exceeds every baseline's.

**D5. Ring chain beyond head nouns.**
- **Items.** The six items whose names end in "domain" (integral … Euclidean domain).
- **Test.** Robinson score, with an exact p over all 720 orders (fraction of orders scoring ≥ observed).
- *Order beyond head-noun blocks* if p < .05.
- **Also reported:** the PC1 |ρ| with its exact p, and the lexical baselines on the whole chain.

**D6. Space chain within topical blocks.** The same exact test within T0 … normal space (6 items) and within metrizable … Hilbert space (5 items).

**D7. Block-only simulation.**
- **Representation.** Equidistant clusters for the ring chain's head nouns (rng | ring, commutative ring | 6 × domain | field, algebraically closed field), with no order between or within clusters. Cluster SNR ∈ {0.5, 1, 2}, d = 2048, 40 seeds each.
- **Reported:** how often the preregistered conjunction (Robinson p < .05 and LOO p < .05; 2,000 and 300 permutations) rejects. This measures how diagnostic the chain test is.

**D8. Holm family (correction).**
- `prereg.md` §4 lists 8 tests. Holm now runs over all 8. The angles prediction enters with p = 1 when its decision rule fails, and p = 0 when it is met.
- This replaces the 7-test reading of 2026-10-30. It is more conservative and matches the frozen text.

**D9. Permutation precision.** The LOO-order null uses 10,000 permutations instead of 300. A kernel-form ridge, identical to `sklearn.linear_model.Ridge(alpha=10)` (unit test), makes this affordable. The statistic is unchanged.

**D10. Ridge-penalty sensitivity.**
- The preregistered α = 10 is not scale-invariant: mean ‖x‖² ranges from about 1.4 to 5,466 across models and twins.
- **Sensitivity variant (descriptive).** α_rel = the mean diagonal of the training kernel (mean squared norm of the centred training items) in each fold.
- **Reported:** LOO ρ and p (10,000 permutations) for every chain, model and twin.

**D11. Bootstrap bias and BCa (descriptive).**
- **Bias.** Bootstrap mean − point estimate, for the H-A1 trained ρ, the H-A1 Δ vs twin, and the H-A3 Δ.
- **BCa.** 95% BCa intervals ([Efron 1987, JASA 82:171](https://doi.org/10.1080/01621459.1987.10478410)). The acceleration comes from a leave-one-concept-out jackknife.
- The percentile CIs remain the preregistered ones.

**D12. Twin seeds (Qwen3.5-2B only).**
- Random-init twins with seeds 1–4, in addition to seed 0, through the same extraction.
- **Statistics:** field ρ and LOO field accuracy (primary concepts), the 4 group-like offsets, and the ring and space Robinson scores.
- *Seed-robust* contrast: the trained value exceeds the maximum over the 5 twin seeds. For H-A1, the smallest lower Δ-CI bound over seeds must also be > 0.

**D13. Nonce modifiers.**
- **Items.** Six heads (magma, quasigroup, loop, semigroup, monoid, group), each with three modifiers: "commutative" (including "commutative group"), "wooden" (a real adjective unrelated to algebra) and "zorbic" (a nonce word). Extracted with the family templates and domain "abstract algebra", for every model and twin.
- **Statistic.** Offset consistency of X → m X over the 6 pairs, for each m. The null draws random pairs from that modifier's own 12 items (6 heads + 6 modified): 2,000 draws.
- Commutativity's consistency counts as *above plain composition* only if "commutative" exceeds both control modifiers in this design.

## Reporting

- Every result above is labelled post-hoc. The preregistered results are reported unchanged next to it.
- D8 and D9 change the inputs of the confirmatory Holm table: D8 the family size, D9 the precision of one p. They are logged in `docs/deviations.md`.
