# Independent audit: word-form controls, twin seeds and test precision

*Generated from `reports/audit_template.md` by `python run.py paper`; every number comes from `results/*/*.json`. Everything here is **post-hoc**: the audit ran after all preregistered results were known. Its tests were defined in [`prereg_addendum_lexical.md`](prereg_addendum_lexical.md), committed before the `lexical` and `twins` phases ran (one amendment, made from a unit test before any real-data run, is recorded there). The log of changes is in [`docs/deviations.md`](../docs/deviations.md).*

## Why an audit was needed

The preregistration uses a random-init twin (same architecture and tokenizer, random weights) as the measure of "form-only similarity". It is not one. In the Qwen3.5-2B twin, the pairs magma → semigroup and commutative magma → commutative semigroup differ only by the literal word "commutative", yet their difference vectors have cosine 0.04. The trained model gives 0.86, and a bag of words gives 1. A random network scrambles even exact word composition, so "beats the twin" never ruled out that a result follows from the spelling of the names. Mathematical names often spell out their own structure ("commutative monoid", "topological group", "… domain"), so this matters here more than for days or months.

## Method

Every test is rerun, with the same statistic, centring, null and seed as for the trained model, on three **word-form baselines**:

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
| H-A1 field map (primary, confirmatory) | beats the twin in all 5 models | beats all three baselines in all 5 models | **holds** |
| H-B space chain (confirmatory) | ordered in all 5 models | spelling does not order it; ordered within metrizable … Hilbert in all 5 models and within T0 … normal in 1 of 5 models (Qwen3.5-2B) | **holds** |
| H-B ring chain (confirmatory) | ordered in 3 of 5 models (all but Qwen3.5-0.8B and Gemma-4-E2B) | head-noun clusters without order pass the test in up to 55% of simulations; the six "… domain" items are ordered in 1 of 5 models (Qwen2.5-Math-1.5B) | **not established beyond head nouns** |
| H-B2 associativity offset (confirmatory) | Holm-significant in all 5 models | above every baseline, disjoint pairs pass, in all 5 models | **suggestive beyond word form** |
| H-B2 commutativity offset (confirmatory) | Holm-significant in all 5 models | beyond word form in none of the 5 models; "commutative" beats "wooden" and "zorbic" in 1 of 5 models (Gemma-4-E2B); the direction predicts abelian group − group in all 5 models | **mostly composition with a word; one held-out pair carries meaning** |
| H-B2 identity offset (confirmatory) | Holm-significant in none of the 5 models | beyond word form in none of the 5 models; disjoint cosines 0.03 to 0.13 | **lexical duplication** |
| H-B2 divisibility offset (confirmatory) | Holm-significant in none of the 5 models | beyond word form in 1 of 5 models (Gemma-4-E2B) | **not beyond word form** |
| Group objects (exploratory) | parallelograms clear their null in all 5 models | above every baseline in none of the 5 models | **word form** |

## D1: field map

| Model | ρ trained | ρ char-3-grams (Mantel p) | ρ word bag (p) | ρ layer-0 token bag (p) | Δ − char-3-grams [CI] | Δ − word bag [CI] | Δ − layer-0 bag [CI] | LOO field acc: trained / char-3 / word / layer-0 |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 0.387 | 0.105 (0.1353) | -0.017 (0.5205) | 0.157 (0.0303) | +0.283 [0.135, 0.398] | +0.404 [0.220, 0.518] | +0.230 [0.139, 0.344] | 0.472 / 0.261 / 0.249 / 0.349 |
| qwen3.5-0.8b | 0.369 | 0.105 (0.1353) | -0.017 (0.5205) | 0.218 (0.0071) | +0.264 [0.117, 0.376] | +0.385 [0.198, 0.490] | +0.151 [0.068, 0.270] | 0.430 / 0.261 / 0.249 / 0.312 |
| qwen2.5-1.5b | 0.406 | 0.105 (0.1353) | -0.017 (0.5205) | 0.178 (0.0276) | +0.302 [0.150, 0.416] | +0.423 [0.238, 0.531] | +0.228 [0.147, 0.363] | 0.565 / 0.261 / 0.249 / 0.337 |
| qwen2.5-math-1.5b | 0.390 | 0.105 (0.1353) | -0.017 (0.5205) | 0.201 (0.0112) | +0.285 [0.140, 0.396] | +0.407 [0.223, 0.506] | +0.189 [0.117, 0.317] | 0.588 / 0.261 / 0.249 / 0.372 |
| gemma-4-e2b | 0.436 | 0.105 (0.1353) | -0.017 (0.5205) | 0.209 (0.0136) | +0.332 [0.190, 0.441] | +0.453 [0.272, 0.566] | +0.227 [0.137, 0.355] | 0.417 / 0.261 / 0.249 / 0.322 |

## D11: bootstrap bias and BCa intervals

Concept resampling duplicates concepts, which pulls resampled ρ toward 0, so the preregistered percentile intervals sit below the point estimates. BCa corrects this; every interval moves up, and no conclusion changes.

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

## D2–D3, D13: axiom offsets

Offsets per axiom: trained (from `lattice.json`), each word-form baseline, and where the cosines come from. *Duplicated* cosines compare two pairs that coincide after deleting "commutative"; *disjoint* cosines are all the others.

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

The 8 names without "commutative" (3 pairs per axiom):

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

Does the direction of the five pairs that add "commutative" predict *abelian group − group*, which shares no word with it? The exact null uses the 72 ordered pairs that do not change commutativity:

| Model | trained: cos (p) | twin: cos (p) | char-3-grams: cos (p) | word bag: cos (p) | layer-0 token bag: cos (p) | semantic transfer |
|---|---|---|---|---|---|---|
| qwen3.5-2b | +0.60 (0.0274) | +0.03 (0.3562) | +0.00 (0.8493) | +0.00 (0.9315) | +0.12 (0.0822) | yes |
| qwen3.5-0.8b | +0.54 (0.0274) | -0.00 (0.5205) | +0.00 (0.8493) | +0.00 (0.9315) | +0.15 (0.0959) | yes |
| qwen2.5-1.5b | +0.51 (0.0274) | +0.02 (0.3288) | +0.00 (0.8493) | +0.00 (0.9315) | +0.08 (0.0959) | yes |
| qwen2.5-math-1.5b | +0.55 (0.0274) | +0.03 (0.3288) | +0.00 (0.8493) | +0.00 (0.9315) | +0.15 (0.0822) | yes |
| gemma-4-e2b | +0.73 (0.0274) | +0.01 (0.4384) | +0.00 (0.8493) | +0.00 (0.9315) | +0.31 (0.0822) | yes |

How consistent is adding *any* modifier? X → m X for the same six heads:

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

## D4: group objects

| Model | trained cos (p) | char-3-grams cos (p) | word bag cos (p) | layer-0 token bag cos (p) | beyond word form |
|---|---|---|---|---|---|
| qwen3.5-2b | +0.398 (0.0045) | +0.611 (0.0005) | +0.503 (0.0005) | +0.678 (0.0005) | no |
| qwen3.5-0.8b | +0.457 (0.0025) | +0.611 (0.0005) | +0.503 (0.0005) | +0.688 (0.0005) | no |
| qwen2.5-1.5b | +0.406 (0.0035) | +0.611 (0.0005) | +0.503 (0.0005) | +0.680 (0.0005) | no |
| qwen2.5-math-1.5b | +0.409 (0.0015) | +0.611 (0.0005) | +0.503 (0.0005) | +0.674 (0.0005) | no |
| gemma-4-e2b | +0.484 (0.0085) | +0.611 (0.0005) | +0.503 (0.0005) | +0.522 (0.0005) | no |

## D5–D7: chains

Exact order tests within blocks of constant head noun or topic, over all n! orders:

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

How often do clusters of the ring chain's head nouns, with no order at all, pass the preregistered chain test (Robinson p < .05 and leave-one-out p < .05)?

| head-noun cluster SNR | conjunction rejects (Robinson p < .05 and LOO p < .05) | mean Robinson | mean LOO ρ | simulations |
|---|---|---|---|---|
| 0.5 | 0.03 | 0.44 | -0.34 | 40 |
| 1.0 | 0.40 | 0.44 | +0.30 | 40 |
| 2.0 | 0.55 | 0.45 | +0.42 | 40 |

Ordered families on word-form baselines (Qwen3.5-2B):

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

## D8: Holm over all 8 preregistered tests

Together with the 10,000-permutation LOO p-values (D9), this changes three decisions relative to the pre-audit rules (7-test Holm, 300 permutations): Gemma-4-E2B's space chain becomes Holm-significant, and identity and divisibility lose Holm significance in Qwen2.5-Math-1.5B. The confirmatory model:

| Test | Model | p | Holm-adjusted p | reject at α=0.05 |
|---|---|---|---|---|
| H-A1 Δρ trained − random-init ≤ 0 (bootstrap) | qwen3.5-2b | 0.0005 | 0.0040 | yes |
| H-B ring chain (Robinson and LOO order) | qwen3.5-2b | 0.0105 | 0.0420 | yes |
| H-B space chain (Robinson and LOO order) | qwen3.5-2b | 0.0005 | 0.0040 | yes |
| H-B2 associative offset consistency | qwen3.5-2b | 0.0035 | 0.0175 | yes |
| H-B2 identity offset consistency | qwen3.5-2b | 0.0425 | 0.0850 | no |
| H-B2 divisible offset consistency | qwen3.5-2b | 0.0275 | 0.0825 | no |
| H-B2 commutative offset consistency | qwen3.5-2b | 0.0010 | 0.0060 | yes |
| H-B angles circle prediction (decision rule: closure 2.11 < 2 and circle ρ 0.29 > line ρ 0.56; counted with p = 1) | qwen3.5-2b | — | — | not met (prediction falsified) |

## D9–D10: leave-one-out order, 10,000 permutations, two ridge penalties

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

## D12: twin seeds

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

## What changed in the claims

- The field map and the space chain are now the lead results; both survive every word-form control.
- "Axioms act as directions" is replaced by "axiom offsets are consistent, but mostly because the names encode the axioms". An associativity direction, and a commutativity direction that reaches "abelian group", are suggestive exceptions.
- "Group objects combine their parents" is withdrawn as evidence of mathematical knowledge: the parallelograms follow shared words.
- The ring chain's order is reported as not established beyond its head nouns.
- "The field map is not specific to mathematics" is withdrawn. The control fields are whole sciences, so the comparison confounds domain with granularity, and specificity is untested.
- "The twin measures form-only similarity" is corrected everywhere except in the frozen preregistration.

## Limitations of the audit

- **Post-hoc.** It was prompted by the results. Its tests were fixed in writing before they ran, but the audit's read-only exploration had already seen several of their values (listed in the addendum).
- **Baselines are lower bounds on word form.** Character 3-grams, bags of words and context-free token embeddings do not exhaust what spelling can carry. Learned morphology, for example, could still contribute to the field map and the space chain.
- **Granularity.** A granularity-matched specificity control, sub-fields of another science, was not run.
