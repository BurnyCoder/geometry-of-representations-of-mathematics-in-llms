# Summary of all experiments

*This file is generated from `reports/summary_template.md` by `python run.py report`. Every number comes from `results/*/*.json`. For what we tried, why, and what went wrong, see [`experiments.md`](experiments.md).*

## In plain words

LLMs store words as long lists of numbers (vectors), and related ideas end up near each other. Earlier work found that the days of the week sit on a *circle* inside an LLM. We asked whether mathematics has a shape too.

1. **Do math fields form a map?** Yes. In the models we tested, algebraic geometry sits near commutative algebra and number theory, and probability sits near numerical analysis. This matches how mathematicians themselves combine fields on arXiv. A copy of the same network with random, untrained weights shows no such map, so the map is learned, not an accident of spelling. Training on a trillion extra tokens of math did **not** make the map sharper. Science topics such as biology and geology are organized about as strongly, so the map is general topical knowledge rather than something special about mathematics.
2. **Do ordered families have shapes?** Partly.
   - "rng ⊃ ring ⊃ … ⊃ field" and "T0 ⊃ … ⊃ Hilbert space" are laid out in order.
   - Days and months form clean circles, but only once we look at the two most important directions.
   - **Angles do not form a circle.** The model treats "330 degrees" like the number 330, far from "0 degrees". This was a prediction we wrote down in advance, and it failed.
3. **Are axioms directions?** Mostly yes, for magma/semigroup/monoid/group-like structures. "Adding associativity" and "adding commutativity" each correspond to a roughly fixed arrow. For example, the commutativity arrow takes monoid → commutative monoid the same way it takes group → abelian group. "Adding an identity" and "adding inverses (divisibility)" point the same way too, but more weakly; the evidence there is suggestive, not conclusive. The untrained copy shows no such arrows.

## Key terms

| Term | Meaning |
|---|---|
| Residual stream | The running vector the model updates layer by layer. We read it at the concept's last token. |
| Random-init twin | The same network architecture and tokenizer with random weights. It shows what geometry comes from word form alone. |
| RDM | Representational dissimilarity matrix: all pairwise distances between items. |
| RSA, field ρ | Spearman correlation between the model's field-centroid RDM and the arXiv-derived field distance. |
| Ochiai cross-listing distance | 1 − (papers in both fields) / √(papers in A × papers in B), over 3.2M arXiv papers. |
| LOO field accuracy | How often a concept is closest to its own field's centroid, computed without that concept. |
| Robinson score | The fraction of item triples whose distances agree with a 1-D ordering. Chance is 1/3. |
| Offset consistency | The mean cosine between difference vectors of pairs that differ by one axiom. 0 means unrelated; 1 means parallel. |
| PC-plane cycle score | The fraction of items whose neighbour around the 2-D principal-component plane is the true next item. 1 means a perfect circle. |
| Post-hoc | An analysis added after seeing data. It is labelled as such and is not confirmatory. |

## Models and headline numbers

All models are base (pretrained) checkpoints, measured at relative depth 0.5. Field ρ uses primary concepts from the confirmatory split.

| Model | Released | Params | Role | Field ρ trained | Field ρ random-init | LOO field acc (chance 0.059) |
|---|---|---|---|---|---|---|
| Qwen3.5-2B-Base | Feb 2026 | 1.88B (text) | confirmatory | 0.387 | -0.118 | 0.472 |
| Qwen3.5-0.8B-Base | Feb 2026 | 0.75B | scale | 0.369 | -0.182 | 0.430 |
| Qwen2.5-1.5B | Sep 2024 | 1.54B | math ablation, base | 0.406 | -0.093 | 0.565 |
| Qwen2.5-Math-1.5B | Sep 2024 | 1.54B | math ablation, +1T math tokens | 0.390 | -0.087 | 0.588 |
| Gemma-4-E2B | Mar 2026 | 5.1B total; 2.3B decoder | second model family | 0.436 | -0.021 | 0.417 |

## Hypotheses: what was tested, supported, falsified

| Hypothesis | Pre-registered? | Outcome | Evidence type |
|---|---|---|---|
| H-A1 field map tracks arXiv relatedness beyond the random-init twin | yes | **supported**: Δρ CI [0.28, 0.58] | correlation; paired control |
| H-A2 hierarchy (super-tree, cophenetic correlation) | yes (branch membership fixed after the tag, before analysis) | **not supported**: beats shuffled trees (p = 0.020) but not the random-init twin (Δ CI [-0.08, 0.35]), and is not more tree-like than a spectrum-matched Gaussian | correlation |
| Field map stronger than non-LLM baselines | added (plan) | **vs fastText: yes** (Δ CI [0.07, 0.30]). **vs Qwen3-Embedding: no** (ρ 0.45 vs LLM 0.39; Δ CI [-0.13, 0.04]) | baseline comparison |
| H-A3 math pretraining sharpens the map | yes | **not supported (null)**: Δρ CI [-0.06, 0.02] | ablation (paired models) |
| Field map specific to math | yes (specificity check) | **no**: science topics are organized comparably (post-hoc matched test) | correlation |
| Days/months circle (positive control) | yes | **full-space criterion failed**; post-hoc PC-plane cycle = 1.00 (days) and 1.00 (months) | correlation |
| Ring/space chains ordered | yes | **supported**: Robinson p = 0.010 / 0.000; twin ≈ chance | correlation |
| Chain order survives removing word frequency and token count | added (plan) | **yes in the 4 Qwen models, no in Gemma**: ring/space ρ = 0.94 / 0.81 in Qwen3.5-2B (p = 0.003 / 0.017); two twins also order the ring chain (word form) | correlation; cross-validated confound regression |
| Angles form a circle | yes | **falsified**: line-like; ρ with the digit RDM = 0.84 | correlation |
| Group-like axioms are consistent directions | yes | **supported for associativity and commutativity** (they survive Holm correction). Divisibility and identity are positive in all 5 models but significant only before correction (5/5 and 3/5 models). | correlation; paired control |
| Axiom offsets are not a word-length artifact | added (plan) | **yes**: with the token-count direction removed, associativity, commutativity and divisibility stay significant in all 5 models, identity in 4 | correlation; distractor removal |
| "Associative quasigroup" collapses onto "group" (a non-empty one is a group) | added (plan; exploratory) | **no** in any model: 2.8× farther from group than from quasigroup (Qwen3.5-2B) | correlation |
| Partial-operation lattice (semigroupoid, small category, groupoid); identity transfer to lattice-like pairs | added (plan; exploratory) | **no evidence** in any model (2–3 pairs per axis: weak tests) | correlation |
| Lie/topological/algebraic groups combine their parents (parallelograms) | yes (exploratory) | **supported** in all 5 models: cos = 0.40 (Qwen3.5-2B); twins ≈ 0 to weak | correlation; paired control |
| Commutativity direction transfers from group-like to ring-like structures | yes (secondary) | **not established**: weak and partly lexical | correlation |
| Concrete↔abstract is the top principal axis (H-A4) | yes (secondary) | **supported** (LLM-judge ratings): PC1 ρ = 0.49 vs twin 0.06 | correlation |
| LLM-judge abstraction ratings agree with Mathlib definitional depth | added (plan; specified before computing) | **weakly supported**: ρ = 0.37, CI [0.07, 0.62], but only 39 of 655 concepts could be matched to Mathlib | correlation (external validity check) |
| Pure↔applied is a main axis (H-A4) | yes (secondary) | **supported**: PC2 ρ = -0.41 | correlation |
| Chain geometry is used causally | yes | **inconclusive**: ring chain below the accuracy gate; space chain shows a small, non-significant effect | intervention (noise injection) |
| Field map visible in SAE features | exploratory | **weak**: SAE-code ρ = 0.15; field-selective features exist | correlation |

## Figures, in plain words

The figures show what the model's internal vectors look like, squashed down to two dimensions. Blue always means a trained model, grey its untrained twin. The [interactive explorer](https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/) lets you move through the model's layers and switch between models.

![at a glance](figures/hero.png)

**All results in one picture.**
- **A:** every trained model has a map of mathematics; no twin has one.
- **B:** days of the week form a circle.
- **C:** angles do not form a circle, although we predicted they would.
- **D:** adding an axiom moves a structure in a consistent direction.

![circle forming](figures/circle_depth_qwen3.5-2b.gif)

**The circle through the layers.** Days and months, layer by layer. The circle is there almost from the start.

![field maps](figures/field_maps_qwen3.5-2b.png)

**The map of mathematics.** Each dot is a field. The grey lines join fields that mathematicians often combine. In the trained model most of these lines stay short; in the untrained twin they cross the whole picture.

![axioms](figures/lattice_axes_qwen3.5-2b.png)

**Axioms as arrows.** Each panel highlights all pairs of structures that differ by one axiom. In the trained model those arrows point roughly the same way.

![effect sizes](figures/forest.png)

**How big the effects are.** Each line is a range of plausible values (95% confidence interval). A blue line does not cross zero, so the effect is reliably there.
