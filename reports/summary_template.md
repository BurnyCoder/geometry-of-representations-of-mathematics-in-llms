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

| Model | Released | Params | Role | Field ρ trained | Field ρ random-init | LOO field acc (chance {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.chance:.3f}}) |
|---|---|---|---|---|---|---|
| Qwen3.5-2B-Base | Feb 2026 | 1.88B (text) | confirmatory | {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}} | {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.random_init:.3f}} | {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.trained.acc:.3f}} |
| Qwen3.5-0.8B-Base | Feb 2026 | 0.75B | scale | {{num:robustness.map_fields.qwen3.5-0.8b.math_primary.rsa.trained:.3f}} | {{num:robustness.map_fields.qwen3.5-0.8b.math_primary.rsa.random_init:.3f}} | {{num:robustness.map_fields.qwen3.5-0.8b.math_primary.loo_acc.trained.acc:.3f}} |
| Qwen2.5-1.5B | Sep 2024 | 1.54B | math ablation, base | {{num:robustness.map_fields.qwen2.5-1.5b.math_primary.rsa.trained:.3f}} | {{num:robustness.map_fields.qwen2.5-1.5b.math_primary.rsa.random_init:.3f}} | {{num:robustness.map_fields.qwen2.5-1.5b.math_primary.loo_acc.trained.acc:.3f}} |
| Qwen2.5-Math-1.5B | Sep 2024 | 1.54B | math ablation, +1T math tokens | {{num:robustness.map_fields.qwen2.5-math-1.5b.math_primary.rsa.trained:.3f}} | {{num:robustness.map_fields.qwen2.5-math-1.5b.math_primary.rsa.random_init:.3f}} | {{num:robustness.map_fields.qwen2.5-math-1.5b.math_primary.loo_acc.trained.acc:.3f}} |
| Gemma-4-E2B | Mar 2026 | 5.1B total; 2.3B decoder | second model family | {{num:gemma.map_fields.gemma-4-e2b.math_primary.rsa.trained:.3f}} | {{num:gemma.map_fields.gemma-4-e2b.math_primary.rsa.random_init:.3f}} | {{num:gemma.map_fields.gemma-4-e2b.math_primary.loo_acc.trained.acc:.3f}} |

## Hypotheses: what was tested, supported, falsified

| Hypothesis | Pre-registered? | Outcome | Evidence type |
|---|---|---|---|
| H-A1 field map tracks arXiv relatedness beyond the random-init twin | yes | **supported**: Δρ CI [{{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.0:.2f}}, {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.1:.2f}}] | correlation; paired control |
| H-A2 hierarchy (super-tree, cophenetic correlation) | yes (branch membership fixed after the tag, before analysis) | **not supported**: beats shuffled trees (p = {{num:mvp.hierarchy.qwen3.5-2b.coph_match_p_shuffled_tree:.3f}}) but not the random-init twin (Δ CI [{{num:mvp.hierarchy.qwen3.5-2b.coph_match_delta_vs_random_ci.0:.2f}}, {{num:mvp.hierarchy.qwen3.5-2b.coph_match_delta_vs_random_ci.1:.2f}}]), and is not more tree-like than a spectrum-matched Gaussian | correlation |
| Field map stronger than non-LLM baselines | added (plan) | **vs fastText: yes** (Δ CI [{{num:mvp.baselines.qwen3.5-2b.fasttext_covered.boot.delta_vs_fasttext.ci.0:.2f}}, {{num:mvp.baselines.qwen3.5-2b.fasttext_covered.boot.delta_vs_fasttext.ci.1:.2f}}]). **vs Qwen3-Embedding: no** (ρ {{num:mvp.baselines.qwen3.5-2b.all_primary.rsa.qwen3_embedding:.2f}} vs LLM {{num:mvp.baselines.qwen3.5-2b.all_primary.rsa.llm_trained:.2f}}; Δ CI [{{num:mvp.baselines.qwen3.5-2b.all_primary.boot.delta_vs_qwen3_embedding.ci.0:.2f}}, {{num:mvp.baselines.qwen3.5-2b.all_primary.boot.delta_vs_qwen3_embedding.ci.1:.2f}}]) | baseline comparison |
| H-A3 math pretraining sharpens the map | yes | **not supported (null)**: Δρ CI [{{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.0:.2f}}, {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.1:.2f}}] | ablation (paired models) |
| Field map specific to math | yes (specificity check) | **no**: science topics are organized comparably (post-hoc matched test) | correlation |
| Days/months circle (positive control) | yes | **full-space criterion failed**; post-hoc PC-plane cycle = {{num:mvp.posthoc.qwen3.5-2b.pc_plane_cycle.days.11.score:.2f}} (days) and {{num:mvp.posthoc.qwen3.5-2b.pc_plane_cycle.months.11.score:.2f}} (months) | correlation |
| Ring/space chains ordered | yes | **supported**: Robinson p = {{num:mvp.shapes.qwen3.5-2b.ring_chain.robinson_p:.3f}} / {{num:mvp.shapes.qwen3.5-2b.space_chain.robinson_p:.3f}}; twin ≈ chance | correlation |
| Chain order survives removing word frequency and token count | added (plan) | **yes in the 4 Qwen models, no in Gemma**: ring/space ρ = {{num:mvp.chain_frequency.qwen3.5-2b.ring_chain.loo_order_beyond:.2f}} / {{num:mvp.chain_frequency.qwen3.5-2b.space_chain.loo_order_beyond:.2f}} in Qwen3.5-2B (p = {{num:mvp.chain_frequency.qwen3.5-2b.ring_chain.p:.3f}} / {{num:mvp.chain_frequency.qwen3.5-2b.space_chain.p:.3f}}); two twins also order the ring chain (word form) | correlation; cross-validated confound regression |
| Angles form a circle | yes | **falsified**: line-like; ρ with the digit RDM = {{num:mvp.shapes.qwen3.5-2b.angles.beyond_reference.rho_fam_ref:.2f}} | correlation |
| Group-like axioms are consistent directions | yes | **supported for associativity and commutativity** (they survive Holm correction). Divisibility and identity are positive in all 5 models but significant only before correction (5/5 and 3/5 models). | correlation; paired control |
| Axiom offsets are not a word-length artifact | added (plan) | **yes**: with the token-count direction removed, associativity, commutativity and divisibility stay significant in all 5 models, identity in 4 | correlation; distractor removal |
| "Associative quasigroup" collapses onto "group" (a non-empty one is a group) | added (plan; exploratory) | **no** in any model: {{num:mvp.lattice.qwen3.5-2b.audit2.associative_quasigroup.ratio:.1f}}× farther from group than from quasigroup (Qwen3.5-2B) | correlation |
| Partial-operation lattice (semigroupoid, small category, groupoid); identity transfer to lattice-like pairs | added (plan; exploratory) | **no evidence** in any model (2–3 pairs per axis: weak tests) | correlation |
| Lie/topological/algebraic groups combine their parents (parallelograms) | yes (exploratory) | **supported** in all 5 models: cos = {{num:mvp.structures.qwen3.5-2b.group_objects.parallelograms.cos:.2f}} (Qwen3.5-2B); twins ≈ 0 to weak | correlation; paired control |
| Commutativity direction transfers from group-like to ring-like structures | yes (secondary) | **not established**: weak and partly lexical | correlation |
| Concrete↔abstract is the top principal axis (H-A4) | yes (secondary) | **supported** (LLM-judge ratings): PC1 ρ = {{num:mvp.axes.qwen3.5-2b.pcs.0.rho_abstraction:.2f}} vs twin {{num:mvp.axes.qwen3.5-2b__random.pcs.0.rho_abstraction:.2f}} | correlation |
| LLM-judge abstraction ratings agree with Mathlib definitional depth | added (plan; specified before computing) | **weakly supported**: ρ = {{num:mvp.mathlib.rho_mean_rating_depth:.2f}}, CI [{{num:mvp.mathlib.ci95.0:.2f}}, {{num:mvp.mathlib.ci95.1:.2f}}], but only {{num:mvp.mathlib.n_matched}} of {{num:mvp.mathlib.n_math_concepts}} concepts could be matched to Mathlib | correlation (external validity check) |
| Pure↔applied is a main axis (H-A4) | yes (secondary) | **supported**: PC2 ρ = {{num:mvp.axes.qwen3.5-2b.pcs.1.rho_applied:.2f}} | correlation |
| Chain geometry is used causally | yes | **inconclusive**: ring chain below the accuracy gate; space chain shows a small, non-significant effect | intervention (noise injection) |
| Field map visible in SAE features | exploratory | **weak**: SAE-code ρ = {{num:mvp.sae.qwen3.5-2b.math.rsa_sae:.2f}}; field-selective features exist | correlation |

## Figures, in plain words

The figures show what the model's internal vectors look like, squashed down to two dimensions. Blue always means a trained model, grey its untrained twin. The [interactive explorer](https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/) lets you move through the model's layers and switch between models.

![at a glance]({{fig:hero}})

**All results in one picture.**
- **A:** every trained model has a map of mathematics; no twin has one.
- **B:** in that map, fields that mathematicians often combine sit close together.
- **C:** days of the week form a circle.
- **D:** angles do not form a circle, although we predicted they would.
- **E:** adding an axiom moves a structure in a consistent direction.

![circle forming]({{fig:circle_depth_qwen3.5-2b.gif}})

**The circle through the layers.** Days and months, layer by layer. The circle is there almost from the start.

![field maps]({{fig:field_maps_qwen3.5-2b}})

**The map of mathematics.** Each dot is a field. The grey lines join fields that mathematicians often combine. In the trained model most of these lines stay short; in the untrained twin they cross the whole picture.

![axioms]({{fig:lattice_axes_qwen3.5-2b}})

**Axioms as arrows.** Each panel highlights all pairs of structures that differ by one axiom. In the trained model those arrows point roughly the same way.

![effect sizes]({{fig:forest}})

**How big the effects are.** Each line is a range of plausible values (95% confidence interval). A blue line does not cross zero, so the effect is reliably there.
