# Summary of all experiments

*This file is generated from `reports/summary_template.md` by `python run.py paper`. Every statistic and every count of models comes from `results/*/*.json`; release dates and sizes come from the linked model cards and the extraction logs. For what we tried, why, and what went wrong, see [`experiments.md`](experiments.md).*

## In plain words

LLMs store words as long lists of numbers (vectors), and related ideas end up near each other. Earlier work found that the days of the week sit on a *circle* inside an LLM. We asked whether mathematics has a shape too. We list the axiom result first; our main prediction written down in advance was the map of fields (question 2), and the order changes nothing about what was tested or found.

1. **Are axioms directions?** Mostly yes, for magma/semigroup/monoid/group-like structures. A structure is defined by which rules (axioms) it obeys, so we compared pairs of structures that differ in exactly one rule and drew the change as an arrow. "Adding associativity" and "adding commutativity" each correspond to a roughly fixed arrow. For example, the commutativity arrow takes monoid → commutative monoid the same way it takes group → abelian group. "Adding an identity" and "adding inverses (divisibility)" point the same way too, but more weakly; the evidence there is suggestive, not conclusive. The untrained copy shows much weaker arrows; in {{fact:twins_any_offset}}, one or two axioms still pass the random-pair test before correction, from word form alone.
2. **Do math fields form a map?** Yes, moderately (ρ ≈ {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.1f}}). Fields that mathematicians often combine on arXiv tend to sit closer together in the model. For example, probability's nearest field is numerical analysis in every model, and numerical analysis is also among its three nearest on arXiv. The match is far from perfect: on arXiv, algebraic geometry's nearest fields are commutative algebra and number theory, but in every model its nearest field is set theory & logic. A copy of the same network with random, untrained weights shows no such map, so the map is learned, not an accident of spelling. Training on over a trillion extra tokens of math ([Qwen2.5-Math report](https://arxiv.org/abs/2409.12122)) did **not** make the map sharper. Science topics (biology, chemistry, geology, astronomy) cluster by field even more cleanly in most models, so the clustering is general topical knowledge rather than something special about mathematics.
3. **Do ordered families have shapes?** Partly.
   - "rng ⊃ ring ⊃ … ⊃ field ⊃ algebraically closed field" and "T0 ⊃ … ⊃ Hilbert space" are laid out in order in the main model. The full pre-registered test holds for the ring chain in {{fact:chain_ordered.ring_chain}} and for the space chain in {{fact:chain_ordered.space_chain}}.
   - Days go round in order in the two most important directions in every model, months in {{fact:top2_cycle_perfect.months}}. A stricter test, which also needs the loop to close, finds a closed days loop in {{fact:closes_top2.days}} and a closed months loop (in other directions) in {{fact:closes_lower_plane.months}}. Our pre-registered test for circles, which uses all directions at once, failed.
   - **Angles do not form a circle.** The model treats "330 degrees" like the number 330, far from "0 degrees". This was a prediction we wrote down in advance, and it failed.

## Key terms

| Term | Meaning |
|---|---|
| Residual stream | The running vector the model updates layer by layer. We read it at the concept's last token. |
| Random-init twin | The same network architecture and tokenizer with random weights. It shows what geometry comes from word form alone. |
| Difference vector (arrow) | The representation of one structure minus that of another, e.g. "abelian group" − "group". |
| Offset consistency | The mean cosine between difference vectors of pairs that differ by one axiom. 0 means unrelated; 1 means parallel. |
| RDM | Representational dissimilarity matrix: all pairwise distances between items. |
| RSA, field ρ | Spearman correlation between the model's field-centroid RDM and the arXiv-derived field distance. |
| Ochiai cross-listing distance | 1 − (papers in both fields) / √(papers in A × papers in B), over 3.2M arXiv papers. |
| LOO field accuracy | How often a concept is closest to its own field's centroid, computed without that concept. |
| Robinson score | The fraction of item triples whose distances agree with a 1-D ordering. Chance is 1/3. |
| PC-plane cycle score | The fraction of items whose neighbour around the 2-D principal-component plane is the true next item. 1 means the items go round in order; an open arc (a horseshoe) can also score 1. |
| Closed-cycle test | The cycle score counted only when the loop also closes in the plane (the last item is no further from the first than about two neighbour steps), checked in every plane of the top-5 principal components. Post-hoc, but written down before it ran. |
| Post-hoc | An analysis added after seeing data. It is labelled as such and is not confirmatory. |

## Models and headline numbers

All models are base (pretrained) checkpoints, measured at relative depth 0.5. Field ρ uses primary concepts from the confirmatory split.

| Model | On Hugging Face since | Params | Role | Commutativity offset trained / random-init | Field ρ trained | Field ρ random-init | LOO field acc (chance {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.chance:.3f}}) |
|---|---|---|---|---|---|---|---|
| [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | Feb 2026 | 1.88B text-only (2.27B with vision) | confirmatory | {{num:mvp.lattice.qwen3.5-2b.group_like.offsets.commutative.consistency:.2f}} / {{num:mvp.lattice.qwen3.5-2b__random.group_like.offsets.commutative.consistency:.2f}} | {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}} | {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.random_init:.3f}} | {{num:mvp.map_fields.qwen3.5-2b.math_primary.loo_acc.trained.acc:.3f}} |
| [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | Feb 2026 | 0.75B text-only (0.87B with vision) | scale | {{num:robustness.lattice.qwen3.5-0.8b.group_like.offsets.commutative.consistency:.2f}} / {{num:robustness.lattice.qwen3.5-0.8b__random.group_like.offsets.commutative.consistency:.2f}} | {{num:robustness.map_fields.qwen3.5-0.8b.math_primary.rsa.trained:.3f}} | {{num:robustness.map_fields.qwen3.5-0.8b.math_primary.rsa.random_init:.3f}} | {{num:robustness.map_fields.qwen3.5-0.8b.math_primary.loo_acc.trained.acc:.3f}} |
| [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | Sep 2024 | 1.54B | math ablation, base | {{num:robustness.lattice.qwen2.5-1.5b.group_like.offsets.commutative.consistency:.2f}} / {{num:robustness.lattice.qwen2.5-1.5b__random.group_like.offsets.commutative.consistency:.2f}} | {{num:robustness.map_fields.qwen2.5-1.5b.math_primary.rsa.trained:.3f}} | {{num:robustness.map_fields.qwen2.5-1.5b.math_primary.rsa.random_init:.3f}} | {{num:robustness.map_fields.qwen2.5-1.5b.math_primary.loo_acc.trained.acc:.3f}} |
| [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | Sep 2024 | 1.54B | math ablation, +1T math tokens ([report](https://arxiv.org/abs/2409.12122)) | {{num:robustness.lattice.qwen2.5-math-1.5b.group_like.offsets.commutative.consistency:.2f}} / {{num:robustness.lattice.qwen2.5-math-1.5b__random.group_like.offsets.commutative.consistency:.2f}} | {{num:robustness.map_fields.qwen2.5-math-1.5b.math_primary.rsa.trained:.3f}} | {{num:robustness.map_fields.qwen2.5-math-1.5b.math_primary.rsa.random_init:.3f}} | {{num:robustness.map_fields.qwen2.5-math-1.5b.math_primary.loo_acc.trained.acc:.3f}} |
| [Gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) | Mar 2026 | 2.3B effective; 5.1B with embeddings | second model family | {{num:gemma.lattice.gemma-4-e2b.group_like.offsets.commutative.consistency:.2f}} / {{num:gemma.lattice.gemma-4-e2b__random.group_like.offsets.commutative.consistency:.2f}} | {{num:gemma.map_fields.gemma-4-e2b.math_primary.rsa.trained:.3f}} | {{num:gemma.map_fields.gemma-4-e2b.math_primary.rsa.random_init:.3f}} | {{num:gemma.map_fields.gemma-4-e2b.math_primary.loo_acc.trained.acc:.3f}} |

## Hypotheses: what was tested, supported, falsified

Axiom results come first; the order is a presentation choice, and the "Pre-registered?" column gives each test's status.

| Hypothesis | Pre-registered? | Outcome | Evidence type |
|---|---|---|---|
| Group-like axioms are consistent directions | yes | **supported for associativity and commutativity** (they survive Holm correction in {{fact:axiom_holm.associative}} and {{fact:axiom_holm.commutative}}). Divisibility and identity are positive in all 5 models; significant before correction in {{fact:axiom_raw.divisible}} and {{fact:axiom_raw.identity}}, after Holm correction in {{fact:axiom_holm.divisible}} and {{fact:axiom_holm.identity}}. | correlation; paired control |
| Axiom offsets are not a word-length artifact | added (plan) | **yes**: with the token-count direction removed, the offsets stay significant (before correction) for associativity in {{fact:axiom_length_projected.associative}}, commutativity in {{fact:axiom_length_projected.commutative}}, divisibility in {{fact:axiom_length_projected.divisible}}, and identity in {{fact:axiom_length_projected.identity}} | correlation; distractor removal |
| "Associative quasigroup" collapses onto "group" (a non-empty one is a group) | added (plan; exploratory) | **no** in any model: {{num:mvp.lattice.qwen3.5-2b.audit2.associative_quasigroup.ratio:.1f}}× farther from group than from quasigroup (Qwen3.5-2B) | correlation |
| Partial-operation lattice (semigroupoid, small category, groupoid); identity transfer to lattice-like pairs | added (plan; exploratory) | **no evidence** in any model (2–3 pairs per axis: weak tests) | correlation |
| Lie/topological/algebraic groups combine their parents (parallelograms) | yes (exploratory) | **supported** in {{fact:parallelograms}}: cos = {{num:mvp.structures.qwen3.5-2b.group_objects.parallelograms.cos:.2f}} (Qwen3.5-2B); twins ≈ 0 to weak, but the twins of Qwen2.5-1.5B and Qwen2.5-Math also pass their own nulls (shared words) | correlation; paired control |
| Commutativity direction transfers from group-like to ring-like structures | yes (exploratory) | **not established**: weak and partly lexical | correlation |
| H-A1 field map tracks arXiv relatedness beyond the random-init twin (the primary preregistered hypothesis) | yes | **supported**: Δρ CI [{{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.0:.2f}}, {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa_boot.delta_vs_random_init.ci.1:.2f}}] | correlation; paired control |
| H-A2 hierarchy (super-tree, cophenetic correlation) | yes (branch membership fixed after the tag, before analysis) | **not supported**: beats shuffled trees (p = {{num:mvp.hierarchy.qwen3.5-2b.coph_match_p_shuffled_tree:.3f}}; {{fact:beats_shuffled_trees}}) but not the random-init twin (Δ CI [{{num:mvp.hierarchy.qwen3.5-2b.coph_match_delta_vs_random_ci.0:.2f}}, {{num:mvp.hierarchy.qwen3.5-2b.coph_match_delta_vs_random_ci.1:.2f}}]). More tree-like than a spectrum-matched Gaussian by cophenetic correlation in {{fact:gaussian_more_treelike.coph_c}}, but by relative Gromov δ in {{fact:gaussian_more_treelike.delta_rel}} | correlation |
| Field map stronger than embedding baselines (fastText; Qwen3-Embedding, an LLM-based embedder), Qwen3.5-2B only | added (plan) | **vs fastText: yes** (Δ CI [{{num:mvp.baselines.qwen3.5-2b.fasttext_covered.boot.delta_vs_fasttext.ci.0:.2f}}, {{num:mvp.baselines.qwen3.5-2b.fasttext_covered.boot.delta_vs_fasttext.ci.1:.2f}}]). **vs Qwen3-Embedding: no** (ρ {{num:mvp.baselines.qwen3.5-2b.all_primary.rsa.qwen3_embedding:.2f}} vs LLM {{num:mvp.baselines.qwen3.5-2b.all_primary.rsa.llm_trained:.2f}}; Δ CI [{{num:mvp.baselines.qwen3.5-2b.all_primary.boot.delta_vs_qwen3_embedding.ci.0:.2f}}, {{num:mvp.baselines.qwen3.5-2b.all_primary.boot.delta_vs_qwen3_embedding.ci.1:.2f}}]) | baseline comparison |
| H-A3 math pretraining sharpens the map | yes | **not supported (null)**: Δρ CI [{{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.0:.2f}}, {{num:robustness.compare.qwen2.5-math-1.5b_vs_qwen2.5-1.5b.boot.delta_vs_qwen2.5-1.5b.ci.1:.2f}}] | ablation (paired models) |
| Field map specific to math | yes (specificity check) | **no**: the control fields cluster more strongly than most random sets of four math fields in {{fact:control_beats_most_math_subsets}} (post-hoc matched test; LOO clustering, not the arXiv map) | correlation |
| Days/months circle (positive control) | yes | **full-space criterion failed**; post-hoc PC-plane cycle = {{num:mvp.posthoc.qwen3.5-2b.pc_plane_cycle.days.11.score:.2f}} (days) and {{num:mvp.posthoc.qwen3.5-2b.pc_plane_cycle.months.11.score:.2f}} (months); a loop that closes: days in {{fact:closes_top2.days}} (top-2 plane), months in {{fact:closes_lower_plane.months}} (a lower plane) | correlation |
| Ring/space chains ordered (Robinson and LOO order, Holm, above the twin) | yes | **supported in Qwen3.5-2B**: Robinson p = {{num:mvp.shapes.qwen3.5-2b.ring_chain.robinson_p:.3f}} / {{num:mvp.shapes.qwen3.5-2b.space_chain.robinson_p:.3f}}, LOO order p = {{num:mvp.shapes.qwen3.5-2b.ring_chain.loo_order_p:.3f}} / {{num:mvp.shapes.qwen3.5-2b.space_chain.loo_order_p:.3f}}; twin ≈ chance. Across models: ring chain in {{fact:chain_ordered.ring_chain}}, space chain in {{fact:chain_ordered.space_chain}} | correlation |
| Chain order survives removing word frequency and token count | added (plan) | **ring chain in {{fact:chain_beyond_frequency.ring_chain}}, space chain in {{fact:chain_beyond_frequency.space_chain}}**: ring/space ρ = {{num:mvp.chain_frequency.qwen3.5-2b.ring_chain.loo_order_beyond:.2f}} / {{num:mvp.chain_frequency.qwen3.5-2b.space_chain.loo_order_beyond:.2f}} in Qwen3.5-2B (p = {{num:mvp.chain_frequency.qwen3.5-2b.ring_chain.p:.3f}} / {{num:mvp.chain_frequency.qwen3.5-2b.space_chain.p:.3f}}); the ring chain is also ordered in {{fact:twin_chain_beyond_frequency.ring_chain}} (word form) | correlation; cross-validated confound regression |
| Angles form a circle | yes | **falsified**: line-like; ρ with the digit RDM = {{num:mvp.shapes.qwen3.5-2b.angles.beyond_reference.rho_fam_ref:.2f}} | correlation |
| Concrete↔abstract is the top principal axis (H-A4) | yes (secondary) | **supported** (LLM-judge ratings): PC1 ρ = {{num:mvp.axes.qwen3.5-2b.pcs.0.rho_abstraction:.2f}} vs twin {{num:mvp.axes.qwen3.5-2b__random.pcs.0.rho_abstraction:.2f}} | correlation |
| LLM-judge abstraction ratings agree with Mathlib definitional depth | added (plan; specified before computing) | **weakly supported**: ρ = {{num:mvp.mathlib.rho_mean_rating_depth:.2f}}, CI [{{num:mvp.mathlib.ci95.0:.2f}}, {{num:mvp.mathlib.ci95.1:.2f}}], but only {{num:mvp.mathlib.n_matched}} of {{num:mvp.mathlib.n_math_concepts}} concepts could be matched to Mathlib | correlation (external validity check) |
| Pure↔applied is a main axis (H-A4) | yes (secondary) | **supported**: PC2 ρ = {{num:mvp.axes.qwen3.5-2b.pcs.1.rho_applied:.2f}} | correlation |
| Chain geometry is used causally | yes | **inconclusive**: ring chain below the accuracy gate ({{num:mvp.causal.qwen3.5-2b.ring_chain.baseline_acc:.2f}} < {{num:mvp.causal.qwen3.5-2b.ring_chain.gate:.2f}}); for the space chain, noise in the chain subspace lowers accuracy more than random-subspace noise at the largest scale (s = 4: {{num:mvp.causal.qwen3.5-2b.space_chain.scales.4.0.acc_chain_subspace:.2f}} vs a random mean of {{num:mvp.causal.qwen3.5-2b.space_chain.scales.4.0.acc_random_subspace_mean:.2f}}) but less at s = 1; with 5 random subspaces this is descriptive, not a test | intervention (noise injection) |
| Field map visible in SAE features | exploratory | **weak**: SAE-code ρ = {{num:mvp.sae.qwen3.5-2b.math.rsa_sae:.2f}}; field-selective features exist | correlation |

## Figures, in plain words

The figures show what the model's internal vectors look like, squashed down to two or three dimensions. Blue always means a trained model, grey its untrained twin. Where both exist, the 3-D view comes first. The [interactive explorer](https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/) lets you turn the 3-D views, move through the model's layers and switch between models.

![at a glance]({{fig:hero}})

**All results in one picture.**
- **Top row, axioms as arrows.** Left: every pair of structures that differ only in commutativity, drawn as an arrow from one starting point. In the model the arrows point roughly the same way; in the untrained twin they point every way. Right: the same measurement for all four axioms in every model and twin.
- **Middle row, the map of mathematics.** Left: every trained model has a map of mathematics; no twin has one. Right: the map itself. On the left is how mathematicians combine fields on arXiv, in the middle the model's map, on the right the untrained twin's. Grey lines join fields that mathematicians often combine; they stay short in the model's map and cross the twin's.
- **Bottom row, shapes, in 3-D.** Days of the week form a circle. Angles do not, although we predicted they would. The ring and space chains are laid out in order.

![how the axiom test works]({{fig:axioms_explainer_qwen3.5-2b}})

**How we test whether an axiom is a direction.**
1. A structure is defined by which of four rules (axioms) it obeys, so the structures form a lattice.
2. Take every pair that differs in just one rule, here commutativity (whether a·b = b·a always holds), and draw an arrow from one to the other.
3. Move all these arrows to one starting point. In the model they point roughly the same way; in the untrained twin they point every way.
4. Check it properly, using all of the model's dimensions instead of three: measure each arrow's angle to the average of the others. Near 0° means "the same direction", 90° means "unrelated". Most of the model's arrows stay well below 90°, most clearly for commutativity and associativity; the twin's sit near 90°.

![axiom arrows, animated]({{fig:axiom_arrows_qwen3.5-2b.gif}})

**The same idea, animated.** One rule at a time, the arrows appear between the structures and slide to one starting point: the model on the left, its untrained twin on the right.

![axioms in 3-D]({{fig:lattice3d_qwen3.5-2b}})

**Axioms as arrows, in 3-D.** Each panel highlights all pairs of structures that differ by one axiom, in the three main directions of the model's internal space. In the trained model those arrows point roughly the same way; in the twin they do not.

![axioms in 2-D]({{fig:lattice_axes_qwen3.5-2b}})

**The same, flattened to two dimensions.** The leftmost panel shows the ideal picture, in which every axiom is the same step.

![arrow similarity]({{fig:lattice_cosines_qwen3.5-2b}})

**How parallel the arrows are.** Red squares along the diagonal mean "adding this axiom" always points the same way. It is strongest for associativity and commutativity.

![Lie groups]({{fig:group_objects}})

**Lie groups sit between their parents.** A Lie group is both a group and a smooth manifold, and the model places it near both. The untrained twin places it near "group" because the two names share a word.

![partial operations]({{fig:partial_operation}})

**A test that found nothing.** Turning a "partial" structure (like a small category) into a "total" one (like a monoid) is not a consistent arrow. With so few examples, the test is weak.

![effect sizes]({{fig:forest}})

**How big the effects are.** Each line is a range of plausible values (95% confidence interval). A blue line does not cross zero, so the effect is reliably there.

![field maps]({{fig:field_maps_qwen3.5-2b}})

**The map of mathematics.** Each dot is a field. The grey lines join fields that mathematicians often combine. In the trained model most of these lines stay short; in the untrained twin they cross the whole picture.

![distance matrices]({{fig:field_rdms_qwen3.5-2b}})

**The same map as a table of colours.** Each square shows how far apart two fields are; light means close. The model's table partly repeats the pattern of the mathematicians' table; the twin's is random.

![map across layers]({{fig:depth_curves}})

**When the map appears.** For each model, how well the map matches arXiv at every layer, from input to output. In {{fact:clears_null_every_layer}} it is above the significance line at every layer; Gemma is at or below it in {{fact:below_null_runs.gemma-4-e2b}}.

![3-D turntable]({{fig:turntable_qwen3.5-2b.gif}})

**Days, months and angles in 3-D.** The three families in their three main directions, turning once (model on top, untrained twin below). A third direction does not turn the angles into a circle. We checked this properly, not by eye: no pair of the top five directions holds a significant closed loop of angles in any model. We wrote that test down before running it.

![chains in 3-D]({{fig:chains3d_qwen3.5-2b}})

**Chains of definitions in 3-D.** Each chain goes from general to special, such as "ring ⊃ commutative ring ⊃ … ⊃ field". In the trained model the ring and space chains run as paths; in the twin they bunch up.

![days, months, angles]({{fig:cyclic_qwen3.5-2b}})

**Circles, flattened to two dimensions.** Days and months go round in order; the untrained twin scrambles them. Angles do not close: 330° lands far from 0°. In this flat view the months' loop is open between December and January; it closes in another pair of directions.

![circle forming]({{fig:circle_depth_qwen3.5-2b.gif}})

**The circle through the layers.** Days and months, layer by layer, in the model (top row) and in its untrained twin (bottom row). The circle is there almost from the start; the twin never forms one.

![angles]({{fig:angles_qwen3.5-2b}})

**Our failed prediction.** We expected a circle (left). The model instead arranges angles like the plain numbers 0–330 (third panel), so "330 degrees" is far from "0 degrees".

![chains]({{fig:chains_qwen3.5-2b}})

**Chains of definitions, flattened.** The model lays them out in order; the twin does not.

![abstraction]({{fig:abstraction_qwen3.5-2b}})

**Concrete to abstract.** The model's biggest axis runs from concrete ideas ("bar chart") to abstract ones ("generalized cohomology theory"). On the right, the judges' abstraction ratings agree with how deep each idea sits in a formal mathematics library (Mathlib).
