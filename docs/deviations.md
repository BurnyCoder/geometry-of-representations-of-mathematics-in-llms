# Deviations from the preregistration

Every change made to the analysis after the `prereg-v1` tag is listed here, with a date and a reason. Changes made *before* the tag are design decisions and appear in [`prereg.md`](prereg.md). The main ones:

- The primary tests are paired contrasts against the random-init twin. We moved away from partial regression after simulations showed it inflates type-I error.
- Families use Euclidean geometry.
- Common single words are flagged as polysemous.

| Date | Change | Reason | Affects |
|---|---|---|---|
| 2026-10-29 | Field → super-tree branch assignment for H-A2 written into `configs/fields.yaml` (`branch:`): algebra & number theory = {group, ring/module, commutative algebra, Lie/representation, number theory, algebraic geometry}; geometry & topology = {general topology, algebraic topology, differential geometry}; analysis = {real/complex, functional}; foundations & discrete = {set theory/logic, category theory, graph theory}; applied = {probability/statistics, numerical analysis, game theory}. | `prereg-v1` named the five branches but not the membership; fixed before any H-A2 statistic was computed. | H-A2 |
| 2026-10-29 | Gemma-4-E2B loaded with a CPU-offloaded per-layer-embedding/audio/vision device map (`offload_ple`). | The full multimodal checkpoint (5.1B params) does not fit in 8 GB; the decoder runs on GPU unchanged. | Robustness |
