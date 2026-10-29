---
license: cc-by-sa-4.0
language: [en]
pretty_name: Geometry of Mathematics in LLMs — concepts, families, activations
tags: [interpretability, representation-geometry, mathematics, llm, activations]
---

# Geometry of Mathematics in LLMs: dataset card

These are the data behind [github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms). The project asks whether LLM residual streams encode three kinds of geometry:
- a *map of mathematical fields*;
- *shapes* for ordered mathematical families;
- *axiom lattices* for algebraic structures.

## Contents

| File | What it holds | Source / licence |
|---|---|---|
| `concepts.csv` | One row per concept, with these columns: <ul><li>`field`, `kind` (math/control), `pa` (pure/applied)</li><li>flags: `leak_own`, `eponym`, `polysemous`, and `primary` (none of the flags)</li><li>`head` noun and `zipf` frequency</li><li>`split` (pilot/confirmatory)</li><li>`page` and `revid` of the source Wikipedia page</li></ul> | Wikipedia glossaries at pinned revisions, CC BY-SA 4.0 |
| `wiki_revisions.json` | Page title → revision id | Wikipedia |
| `field_distance.json` | Field × field Ochiai distance from arXiv cross-listing, with the dataset commit and per-field paper counts | `librarian-bots/arxiv-metadata-snapshot`, CC0 |
| `families.yaml` | Chains, angles, n-gons, and axiom lattices with ground-truth bits; each entry cites its source | Wikipedia / nLab definitions |
| `judge_ratings.csv` | 1–7 abstraction ratings from three judge LLMs, each the expectation over digit probabilities | Model outputs |
| `acts/<model>/<condition>.npz` | float16 arrays of shape `[items × (layers+1) × d_model]`. Keys: <ul><li>`last`: concept's last token, averaged over 8 templates</li><li>`last_a` / `last_b`: averages over the two template halves</li><li>`pool`: mean over the concept's tokens</li><li>`n_tokens`</li></ul> | Computed |
| `logs/*_prompts.jsonl` | Every prompt sent to every model, with its greedy next token, or the judge's digit distribution | Computed |

`<model>` is a model key from the run config. `<model>__random` is its random-init twin: the same architecture and tokenizer, with weights redrawn using seed 0.

## How the data were built

The full pipeline is `run.py`:
1. Fetch glossaries at pinned revisions.
2. Clean the terms and flag leakage.
3. Sample at most 30 primary concepts per field.
4. Split the concepts 20% pilot / 80% confirmatory.
5. Compute arXiv field distances.
6. Run 8 templates per item through each model and its random-init twin.

## Intended use and limitations

The data are for interpretability research on representation geometry. Known limitations:
- Glossary terms are a convenience sample of each field. Some junk terms survive the cleaning rules.
- Abstraction ratings come from LLM judges, not from humans.
- The template-averaged residuals at the concept's last token are one specific read-out; others may behave differently.
