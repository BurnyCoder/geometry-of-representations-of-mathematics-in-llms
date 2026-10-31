---
license: cc-by-sa-4.0
language: [en]
pretty_name: Geometry of Mathematics in LLMs — concepts, families, activations
tags: [interpretability, representation-geometry, mathematics, llm, activations]
---

# Geometry of Mathematics in LLMs: dataset card

These are the data behind [github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms) (paper: [`paper/paper.pdf`](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/blob/master/paper/paper.pdf); interactive explorer: <https://burnycoder.github.io/geometry-of-representations-of-mathematics-in-llms/>). The project asks whether LLM residual streams encode three kinds of geometry:
- a *map of mathematical fields*;
- *shapes* for ordered mathematical families;
- *axiom lattices* for algebraic structures.

## Contents

| File | What it holds | Source / licence |
|---|---|---|
| `concepts.csv` | One row per concept, with these columns: <ul><li>`concept` (the cleaned term), `field`, `kind` (math/control), `pa` (pure/applied)</li><li>flags: `leak_own`, `eponym`, `polysemous`, and `primary` (none of these three); also `leak_any` (the name contains any math field's root word; reported, does not exclude) and `multi_field` (always false here: terms listed under more than one field are dropped before this table is written)</li><li>`head` noun and `zipf` frequency</li><li>`split` (pilot/confirmatory)</li><li>`page` and `revid` of the source Wikipedia page</li></ul> | Wikipedia glossaries at pinned revisions, CC BY-SA 4.0 |
| `wiki_revisions.json` | Page title → revision id | Wikipedia |
| `fields.yaml` | The 17 math and 4 control fields: source pages, arXiv categories, pure/applied label, super-tree branch, leak root words | Hand-written config |
| `field_distance.json` | Field × field Ochiai distance from arXiv cross-listing, with the dataset commit and per-field paper counts | `librarian-bots/arxiv-metadata-snapshot`, CC0 |
| `families.yaml` | Chains, angles, n-gons, and axiom lattices with ground-truth bits; most entries cite their source | Wikipedia, nLab, Crans 2004, Munkres 2000 |
| `infinigram_counts.json` | Per-concept n-gram counts from the [infini-gram API](https://infini-gram.readthedocs.io/en/latest/api.html) (index `v4_dolma-v1_7_llama`, i.e. Dolma v1.7), a pretraining-scale frequency covariate | infini-gram API |
| `judge_ratings.csv` | 1–7 abstraction ratings from three judge LLMs, each the expectation over digit probabilities | Model outputs |
| `mathlib_depth.csv` | Per math concept: matched Mathlib declaration and module (if any), match source (curated YAML or CamelCase), module import depth, mean judge rating | [Mathlib4](https://github.com/leanprover-community/mathlib4) at the commit in `configs/mvp.yaml`, Apache-2.0 |
| `acts/family_index.json` | Which rows of the `family` activations belong to which family item. The last 24 rows (family `nonce_modifier`: six heads, each also with "commutative", "wooden" and the nonce word "zorbic") were appended by the independent audit; all earlier rows are unchanged | Computed |
| `acts/<model>/<condition>.npz` | float16 arrays of shape `[items × (layers+1) × d_model]`. Keys: <ul><li>`last`: concept's last token, averaged over 8 templates</li><li>`last_a` / `last_b`: averages over the two template halves</li><li>`pool`: mean over the concept's tokens</li><li>`n_tokens`</li></ul> | Computed |
| `logs/*_prompts.jsonl` | Every prompt sent to every model, with its greedy next token, or the judge's digit distribution | Computed |
| `results/<config>/*.json` | Every statistic of the study (the files the README, paper and site read their numbers from) | Computed |

`<model>` is a model key from the run config. `<model>__random` is its random-init twin: the same architecture and tokenizer, with weights redrawn using seed 0. `qwen3.5-2b__random_s1` … `qwen3.5-2b__random_s4` are four more random-init twins of the confirmatory model, with seeds 1–4 (`field_neutral` and `family` only), added by the independent audit to measure how much twin statistics depend on the random draw (`configs/twin_seeds.yaml`; `results/twin_seeds/twins.json`).

**Models** (base checkpoints): [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (confirmatory), [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) and [Gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B), all Apache-2.0. Judges: [granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro) and [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) (Apache-2.0), [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) (MIT).

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
- The `eponym` flag missed five primary concepts (tate conjecture, tate curve, fourier, kan complex, boolean), because cleaning lowercases a term's first word when it is also a common word. They are flagged `eponym = False` in `concepts.csv` and stay in the primary set, which was frozen at the preregistration tag.
- The template-averaged residuals at the concept's last token are one specific read-out; others may behave differently.
- A random-init twin is not a control for word form: it can scramble even exact word composition (strongly in the Qwen3.5 twins, hardly in Gemma's). The [independent audit](https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/blob/master/reports/audit_lexical.md) therefore compares every test with word-form baselines (character 3-grams, word bags, and the sum of the input embeddings over a name's tokens, which is `pool[:, 0] · n_tokens` in these arrays).
