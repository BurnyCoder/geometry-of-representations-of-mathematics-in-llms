# Experiment log and learnings

This file records **what we tried, why, what went wrong, how we found out, and what we chose instead**. It is hand-written prose and deliberately contains **no result numbers**. Every number lives in auto-generated files, regenerated with `python run.py report --config configs/<cfg>.yaml`:
- [`<cfg>_results.md`](.) tables;
- [`../results/`](../results) JSON.

Experiments are listed in the order they were run.

---

## E0: Method validation before touching LLM data (calibration study)

**Question.** Can partial regression on distance matrices remove lexical and frequency confounds? Concretely: regress the concept RDM on a field-distance RDM plus nuisance RDMs, with Freedman–Lane permutation.

**Why we asked.**
- Field maps are easy to fake. Names leak fields ("topological space"), and rare jargon clusters by frequency.
- Partial Mantel tests have a known type-I problem ([Guillot & Rousset 2013](https://doi.org/10.1111/2041-210X.12018)).

**Method.** [`src/mathgeom/analyses/calibration.py`](../src/mathgeom/analyses/calibration.py) simulates five scenarios, 200 runs each, and records how often p < .05:
- the nuisance is correctly specified;
- the nuisance is misspecified (item-level noise, nonlinear link);
- field level, pure null;
- field level, where the nuisance is only a noisy proxy of a confounder;
- field level with a real effect (power).

**What happened.** The test is calibrated only when the nuisance RDM is exactly the confounder.
- With item-level noise, or a proxy nuisance, false positives multiply. See the *Method validation* table in [`mvp_results.md`](mvp_results.md).
- A cubic basis of the nuisance ranks did not fix it. We tried this interactively before writing the phase.

**Decision.** Every primary test became a **paired contrast against a random-init twin**: the same architecture and tokenizer, with weights redrawn at random.
- The twin measures form-only similarity (shared sub-word tokens, word length, morphology) *on the same concepts*.
- So it does not depend on how well a nuisance RDM approximates the confounder.
- Uncertainty comes from a stratified concept bootstrap with identical resamples for both models.
- Partial-regression coefficients are kept, but only as *secondary/descriptive* numbers.

This was decided before the preregistration was tagged ([`docs/prereg.md`](../docs/prereg.md)).

## E0b: Synthetic shape tests (TDD)

**What we tried.** Cosine distance after mean-centering, the usual RSA choice, for small ordered families.

**What went wrong.** The unit test with a planted line failed. After centering on a family's own mean, collinear points on one side of the centroid all point the same way, so cosine distance collapses to roughly {0, 2}.

**Decision.** Families and lattices use **Euclidean** distance, which is what PCA plots ([Engels et al. 2024](https://arxiv.org/abs/2405.14860)) and SMDS stress ([Tiblias et al. 2025](https://arxiv.org/abs/2510.01025)) use. The field map keeps cosine distance on globally centered concepts.

A second test failure was a bug in the test itself: the synthetic shape was smaller than the high-dimensional noise norm. We scaled the shapes; the statistic was unchanged.

## E1: Concept data (Wikipedia glossaries)

**Why glossaries.** They are curated, field-labelled lists of concept names, available at pinned revisions under CC BY-SA. They cover 17 math fields and 4 non-math control fields.

**Problems we found by reading the term lists before any extraction, and their fixes:**

| Problem | Fix |
|---|---|
| Sentence-initial capitals were lowercased, which hid eponyms ("Plücker", "Kuranishi") | A capital is kept when the page never uses the word in lowercase mid-sentence |
| Sentence-case glossaries made "Homeomorphism" look like a name | The same mid-sentence check fixes this |
| Fragments and junk: "dom", "adic", "say", "such that", "pseudo-", "faithfully", "player i's strategy space" | Drop single words of ≤ 4 characters, single-letter tokens, prefixes ending in "-", and adverbs ending in "-ly" |
| Two differential-geometry glossaries use `<p><b>term</b>` markup instead of `<dt>` | Added a fallback parser |
| Everyday polysemous words ("mean", "walk", "state") | Flagged `polysemous` (single word, wordfreq Zipf ≥ 4) and kept out of the primary set |

**Field-leak and eponym flags** follow the plan's design review: concepts whose name contains their own field's root word, or a proper name, are excluded from the primary set. Flagged concepts are kept for the "math_all" sensitivity analysis.

## E2: Ground-truth field distance (arXiv cross-listing)

**Why not MSC2020.** Above its 2-digit codes it is flat, so a tree distance would be only same/different.

**What we used.** Ochiai similarity of the paper sets in each field's arXiv categories, computed from the CC0 arXiv metadata snapshot (pinned by commit). We read only the `categories` column of the parquet files.

**Sanity check.** Nearest neighbours match how mathematicians think of the fields: algebraic geometry ↔ commutative algebra ↔ number theory, category theory ↔ algebraic topology, real ↔ functional analysis, general topology ↔ logic.

## E3: MVP extraction and analysis (Qwen3.5-2B-Base + random-init twin)

See the results files. The interpretation is added after the run, in [`summary.md`](summary.md).

## Infrastructure learnings

- **Disk.** C: was 99% full. The venv, HF cache and activations live on D: through directory junctions (`.venv` and `artifacts` → `D:\AI\geometry-math-llms`). D: is a slow disk, so the first `import torch` and model loads are slow.
- **Windows MAX_PATH.** A uv cache inside the long scratchpad path produced broken numpy and pytest installs (`_multiarray_umath` missing). A short cache path fixed it.
- **HF auth.** Moving `HF_HOME` hides the CLI token, so `HF_TOKEN_PATH` must point at the original token file.
- **Parallel downloads.** Pre-downloading robustness models in parallel starved the MVP download. Download sequentially.
- **Model classes.** `AutoModelForCausalLM` builds a text-only class for Qwen3.5 but the full multimodal class for Gemma 4. The Gemma per-layer-embedding tables and towers are therefore offloaded to the CPU (`offload_ple`).
