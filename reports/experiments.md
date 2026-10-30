# Experiment log and learnings

This file records **what we tried, why, what went wrong, how we found out, and what we chose instead**. It is hand-written prose. It quotes a few numbers where they explain a decision (e.g. simulation rejection rates, a leaked statistic), each from the file named next to it; every reported result lives in auto-generated files, regenerated with `python run.py report --config configs/<cfg>.yaml`:
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

**What ran.**
- 815 concepts and 171 family items, 8 templates each, through the trained model and its twin.
- All 25 hidden states were stored. Extraction took about 1 s per item, because Qwen3.5's hybrid linear-attention layers fall back to the PyTorch implementation.

**Field map.** It is clearly present and absent in the twin. Results: [`mvp_results.md`](mvp_results.md).

**Problem 1: all-but-the-top reversed the effect.**
- The preregistered all-but-the-top variant (remove the top d/100 principal components) *reversed* the field-map correlation.
- *Hypothesis:* the map lives in the dominant components rather than in a few rogue dimensions, so removing 20 components removes the map.
- *Test:* a post-hoc sweep over the removal depth plus per-component field variance (η²). Removing the first component helps, removing more steadily destroys the map, and field variance is spread over the top ~10 components.
- *Decision:* report the preregistered variant as it came out, and add the sweep as an explicitly post-hoc explanation.

**Problem 2: the positive control (days/months) failed its preregistered criterion.**
- Circle-template ρ was significant and the closure ratio was near 1. But "circle ρ > line ρ" failed at the preregistered layer.
- The twin also had a closure ratio near 1, which revealed that the closure ratio cannot tell a circle from structureless points.
- Following the prereg, we debugged before interpreting anything else. Engels et al. found circles in low-dimensional projections, so we tested cyclic order in the top-2-PC plane.
- Days and months are *perfect* cycles there in the trained model, and nothing in the twin. So the pipeline works and the full-space criterion was the weak link.
- *Decision:* keep the failed criterion in the report, add the PC-plane statistic as post-hoc, and add `closure_far` as post-hoc as well.

**Problem 3: the specificity comparison was confounded.**
- 17 math fields were being compared with 4 control fields, so chance levels differ.
- We added a post-hoc matched comparison over random 4-field subsets of the math fields.
- Math fields cluster about as strongly as science topics, not more.

**Bug.** Activations are stored as float16, which LAPACK rejects. A regression test now covers this.

## E4: Robustness (scale, model family, math training)

**Qwen.** Qwen3.5-0.8B, Qwen2.5-1.5B and Qwen2.5-Math-1.5B ran without changes.

**Gemma-4-E2B needed two attempts.**
- *Attempt 1:* an accelerate `device_map` with "cpu" entries for the 2.35B-parameter per-layer-embedding table. It failed with "Cannot copy out of meta tensor", because accelerate treats "cpu" as offload-to-GPU-on-use and Gemma reads that table directly.
- *Attempt 2:* explicit placement. The table and the unused audio and vision towers stay in CPU RAM, the decoder goes to the GPU, and two forward hooks move the lookup's input and output between devices.
- We also moved Gemma to its own config, so the Qwen analyses could proceed meanwhile.

**H-A3 (math pretraining sharpens the map)** is a clean null with a tight CI. We report it as such.

## E5: Judges, axes, causal test, SAE

**Judges.**
- One judge (Granite) puts essentially all next-token probability on the digits 1–7. The others put most of it there, and we renormalize over the digits.
- Rank agreement is moderate and absolute agreement low, because the judges use different scale offsets. Only rank-based statistics use the ratings.

**Causal test.**
- The preregistered accuracy gate (≥ 0.70 on "which is more general?") was set *before* running it. The ring chain missed it narrowly, so it was not intervened on.
- The space chain passed. Chain-subspace noise hurt only marginally more than random-subspace noise, so we call this inconclusive.
- With five random subspaces, the smallest attainable rank p-value is about 0.17. A stronger test needs more random draws and more questions; this is future work.

**SAE.** Qwen-Scope SAEs show field-selective features, but the graded field map is much weaker in the sparse code than in the dense residual.

## E6: Plan and preregistration audit

Asked to double-check completeness, we compared every commitment in the approved plan and in `docs/prereg.md` against the repository. What we found:

**A wrong primary measure.** H-A2 had been computed as a tree-distance RSA, but the preregistration specifies *cophenetic correlation*.
- With the preregistered measure, a UPGMA dendrogram matches the super-tree better than shuffled trees, but not better than the random-init twin.
- It is also no more tree-like than a spectrum-matched Gaussian cloud, which the plan also required.
- The earlier "supported" claim was withdrawn. The RSA is kept as post-hoc only.
- *Lesson:* implement the preregistered statistic verbatim before any variant.

**Missing baselines (fastText, a sentence embedder).**
- The LLM's field map beats static fastText vectors.
- It does not beat Qwen3-Embedding. The map is a property of contextual text representations in general, not of generative LLMs specifically.

**Missing items, now run:**
- **Infini-gram covariate:** the H-A4 axes survive it.
- **SMDS with shuffled refits:** the twin's line fit is perfect for any labelling, a vivid demonstration of the small-n overfitting trap, and none of its fits beats the null.
- **Template variants:** naming the field inflates clustering in the trained model *and* in its twin, which is the leakage the preregistration guarded against. The graded map barely moves.
- **Manifest, hook test, docs-regeneration lint, per-prompt causal logs:** the causal re-run reproduced its earlier results byte for byte.

**Citations.** Every arXiv ID and DOI was checked through the arXiv API and Crossref. One DOI (Anderson & Legendre 1999) pointed to the wrong paper and was fixed; two papers had been retitled.

**A reporting bug.** A second config reusing a model key overwrote the confirmatory figures and the explorer data. Variant configs now write to their own figure subfolder and skip the site export.

**Not run, with reasons in `docs/deviations.md`:**
- Qwen3.5-4B in 8-bit: its random-init twin does not fit.
- OctoThinker: optional.
- Ripser: no family has ≥30 points.
- Mathlib validation of the judges. *(Run in E7.)*

## E7: Second plan audit

A second line-by-line pass over the approved plan found six items still missing or partial. Each analysis was specified in `docs/deviations.md` before it was computed.

**Mathlib validation of the judges (H-A4).** This was the user's chosen abstraction ground truth ("Judges + Mathlib") and had not been run.
- *Method.* Mathlib4 was cloned sparsely (sources and `docs/` only) at a pinned commit. Each concept was matched to a Lean declaration, first through Mathlib's curated English-name maps (`docs/overview.yaml`, `docs/undergrad.yaml`), then through exact CamelCase names for multi-word concepts. Depth is the defining module's longest import chain, computed with the standard library's `graphlib.TopologicalSorter`.
- *Why module depth.* Declaration-level dependency depth needs a compiled Mathlib, which takes hours. Import depth needs only the sources.
- *What we learned.* Coverage is low: glossary vocabulary ("point of closure", "completely regular") rarely matches Mathlib names, and single words ("kernel", "ideal") are too ambiguous to match by name, so they were excluded by design. Every match was inspected by hand; all name the intended concept, and one ("inductive limit") points to Mathlib's metric-space construction.

**Chain frequency control (plan H-B).** The plan required partialling log-frequency and token count from the chain-order test; the preregistration had dropped it without a logged deviation.
- *Issue.* The first specification residualized the LOO predictions and the ranks on the covariates and kept the usual order-permutation null. Before touching real data, a simulation in which the representation encodes *only* a frequency that tracks the order rejected 42% of the time at α = .05. Permuting the order breaks the order–frequency link that this null hypothesis keeps, so the null distribution was the wrong one.
- *First fix.* Residualize the representation as well as the ranks, then predict the residual order from the residual representation. The same simulation gave 2.5% rejections under the frequency-only null and 92.5% power even when frequency is a noisy copy of the order.
- *Second issue, found on real data.* In the random-init twin, the first fix scored ρ = 1.0 for the true order and for every permuted one: a degenerate statistic. The residualization had used all 11 items. Residuals sum to zero, so each held-out residual is an exact linear combination of the training residuals, for both the representation and the target. The twin's large-scale activations make ridge regularization negligible, so LOO "predicted" any order. This is data leakage across the cross-validation split.
- *Final fix.* Cross-validated confound regression ([Snoek, Miletić & Scholte 2019](https://doi.org/10.1016/j.neuroimage.2018.09.074)): fit the covariate regression on each fold's training items only and apply it to the held-out item. Simulation: 5% / 2.5% rejections under the frequency-only and large-scale-noise nulls, and 100% power. A regression test now covers the large-scale-noise case. Because this change came after seeing the confirmatory output, it is logged as post-hoc, and the leaky version's numbers are kept in `docs/deviations.md`.
- *Lessons.* A permutation null is valid only if it is exchangeable under the specific null hypothesis being tested. Any preprocessing fitted on data, including confound regression, belongs inside the cross-validation fold. Check a new statistic on the random-init twin's regime (large, structureless activations), not only on unit-scale synthetic data.

**Lattice items from the plan that had been extracted or planned but never analysed:**
- associative quasigroup, the semantic-collapse probe (a non-empty one is a group);
- inverse semigroup;
- the partial-operation lattice (semigroupoid, small category, groupoid);
- lattice-like held-out pairs for the identity-direction transfer;
- offsets on mean-pooled tokens, and with the token-count direction removed (a one-direction analogue of Li et al.'s LDA distractor removal).

The new items were appended after the existing ones. Extraction runs at batch size 1, so every earlier activation and every earlier result is unchanged. We verified this in two ways:
- items extracted twice ("semigroup", "monoid", "group", appearing in both the group-like and partial-operation lattices) have bit-identical activations;
- diffing every results file before and after the rerun showed no changed value, only added keys.

## E8: Article-style, interactive visualizations

**Goal.** Make the figures readable and explained in the style of Anthropic's Transformer Circuits articles and Goodfire's research posts, and make the explorer interactive. Two references were studied for their visual devices:
- [Gurnee et al. 2025, "When Models Manipulate Manifolds"](https://transformer-circuits.pub/2025/linebreaks/index.html): colour along the manifold coordinate; the same object shown as a projection and as a similarity matrix; takeaway-first captions.
- [Goodfire, "Finding the Tree of Life in Evo 2"](https://www.goodfire.com/papers/phylogeny-manifold): a hero figure, an animated view, small multiples.

**A bug the redesign found.** The old report code loaded activations directly instead of through the config's template mapping, so every "variant" figure duplicated the confirmatory one. The loader is now public (`pipeline.load_acts`) and used everywhere. A consistency test checks every config: the figure data must reproduce the reported field ρ, the offset consistencies and the cycle scores exactly. The loader also now reads each array on first use, which made figure building several times faster on the slow data disk.

**Design decisions and why:**
- **Colour.** The palette validator showed that no four-colour set separates all pairs for colour-vision deficiency. So no plot uses more than three hues; the four axioms are shown as small multiples, one axiom highlighted per panel, instead of four colours on one plot.
- **Unsupervised views only.** With 13 items in more than 1,000 dimensions, a projection fitted to labels draws a lattice even for random vectors. All pictures are PCA or MDS. The quantitative evidence is drawn as the cosine matrix of difference vectors in the full space.
- **A visual artifact we guard against.** In that matrix, two difference vectors that share a structure have cosine ≈ ±½ even for random vectors, so those cells are hatched.
- **Fixed matrix order.** Distance matrices use one fixed order (super-tree branch, then field). Re-clustering each matrix separately would create blocks out of noise.
- **Alignment is display-only.** Panels and layers are aligned by rotation and reflection only, so the layer slider doesn't flip. Every statistic shown is read from the results files, never recomputed from a 2-D picture.
- **Exact browser-side geometry.** For a family of n items, the (n − 1)-dimensional PCA coordinates preserve every distance and cosine of the centred vectors. The site's analogy playground therefore computes exact ranks in the browser; a test checks this against the Python statistics.

**What went wrong along the way:**
- Statistic badges inside panels covered data points; they moved under the panels.
- A hand-typed paper count slipped into a figure note and was replaced by the value from `field_distance.json`.
- Two README captions overclaimed: that the field map clears the null "from the first layers" (false for Gemma), and that the days cycle "persists to the final layer" (it breaks at the last layer). Both were corrected after checking them against the figures.
- The browser pane does not repaint a hidden page after scrolling, so the site was checked with full-page screenshots from headless Edge plus scripted interaction tests: linked hover, the layer slider, model switching, search, chips and the analogy playground. At phone width it was checked by measuring element widths.

**A plan audit of the visualizations found these gaps and errors, all fixed:**
- *Palette.* The plan required running the colour-vision validator on every palette used; it had only been run on candidate four-colour sets. Running it found a bug: the ordinal ramp used a discrete colormap, so with 12 items two pairs of neighbours shared a colour. The ramp now interpolates. A 7-step ramp passes; a 12-step ramp cannot keep every adjacent lightness gap above the threshold, so order is always also shown by labels and the path.
- *Missing pieces.*
  - the hero's field-map panel and its word-length control;
  - the group-object figure;
  - the twin's values on the abstraction figure;
  - table views for most interactive charts;
  - two-way linked hover (matrices → map → concept cloud);
  - the ideal-shape overlay and per-layer statistics in the family tooltips;
  - smooth transitions between layers;
  - links from each site figure to its paper section;
  - interactive links in most README and paper captions;
  - a documentation page (`docs/figures.md`).
- *Overclaims caught on the replication models' figures* (viewed as contact sheets):
  - "twins place 'Lie group' next to 'group' only" is false for Gemma's twin, which ranks "smooth manifold" 6th.
  - "a circle needs closure < 2" is misleading for Qwen2.5-Math, which meets that half (1.84) but fails "circle ρ > line ρ". The angles figure now shows both halves of the preregistered rule.
- *A broken link.* The paper linked to a site anchor that did not exist. A test now checks every link from the README, summary and paper into the site.
- *Deferred at first.* The plan's global trained ↔ twin toggle was first skipped, because every site figure already shows both side by side. The third pass built it (below).

**A second audit pass, against the plan text rather than memory, found more:**
- *Dark-mode contrast.* The site used the light-theme ordinal ramp on the dark surface, where its darkest step has 1.46:1 contrast (floor 2:1). Dark mode now uses step 600 → 100, which passes the validator.
- *Forest plot.* The plan says verdicts must come from the Holm computation; the plot showed only "CI excludes 0". The H-A1 rows now carry the Holm decision, other rows are marked as outside the Holm family, and the site hover shows n.
- *Controls and negative results.*
  - The angles figure lacked a twin panel.
  - The hero showed only the trained field map, where the plan asked for arXiv vs trained vs twin.
  - The partial-operation lattice, a negative result the plan said must get a figure, had none.
  - All three are added.
- *Site.*
  - added a preregistered/post-hoc status column (and hover) on the scorecard;
  - added a depth-curve inset in the control bar that follows the slider;
  - hovering the concept cloud now highlights the field in the map and matrices;
  - clicking a lattice node shows its axioms;
  - the concept card now lists 10 neighbours (it had 8) and the concept's flag status;
  - the families table view now includes the per-layer cycle scores.
- *A finding the new twin panel exposed.* All five random-init twins *meet* the preregistered angles rule (closure < 2 and circle ρ > line ρ), while all five trained models fail it. A structureless point cloud has closure ≈ 1 and near-zero template correlations, so it passes by chance. The rule is not specific, and the informative result is the trained models' line-like order. Reported in the paper and README; this is the second preregistered shape criterion (after the days/months full-space criterion) that turned out to be insensitive or non-specific.
- *Summary.* It carried only five of the figures; it now has all of them with plain-language captions. Writing them surfaced two loose phrasings, both fixed: "the map appears and stays" (Gemma dips twice) and an example concept that is not labelled in the figure.

**A third audit pass, line by line against the plan, closed the remaining gaps:**
- *Controls in every panel.* The README GIF showed only the trained model; it now has a twin row, so every frame shows its control. The hero's angles panel now gives the twin's closure ratio next to the trained one, and its caption says why a small twin value is not a circle.
- *Scorecard at the top.* The plan put the scorecard next to the hero; it sat inside the results block. The README now has two generated blocks: `TOP` (scorecard and the GIF with its three-part caption, from `reports/readme_top_template.md`) and `RESULTS`. The GIF had also appeared twice; it now appears once.
- *Legends.* The plan asks for a legend whenever a chart has two or more series; the site charts had none. The field map, depth curve, cycle-score curve, forest plot, concept cloud, family planes, lattice and group-object charts now carry one.
- *The global trained ↔ twin morph.* A control-bar button switches the main panels (field map and matrix, families, lattice, concept cloud) to the twin; 2-D points glide with a Plotly transition. To make the motion minimal, the twin's family and lattice planes are rotated onto the trained plane (rotation and reflection only; a test checks that no distance changes).
- *Missing pieces.* The site had no partial-operation chart (now `#fig-partial`, with the selected model, the other models, all twins and the null), the forest plot had no filter by hypothesis (now chips), and the method figure had no caption. The paper's partial-operation caption now links to the site.

## E9: 3-D views, and a closed-cycle test beyond the top-2 plane

**Goal.**
- The user asked for 3-D visualizations where possible (the top 3 principal components), as interactive as possible, and for the scientific results to change if the 3-D views changed them.
- Motivation from the literature:
  - numbers lie on a generalized helix: the top PC is nearly linear in the number, and the periodic parts (periods 2, 5, 10, 100) are found with Fourier features ([Kantamneni & Tegmark 2025](https://arxiv.org/abs/2502.00873));
  - Engels et al. show raw-activation circles in the top two PCs, but their SAE-cluster circles in PCs 2–3, and fit their circular probe on the top five PCs ([arXiv:2405.14860](https://arxiv.org/abs/2405.14860), Figs. 1, 4, 15);
  - Gurnee et al. and Goodfire present manifolds as rotatable or animated 3-D views.

**What was built.**
- *Site:* 3-D by default, with a 2D switch per figure.
  - Field maps use 3-D MDS; families, the lattice and the concept cloud use each layer's top PCs.
  - Interactions:
    - linked cameras across panels, a turntable Spin button, and an axes picker (any 3 of PC1–5);
    - a variance bar chart whose bars put a PC on an axis, and click-to-pin with info lines;
    - the analogy query b − a + c drawn exactly in the lattice;
    - wall-clock tweening for layer, model and trained ↔ twin changes.
- *Static:*
  - fixed-camera 3-D figures for families, chains and the lattice in every model;
  - a turntable GIF for the README.

**Integrity decisions.**
- *True proportions everywhere* (`aspectmode: "data"`, `set_box_aspect`). The old concept cloud used a cube aspect, which inflated PC3.
- *Each 3-D view says how much it shows.* PCA views give the variance per axis; MDS views give stress-1.
- *Axes are each layer's own PCs.*
  - The first version rotated every layer's 5-D view onto the preregistered layer (Procrustes), as the 2-D views do.
  - Testing in the browser showed this makes the printed "PC3 · 14%" wrong everywhere except the trained model at the preregistered layer, because a rotated axis mixes PCs.
  - Axes are now only *sign*-aligned (`figdata.pcs_by_layer`); a test checks that every axis is that layer's PCk.
- *MDS fit measure.*
  - A Spearman "fidelity" of layout distances was tried first. It said the 3-D arXiv layout was *less* faithful than 2-D (0.57 vs 0.63), although 3-D cuts the MDS stress from 0.35 to 0.25. Convergence was checked (3,000 iterations, 8 random starts).
  - The cause: the arXiv distances are skewed, so rank correlation weighs tiny differences among large distances.
  - Replaced by Kruskal's stress-1 with optimal scaling, the measure metric MDS optimizes.

**Scientific follow-up: post-hoc 4.**
- *What we did.*
  - Only one reported result depends on a projection: the post-hoc PC1–2 cycle score.
  - We pre-specified a test of all 10 planes of PC1–5 in [`reports/prereg_addendum_3d.md`](prereg_addendum_3d.md) and committed it before running.
- *Before any real data was run, unit tests forced two amendments:*
  1. *An open arc scores 1.* The angular cycle score gives a perfect 1 to an open arc: the plane of a line and one periodic coordinate is a U. PCA bends lines into such horseshoes ([Diaconis, Goel & Holmes 2008](https://arxiv.org/abs/0811.1477)). So a plane now counts only if the loop also closes in it (in-plane closure ratio < 2).
  2. *The test is conservative.* The closure rule makes it so (1,000 null simulations: 1.8% false positives at α = 0.05 for 12 items, 4.0% for 7). The unit test now checks validity, not exact size.
- *The planted-cycle test failed at first because of a bad fixture.* The ramp was not orthogonal to sin θ, so part of the circle leaked into PC1. Rebuilding the fixture with orthogonal components fixed it; the code was right.
- *Results* (the tables are in the README and paper, generated from `results/*/posthoc.json`):
  - **Angles:** no model meets the pre-specified rule. The best is Gemma, at p ≈ 0.08 before Holm correction. "Angles are not a circle" stands, now for every plane of the top-5 PCs.
  - **Days:** the loop closes in the top-2 plane in 3 of 5 models. In Qwen3.5-0.8B and Gemma the top-2 view is an open arc, although the full-space closure ratio is below 2.
  - **Months:** in Qwen3.5-2B's top-2 plane the Dec → Jan step is 3× a neighbour step, an open arc. A closed loop sits in PC2–PC3 or PC4–PC5 in 3 of 5 models.
  - **Twin:** one twin also shows a closed months loop (p = 0.004). Word form alone can occasionally produce one, so the test is not fully specific.
  - **Existing results:** every existing post-hoc number is unchanged. This was checked by diffing all three results files before and after the rerun; the new test uses its own random stream.
- *What changed in the write-up.*
  - Updated: the README highlights, the paper (abstract, §4.3 at the time, now §4.4, limitations), the summary, and the site's "what failed" list and sidenote.
  - The positive control is now described as replicating partly, not "with a post-hoc 2-D test", and the scorecard has two new post-hoc rows.

**A plan audit of E9 closed these gaps:**
- the no-WebGL fallback now shows a note, tested with WebGL actually disabled;
- pinning works in the lattice and in 2-D, highlights the table row, and the info line adds the family's statistic at that layer;
- every 3-D caption lists the interactions;
- the README and paper captions print the exact variance shares through placeholders (`results/<config>/views.json`; a test checks them against the site data, after it caught the lattice share computed on 15 instead of 13 structures);
- the Plotly limitation is cited;
- new tests: a turntable smoke test, and a test that the earlier post-hoc numbers equal the files from before the addendum.

**Arrowheads.** The user asked for bigger arrowheads so arrow direction reads at a glance. The changes:
- *Static 2-D facets:* filled heads about 2.7× longer.
- *Static 3-D facets:* a filled triangle facing the fixed camera, with a fixed size (9% of the panel's extent, at most 60% of the arrow), replacing `quiver` heads that shrank with short arrows.
- *Site 2-D:* Plotly `arrowsize` 1.9 with a filled head.
- *Site 3-D:* cones in `sizemode: "raw"`, each as long as its vector: 13% of the view's extent, at most 60% of the arrow.
  - The earlier `"absolute"` mode multiplies the size by an internal spacing factor, so the cones were too small to see in the rendered page.
  - An inline comment had also briefly swallowed the cones' colour settings (a red colorscale appeared); caught by looking at a screenshot.
- *A reproducibility bug found on the way.* Regenerating also changed figures that had not been touched. Pixel diffs put every change in label positions.
  - The cause: adjustText stops after 1 s of wall-clock time by default, so placement depended on machine load.
  - The fix: a fixed iteration limit, about as fast. A test checks that two identical calls place labels identically, and a second `run.py report` reproduces every figure's hash.

**Issues found while building, all fixed.**
- *Hidden browser pane.* `requestAnimationFrame` never fires in a hidden pane, so tweens hung; they now use timers on wall-clock time.
- *Stale tween frame.* A stale frame overwrote the 2D switch; purging a panel now cancels pending tweens.
- *URL parameter.* The `?view=2d` parameter was read after the buttons were wired.

## E10: Axioms first, with a picture of how the test works

**Goal.** The user asked to make "axioms as directions" (H-B2) the first highlighted result everywhere, with visuals of how the test works, 3-D shown first, and the other results after it. No analysis was rerun and no number changed; this entry records a change of presentation.

**What changed.**
- *New figures* (built in a parallel session; see [`docs/figures.md`](../docs/figures.md)):
  - `axioms_explainer_<model>`: the test in four steps. The lattice of structures; one axiom's pairs as arrows in the top-3 PCs; the same arrows at one common origin, trained vs twin; and a compass of full-space angles for every axiom.
  - `axiom_arrows_qwen3.5-2b.gif`: the arrows sliding to one common origin in 3-D, at the top of the README.
  - A three-row hero with the axioms in the top row.
- *Order.*
  - README: the axiom GIF, then the scorecard; the axioms bullet leads the highlights, and the results block opens with the axioms section.
  - Paper: §4.1 H-B2 axioms, §4.2 H-A1 map, §4.3 robustness (now also the template variants), §4.4 H-B shapes, §4.5 secondary. The abstract, contributions, discussion and conclusion lead with the axioms.
  - Summary: the plain-words list, the hypotheses table and the figures lead with the axioms.
  - Everywhere, a 3-D view comes before the 2-D view of the same data (lattice, families, chains). The days/months layer GIF moved from the top of the README into the shapes section.

**What did not change, and how the text says so.**
- H-A1 (the field map) stays the primary preregistered hypothesis.
- Every test keeps its status label (confirmatory, secondary, exploratory, added or post-hoc), the Holm family is unchanged, and every number is still a placeholder resolved from `results/*/*.json`.
- The paper's Results section, the README results block, the summary and the hero caption each say in one sentence that the order is a presentation choice. A reader who sees axioms first could otherwise take H-B2 for the primary hypothesis.

**Integrity choices in the new captions.**
- *The picture is not the statistic.* The explainer's steps 2–3 and the GIF are 3-D projections. Every caption says that offset consistency is computed in the full residual stream.
- *The compass is a visual aid, not a test.* It shows each difference vector's angle to the leave-one-out mean of the others, which is not the same quantity as the mean pairwise cosine that is tested. Its numbers are therefore not quoted in the text; only the reported consistencies, null percentiles and p-values are.
- *Holm per model.* While reordering, two hand-written statements turned out to be wrong. The paper abstract said divisibility does not survive Holm correction, and the summary said divisibility and identity are significant only before correction. Both hold for the confirmatory model, but not for every replication model: the scorecard shows divisibility surviving Holm correction in two models and identity in one. The abstract now says "in the confirmatory model neither survives Holm correction", and the summary gives the counts.

**Issues found while doing it.**
- *Renumbered sections.* Renumbering the paper's Results sections changed the anchors the site uses to link figures to paper sections (`data-paper`); the new anchors were handed to the site session. In the paper, one `§` reference pointed at an old number (the closed-cycle test, §4.3 → §4.4). Two vague ones ("see §6 and the causal results", "all tests except §4.5") now name the test they mean.
- *New site anchor.* The new link `#fig-axiom-how` resolves only once the site has the explainer figure. `tests/test_site.py` checks every such link.

## E11: Independent fact-check of every claim

**Goal.** The user asked for an independent check of every claim in the README, docs, reports, paper, Hugging Face card, figures, site and code comments, ignoring earlier audits, with sources wherever possible.

**How.**
- Three read-only checkers worked in parallel: README, summary and docs; paper and reports; site, figures and code. Each re-read the claims against `results/*/*.json`, the configs, the data files and the code.
- Every reported discrepancy was re-verified against the JSON or the code before it was changed.
- External facts were checked against primary sources: every cited arXiv id (arXiv API), every DOI (Crossref), the model cards and licence files on Hugging Face, the Gemma config (per-layer-embedding size), and the cited papers' own text (Engels et al. figures, Qwen2.5-Math corpus size, Kantamneni & Tegmark periods).

**What was already right.** All generated numbers. The README blocks, summary, paper and per-config reports re-rendered byte-identically from the JSON, and the 21,347 numbers in the site data matched the results files.

**What was wrong, and why it slipped through.**
- *A rule that was not the preregistered one.* The preregistration calls a chain ordered only if Robinson p < .05 **and** leave-one-out order p < .05, and the trained Robinson score beats the twin's. The report code tested Robinson p only. Every later summary inherited that, so "chains replicate in all five models" was repeated everywhere.
  - The fix tests the conjunction with its intersection-union p, max(Robinson p, LOO p), inside the Holm family ([Berger 1982](https://doi.org/10.1080/00401706.1982.10487790)). The user chose this strict version over adding a separate LOO check.
  - Under it, the ring chain fails in two models. In Gemma, the space chain and the divisibility offset lose Holm significance, because the step-down adjustment depends on every p in the family.
  - **Lesson:** decision rules should live in one function with a test that quotes the preregistration (`tests/test_report_rules.py`).
- *Hand-typed counts.* Placeholders resolved every statistic, but counts such as "three of five", "two stretches" and "all five models" were typed by hand. Several were wrong from the start (Gemma's depth curve dips in three stretches, not two) or went stale after a rule change.
  - The fix is `{{fact:…}}` placeholders (`render.fact`), which compute these counts and name the exceptions.
  - The README highlights moved into the generated block so that they use the placeholders too.
- *Wrong examples in plain-language text.* The summary illustrated the field map with "algebraic geometry sits near commutative algebra and number theory". That is the arXiv pattern; in every model, algebraic geometry's nearest field is set theory & logic. An illustrative example needs the same check as a statistic.
- *Prose about prompts that did not match the prompts.* The site quoted a template that does not exist, and the paper said templates never name a sub-field, which is true of the field-map templates only.
- *Facts copied without opening the source.*
  - The Qwen-Scope SAE licence was described as "Qwen licence, research use", but the repository's LICENSE file is Apache-2.0.
  - A Wikipedia quote said "smooth manifold" where the article says "differentiable manifold".
  - Two citations had no authors or titles (Golechha et al.; Kvinge et al. at ICML 2025).
- *Ground truth that depends on a convention.* The space chain T0 ⊃ T1 ⊃ Hausdorff ⊃ regular ⊃ … is a chain only if "regular" and "normal" include T1, as in Munkres. The cited Wikipedia page uses the other convention. This is documented as a limitation; renaming the items would need re-extraction and would change a preregistered family.
- *A flagger miss.* Five primary concepts are eponyms ("tate curve", "fourier", …) that the eponym flag missed, because glossary cleaning lowercases a term's first word when it is also a common word. The frozen primary set is unchanged; the miss is listed in `docs/deviations.md`.
- *A PDF text-layer bug.* Cambria Math's fi/fl ligatures have no Unicode mapping, so "first" became "irst" in search and copy. The fix turns off common ligatures (`mainfontoptions=Ligatures=NoCommon`). A first diagnosis also blamed lost minus signs and ρ, but that was the Windows console code page; `pdftotext -enc UTF-8` into a file showed them intact.
- *Measured, not remembered.*
  - Extraction times were re-read from `logs/*_extract.log`; Qwen2.5 takes about 20 minutes, not 15.
  - The disk estimate was re-measured: about 50 GB, mostly model weights.

**Result.** The list of every change is in `docs/deviations.md` (rows dated 2026-10-30, audit). No results file changed; only report code, prose, captions, citations and the card.

## Tooling learnings (continued)

- **Line endings.** On Windows, `Path.write_text` translates `\n` to `\r\n`. Python heredoc scripts that search-and-replace code containing `"\n"` literals therefore corrupted files three times. The fix was to make code edits with the editor tool.
- **Timing.** The shape-statistic negative control (100 random lists, each with a leave-one-out permutation null) dominates analysis time.

## Infrastructure learnings

- **Disk.** C: was 99% full. The venv, HF cache and activations live on D: through directory junctions (`.venv` and `artifacts` → `D:\AI\geometry-math-llms`). D: is a slow disk, so the first `import torch` and model loads are slow.
- **Windows MAX_PATH.** A uv cache inside the long scratchpad path produced broken numpy and pytest installs (`_multiarray_umath` missing). A short cache path fixed it.
- **HF auth.** Moving `HF_HOME` hides the CLI token, so `HF_TOKEN_PATH` must point at the original token file.
- **Parallel downloads.** Pre-downloading robustness models in parallel starved the MVP download. Download sequentially.
- **Model classes.** `AutoModelForCausalLM` builds a text-only class for Qwen3.5 but the full multimodal class for Gemma 4. The Gemma per-layer-embedding tables and towers are therefore offloaded to the CPU (`offload_ple`).
- **Non-ASCII in heredocs.** Python scripts piped through a Bash heredoc on Windows are decoded with the console code page, so characters such as `°`, `×` and `ρ` are mangled. Edits containing them go through the editor tool.
