# Figures and the explorable article

Every figure and every number on the website is generated from `results/<config>/*.json` and the cached activations. Nothing is drawn or typed by hand.

To regenerate the per-model figures, report tables and site data for one config:

```bash
uv run python run.py report --config configs/mvp.yaml
```

To regenerate the cross-model figures, the two generated README blocks (`TOP`: scorecard and GIF, from `reports/readme_top_template.md`; `RESULTS`, from `reports/readme_results_template.md`), the summary and the paper:

```bash
uv run python run.py paper --config configs/mvp.yaml
```

## Visual system (`src/mathgeom/report/style.py`)

- **Palette.** The reference palette of the data-viz guidance used here, checked with its validator for colour-vision deficiency (OKLab ΔE under protan/deutan/tritan simulation) and contrast against the chart surface.
  - Pure/applied (blue `#2a78d6`, orange `#eb6834`; dark mode `#3987e5`, `#d95926`) passes every check in both modes.
  - No four-colour set passes the all-pairs check needed for scatter plots. So no plot uses more than three categorical hues; the four axioms are shown as small multiples, with one axiom highlighted per panel.
- **Order.** A single-hue blue ramp, light → dark, interpolated between steps so that every item gets a distinct colour.
  - A 7-step ramp (days) passes the ordinal check.
  - A 12-step ramp (months, angles) cannot keep every adjacent lightness gap above the validator's threshold. Order is therefore never carried by colour alone: every point is labelled and joined to its neighbours by the ordered path.
  - On the site's dark theme the ramp runs from step 600 to step 100 (dark → light), because the light-theme ramp's darkest step has only 1.5:1 contrast on the dark surface. The 7-step dark ramp passes the ordinal check.
- **Roles.** Trained models are always blue and random-init twins always grey; nulls are grey bands or lines; signed similarity uses a blue ↔ red scale with a grey midpoint.
- **Output.** PNG files have an opaque background so they read correctly in GitHub's dark mode. PDFs embed TrueType fonts (`pdf.fonttype=42`) and carry no timestamps, so regenerating them doesn't change the files.

## Integrity rules for every picture (`src/mathgeom/report/figdata.py`)

- **Unsupervised projections only.** PCA or metric MDS, never a plane fitted to the labels: with 13 items in more than 1,000 dimensions, a fitted plane draws a lattice or a circle even from random vectors.
- **Display-only alignment.** Panels and layers are aligned by orthogonal Procrustes (rotation and reflection, after scaling to unit RMS). Statistics are read from the results files, never recomputed from a 2-D layout.
- **Fixed matrix order.** Distance matrices use one fixed order: super-tree branch, then field.
- **Hatched cells in the difference-vector cosine matrix.** Cells comparing two difference vectors that share a structure are hatched, because isotropic noise alone puts them near ±½.
- **Exact browser-side geometry.** The site's analogy playground uses (n − 1)-dimensional coordinates of the centred vectors. These preserve every distance and cosine exactly, which `tests/test_figdata.py` checks against the Python statistics.
- **Consistency test.** `tests/test_figdata.py` recomputes, from the figure data of every config, the reported field ρ, the offset consistencies and the cycle scores.

### 3-D views (top principal components, 3-D MDS)

- **The same unsupervised projections, with one more axis.**
  - Families, the lattice and the concept cloud use each layer's top principal components. Every axis is that layer's own PC; only its sign is aligned to the preregistered layer (and the twin's to the trained model), so the views do not mirror between layers. This is `figdata.pcs_by_layer`.
  - So the share of variance printed on each axis is exact (`figdata.pca_var`, checked against `sklearn.decomposition.PCA`).
  - Field maps use 3-D metric MDS, aligned like the 2-D maps.
- **True proportions.** Plotly scenes use `aspectmode: "data"` and matplotlib panels use `set_box_aspect` set to the data ranges, so a small PC3 looks small.
- **Every 3-D view says how much it shows:**
  - PCA views give the variance on each axis, and in total in the title.
  - MDS views give Kruskal's stress-1 with optimal scaling (`figdata.stress1`; 0 = distances kept exactly) for the 2-D and 3-D layout.
  - Rank-correlation "fidelity" was tried first and dropped. The arXiv distances are skewed, so it reported the 3-D layout as *less* faithful than the 2-D one, although 3-D cuts the stress from 0.35 to 0.25.
- **One fixed static camera** (`style.VIEW3D`: elevation 22°, azimuth −55°), never chosen per model. The README turntable shows every angle.
- **The analogy point is exact.**
  - The lattice's exported (n − 1)-D coordinates are coordinates in the PC basis.
  - `figdata.view_map` gives the 5×5 map from them to the view at the preregistered layer, so b − a + c is drawn without approximation.
  - Nearness in the picture is still not nearness in the full space.
- **No statistic comes from these views.**
  - The PC1–2 cycle score and the preregistered full-space statistics are unchanged.
  - Whether a cycle hides in another PC plane is tested separately: post-hoc 4, the closed-cycle test over the 10 planes of PC1–5, pre-specified in [`reports/prereg_addendum_3d.md`](../reports/prereg_addendum_3d.md).

## Figure catalogue

| Figure (file in `reports/figures/`) | Shows | Built by | Data |
|---|---|---|---|
| `hero` | Two-row overview: field-map ρ in all models vs twins and the arXiv / trained / twin maps; days, angles and axiom offsets | `figures/overview.py: draw_hero` | `map_fields`, `posthoc`, `shapes`, `lattice` + activations |
| `forest` | Effect sizes with 95% CIs, faceted by metric | `overview.py: draw_forest` | `map_fields`, `baselines`, `compare`, `hierarchy`, `mathlib` |
| `depth_curves` | Field-map ρ at every layer, per model, with the max-over-layers null | `overview.py: draw_depth` | `map_fields` layer curves |
| `group_objects` | Group-object parallelograms and the parent ranks of "Lie group", all models | `overview.py: draw_group_objects` | `structures` |
| `partial_operation` | Negative result: partial- vs total-operation offsets, all models and twins | `overview.py: draw_partial_operation` | `lattice` (`audit2`) |
| `field_maps_<model>` | MDS of the arXiv, trained and twin distance matrices with ground-truth neighbour edges | `figures/maps.py` | activations, `field_distance.json` |
| `field_rdms_<model>` | The same matrices as within-matrix ranks | `maps.py` | same |
| `cyclic_<model>` | Days, months and angles in the PC plane, trained and twin | `figures/families.py` | activations, `posthoc` |
| `circle_depth_<model>` (`.png`, `.gif`) | Days and months at fixed depths; the GIF animates every layer, trained (top row) and twin (bottom row) | `families.py` | activations, `posthoc` |
| `angles_<model>` | Preregistered prediction vs observed angles vs bare numbers, plus the twin as control; both halves of the decision rule | `families.py` | activations, `shapes` |
| `chains_<model>` | Ordered chains with Robinson scores and the frequency control | `families.py` | activations, `shapes`, `chain_frequency` |
| `lattice_axes_<model>` | Ideal axiom lattice and per-axiom PCA facets, trained and twin | `figures/lattice.py` | activations, `lattice` |
| `lattice_cosines_<model>` | Difference-vector cosine matrices and per-axiom offset consistency | `lattice.py` | activations, `lattice` |
| `abstraction_<model>` | Concept cloud on residualized PC1/PC2 by judge abstraction and pure/applied; judges vs Mathlib | `figures/axes.py` | activations, `axes`, `mathlib_depth.csv` |
| `families3d_<model>` | Days, months, angles in their top-3 PCs (true proportions, fixed camera), trained and twin; variance and the closed-cycle result under each panel | `families.py: draw_grid3d` | activations, `posthoc` |
| `chains3d_<model>` | The ordered chains in their top-3 PCs, trained and twin | `families.py: draw_grid3d` | activations |
| `lattice3d_<model>` | Per-axiom 3-D facets: that axiom's pairs as blue 3-D arrows, trained and twin | `lattice.py: draw_facets3d` | activations, `lattice` |
| `turntable_qwen3.5-2b.gif` | Days, months and angles in their top-3 PCs rotating once, trained (top) vs twin (bottom); README | `families.py: save_turntable` (via `overview.build`) | activations |

The Markdown scorecard (`{{table:scorecard:all}}`) and its site version come from `render.scorecard_rows`. It uses the same Holm decisions as the Holm table.

## The explorable article (`site/`)

`site/index.html`, `site/app.js` and `site/style.css` read the following files, written by `src/mathgeom/report/site.py`.

**`data/index.json`:**
- models;
- fields (labels, branches, arXiv ranks and layout, neighbour edges);
- concepts (judge rating, Mathlib match);
- Mathlib points;
- the scorecard, forest rows, and a pruned copy of the results.

**3-D keys** (all written by `site.py`; a test checks shapes, file size under 1.5 MB, and the exact analogy map):
- `fields.<side>.layout3` and `stress` (per layer);
- `families.<f>.pc5 / var / ideal5`;
- `lattice.<side>.pc5 / var / to_view`;
- `concepts.<side>.pc5 / var`;
- in `index.json`: `fields.arxiv_layout3`, `arxiv_stress`, and the per-layer `posthoc.<model>.pc_planes_cycle`.

**`data/<model>.json`:**
- per-layer field layouts and rank matrices;
- concept PCA coordinates and nearest neighbours;
- per-layer family planes;
- lattice planes, exact coordinates and the cosine matrix.

The twin's family and lattice planes are rotated onto the trained model's plane at the preregistered layer (rotation and reflection only), so the **Trained ↔ twin** button in the control bar morphs the main panels with minimal motion. The field maps of both are already aligned to the arXiv layout.

**3-D interactions** (3-D is the default; a **3D view** button per figure switches to 2-D, `?view=2d` deep-links it, and without WebGL the page falls back to 2-D):
- drag to rotate, scroll to zoom, double-click to reset;
- **Link 3-D views** makes sibling panels follow one camera;
- **⟳ Spin** turns a figure until you drag it;
- **Axes** picks 3 of PC1–5, and the variance bar chart can be clicked to put a PC on an axis;
- clicking a field or item pins it across panels, with an info line;
- the lattice draws the analogy point of the playground;
- the layer slider, play and trained ↔ twin morph glide in 3-D too: frames are interpolated on wall-clock time because Plotly's transitions do not animate 3-D traces.

**Interactions.** Model menu; layer slider with play button and a depth inset; the global trained ↔ twin morph; linked hover between field maps, distance matrices and the concept cloud; concept search and a card with 10 nearest neighbours in the model and its twin; family picker with an ideal-shape overlay; axiom chips, a lattice click that shows axiom bits, and the analogy playground; scorecard and forest filters by hypothesis; a table view under every chart; light and dark themes. Every chart with two or more series has a legend.

Prose numbers are `data-num` elements resolved against `index.json`. `tests/test_site.py` checks the data schema, that every `data-num` path resolves, that the page text contains no hand-typed decimals, and that every link from the README, summary or paper into the site points to an existing anchor.

To preview locally:

```bash
uv run python -m http.server 8791 --directory site
```
