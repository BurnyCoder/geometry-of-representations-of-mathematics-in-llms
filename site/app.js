// Explorable article: renders every figure from data/index.json + data/<model>.json (written by
// src/mathgeom/report/site.py). No number is typed into the page: prose numbers are <span data-num>
// elements resolved against index.json, chart values come from the same files.
"use strict";

const S = { I: null, M: {}, key: null, layer: 12, timer: null, edges: true, allModels: false, fam: "days",
            axiom: null, cloudTwin: false, colorMode: "field", field: null, picked: null, hover: null };
const CFG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage"] };
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const P = () => ({ surface: css("--surface"), ink: css("--ink"), ink2: css("--ink2"), muted: css("--muted"), grid: css("--grid"),
  axis: css("--axis"), blue: css("--blue"), orange: css("--orange"), twin: css("--twin"), red: css("--red"), mid: css("--mid"),
  lo: css("--seq-lo"), hi: css("--seq-hi") });

// ---------------------------------------------------------------- data helpers
function get(obj, path) {  // dotted path; keys may contain dots (model names), so match the longest key first
  let o = obj, parts = path.split(".");
  while (parts.length) {
    let hit = false;
    for (let j = parts.length; j > 0; j--) {
      const k = parts.slice(0, j).join(".");
      if (o !== null && typeof o === "object" && k in o) { o = o[k]; parts = parts.slice(j); hit = true; break; }
    }
    if (!hit) return undefined;
  }
  return o;
}
const minus = (s) => s.replace(/^-/, "−");
function fmt(v, f) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  if (f === "d") return String(Math.round(v));
  if (f === "M") return (v / 1e6).toFixed(1) + "M";
  const m = /^\.(\d)f$/.exec(f || ".2f");
  return minus(Number(v).toFixed(m ? +m[1] : 2));
}
const R = (path) => get(S.I.results, path);
const conf = () => S.I.models[0].key;               // the pre-registered confirmatory model
const model = () => S.I.models.find((m) => m.key === S.key);
const label = (f) => S.I.fields.labels[S.I.fields.names.indexOf(f)] || f;
function fillNums(root = document) {
  root.querySelectorAll("[data-num]").forEach((el) => { el.textContent = fmt(get(S.I, el.dataset.num), el.dataset.fmt); });
}
function ramp(n) {  // ordinal blue ramp (light → dark), matching report/style.py ordinal()
  const a = [134, 182, 239], b = [13, 54, 107];
  return Array.from({ length: n }, (_, i) => { const t = n > 1 ? i / (n - 1) : 0; return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * t)).join(",")})`; });
}
function table(el, head, rows) {
  el.innerHTML = `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

// ---------------------------------------------------------------- plotly scaffolding
function layout(title, extra = {}) {
  const p = P();
  return Object.assign({
    title: { text: title, font: { size: 13, color: p.ink }, x: 0.01, xanchor: "left" },
    margin: { l: 10, r: 10, t: 40, b: 10 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", color: p.ink2, size: 11 },
    xaxis: { visible: false }, yaxis: { visible: false, scaleanchor: "x" }, showlegend: false,
    hoverlabel: { bgcolor: p.surface, bordercolor: p.axis, font: { color: p.ink } },
  }, extra);
}
const plot = (id, data, lay) => Plotly.react($(id), data, lay, CFG);
function padded(xy, f = 0.32) {  // axis ranges with room for direct labels at the edges
  const lo = [0, 1].map((k) => Math.min(...xy.map((d) => d[k]))), hi = [0, 1].map((k) => Math.max(...xy.map((d) => d[k])));
  const pad = [0, 1].map((k) => (hi[k] - lo[k]) * f);
  return { xaxis: { visible: false, range: [lo[0] - pad[0], hi[0] + pad[0]] }, yaxis: { visible: false, scaleanchor: "x", range: [lo[1] - pad[1] * 0.35, hi[1] + pad[1] * 0.55] } };
}
function span(rows) {  // forest x-range that always contains 0, so "excludes 0" can be judged by eye
  const lo = Math.min(0, ...rows.map((r) => r.ci[0])), hi = Math.max(0, ...rows.map((r) => r.ci[1])), pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

// render lazily: a figure renders the first time it scrolls into view, then on every state change
const RENDER = {};
const seen = new Set();
function register(figId, fn) { RENDER[figId] = fn; }
function rerender(ids) { (ids || Object.keys(RENDER)).forEach((id) => { if (seen.has(id)) RENDER[id](); }); }
const io = new IntersectionObserver((ents) => ents.forEach((e) => {
  if (e.isIntersecting && !seen.has(e.target.id) && RENDER[e.target.id]) { seen.add(e.target.id); RENDER[e.target.id](); }
}), { rootMargin: "300px" });

// ---------------------------------------------------------------- figure 2–3: tiles, scorecard, forest
function renderTiles() {
  const k = conf(), Lp = S.I.models[0].primary_layer, t = (v, kk, tt) => `<div class="tile"><div class="v">${v}</div><div class="k">${kk}</div><div class="t">${tt}</div></div>`;
  $("tiles").innerHTML = [
    t(fmt(R(`map_fields.${k}.math_primary.rsa.trained`)), "field map vs arXiv (ρ)", `twin ${fmt(R(`map_fields.${k}.math_primary.rsa.random_init`))}`),
    t(fmt(R(`posthoc.${k}.pc_plane_cycle.days.${Lp - 1}`)), "days: cycle score", `twin ${fmt(R(`posthoc.${k}__random.pc_plane_cycle.days.${Lp - 1}`))} · positive control`),
    t(fmt(R(`shapes.${k}.angles.closure`)), "angles: d(0°, 330°) ÷ neighbour distance", "a circle needs < 2 · falsified"),
    t(fmt(R(`lattice.${k}.group_like.offsets.commutative.consistency`)), "“add commutativity” consistency", `twin ${fmt(R(`lattice.${k}__random.group_like.offsets.commutative.consistency`))}`),
  ].join("");
}
function renderScorecard(filter = "all") {
  const sc = S.I.scorecard, groups = { map: /field map/i, shapes: /days|chain|angles|order/i, axioms: /axiom|groups/i };
  const rows = sc.rows.filter((r) => filter === "all" || groups[filter].test(r.test));
  $("scorecard").innerHTML = `<thead><tr><th>Test</th>${sc.models.map((m) => `<th class="num">${m}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr><td>${r.test}</td>${r.cells.map((c) => `<td class="cell num">${minus(c)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("score-legend").textContent = sc.legend;
  $("score-filter").innerHTML = ["all", "map", "shapes", "axioms"].map((g) => `<button data-g="${g}" aria-pressed="${g === filter}">${g[0].toUpperCase() + g.slice(1)}</button>`).join("");
  $("score-filter").querySelectorAll("button").forEach((b) => b.onclick = () => renderScorecard(b.dataset.g));
}
function renderForest() {
  const p = P(), box = $("forest");
  box.innerHTML = S.I.forest.map((_, i) => `<div id="forest-${i}"></div>`).join("");
  const all = [];
  S.I.forest.forEach((f, i) => {
    const traces = [true, false].map((sig) => {
      const rows = f.rows.filter((r) => (r.ci[0] > 0 || r.ci[1] < 0) === sig);
      rows.forEach((r) => all.push([f.title, r.label, fmt(r.est), `[${fmt(r.ci[0])}, ${fmt(r.ci[1])}]`, r.status]));
      return { type: "scatter", mode: "markers", x: rows.map((r) => r.est), y: rows.map((r) => r.label),
        error_x: { type: "data", symmetric: false, array: rows.map((r) => r.ci[1] - r.est), arrayminus: rows.map((r) => r.est - r.ci[0]), color: sig ? p.blue : p.twin, thickness: 3, width: 0 },
        marker: { size: 10, color: sig ? p.blue : p.twin, line: { color: p.surface, width: 1.5 } },
        hovertemplate: rows.map((r) => `<b>${r.label}</b><br>estimate ${fmt(r.est)}<br>95% CI [${fmt(r.ci[0])}, ${fmt(r.ci[1])}]<br>${r.status}<extra></extra>`) };
    });
    plot(`forest-${i}`, traces, layout(f.title, { height: 70 + 30 * f.rows.length, margin: { l: 150, r: 20, t: 34, b: 28 },
      xaxis: { range: span(f.rows), zeroline: true, zerolinecolor: p.ink2, gridcolor: p.grid, color: p.muted }, yaxis: { autorange: "reversed", color: p.ink2, categoryorder: "array", categoryarray: f.rows.map((r) => r.label) } }));
  });
  table($("forest-table"), ["Comparison", "Row", "Estimate", "95% CI", "Status"], all);
}

// ---------------------------------------------------------------- figure 4–6: field map, matrices, depth
function fieldTitle(side) {
  const lc = R(`map_fields.${S.key}.math_primary.layer_curve`);
  const rho = side === "trained" ? lc.trained[S.layer - 1] : lc.random_init[S.layer - 1];
  return `${side === "trained" ? model().name : "random-init twin"} · layer ${S.layer} · ρ = ${fmt(rho)}`;
}
function mapTraces(xy, highlight) {
  const p = P(), F = S.I.fields, n = F.names.length;
  const ex = [], ey = [];
  if (S.edges) F.edges.forEach(([i, j]) => { ex.push(xy[i][0], xy[j][0], null); ey.push(xy[i][1], xy[j][1], null); });
  const hl = (i) => highlight === null || highlight === i;
  return [
    { type: "scatter", mode: "lines", x: ex, y: ey, line: { color: p.axis, width: 1 }, hoverinfo: "skip" },
    { type: "scatter", mode: "markers+text", x: xy.map((d) => d[0]), y: xy.map((d) => d[1]), text: F.labels, textposition: "top center",
      textfont: { size: 10, color: F.names.map((_, i) => (hl(i) ? p.ink2 : p.grid)) },
      marker: { size: F.names.map((_, i) => (highlight === i ? 15 : 10)), color: F.pa.map((x) => (x === "pure" ? p.blue : p.orange)),
        opacity: F.names.map((_, i) => (hl(i) ? 1 : 0.25)), line: { color: p.surface, width: 1.5 } },
      customdata: F.names.map((_, i) => i), hovertemplate: F.labels.map((l, i) => `<b>${l}</b><br>${S.I.fields.branch_labels[F.branch[i]] || F.branch[i]} · ${F.pa[i]}<extra></extra>`) },
  ];
}
function renderMap() {
  const M = S.M[S.key], F = M.fields, L = S.layer - 1;
  plot("map-arxiv", mapTraces(S.I.fields.arxiv_layout, S.hover), layout("arXiv cross-listing (ground truth)", padded(S.I.fields.arxiv_layout)));
  plot("map-trained", mapTraces(F.trained.layout[L], S.hover), layout(fieldTitle("trained"), padded(F.trained.layout[L])));
  plot("map-twin", mapTraces(F.random_init.layout[L], S.hover), layout(fieldTitle("random_init"), padded(F.random_init.layout[L])));
  ["map-arxiv", "map-trained", "map-twin"].forEach((id) => {
    const el = $(id);
    if (el._linked) return;
    el._linked = true;
    el.on("plotly_hover", (e) => { const i = e.points[0].customdata; if (i !== undefined && S.hover !== i) { S.hover = i; rerender(["fig-map", "fig-rdm"]); } });
    el.on("plotly_unhover", () => { S.hover = null; rerender(["fig-map", "fig-rdm"]); });
  });
  const lab = S.I.fields.labels;
  table($("map-table"), ["Field", "Pure/applied", "Branch", "Nearest arXiv neighbours"], lab.map((l, i) => [l, S.I.fields.pa[i], S.I.fields.branch_labels[S.I.fields.branch[i]],
    S.I.fields.edges.filter(([a, b]) => a === i || b === i).map(([a, b]) => lab[a === i ? b : a]).join(", ")]));
}
function rdmTrace(M, order) {
  const p = P(), lab = S.I.fields.labels, o = order;
  const z = o.map((i) => o.map((j) => (i === j ? null : M[i][j])));
  return [{ type: "heatmap", z, x: o.map((i) => lab[i]), y: o.map((i) => lab[i]), zmin: 0, zmax: 1, colorscale: [[0, p.lo], [1, p.hi]], showscale: false,
    hovertemplate: "%{y} ↔ %{x}<br>distance rank %{z:.2f}<extra></extra>", xgap: 0.5, ygap: 0.5 }];
}
function renderRdm() {
  const M = S.M[S.key], o = S.I.fields.order, L = S.layer - 1, p = P();
  const shapes = [];
  if (S.hover !== null) {
    const k = o.indexOf(S.hover);
    shapes.push({ type: "rect", xref: "x", yref: "y", x0: -0.5, x1: o.length - 0.5, y0: k - 0.5, y1: k + 0.5, line: { color: p.orange, width: 2 } },
                { type: "rect", xref: "x", yref: "y", y0: -0.5, y1: o.length - 0.5, x0: k - 0.5, x1: k + 0.5, line: { color: p.orange, width: 2 } });
  }
  const lay = (t, ticks) => layout(t, { margin: { l: ticks ? 150 : 10, r: 10, t: 40, b: 10 }, shapes,
    xaxis: { visible: false }, yaxis: { visible: ticks, autorange: "reversed", tickfont: { size: 9 }, color: p.ink2 } });
  plot("rdm-arxiv", rdmTrace(S.I.fields.arxiv_rank, o), lay("arXiv distances", true));
  plot("rdm-trained", rdmTrace(M.fields.trained.rank[L], o), lay(fieldTitle("trained"), false));
  plot("rdm-twin", rdmTrace(M.fields.random_init.rank[L], o), lay(fieldTitle("random_init"), false));
}
function renderDepth() {
  const p = P(), traces = [], show = S.allModels ? S.I.models : [model()];
  show.forEach((m) => {
    const lc = R(`map_fields.${m.key}.math_primary.layer_curve`), x = lc.trained.map((_, i) => (i + 1) / lc.trained.length), me = m.key === S.key;
    if (me) {
      traces.push({ x, y: lc.random_init, mode: "lines", line: { color: p.twin, width: 2 }, name: "twin", hovertemplate: "twin · depth %{x:.2f}: ρ %{y:.2f}<extra></extra>" });
      traces.push({ x: [0, 1], y: [lc.max_null95, lc.max_null95], mode: "lines", line: { color: p.muted, width: 1 }, hoverinfo: "skip" });
    }
    traces.push({ x, y: lc.trained, mode: "lines", line: { color: me ? p.blue : p.axis, width: me ? 3 : 1.5 }, name: m.name,
      hovertemplate: `${m.name} · depth %{x:.2f}: ρ %{y:.2f}<extra></extra>` });
  });
  const L = model().n_layers;
  plot("depth", traces, layout(`${model().name}: field map vs depth${S.allModels ? " (others in grey)" : ""}`, { height: 330, margin: { l: 50, r: 16, t: 40, b: 40 },
    xaxis: { title: "relative depth", range: [0, 1], gridcolor: p.grid, color: p.muted }, yaxis: { title: "ρ with arXiv", gridcolor: p.grid, color: p.muted, zerolinecolor: p.axis },
    shapes: [{ type: "line", x0: S.layer / L, x1: S.layer / L, y0: 0, y1: 1, yref: "paper", line: { color: p.orange, width: 2 } },
             { type: "line", x0: model().primary_layer / L, x1: model().primary_layer / L, y0: 0, y1: 1, yref: "paper", line: { color: p.axis, width: 1 } }],
    annotations: [{ x: model().primary_layer / L, y: 1, yref: "paper", text: "pre-registered", showarrow: false, font: { size: 10, color: p.muted }, yanchor: "bottom" },
                  { x: S.layer / L, y: 0, yref: "paper", text: `layer ${S.layer}`, showarrow: false, font: { size: 10, color: p.orange }, yanchor: "top" }] }));
}

// ---------------------------------------------------------------- figure 7: concept cloud + card
function renderCloud() {
  const p = P(), C = S.I.concepts, side = S.cloudTwin ? "random_init" : "trained", xyz = S.M[S.key].concepts[side].xyz;
  const on = C.field.map((f) => f === S.field);
  const marker = S.colorMode === "abs"
    ? { size: 3.5, color: C.abstraction, colorscale: [[0, p.lo], [1, p.hi]], colorbar: { title: { text: "abstraction", side: "right" }, thickness: 10, len: 0.6, tickfont: { color: p.muted } } }
    : { size: on.map((b) => (b ? 5 : 2.6)), color: on.map((b) => (b ? p.blue : p.twin)), opacity: 0.85 };
  const traces = [{ type: "scatter3d", mode: "markers", x: xyz.map((d) => d[0]), y: xyz.map((d) => d[1]), z: xyz.map((d) => d[2]), marker,
    text: C.concept, customdata: C.concept.map((_, i) => i), hovertemplate: C.concept.map((c, i) => `<b>${c}</b><br>${label(C.field[i])}<extra></extra>`) }];
  if (S.picked !== null) traces.push({ type: "scatter3d", mode: "markers+text", x: [xyz[S.picked][0]], y: [xyz[S.picked][1]], z: [xyz[S.picked][2]], text: [C.concept[S.picked]],
    marker: { size: 8, color: p.orange }, textfont: { color: p.ink }, hoverinfo: "skip" });
  const ax = { showgrid: false, zeroline: false, showticklabels: false, title: "", showbackground: false };
  plot("cloud", traces, layout(`${model().name}${S.cloudTwin ? " · random-init twin" : ""} · layer ${model().primary_layer} · PC1–3`,
    { margin: { l: 0, r: 0, t: 40, b: 0 }, scene: { xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: "cube", camera: { eye: { x: 0.95, y: 0.95, z: 0.7 } } } }));
  const el = $("cloud");
  if (!el._linked) { el._linked = true; el.on("plotly_click", (e) => { const i = e.points[0].customdata; if (i !== undefined) pick(i); }); }
}
function pick(i) {
  S.picked = i;
  const C = S.I.concepts, M = S.M[S.key].concepts, nn = (side) => M[side].nn[i].map((j) => `<li>${C.concept[j]} <span class="muted">· ${label(C.field[j])}</span></li>`).join("");
  const ml = C.mathlib[i];
  $("concept-card").innerHTML = `<h4>${C.concept[i]}</h4><div class="muted">${label(C.field[i])} · judge abstraction ${fmt(C.abstraction[i])} / 7${ml ? ` · Mathlib <code>${ml.decl}</code>, import depth ${ml.depth}` : ""}</div>
    <div class="nn"><div><b>Nearest in ${model().name}</b><ol>${nn("trained")}</ol></div><div><b>Nearest in the twin</b><ol>${nn("random_init")}</ol></div></div>`;
  rerender(["fig-concepts"]);
}

// ---------------------------------------------------------------- figure 8: families
const FAM_TAKE = {
  days: "Positive control: the days close into a cycle at almost every depth; the twin never does.",
  months: "Positive control: the months form a cycle at most depths; the twin never does.",
  angles: "Falsified prediction: angles follow the number line, so 330° lands far from 0° (orange edge).",
  ring_chain: "Ordered: the chain of ring definitions runs as a path; the twin is a tangle.",
  space_chain: "Ordered: the separation-axiom chain runs as a path; the twin is a tangle.",
  number_systems: "Exploratory (7 items): ordered but frequency-confounded.",
  ngons: "Follows the number words three … twelve.", dimensions: "Follows the number words zero … ten.",
};
function famTraces(xy, items, closed, isAngles) {
  const p = P(), seq = closed ? [...xy.keys(), 0] : [...xy.keys()];
  const t = [{ type: "scatter", mode: "lines", x: seq.map((i) => xy[i][0]), y: seq.map((i) => xy[i][1]), line: { color: p.axis, width: 1.5 }, hoverinfo: "skip" }];
  if (isAngles) t.push({ type: "scatter", mode: "lines", x: [xy[xy.length - 1][0], xy[0][0]], y: [xy[xy.length - 1][1], xy[0][1]], line: { color: p.orange, width: 2.5 }, hoverinfo: "skip" });
  t.push({ type: "scatter", mode: "markers+text", x: xy.map((d) => d[0]), y: xy.map((d) => d[1]), text: items.map((s) => s.replace(" degrees", "°").replace("-dimensional", "-d")),
    textposition: "top center", textfont: { size: 10, color: p.ink2 }, marker: { size: 11, color: ramp(xy.length), line: { color: p.surface, width: 1.5 } },
    hovertemplate: items.map((s, i) => `<b>${s}</b><br>position ${i + 1} of ${items.length}<extra></extra>`) });
  return t;
}
function renderFamilies() {
  const p = P(), f = S.fam, F = S.M[S.key].families[f], L = S.layer - 1, closed = F.expected === "circle", ang = f === "angles";
  const name = (S.I.family_labels[f] || f).split("\n")[0];
  plot("fam-trained", famTraces(F.trained[L], F.items, closed, ang), layout(`${name} · ${model().name} · layer ${S.layer}`));
  plot("fam-twin", famTraces(F.random_init[L], F.items, closed, ang), layout(`${name} · random-init twin`));
  $("fam-takeaway").innerHTML = `<b>${FAM_TAKE[f] || ""}</b>`;
  const cyc = R(`posthoc.${S.key}.pc_plane_cycle.${f}`);
  if (cyc) {
    const tw = R(`posthoc.${S.key}__random.pc_plane_cycle.${f}`), x = cyc.map((_, i) => (i + 1) / cyc.length);
    plot("fam-curve", [{ x, y: tw, mode: "lines", line: { color: p.twin, width: 2 }, hovertemplate: "twin · depth %{x:.2f}: %{y:.2f}<extra></extra>" },
                       { x, y: cyc, mode: "lines", line: { color: p.blue, width: 3 }, hovertemplate: "trained · depth %{x:.2f}: %{y:.2f}<extra></extra>" }],
      layout("cycle score by depth (post-hoc)", { margin: { l: 44, r: 12, t: 40, b: 40 }, xaxis: { title: "relative depth", range: [0, 1], gridcolor: p.grid, color: p.muted },
        yaxis: { range: [0, 1.08], gridcolor: p.grid, color: p.muted },
        shapes: [{ type: "line", x0: S.layer / cyc.length, x1: S.layer / cyc.length, y0: 0, y1: 1, yref: "paper", line: { color: p.orange, width: 2 } }] }));
  } else {
    const sh = R(`shapes.${S.key}.${f}`), tw = R(`shapes.${S.key}__random.${f}`), cf = R(`chain_frequency.${S.key}.${f}`);
    plot("fam-curve", [{ type: "bar", x: ["trained", "twin"], y: [sh.robinson, tw.robinson], marker: { color: [p.blue, p.twin] }, width: 0.5,
      hovertemplate: ["trained", "twin"].map((s, i) => `${s}: Robinson ${fmt([sh, tw][i].robinson)}, p ${fmt([sh, tw][i].robinson_p, ".3f")}<extra></extra>`) }],
      layout(`Robinson score (layer ${model().primary_layer})${cf ? ` · beyond frequency p = ${fmt(cf.p, ".3f")}` : ""}`, { margin: { l: 44, r: 12, t: 40, b: 40 },
        xaxis: { visible: true, color: p.ink2 }, yaxis: { visible: true, range: [0, 1], gridcolor: p.grid, color: p.muted, scaleanchor: null },
        shapes: [{ type: "line", x0: -0.5, x1: 1.5, y0: 1 / 3, y1: 1 / 3, line: { color: p.muted, width: 1 } }],
        annotations: [{ x: 1.5, y: 1 / 3, text: "chance", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10, color: p.muted } }] }));
  }
  $("fam-chips").innerHTML = Object.keys(S.M[S.key].families).map((k) => `<button data-f="${k}" aria-pressed="${k === f}">${(S.I.family_labels[k] || k).split("\n")[0]}</button>`).join("");
  $("fam-chips").querySelectorAll("button").forEach((b) => b.onclick = () => { S.fam = b.dataset.f; renderFamilies(); });
}

// ---------------------------------------------------------------- figures 9–11: axiom lattice
function latTraces(side) {
  const p = P(), Lt = S.M[S.key].lattice, xy = Lt[side].planes[S.layer - 1], a = Lt.axioms.indexOf(S.axiom);
  const gx = [], gy = [], ann = [];
  Lt.pairs.forEach(([i, j], k) => {
    if (Lt.group[k] === a) ann.push({ x: xy[j][0], y: xy[j][1], ax: xy[i][0], ay: xy[i][1], xref: "x", yref: "y", axref: "x", ayref: "y", showarrow: true, arrowhead: 3, arrowsize: 1.1, arrowwidth: 2.2, arrowcolor: p.blue, standoff: 6, startstandoff: 6 });
    else { gx.push(xy[i][0], xy[j][0], null); gy.push(xy[i][1], xy[j][1], null); }
  });
  const has = (b) => Lt.axioms.filter((_, q) => b[q]).join(", ") || "none";
  return { data: [{ type: "scatter", mode: "lines", x: gx, y: gy, line: { color: p.grid, width: 1 }, hoverinfo: "skip" },
    { type: "scatter", mode: "markers+text", x: xy.map((d) => d[0]), y: xy.map((d) => d[1]), text: Lt.items.map((s) => s.replace("commutative ", "comm. ")), textposition: "top center",
      textfont: { size: 10, color: p.ink2 }, marker: { size: 9, color: p.ink2, line: { color: p.surface, width: 1.5 } },
      hovertemplate: Lt.items.map((s, i) => `<b>${s}</b><br>axioms: ${has(Lt.bits[i])}<extra></extra>`) }], ann };
}
function renderLattice() {
  const Lt = S.M[S.key].lattice;
  if (!S.axiom) S.axiom = Lt.axioms[0];
  ["trained", "random_init"].forEach((side) => {
    const t = latTraces(side);
    plot(side === "trained" ? "lat-trained" : "lat-twin", t.data, layout(`${side === "trained" ? model().name : "random-init twin"} · PCA plane at layer ${S.layer} · + ${S.axiom}`, { annotations: t.ann }));
  });
  const o = R(`lattice.${S.key}.group_like.offsets.${S.axiom}`), ot = R(`lattice.${S.key}__random.group_like.offsets.${S.axiom}`);
  $("lat-note").innerHTML = `<b>+ ${S.axiom}:</b> offset consistency ${fmt(o.consistency)} (random-pair null 95th percentile ${fmt(o.null95)}, p ${fmt(o.p, ".3f")}); twin ${fmt(ot.consistency)}. Values at the preregistered layer.`;
  $("axiom-chips").innerHTML = Lt.axioms.map((a) => `<button data-a="${a}" aria-pressed="${a === S.axiom}">+ ${a}</button>`).join("");
  $("axiom-chips").querySelectorAll("button").forEach((b) => b.onclick = () => { S.axiom = b.dataset.a; renderLattice(); });
}
function renderCosines() {
  const p = P(), Lt = S.M[S.key].lattice, n = Lt.pairs.length, nm = (k) => `${Lt.items[Lt.pairs[k][0]]} → ${Lt.items[Lt.pairs[k][1]]}`;
  const starts = Lt.group.map((g, i) => (i === 0 || g !== Lt.group[i - 1] ? i : null)).filter((x) => x !== null);
  const centers = starts.map((s, i) => (s + (starts[i + 1] ?? n) - 1) / 2);
  const make = (side, title) => {
    const C = Lt[side].cos, z = C.map((r, i) => r.map((v, j) => (i === j ? null : v)));
    const sx = [], sy = [];
    Lt.shared.forEach((r, i) => r.forEach((s, j) => { if (s) { sx.push(j); sy.push(i); } }));
    const lines = starts.slice(1).flatMap((s) => [{ type: "line", x0: s - 0.5, x1: s - 0.5, y0: -0.5, y1: n - 0.5, line: { color: p.ink2, width: 1 } },
                                                 { type: "line", y0: s - 0.5, y1: s - 0.5, x0: -0.5, x1: n - 0.5, line: { color: p.ink2, width: 1 } }]);
    plot(side === "trained" ? "cos-trained" : "cos-twin", [
      { type: "heatmap", z, zmin: -1, zmax: 1, colorscale: [[0, p.blue], [0.5, p.mid], [1, p.red]], showscale: side === "trained", colorbar: { thickness: 10, len: 0.6, tickfont: { color: p.muted } },
        text: z.map((r, i) => r.map((_, j) => `${nm(i)}<br>vs ${nm(j)}`)), hovertemplate: "%{text}<br>cos %{z:.2f}<extra></extra>" },
      { type: "scatter", mode: "markers", x: sx, y: sy, marker: { symbol: "x-thin", size: 5, line: { width: 1, color: p.surface } }, hoverinfo: "skip" }],
      layout(title, { margin: { l: 70, r: 10, t: 40, b: 40 }, shapes: lines,
        xaxis: { visible: true, tickvals: centers, ticktext: Lt.axioms, color: p.ink2, showgrid: false, zeroline: false },
        yaxis: { visible: true, tickvals: centers, ticktext: Lt.axioms, autorange: "reversed", color: p.ink2, showgrid: false, zeroline: false, scaleanchor: "x" } }));
  };
  make("trained", `${model().name} · layer ${model().primary_layer}`);
  make("random_init", "random-init twin");
  table($("cos-table"), ["Axiom", "Pairs", "Consistency (trained)", "Null 95th pct", "p", "Consistency (twin)"], Lt.axioms.map((a) => {
    const o = R(`lattice.${S.key}.group_like.offsets.${a}`), t = R(`lattice.${S.key}__random.group_like.offsets.${a}`);
    return [a, o.n_pairs, fmt(o.consistency), fmt(o.null95), fmt(o.p, ".3f"), fmt(t.consistency)];
  }));
}
function analogy(E, a, b, c) {  // rank of every candidate for b − a + c by cosine (as analyses/lattice analogy_rank)
  const q = E[b].map((v, k) => v - E[a][k] + E[c][k]), nq = Math.hypot(...q);
  return E.map((e, i) => ({ i, cos: e.reduce((s, v, k) => s + v * q[k], 0) / (Math.hypot(...e) * nq) }))
    .filter((r) => ![a, b, c].includes(r.i)).sort((x, y) => y.cos - x.cos);
}
function renderAnalogy() {
  const Lt = S.M[S.key].lattice, sel = ["an-a", "an-b", "an-c"].map((id) => $(id));
  if (!sel[0].options.length) {
    sel.forEach((s, k) => { s.innerHTML = Lt.items.map((it, i) => `<option value="${i}">${it}</option>`).join(""); s.value = Lt.items.indexOf(["semigroup", "monoid", "quasigroup"][k]); s.onchange = renderAnalogy; });
  }
  const [a, b, c] = sel.map((s) => +s.value);
  const want = Lt.bits[b].map((v, k) => v - Lt.bits[a][k] + Lt.bits[c][k]);
  const target = Lt.bits.findIndex((bt) => bt.every((v, k) => v === want[k]));
  ["trained", "random_init"].forEach((side) => {
    const rows = analogy(Lt[side].exact, a, b, c).slice(0, 8);
    const rank = analogy(Lt[side].exact, a, b, c).findIndex((r) => r.i === target) + 1;
    $(side === "trained" ? "an-trained" : "an-twin").innerHTML = `<b>${side === "trained" ? model().name : "random-init twin"}</b>` +
      (target >= 0 ? ` <span class="muted">· predicted answer “${Lt.items[target]}” ranks ${rank}</span>` : ` <span class="muted">· no structure has the predicted axioms</span>`) +
      rows.map((r) => `<div class="rank-row${r.i === target ? " target" : ""}"><span>${Lt.items[r.i]}</span><div class="rank-bar ${side === "trained" ? "trained" : ""}" style="width:${Math.max(2, 100 * (r.cos + 1) / 2)}%"></div><span class="num">${fmt(r.cos)}</span></div>`).join("");
  });
}

// ---------------------------------------------------------------- figure 12: Mathlib
function renderMathlib() {
  const p = P(), pts = S.I.mathlib_points, ml = R("mathlib");
  plot("mathlib", [{ type: "scatter", mode: "markers", x: pts.map((d) => d.depth), y: pts.map((d) => d.rating), marker: { size: 10, color: p.blue, line: { color: p.surface, width: 1.5 } },
    hovertemplate: pts.map((d) => `<b>${d.concept}</b> · ${d.field}<br>Mathlib <code>${d.decl}</code><br>import depth ${d.depth} · judge ${fmt(d.rating)}<extra></extra>`) }],
    layout(`Judges vs Mathlib: ρ = ${fmt(ml.rho_mean_rating_depth)}, 95% CI [${fmt(ml.ci95[0])}, ${fmt(ml.ci95[1])}], n = ${ml.n_matched}`,
      { height: 380, margin: { l: 54, r: 16, t: 40, b: 46 }, xaxis: { title: "Mathlib import depth", gridcolor: p.grid, color: p.muted },
        yaxis: { title: "mean judge abstraction (1–7)", gridcolor: p.grid, color: p.muted, scaleanchor: null } }));
}

// ---------------------------------------------------------------- controls, state, boot
function setLayer(l) {
  S.layer = Math.max(1, Math.min(model().n_layers, l));
  $("layer").value = S.layer;
  $("layer-readout").textContent = `layer ${S.layer} of ${model().n_layers}${S.layer === model().primary_layer ? " · pre-registered" : ""}`;
  history.replaceState(null, "", `?model=${S.key}&layer=${S.layer}${location.hash}`);
  rerender(["fig-map", "fig-rdm", "fig-depth", "fig-families", "fig-lattice"]);
}
async function setModel(key) {
  if (!S.M[key]) S.M[key] = await (await fetch(`data/${key}.json`)).json();
  S.key = key;
  $("model").value = key;
  $("layer").max = model().n_layers;
  ["an-a", "an-b", "an-c"].forEach((id) => { $(id).innerHTML = ""; });
  S.picked = null;
  setLayer(Math.min(S.layer, model().n_layers));
  rerender();
}
function wire() {
  $("model").innerHTML = S.I.models.map((m) => `<option value="${m.key}">${m.name}${m.key === conf() ? " ★" : ""}</option>`).join("");
  $("model").onchange = (e) => setModel(e.target.value);
  $("layer").oninput = (e) => setLayer(+e.target.value);
  $("reset-layer").onclick = () => setLayer(model().primary_layer);
  $("play").onclick = (e) => {
    if (S.timer) { clearInterval(S.timer); S.timer = null; e.target.setAttribute("aria-pressed", "false"); e.target.textContent = "▶ Play"; return; }
    e.target.setAttribute("aria-pressed", "true"); e.target.textContent = "❚❚ Pause";
    S.timer = setInterval(() => setLayer(S.layer >= model().n_layers ? 1 : S.layer + 1), 750);
  };
  $("theme").onclick = () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (_) { /* storage may be blocked */ }
    rerender();
  };
  $("edges-toggle").onclick = (e) => { S.edges = !S.edges; e.target.setAttribute("aria-pressed", String(S.edges)); rerender(["fig-map"]); };
  $("all-models").onclick = (e) => { S.allModels = !S.allModels; e.target.setAttribute("aria-pressed", String(S.allModels)); rerender(["fig-depth"]); };
  document.querySelectorAll("[data-table]").forEach((b) => b.onclick = () => $(b.dataset.table).classList.toggle("open"));
  const C = S.I.concepts, fields = [...new Set(C.field)];
  $("cloud-field").innerHTML = fields.map((f) => `<option value="${f}">${label(f)}</option>`).join("");
  S.field = fields[0];
  $("cloud-field").onchange = (e) => { S.field = e.target.value; rerender(["fig-concepts"]); };
  $("cloud-color").onchange = (e) => { S.colorMode = e.target.value; $("cloud-field").disabled = S.colorMode !== "field"; rerender(["fig-concepts"]); };
  $("cloud-twin").onclick = (e) => { S.cloudTwin = !S.cloudTwin; e.target.setAttribute("aria-pressed", String(S.cloudTwin)); rerender(["fig-concepts"]); };
  $("concept-list").innerHTML = C.concept.map((c) => `<option value="${c}">`).join("");
  $("search").onchange = (e) => {  // exact name first, then the first concept containing the query (case-insensitive)
    const q = e.target.value.trim().toLowerCase(), lc = C.concept.map((c) => c.toLowerCase());
    const i = lc.indexOf(q) >= 0 ? lc.indexOf(q) : lc.findIndex((c) => c.includes(q));
    if (q && i >= 0) pick(i); else if (q) $("concept-card").innerHTML = `<span class="muted">No primary concept matches “${e.target.value}”.</span>`;
  };
}
async function boot() {
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (_) { /* ignore */ }
  S.I = await (await fetch("data/index.json")).json();
  fillNums();
  wire();
  renderTiles();
  renderScorecard();
  register("fig-forest", renderForest); register("fig-map", renderMap); register("fig-rdm", renderRdm); register("fig-depth", renderDepth);
  register("fig-concepts", renderCloud); register("fig-families", renderFamilies); register("fig-lattice", renderLattice);
  register("fig-cosines", renderCosines); register("fig-analogy", renderAnalogy); register("fig-abstraction", renderMathlib);
  const q = new URLSearchParams(location.search);
  const key = S.I.models.some((m) => m.key === q.get("model")) ? q.get("model") : conf();
  S.layer = +q.get("layer") || S.I.models.find((m) => m.key === key).primary_layer;
  if (["light", "dark"].includes(q.get("theme"))) document.documentElement.dataset.theme = q.get("theme");
  await setModel(key);
  Object.keys(RENDER).forEach((id) => io.observe($(id)));
  // fallback: render anything not yet seen, one figure at a time, so figures never depend on the
  // observer alone (background tabs, printing, browsers that throttle intersection callbacks)
  setTimeout(() => Object.keys(RENDER).forEach((id, i) => setTimeout(() => { if (!seen.has(id)) { seen.add(id); RENDER[id](); } }, 300 * i)), 2500);
}
boot();
