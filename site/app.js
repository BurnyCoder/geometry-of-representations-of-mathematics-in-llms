// Explorable article: renders every figure from data/index.json + data/<model>.json (written by
// src/mathgeom/report/site.py). No number is typed into the page: prose numbers are <span data-num>
// elements resolved against index.json, chart values come from the same files.
"use strict";

const S = { I: null, M: {}, key: null, layer: 12, timer: null, edges: true, allModels: false, fam: "days", ideal: false,
            axiom: null, cloudTwin: false, colorMode: "field", field: null, picked: null, hover: null, twin: false, hyp: "all" };
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
function ramp(n) {  // ordinal ramp, piecewise-linear through the --ord stops (light and dark steps validated separately)
  const stops = css("--ord").split(",").map((h) => [1, 3, 5].map((k) => parseInt(h.trim().slice(k, k + 2), 16)));
  return Array.from({ length: n }, (_, i) => {
    const t = (n > 1 ? i / (n - 1) : 0) * (stops.length - 1), j = Math.min(stops.length - 2, Math.floor(t)), u = t - j;
    return `rgb(${stops[j].map((x, k) => Math.round(x + (stops[j + 1][k] - x) * u)).join(",")})`;
  });
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
function wrap(text, max) {  // greedy word wrap for Plotly titles (they never wrap on their own)
  return text.split(" ").reduce((ls, w) => ((ls[ls.length - 1] + " " + w).trim().length > max ? [...ls, w] : [...ls.slice(0, -1), (ls[ls.length - 1] + " " + w).trim()]), [""]);
}
function plot(id, data, lay) {  // fit the layout to its element: wrap long titles; put "below" legends a fixed pixel gap under the plot
  const el = $(id), W = el.clientWidth || 600, H = lay.height || el.clientHeight || 400, t = lay.title && lay.title.text;
  const max = Math.max(24, Math.floor((W - 24) / 6.9));  // ≈ characters per line at the 13 px title font
  if (t && t.length > max) {
    const lines = wrap(t, max);
    lay = { ...lay, title: { ...lay.title, text: lines.join("<br>") }, margin: { ...lay.margin, t: lay.margin.t + 16 * (lines.length - 1) } };
  }
  if (lay.legend && lay.legend.below !== undefined) {
    const { below, ...lg } = lay.legend;
    lay = { ...lay, legend: { ...lg, y: -below / Math.max(60, H - lay.margin.t - lay.margin.b) } };
  }
  return Plotly.react(el, data, lay, CFG).then(() => (lay.showlegend && lay.legend && lay.legend.y < 0 ? settleLegend(el) : el));
}
async function settleLegend(el, tries = 3) {
  // Auto-margins resize the plot after the legend's position is set, so measure the drawn chart and move the
  // legend down until it clears the x tick labels and axis title (exact, in the chart's final pixels).
  const box = (e) => e && e.getBoundingClientRect(), lg = box(el.querySelector(".legend"));
  const ax = [".xaxislayer-above", ".g-xtitle"].map((q) => box(el.querySelector(q))).filter((r) => r && r.height > 0);
  if (!lg || !ax.length || !tries) return el;
  const gap = lg.top - Math.max(...ax.map((r) => r.bottom));
  if (gap >= 6) return el;
  await Plotly.relayout(el, { "legend.y": el._fullLayout.legend.y - (10 - gap) / el._fullLayout._size.h });
  return settleLegend(el, tries - 1);
}
const TRANSITION = { duration: 350, easing: "cubic-in-out" };  // points glide between layers instead of jumping
const main = () => (S.twin ? "random_init" : "trained");     // what the main panels show (global trained ↔ twin morph)
function legend(below = 4) {  // horizontal legend `below` px under the plot area (plot() converts to paper units); Plotly pushes the margin to fit
  return { showlegend: true, legend: { orientation: "h", x: 0, xanchor: "left", below, yanchor: "top", font: { size: 10, color: P().ink2 }, bgcolor: "rgba(0,0,0,0)" } };
}
function padded(xy, f = 0.32) {  // axis ranges with room for direct labels at the edges
  const lo = [0, 1].map((k) => Math.min(...xy.map((d) => d[k]))), hi = [0, 1].map((k) => Math.max(...xy.map((d) => d[k])));
  const pad = [0, 1].map((k) => (hi[k] - lo[k]) * f);
  return { xaxis: { visible: false, range: [lo[0] - pad[0], hi[0] + pad[0]] }, yaxis: { visible: false, scaleanchor: "x", range: [lo[1] - pad[1] * 0.35, hi[1] + pad[1] * 0.55] } };
}
const SPOTS = ["top center", "bottom center", "middle right", "middle left", "top right", "top left", "bottom right", "bottom left"];
function placeLabels(xy, labels, el, mx = 0.04, my = 0.04, size = 10) {
  // Direct labels without overlaps: for each point, in order, take the first of 8 positions whose text box
  // (estimated from the character count) hits no earlier label, no marker and stays inside the plotted range
  // (data extent plus mx / my of it, as the panel's axis range has); else the least-bad one.
  const xs = xy.map((d) => d[0]), ys = xy.map((d) => d[1]), ext = (v) => Math.max(...v) - Math.min(...v) || 1;
  const k = Math.min((el.clientWidth || 400) / (ext(xs) * (1 + 2 * mx) * 1.15), (el.clientHeight || 400) / (ext(ys) * (1 + 2 * my) * 1.4));  // px per data unit (approx.)
  const r = 6, h = size + 2, boxes = xy.map(([x, y]) => [x * k - r, -y * k - r, x * k + r, -y * k + r]);
  const hit = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const frame = [(Math.min(...xs) - mx * ext(xs)) * k, -(Math.max(...ys) + my * ext(ys)) * k, (Math.max(...xs) + mx * ext(xs)) * k, -(Math.min(...ys) - my * ext(ys)) * k];
  return xy.map(([x, y], i) => {
    const X = x * k, Y = -y * k, w = 0.56 * size * String(labels[i]).length;
    const cand = SPOTS.map((sp) => {
      const [v, hz] = sp.split(" "), x0 = hz === "center" ? X - w / 2 : hz === "right" ? X + r : X - r - w, y0 = v === "middle" ? Y - h / 2 : v === "top" ? Y - r - h : Y + r;
      const b = [x0, y0, x0 + w, y0 + h];
      return { sp, b, cost: boxes.reduce((s, o, j) => s + (j === i ? 0 : hit(b, o)), 0) + 4 * (w * h - hit(b, frame)) };  // clipped text costs more
    });
    const best = cand.find((c) => c.cost === 0) || cand.reduce((a, c) => (c.cost < a.cost ? c : a));
    boxes.push(best.b);
    return best.sp;
  });
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
    t(fmt(R(`shapes.${k}.angles.line_rho`)), "angles follow the number line (line ρ)", `circle ρ ${fmt(R(`shapes.${k}.angles.circle_rho`))} · twin line ρ ${fmt(R(`shapes.${k}__random.angles.line_rho`))} · circle falsified`),
    t(fmt(R(`lattice.${k}.group_like.offsets.commutative.consistency`)), "“add commutativity” consistency", `twin ${fmt(R(`lattice.${k}__random.group_like.offsets.commutative.consistency`))}`),
  ].join("");
}
function renderScorecard(filter = "all") {
  const sc = S.I.scorecard, groups = { map: /field map/i, shapes: /days|chain|angles|order/i, axioms: /axiom|groups/i };
  const rows = sc.rows.filter((r) => filter === "all" || groups[filter].test(r.test));
  $("scorecard").innerHTML = `<thead><tr><th>Test</th><th>Status</th>${sc.models.map((m) => `<th class="num">${m}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr><td>${r.test}</td><td class="muted">${r.status}</td>${r.cells.map((c, j) => `<td class="cell num" title="${sc.models[j]} · ${r.test} · ${r.status}">${minus(c)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("score-legend").textContent = sc.legend;
  $("score-filter").innerHTML = ["all", "map", "shapes", "axioms"].map((g) => `<button data-g="${g}" aria-pressed="${g === filter}">${g[0].toUpperCase() + g.slice(1)}</button>`).join("");
  $("score-filter").querySelectorAll("button").forEach((b) => b.onclick = () => renderScorecard(b.dataset.g));
}
const holmText = (r) => (r.holm === null ? "not in the Holm family" : `Holm: ${r.holm ? "significant" : "not significant"}`);
const hypOf = (title) => (title.match(/H-A\d/) || ["Baselines"])[0];  // facet → hypothesis tag (filter chips)
function renderForest() {
  const p = P(), box = $("forest"), tags = [...new Set(S.I.forest.map((f) => hypOf(f.title)))];
  $("forest-filter").innerHTML = ["all", ...tags].map((g) => `<button data-h="${g}" aria-pressed="${g === S.hyp}">${g === "all" ? "All" : g}</button>`).join("");
  $("forest-filter").querySelectorAll("button").forEach((b) => b.onclick = () => { S.hyp = b.dataset.h; renderForest(); });
  const shown = S.I.forest.map((f, i) => [f, i]).filter(([f]) => S.hyp === "all" || hypOf(f.title) === S.hyp);
  box.innerHTML = shown.map(([, i]) => `<div id="forest-${i}"></div>`).join("");
  const all = [];
  shown.forEach(([f, i], n) => {
    const traces = [true, false].map((sig) => {
      const rows = f.rows.filter((r) => (r.ci[0] > 0 || r.ci[1] < 0) === sig);
      rows.forEach((r) => all.push([f.title, r.label, fmt(r.est), `[${fmt(r.ci[0])}, ${fmt(r.ci[1])}]`, r.n, r.status, holmText(r)]));
      return { type: "scatter", mode: "markers", x: rows.length ? rows.map((r) => r.est) : [null], y: rows.length ? rows.map((r) => r.label) : [null], name: sig ? "95% CI excludes 0" : "95% CI includes 0",
        error_x: { type: "data", symmetric: false, array: rows.map((r) => r.ci[1] - r.est), arrayminus: rows.map((r) => r.est - r.ci[0]), color: sig ? p.blue : p.twin, thickness: 3, width: 0 },
        marker: { size: 10, color: sig ? p.blue : p.twin, line: { color: p.surface, width: 1.5 } },
        hovertemplate: rows.map((r) => `<b>${r.label}</b><br>estimate ${fmt(r.est)}<br>95% CI [${fmt(r.ci[0])}, ${fmt(r.ci[1])}] (unadjusted)<br>n = ${r.n}<br>${r.status}${r.holm === null ? "" : "<br>" + holmText(r)}<extra></extra>`) };
    });
    const last = n === shown.length - 1, H = 70 + 30 * f.rows.length + (last ? 56 : 0), B = last ? 84 : 28;  // legend once, under the last facet (room for 2 rows on phones)
    plot(`forest-${i}`, traces, layout(f.title, { height: H, margin: { l: 150, r: 20, t: 34, b: B },
      xaxis: { range: span(f.rows), zeroline: true, zerolinecolor: p.ink2, gridcolor: p.grid, color: p.muted }, yaxis: { type: "category", autorange: "reversed", color: p.ink2, categoryorder: "array", categoryarray: f.rows.map((r) => r.label) },
      ...(last ? legend(34) : {}) }));
  });
  table($("forest-table"), ["Comparison", "Row", "Estimate", "95% CI (unadjusted)", "n", "Status", "Holm decision"], all);
}

// ---------------------------------------------------------------- figure 4–6: field map, matrices, depth
function fieldTitle(side, morphed = false) {
  const lc = R(`map_fields.${S.key}.math_primary.layer_curve`);
  const rho = side === "trained" ? lc.trained[S.layer - 1] : lc.random_init[S.layer - 1];
  return `${side === "trained" ? model().name : "random-init twin"}${morphed ? " (morphed)" : ""} · layer ${S.layer} · ρ = ${fmt(rho)}`;
}
function mapTraces(xy, highlight, keys = false, el = null) {  // keys: add legend entries (pure / applied / edges)
  const p = P(), F = S.I.fields, n = F.names.length;
  const ex = [], ey = [];
  if (S.edges) F.edges.forEach(([i, j]) => { ex.push(xy[i][0], xy[j][0], null); ey.push(xy[i][1], xy[j][1], null); });
  const hl = (i) => highlight === null || highlight === i;
  return [
    { type: "scatter", mode: "lines", x: ex, y: ey, line: { color: p.axis, width: 1 }, hoverinfo: "skip", name: "2 nearest arXiv neighbours", showlegend: keys },
    { type: "scatter", mode: "markers", x: [null], y: [null], marker: { size: 10, color: p.blue }, name: "pure field", showlegend: keys, hoverinfo: "skip" },
    { type: "scatter", mode: "markers", x: [null], y: [null], marker: { size: 10, color: p.orange }, name: "applied field", showlegend: keys, hoverinfo: "skip" },
    { type: "scatter", mode: "markers+text", showlegend: false, x: xy.map((d) => d[0]), y: xy.map((d) => d[1]), text: F.labels, textposition: el ? placeLabels(xy, F.labels, el, 0.32, 0.11) : "top center", cliponaxis: false,
      textfont: { size: 10, color: F.names.map((_, i) => (hl(i) ? p.ink2 : p.grid)) },
      marker: { size: F.names.map((_, i) => (highlight === i ? 15 : 10)), color: F.pa.map((x) => (x === "pure" ? p.blue : p.orange)),
        opacity: F.names.map((_, i) => (hl(i) ? 1 : 0.25)), line: { color: p.surface, width: 1.5 } },
      customdata: F.names.map((_, i) => i), hovertemplate: F.labels.map((l, i) => `<b>${l}</b><br>${S.I.fields.branch_labels[F.branch[i]] || F.branch[i]} · ${F.pa[i]}<extra></extra>`) },
  ];
}
function renderMap() {
  const M = S.M[S.key], F = M.fields, L = S.layer - 1, tr = { transition: TRANSITION };
  plot("map-arxiv", mapTraces(S.I.fields.arxiv_layout, S.hover, true, $("map-arxiv")), layout("arXiv cross-listing (ground truth)", { ...padded(S.I.fields.arxiv_layout), ...legend() }));
  plot("map-trained", mapTraces(F[main()].layout[L], S.hover, false, $("map-trained")), layout(fieldTitle(main(), S.twin), { ...padded(F[main()].layout[L]), ...tr }));
  plot("map-twin", mapTraces(F.random_init.layout[L], S.hover, false, $("map-twin")), layout(fieldTitle("random_init"), { ...padded(F.random_init.layout[L]), ...tr }));
  ["map-arxiv", "map-trained", "map-twin"].forEach((id) => {
    const el = $(id);
    if (el._linked) return;
    el._linked = true;
    el.on("plotly_hover", (e) => { const i = e.points[0].customdata; if (i !== undefined && S.hover !== i) hoverField(i); });
    el.on("plotly_unhover", () => hoverField(null));
  });
  const lab = S.I.fields.labels;
  table($("map-table"), ["Field", "Pure/applied", "Branch", "Nearest arXiv neighbours"], lab.map((l, i) => [l, S.I.fields.pa[i], S.I.fields.branch_labels[S.I.fields.branch[i]],
    S.I.fields.edges.filter(([a, b]) => a === i || b === i).map(([a, b]) => lab[a === i ? b : a]).join(", ")]));
}
function hoverField(i) {  // linked highlight: field maps ↔ distance matrices → concept cloud (that field's concepts)
  S.hover = i;
  if (i !== null && S.colorMode === "field") { S.field = S.I.fields.names[i]; $("cloud-field").value = S.field; }
  rerender(["fig-map", "fig-rdm", "fig-concepts"]);
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
  plot("rdm-trained", rdmTrace(M.fields[main()].rank[L], o), lay(fieldTitle(main(), S.twin), false));
  plot("rdm-twin", rdmTrace(M.fields.random_init.rank[L], o), lay(fieldTitle("random_init"), false));
  ["rdm-arxiv", "rdm-trained", "rdm-twin"].forEach((id) => {  // hovering a row highlights that field everywhere
    const el = $(id);
    if (el._linked) return;
    el._linked = true;
    el.on("plotly_hover", (e) => { const i = S.I.fields.labels.indexOf(e.points[0].y); if (i >= 0 && S.hover !== i) hoverField(i); });
    el.on("plotly_unhover", () => hoverField(null));
  });
  const lab = S.I.fields.labels, rows = [];
  o.forEach((i, a) => o.slice(a + 1).forEach((j) => rows.push([`${lab[i]} ↔ ${lab[j]}`, fmt(S.I.fields.arxiv_rank[i][j]), fmt(M.fields.trained.rank[L][i][j]), fmt(M.fields.random_init.rank[L][i][j])])));
  table($("rdm-table"), ["Field pair", "arXiv rank", `${model().name} rank (layer ${S.layer})`, "Twin rank"], rows);
}
function renderDepth() {
  const p = P(), traces = [], show = S.allModels ? S.I.models : [model()];
  let others = 0;
  show.forEach((m) => {
    const lc = R(`map_fields.${m.key}.math_primary.layer_curve`), x = lc.trained.map((_, i) => (i + 1) / lc.trained.length), me = m.key === S.key;
    if (me) {
      traces.push({ x, y: lc.random_init, mode: "lines", line: { color: p.twin, width: 2 }, name: "random-init twin", hovertemplate: "twin · depth %{x:.2f}: ρ %{y:.2f}<extra></extra>" });
      traces.push({ x: [0, 1], y: [lc.max_null95, lc.max_null95], mode: "lines", line: { color: p.muted, width: 1, dash: "dot" }, name: "null: 95th pct of max over layers", hoverinfo: "skip" });
    }
    traces.push({ x, y: lc.trained, mode: "lines", line: { color: me ? p.blue : p.axis, width: me ? 3 : 1.5 }, name: me ? m.name : "other models",
      legendgroup: me ? m.key : "others", showlegend: me || !others++, hovertemplate: `${m.name} · depth %{x:.2f}: ρ %{y:.2f}<extra></extra>` });
  });
  const L = model().n_layers;
  plot("depth", traces, layout(`${model().name}: field map vs depth${S.allModels ? " (others in grey)" : ""}`, { height: 360, margin: { l: 50, r: 16, t: 40, b: 40 }, ...legend(74),
    xaxis: { title: "relative depth", range: [0, 1], gridcolor: p.grid, color: p.muted }, yaxis: { title: "ρ with arXiv", gridcolor: p.grid, color: p.muted, zerolinecolor: p.axis },
    shapes: [{ type: "line", x0: S.layer / L, x1: S.layer / L, y0: 0, y1: 1, yref: "paper", line: { color: p.orange, width: 2 } },
             { type: "line", x0: model().primary_layer / L, x1: model().primary_layer / L, y0: 0, y1: 1, yref: "paper", line: { color: p.axis, width: 1 } }],
    annotations: [{ x: model().primary_layer / L, y: 1, yref: "paper", text: "pre-registered", showarrow: false, font: { size: 10, color: p.muted }, yanchor: "bottom" },
                  { x: S.layer / L, y: 0, yref: "paper", text: `layer ${S.layer}`, showarrow: false, font: { size: 10, color: p.orange }, yanchor: "top" }] }));
  const lc = R(`map_fields.${S.key}.math_primary.layer_curve`);
  table($("depth-table"), ["Layer", "Relative depth", "ρ trained", "ρ twin", "95th pct of max-over-layers null"],
    lc.trained.map((v, i) => [i + 1, fmt((i + 1) / lc.trained.length), fmt(v), fmt(lc.random_init[i]), fmt(lc.max_null95)]));
}

// ---------------------------------------------------------------- figure 7: concept cloud + card
function renderCloud() {
  const p = P(), C = S.I.concepts, side = S.cloudTwin ? "random_init" : "trained", xyz = S.M[S.key].concepts[side].xyz;
  const cloud = (idx, marker, name) => ({ type: "scatter3d", mode: "markers", name, x: idx.map((i) => xyz[i][0]), y: idx.map((i) => xyz[i][1]), z: idx.map((i) => xyz[i][2]), marker,
    customdata: idx, hovertemplate: idx.map((i) => `<b>${C.concept[i]}</b><br>${label(C.field[i])}<extra></extra>`) });
  const all = C.concept.map((_, i) => i);
  const traces = S.colorMode === "abs"   // one series coloured by rating, or the highlighted field against the rest (two series)
    ? [cloud(all, { size: 3.5, color: C.abstraction, colorscale: [[0, p.lo], [1, p.hi]], colorbar: { title: { text: "abstraction", side: "right" }, thickness: 10, len: 0.6, tickfont: { color: p.muted } } }, "concepts")]
    : [cloud(all.filter((i) => C.field[i] !== S.field), { size: 2.6, color: p.twin, opacity: 0.7 }, "other fields"),
       cloud(all.filter((i) => C.field[i] === S.field), { size: 5, color: p.blue, opacity: 0.95 }, label(S.field))];
  if (S.picked !== null) traces.push({ type: "scatter3d", mode: "markers+text", name: "selected concept", x: [xyz[S.picked][0]], y: [xyz[S.picked][1]], z: [xyz[S.picked][2]], text: [C.concept[S.picked]],
    marker: { size: 8, color: p.orange }, textfont: { color: p.ink }, hoverinfo: "skip" });
  const ax = { showgrid: false, zeroline: false, showticklabels: false, title: "", showbackground: false };
  plot("cloud", traces, layout(`${model().name}${S.cloudTwin ? " · random-init twin" : ""} · layer ${model().primary_layer} · PC1–3`,
    { margin: { l: 0, r: 0, t: 40, b: 0 }, ...(S.colorMode === "abs" ? {} : legend(4)), scene: { xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: "cube", camera: { eye: { x: 0.95, y: 0.95, z: 0.7 } } } }));
  const el = $("cloud");
  if (!el._linked) {
    el._linked = true;
    el.on("plotly_click", (e) => { const i = e.points[0].customdata; if (i !== undefined) pick(i); });
    el.on("plotly_hover", (e) => { const i = e.points[0].customdata; if (i === undefined) return; S.hover = S.I.fields.names.indexOf(C.field[i]); rerender(["fig-map", "fig-rdm"]); });
    el.on("plotly_unhover", () => { S.hover = null; rerender(["fig-map", "fig-rdm"]); });
  }
  if (!$("cloud-table").childElementCount) table($("cloud-table"), ["Concept", "Field", "Judge abstraction (1–7)", "Mathlib declaration (import depth)"],
    C.concept.map((c, i) => [c, label(C.field[i]), fmt(C.abstraction[i]), C.mathlib[i] ? `<code>${C.mathlib[i].decl}</code> (${C.mathlib[i].depth})` : "—"]));
}
function pick(i) {
  S.picked = i;
  const C = S.I.concepts, M = S.M[S.key].concepts, nn = (side) => M[side].nn[i].map((j) => `<li>${C.concept[j]} <span class="muted">· ${label(C.field[j])}</span></li>`).join("");
  const ml = C.mathlib[i];
  $("concept-card").innerHTML = `<h4>${C.concept[i]}</h4><div class="muted">${label(C.field[i])} · flags: none (primary set: name does not leak the field, no eponym, not an everyday word) · judge abstraction ${fmt(C.abstraction[i])} / 7${ml ? ` · Mathlib <code>${ml.decl}</code>, import depth ${ml.depth}` : " · no Mathlib match"}</div>
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
function idealFit(xy, closed) {
  // Display aid: the ideal template (equally spaced circle, or evenly spaced line) in true order, rotated,
  // reflected and scaled onto the observed points (2-D orthogonal Procrustes, closed form). It is drawn
  // the same way for the twin, so a good fit there would show what chance looks like; no statistic uses it.
  const n = xy.length, c = [0, 1].map((k) => xy.reduce((s, d) => s + d[k], 0) / n), X = xy.map((d) => [d[0] - c[0], d[1] - c[1]]);
  const T0 = xy.map((_, i) => (closed ? [Math.cos((2 * Math.PI * i) / n), Math.sin((2 * Math.PI * i) / n)] : [i - (n - 1) / 2, 0]));
  const rms = (A) => Math.sqrt(A.reduce((s, d) => s + d[0] ** 2 + d[1] ** 2, 0) / n), sx = rms(X);
  let best = null;
  for (const flip of [1, -1]) {
    const T = T0.map((d) => [d[0], flip * d[1]]), st = rms(T);
    const a = T.reduce((s, d, i) => s + d[0] * X[i][0] + d[1] * X[i][1], 0), b = T.reduce((s, d, i) => s + d[0] * X[i][1] - d[1] * X[i][0], 0);
    const th = Math.atan2(b, a), k = sx / st, Y = T.map((d) => [k * (Math.cos(th) * d[0] - Math.sin(th) * d[1]), k * (Math.sin(th) * d[0] + Math.cos(th) * d[1])]);
    const err = Y.reduce((s, d, i) => s + (d[0] - X[i][0]) ** 2 + (d[1] - X[i][1]) ** 2, 0);
    if (!best || err < best.err) best = { err, Y };
  }
  return best.Y.map((d) => [d[0] + c[0], d[1] + c[1]]);
}
function famTraces(xy, items, closed, isAngles, cycle, el) {
  const p = P(), seq = closed ? [...xy.keys(), 0] : [...xy.keys()];
  const t = [];
  if (S.ideal) {
    const Y = idealFit(xy, closed), ys = closed ? [...Y.keys(), 0] : [...Y.keys()];
    t.push({ type: "scatter", mode: "lines+markers", name: `ideal ${closed ? "circle" : "line"} (fitted)`, x: ys.map((i) => Y[i][0]), y: ys.map((i) => Y[i][1]), line: { color: p.grid, width: 6 }, marker: { size: 5, color: p.axis }, hoverinfo: "skip" });
  }
  t.push({ type: "scatter", mode: "lines", name: "path in true order", x: seq.map((i) => xy[i][0]), y: seq.map((i) => xy[i][1]), line: { color: p.axis, width: 1.5 }, hoverinfo: "skip" });
  if (isAngles) t.push({ type: "scatter", mode: "lines", name: "330° → 0° (predicted neighbours)", x: [xy[xy.length - 1][0], xy[0][0]], y: [xy[xy.length - 1][1], xy[0][1]], line: { color: p.orange, width: 2.5 }, hoverinfo: "skip" });
  const stat = cycle === undefined ? "" : `<br>cycle score at this layer ${fmt(cycle)} (post-hoc)`;
  t.push({ type: "scatter", mode: "markers+text", showlegend: false, x: xy.map((d) => d[0]), y: xy.map((d) => d[1]), text: items.map((s) => s.replace(" degrees", "°").replace("-dimensional", "-d")),
    textposition: placeLabels(xy, items.map((s) => s.replace(" degrees", "°").replace("-dimensional", "-d")), el, 0.3, 0.1), cliponaxis: false, textfont: { size: 10, color: p.ink2 }, marker: { size: 11, color: ramp(xy.length), line: { color: p.surface, width: 1.5 } },
    hovertemplate: items.map((s, i) => `<b>${s}</b><br>position ${i + 1} of ${items.length}${stat}<extra></extra>`) });
  return t;
}
function renderFamilies() {
  const p = P(), f = S.fam, F = S.M[S.key].families[f], L = S.layer - 1, closed = F.expected === "circle", ang = f === "angles";
  const name = (S.I.family_labels[f] || f).split("\n")[0], tr = { transition: TRANSITION };
  const cyc = R(`posthoc.${S.key}.pc_plane_cycle.${f}`), cycT = R(`posthoc.${S.key}__random.pc_plane_cycle.${f}`);
  const mc = S.twin ? cycT : cyc;  // the main panel follows the global trained ↔ twin morph
  plot("fam-trained", famTraces(F[main()][L], F.items, closed, ang, mc && mc[L], $("fam-trained")), layout(`${name} · ${S.twin ? "random-init twin (morphed)" : model().name} · layer ${S.layer}`, { ...tr, ...legend() }));
  plot("fam-twin", famTraces(F.random_init[L], F.items, closed, ang, cycT && cycT[L], $("fam-twin")), layout(`${name} · random-init twin`, tr));
  table($("fam-table"), ["Item", "Position", `${model().name} PC1`, "PC2", "Twin PC1", "Twin PC2"],
    F.items.map((it, i) => [it, i + 1, fmt(F.trained[L][i][0]), fmt(F.trained[L][i][1]), fmt(F.random_init[L][i][0]), fmt(F.random_init[L][i][1])]));
  if (cyc) $("fam-table").insertAdjacentHTML("beforeend", `<table><thead><tr><th>Layer</th><th>Cycle score, trained</th><th>Cycle score, twin</th></tr></thead><tbody>${
    cyc.map((v, l) => `<tr><td>${l + 1}</td><td>${fmt(v)}</td><td>${fmt(cycT[l])}</td></tr>`).join("")}</tbody></table>`);
  $("fam-takeaway").innerHTML = `<b>${FAM_TAKE[f] || ""}</b>`;
  if (cyc) {
    const tw = cycT, x = cyc.map((_, i) => (i + 1) / cyc.length);
    plot("fam-curve", [{ x, y: tw, mode: "lines", name: "random-init twin", line: { color: p.twin, width: 2 }, hovertemplate: "twin · depth %{x:.2f}: %{y:.2f}<extra></extra>" },
                       { x, y: cyc, mode: "lines", name: model().name, line: { color: p.blue, width: 3 }, hovertemplate: "trained · depth %{x:.2f}: %{y:.2f}<extra></extra>" }],
      layout("cycle score by depth (post-hoc)", { margin: { l: 44, r: 12, t: 40, b: 40 }, ...legend(74), xaxis: { title: "relative depth", range: [0, 1], gridcolor: p.grid, color: p.muted },
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
function latTraces(side, keys = false, el = null) {  // keys: legend entries for the highlighted and the other single-axiom pairs
  const p = P(), Lt = S.M[S.key].lattice, xy = Lt[side].planes[S.layer - 1], a = Lt.axioms.indexOf(S.axiom);
  const gx = [], gy = [], ann = [];
  Lt.pairs.forEach(([i, j], k) => {
    if (Lt.group[k] === a) ann.push({ x: xy[j][0], y: xy[j][1], ax: xy[i][0], ay: xy[i][1], xref: "x", yref: "y", axref: "x", ayref: "y", showarrow: true, arrowhead: 3, arrowsize: 1.1, arrowwidth: 2.2, arrowcolor: p.blue, standoff: 6, startstandoff: 6 });
    else { gx.push(xy[i][0], xy[j][0], null); gy.push(xy[i][1], xy[j][1], null); }
  });
  const has = (b) => Lt.axioms.filter((_, q) => b[q]).join(", ") || "none";
  return { data: [{ type: "scatter", mode: "lines", x: gx, y: gy, line: { color: p.grid, width: 1 }, hoverinfo: "skip", name: "other single-axiom pairs", showlegend: keys },
    { type: "scatter", mode: "markers+text", showlegend: false, x: xy.map((d) => d[0]), y: xy.map((d) => d[1]), text: Lt.items.map((s) => s.replace("commutative ", "comm. ")), textposition: placeLabels(xy, Lt.items.map((s) => s.replace("commutative ", "comm. ")), el || {}, 0.3, 0.1), cliponaxis: false,
      textfont: { size: 10, color: p.ink2 }, marker: { size: 9, color: p.ink2, line: { color: p.surface, width: 1.5 } },
      hovertemplate: Lt.items.map((s, i) => `<b>${s}</b><br>axioms: ${has(Lt.bits[i])}<extra></extra>`) },
    { type: "scatter", mode: "lines", x: [null], y: [null], line: { color: p.blue, width: 2.2 }, name: `pairs differing only in ${S.axiom} (arrows)`, showlegend: keys, hoverinfo: "skip" }], ann };
}
function renderLattice() {
  const Lt = S.M[S.key].lattice;
  if (!S.axiom) S.axiom = Lt.axioms[0];
  [["lat-trained", main()], ["lat-twin", "random_init"]].forEach(([id, side]) => {  // the main panel follows the global morph
    const t = latTraces(side, id === "lat-trained", $(id)), who = side === "trained" ? model().name : `random-init twin${id === "lat-trained" ? " (morphed)" : ""}`;
    plot(id, t.data, layout(`${who} · PCA plane at layer ${S.layer} · + ${S.axiom}`, { annotations: t.ann, transition: TRANSITION, ...(id === "lat-trained" ? legend() : {}) }));
  });
  table($("lat-table"), ["Structure", ...Lt.axioms, `+ ${S.axiom} leads to`], Lt.items.map((it, i) => [it, ...Lt.bits[i].map((b) => (b ? "✓" : "")),
    Lt.pairs.filter(([a], k) => a === i && Lt.group[k] === Lt.axioms.indexOf(S.axiom)).map(([, b]) => Lt.items[b]).join(", ") || "—"]));
  const o = R(`lattice.${S.key}.group_like.offsets.${S.axiom}`), ot = R(`lattice.${S.key}__random.group_like.offsets.${S.axiom}`);
  ["lat-trained", "lat-twin"].forEach((id) => {
    const el = $(id);
    if (el._linked) return;
    el._linked = true;
    el.on("plotly_click", (e) => {
      const i = e.points[0].pointIndex, Ltc = S.M[S.key].lattice;
      if (e.points[0].curveNumber !== 1) return;
      $("lat-bits").innerHTML = `<b>${Ltc.items[i]}</b>: ` + Ltc.axioms.map((a, q) => `${a} ${Ltc.bits[i][q] ? "✓" : "✗"}`).join(" · ");
    });
  });
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
      layout(title, { margin: { l: 88, r: 10, t: 40, b: 40 }, shapes: lines,
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
  table($("ml-table"), ["Concept", "Field", "Mathlib declaration", "Import depth", "Judge rating"], pts.map((d) => [d.concept, d.field, `<code>${d.decl}</code>`, d.depth, fmt(d.rating)]));
}

// ---------------------------------------------------------------- figure 12: group objects
function renderGroups() {
  const p = P(), st = S.I.results.structures, ms = S.I.models, y = ms.map((m) => m.name);
  const g = (k) => st[k].group_objects;
  const dots = (k, side, color, name) => ({ type: "scatter", mode: "markers", x: ms.map((m) => g(m.key + side).parallelograms.cos), y, name, marker: { size: side ? 9 : 11, color, line: { color: p.surface, width: 1.5 } },
    hovertemplate: ms.map((m) => `${m.name} · ${name}<br>cos ${fmt(g(m.key + side).parallelograms.cos)} (null 95th pct ${fmt(g(m.key + side).parallelograms.null95)}, p ${fmt(g(m.key + side).parallelograms.p, ".3f")})<extra></extra>`) });
  plot("groups-par", [{ type: "bar", orientation: "h", x: ms.map((m) => g(m.key).parallelograms.null95), y, marker: { color: p.grid }, name: "null 95th pct", hovertemplate: "null 95th percentile %{x:.2f}<extra></extra>" },
    dots("", "__random", p.twin, "twin"), dots("", "", p.blue, "trained")],
    layout("Parallelogram consistency (mean cosine) vs null", { margin: { l: 120, r: 16, t: 40, b: 40 }, barmode: "overlay", ...legend(34), xaxis: { visible: true, gridcolor: p.grid, color: p.muted, zerolinecolor: p.axis },
      yaxis: { visible: true, autorange: "reversed", color: p.ink2, scaleanchor: null } }));
  const rank = (side, parent, color, symbol, name) => ({ type: "scatter", mode: "markers", y, name, x: ms.map((m) => g(m.key + side).parent_ranks["Lie group"][parent]),
    marker: { size: 10, color, symbol, line: { color: p.surface, width: 1.5 } }, hovertemplate: `${name}: rank %{x}<extra></extra>` });
  plot("groups-rank", [rank("", 0, p.blue, "circle", "trained · “group”"), rank("", 1, p.blue, "diamond", "trained · “smooth manifold”"),
    rank("__random", 0, p.twin, "circle", "twin · “group”"), rank("__random", 1, p.twin, "diamond", "twin · “smooth manifold”")],
    layout("Where the parents of “Lie group” rank (1 = nearest; log scale)", { margin: { l: 120, r: 16, t: 40, b: 40 }, ...legend(34), xaxis: { visible: true, type: "log", gridcolor: p.grid, color: p.muted },
      yaxis: { visible: true, autorange: "reversed", color: p.ink2, scaleanchor: null } }));
  table($("groups-table"), ["Model", "Parallelogram cos", "Null 95th pct", "p", "Twin cos", "“group” rank (twin)", "“smooth manifold” rank (twin)"], ms.map((m) => {
    const a = g(m.key), t = g(m.key + "__random");
    return [m.name, fmt(a.parallelograms.cos), fmt(a.parallelograms.null95), fmt(a.parallelograms.p, ".3f"), fmt(t.parallelograms.cos),
      `${a.parent_ranks["Lie group"][0]} (${t.parent_ranks["Lie group"][0]})`, `${a.parent_ranks["Lie group"][1]} (${t.parent_ranks["Lie group"][1]})`];
  }));
}

// ---------------------------------------------------------------- figure 14: partial-operation lattice (negative result)
function renderPartial() {
  const p = P(), la = S.I.results.lattice, off = (k) => la[k].audit2.partial_operation.offsets, axes = Object.keys(off(S.key));
  const others = S.I.models.filter((m) => m.key !== S.key), me = off(S.key), clears = axes.filter((a) => me[a].consistency > me[a].null95);
  const dots = (keys, marker, name) => ({ type: "scatter", mode: "markers", name, marker,
    x: keys.flatMap((k) => axes.map((a) => off(k)[a].consistency)), y: keys.flatMap(() => axes.map((a) => `+ ${a}`)),
    hovertemplate: keys.flatMap((k) => axes.map((a) => `${(S.I.models.find((m) => m.key === k.replace("__random", "")) || {}).name}${k.endsWith("__random") ? " twin" : ""} · + ${a}<br>consistency ${fmt(off(k)[a].consistency)} (${off(k)[a].n_pairs} pairs)<extra></extra>`)) });
  plot("partial", [
    { type: "bar", orientation: "h", base: axes.map((a) => me[a].null_mean), x: axes.map((a) => me[a].null95 - me[a].null_mean), y: axes.map((a) => `+ ${a}`), width: 0.25,
      marker: { color: p.grid }, name: `${model().name} null: mean → 95th pct`, hovertemplate: axes.map((a) => `random-pair null: mean ${fmt(me[a].null_mean)}, 95th pct ${fmt(me[a].null95)}<extra></extra>`) },
    dots(S.I.models.map((m) => m.key + "__random"), { size: 7, color: p.twin }, "all twins"),
    dots(others.map((m) => m.key), { size: 9, color: p.surface, line: { color: p.blue, width: 1.5 } }, "other models"),
    dots([S.key], { size: 12, color: p.blue, line: { color: p.surface, width: 1.5 } }, model().name)],
    layout(`${model().name}: ${clears.length ? clears.map((a) => `+ ${a}`).join(", ") + " above" : "no axis clears"} its null 95th percentile (offset consistency, mean cosine)`, { height: 300, margin: { l: 96, r: 16, t: 40, b: 40 }, ...legend(34),
      xaxis: { visible: true, gridcolor: p.grid, color: p.muted, zerolinecolor: p.axis },
      yaxis: { visible: true, autorange: "reversed", color: p.ink2, scaleanchor: null } }));
  table($("partial-table"), ["Model", ...axes.flatMap((a) => [`+ ${a}`, "null 95th pct", "p"]), "Twin: " + axes.join(" / ")], S.I.models.map((m) => [m.name,
    ...axes.flatMap((a) => [fmt(off(m.key)[a].consistency), fmt(off(m.key)[a].null95), fmt(off(m.key)[a].p, ".3f")]), axes.map((a) => fmt(off(m.key + "__random")[a].consistency)).join(" / ")]));
}

// ---------------------------------------------------------------- figure decorations: paper links, table views
const PAPER = "https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms/blob/master/paper/paper.md";
function decorate() {
  document.querySelectorAll("figure.fig[data-paper]").forEach((f) => {
    f.querySelector("figcaption")?.insertAdjacentHTML("beforeend", `<span><a href="${PAPER}${f.dataset.paper}">Details in the paper, §${f.dataset.section} →</a></span>`);
  });
  document.querySelectorAll("figure.fig[data-tableid]").forEach((f) => {  // every chart gets a table-view twin
    const id = f.dataset.tableid;
    if (!$(id)) {
      let tools = f.querySelector(".fig-tools");
      if (!tools) { tools = document.createElement("div"); tools.className = "fig-tools"; f.querySelector(".fig-head").appendChild(tools); }
      tools.insertAdjacentHTML("beforeend", `<button data-table="${id}">Table view</button>`);
      f.querySelector("figcaption").insertAdjacentHTML("beforebegin", `<div class="table-view" id="${id}"></div>`);
    }
  });
  $("fig-families").querySelector(".fig-head").insertAdjacentHTML("beforeend", `<div class="fig-tools"><button id="ideal" aria-pressed="false" title="Overlay the ideal circle or line, rotated and scaled onto the points">Ideal shape</button></div>`);
}

// ---------------------------------------------------------------- controls, state, boot
function drawSpark() {  // depth inset: trained (blue) and twin (grey) ρ with arXiv by layer; dot = current layer
  const p = P(), lc = R(`map_fields.${S.key}.math_primary.layer_curve`), W = 150, H = 28, n = lc.trained.length;
  const lo = Math.min(...lc.random_init, 0), hi = Math.max(...lc.trained), X = (i) => 3 + (i * (W - 6)) / (n - 1), Y = (v) => H - 3 - ((v - lo) * (H - 6)) / (hi - lo || 1);
  const line = (ys, c, w) => `<polyline fill="none" stroke="${c}" stroke-width="${w}" points="${ys.map((v, i) => `${X(i)},${Y(v)}`).join(" ")}"/>`;
  $("spark").innerHTML = `<title>ρ with arXiv by layer (trained blue, twin grey)</title>${line(lc.random_init, p.twin, 1.2)}${line(lc.trained, p.blue, 1.8)}` +
    `<line x1="${X(model().primary_layer - 1)}" x2="${X(model().primary_layer - 1)}" y1="2" y2="${H - 2}" stroke="${p.axis}"/>` +
    `<circle cx="${X(S.layer - 1)}" cy="${Y(lc.trained[S.layer - 1])}" r="3.5" fill="${p.orange}"/>`;
}
function setLayer(l) {
  S.layer = Math.max(1, Math.min(model().n_layers, l));
  $("layer").value = S.layer;
  $("layer-readout").textContent = `layer ${S.layer} of ${model().n_layers}${S.layer === model().primary_layer ? " · pre-registered" : ""}`;
  history.replaceState(null, "", `?model=${S.key}&layer=${S.layer}${location.hash}`);
  drawSpark();
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
    drawSpark();
  };
  $("morph").onclick = (e) => {  // global trained ↔ twin: main panels glide to the twin's (aligned) layout and back
    S.twin = S.cloudTwin = !S.twin;
    e.target.setAttribute("aria-pressed", String(S.twin));
    $("cloud-twin").setAttribute("aria-pressed", String(S.cloudTwin));
    rerender(["fig-map", "fig-rdm", "fig-families", "fig-lattice", "fig-concepts"]);
  };
  $("edges-toggle").onclick = (e) => { S.edges = !S.edges; e.target.setAttribute("aria-pressed", String(S.edges)); rerender(["fig-map"]); };
  $("all-models").onclick = (e) => { S.allModels = !S.allModels; e.target.setAttribute("aria-pressed", String(S.allModels)); rerender(["fig-depth"]); };
  $("ideal").onclick = (e) => { S.ideal = !S.ideal; e.target.setAttribute("aria-pressed", String(S.ideal)); rerender(["fig-families"]); };
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
  decorate();
  wire();
  renderTiles();
  renderScorecard();
  register("fig-forest", renderForest); register("fig-map", renderMap); register("fig-rdm", renderRdm); register("fig-depth", renderDepth);
  register("fig-concepts", renderCloud); register("fig-families", renderFamilies); register("fig-lattice", renderLattice);
  register("fig-cosines", renderCosines); register("fig-analogy", renderAnalogy); register("fig-groups", renderGroups); register("fig-abstraction", renderMathlib);
  register("fig-partial", renderPartial);
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
