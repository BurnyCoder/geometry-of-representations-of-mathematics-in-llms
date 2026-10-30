// Explorable article: renders every figure from data/index.json + data/<model>.json (written by
// src/mathgeom/report/site.py). No statistic is typed into the page (only structural counts such as "13 structures"): prose numbers are <span data-num>
// elements resolved against index.json, chart values come from the same files.
"use strict";

const S = { I: null, M: {}, key: null, layer: 12, timer: null, edges: true, allModels: false, fam: "days", ideal: false,
            axiom: null, cloudTwin: false, colorMode: "field", field: null, picked: null, hover: null, twin: false, hyp: "all",
            view: { map: "3d", fam: "3d", lat: "3d" }, axes: { fam: [0, 1, 2], lat: [0, 1, 2], cloud: [0, 1, 2], how: [0, 1, 2] }, cam: {}, spin: {}, link: true, camSync: false, pin: { map: null, fam: null, lat: null },
            how: 1, howTimer: null };  // how: the step (1–4) of the axiom-test explainer
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
const pval = (p) => (p < 0.001 ? "< 0.001" : fmt(p, ".3f"));  // permutation p is never 0, so do not print "0.000"
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
// ---------------------------------------------------------------- 3-D views (top principal components / 3-D MDS)
// Display only: the same unsupervised projections as the 2-D views, with one more axis. Scenes keep true
// proportions (aspectmode "data", so a small PC3 looks small) and every axis states its share of variance.
const HAS_GL = (() => { try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); } catch (_) { return false; } })();
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const CAM0 = { eye: { x: 1.35, y: -1.35, z: 0.85 }, up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: 0 } };
const COMBOS = [];  // the 10 choices of 3 axes out of PC1–5
for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) for (let c = b + 1; c < 5; c++) COMBOS.push([a, b, c]);
const is3d = (fig) => HAS_GL && S.view[fig] === "3d";
const pick3 = (rows, ax) => rows.map((r) => ax.map((k) => r[k]));  // choose the displayed PCs from 5-column rows
const cols = (rows) => [0, 1, 2].map((k) => rows.map((r) => (r === null ? null : r[k])));
function scene3d(fig, titles) {  // axis titles carry the variance; the camera is the figure's shared camera
  const p = P(), ax = (t) => ({ title: { text: t, font: { size: 10, color: p.muted } }, showticklabels: false, showgrid: true, gridcolor: p.grid,
    zeroline: false, showbackground: false, showspikes: false, color: p.muted });
  return { xaxis: ax(titles[0]), yaxis: ax(titles[1]), zaxis: ax(titles[2]), aspectmode: "data", camera: S.cam[fig] || CAM0 };
}
const pcTitles = (ax, v) => ax.map((k) => `PC${k + 1} · ${Math.round(100 * v[k])}%`);
const held = (ax, v) => Math.round(100 * ax.reduce((s, k) => s + v[k], 0));
function linkCameras(fig, ids) {  // rotate / zoom one panel and its siblings follow (and Spin stops)
  ids.forEach((id) => {
    const el = $(id);
    if (el._camLinked || !el.on) return;  // not plotted yet (e.g. a panel hidden in this step): linked on its first render
    el._camLinked = true;
    const follow = (ev) => {
      const cam = ev["scene.camera"];
      if (!cam || S.camSync) return;
      if (S.spin[fig]) toggleSpin(fig);
      S.cam[fig] = cam;
      if (!S.link) return;
      S.camSync = true;
      Promise.all(ids.filter((o) => o !== id && $(o)._fullLayout && $(o)._fullLayout.scene).map((o) => Plotly.relayout($(o), { "scene.camera": cam })))
        .finally(() => { S.camSync = false; });
    };
    el.on("plotly_relayout", follow);
    el.on("plotly_relayouting", follow);
  });
}
function toggleSpin(fig) {  // turntable: rotate the shared camera about the vertical axis in every linked panel
  const btn = document.querySelector(`[data-spin="${fig}"]`);
  if (S.spin[fig]) { cancelAnimationFrame(S.spin[fig]); S.spin[fig] = null; if (btn) btn.setAttribute("aria-pressed", "false"); return; }
  if (btn) btn.setAttribute("aria-pressed", "true");
  let t0 = performance.now();
  const step = (t) => {
    const c = S.cam[fig] || CAM0, e = c.eye, r = Math.hypot(e.x, e.y), a = Math.atan2(e.y, e.x) + 0.0006 * (t - t0);
    t0 = t;
    S.cam[fig] = { ...c, eye: { x: r * Math.cos(a), y: r * Math.sin(a), z: e.z } };
    S.camSync = true;
    Promise.all(SPIN_IDS[fig].filter((id) => $(id)._fullLayout && $(id)._fullLayout.scene).map((id) => Plotly.relayout($(id), { "scene.camera": S.cam[fig] })))
      .finally(() => { S.camSync = false; if (S.spin[fig]) S.spin[fig] = requestAnimationFrame(step); });
  };
  S.spin[fig] = requestAnimationFrame(step);
}
const SPIN_IDS = { map: ["map-arxiv", "map-trained", "map-twin"], fam: ["fam-trained", "fam-twin"], lat: ["lat-trained", "lat-twin"], cloud: ["cloud"] };
function plot3d(id, data, lay) {
  // Plotly's layout transitions do not animate 3-D traces ("only scatter traces may be smoothly transitioned",
  // https://plotly.com/javascript/animations/), so glide here: interpolate every trace's x/y/z
  // from the previous render to this one over a few frames (layer slider, play, model switch, trained ↔ twin)
  const el = $(id), prev = el._xyz, next = data.map((t) => ({ x: t.x, y: t.y, z: t.z }));
  el._xyz = next;
  const tok = (el._tok = (el._tok || 0) + 1);
  const same = prev && prev.length === next.length && prev.every((q, i) => !q.x || (next[i].x && q.x.length === next[i].x.length));
  const moved = same && prev.some((q, i) => q.x && q.x.some((v, k) => v !== next[i].x[k] || q.y[k] !== next[i].y[k] || q.z[k] !== next[i].z[k]));
  if (!moved || REDUCED || document.hidden) return plot(id, data, lay);  // nothing moved (e.g. a hover highlight) or nobody watching: draw at once
  const DUR = 320, t0 = performance.now(), lerp = (a, b, u) => b.map((v, k) => (v === null || a[k] === null || a[k] === undefined ? v : a[k] + (v - a[k]) * u));
  const step = () => {
    if (el._tok !== tok) return;
    const u = Math.min(1, (performance.now() - t0) / DUR), e = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;  // wall-clock ease in-out
    const frame = u >= 1 ? data : data.map((t, i) => (t.x ? { ...t, x: lerp(prev[i].x, t.x, e), y: lerp(prev[i].y, t.y, e), z: lerp(prev[i].z, t.z, e) } : t));
    plot(id, frame, lay).then(() => { if (u < 1 && el._tok === tok) setTimeout(step, 16); });  // timers, not rAF: rAF stalls in hidden panes
  };
  setTimeout(step, 0);
  return Promise.resolve(el);
}
function pinRow(tableId, i) {  // the pinned item's row in the figure's table view
  $(tableId).querySelectorAll("table:first-of-type tbody tr").forEach((tr, k) => tr.classList.toggle("pinned", k === i));
}
function purge(ids) {  // switching 2-D ↔ 3-D: cancel pending tweens, free the WebGL context, re-attach listeners on the next render
  ids.forEach((id) => {
    const el = $(id);
    el._tok = (el._tok || 0) + 1;
    if (el._fullLayout) Plotly.purge(el);
    Object.assign(el, { _linked: false, _camLinked: false, _xyz: null });
  });
}
function axesPicker(fig) {  // one select with the 10 choices of 3 axes from PC1–5
  const sel = $(`axes-${fig}`);
  if (!sel || sel.options.length) return;
  sel.innerHTML = COMBOS.map((c, i) => `<option value="${i}">${c.map((k) => `PC${k + 1}`).join(" · ")}</option>`).join("");
  sel.onchange = () => { S.axes[fig] = COMBOS[+sel.value]; rerender([FIG_OF[fig]]); };
}
const FIG_OF = { fam: "fig-families", lat: "fig-lattice", cloud: "fig-concepts", map: "fig-map", how: "fig-axiom-how" };
function scree(fig, vt, vw) {  // PC1–5 variance, trained vs twin; displayed PCs emphasized; click a bar to put that PC on an axis
  const p = P(), el = $(`scree-${fig}`), ax = S.axes[fig], x = [1, 2, 3, 4, 5].map((k) => `PC${k}`);
  if (!el) return;
  const on = (k) => ax.includes(k);
  plot(`scree-${fig}`, [
    { type: "bar", name: "twin", x, y: vw, marker: { color: x.map((_, k) => (on(k) ? p.twin : p.grid)) }, hovertemplate: "twin · %{x}: %{y:.0%} of variance<extra></extra>" },
    { type: "bar", name: "trained", x, y: vt, marker: { color: x.map((_, k) => (on(k) ? p.blue : p.axis)) }, hovertemplate: "trained · %{x}: %{y:.0%} of variance<extra></extra>" }],
    layout("variance per PC: model blue, twin grey (click a bar to show it)", { height: 150, margin: { l: 34, r: 8, t: 26, b: 22 }, barmode: "group", bargap: 0.25, title: { text: "variance per PC: model blue, twin grey (click a bar to show it)", font: { size: 11, color: p.ink2 }, x: 0.01, xanchor: "left" },
      xaxis: { visible: true, color: p.muted, fixedrange: true }, yaxis: { visible: true, tickformat: ".0%", gridcolor: p.grid, color: p.muted, fixedrange: true, scaleanchor: null } }));
  if (!el._linked) {
    el._linked = true;
    el.on("plotly_click", (e) => {
      const k = e.points[0].pointIndex, cur = S.axes[fig];
      if (cur.includes(k)) return;
      S.axes[fig] = [cur[0], cur[1], k].sort((a, b) => a - b);
      $(`axes-${fig}`).value = COMBOS.findIndex((c) => c.join() === S.axes[fig].join());
      rerender([FIG_OF[fig]]);
    });
  }
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

// ---------------------------------------------------------------- at a glance: tiles, scorecard, forest
function renderTiles() {
  const k = conf(), Lp = S.I.models[0].primary_layer, t = (v, kk, tt) => `<div class="tile"><div class="v">${v}</div><div class="k">${kk}</div><div class="t">${tt}</div></div>`;
  $("tiles").innerHTML = [  // the axiom result leads, as in the TL;DR
    t(fmt(R(`lattice.${k}.group_like.offsets.commutative.consistency`)), "“add commutativity” consistency", `twin ${fmt(R(`lattice.${k}__random.group_like.offsets.commutative.consistency`))} · null 95th pct ${fmt(R(`lattice.${k}.group_like.offsets.commutative.null95`))}`),
    t(fmt(R(`map_fields.${k}.math_primary.rsa.trained`)), "field map vs arXiv (ρ)", `twin ${fmt(R(`map_fields.${k}.math_primary.rsa.random_init`))}`),
    t(fmt(R(`posthoc.${k}.pc_plane_cycle.days.${Lp - 1}`)), "days: cycle score", `twin ${fmt(R(`posthoc.${k}__random.pc_plane_cycle.days.${Lp - 1}`))} · positive control`),
    t(fmt(R(`shapes.${k}.angles.line_rho`)), "angles follow the number line (line ρ)", `circle ρ ${fmt(R(`shapes.${k}.angles.circle_rho`))} · twin line ρ ${fmt(R(`shapes.${k}__random.angles.line_rho`))} · circle falsified`),
  ].join("");
}
function renderScorecard(filter = "all") {
  const sc = S.I.scorecard, groups = { map: /field map/i, shapes: /days|chain|angles|order/i, axioms: /axiom|groups/i };
  const rows = sc.rows.filter((r) => filter === "all" || groups[filter].test(r.test));
  $("scorecard").innerHTML = `<thead><tr><th>Test</th><th>Status</th>${sc.models.map((m) => `<th class="num">${m}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr><td>${r.test}</td><td class="muted">${r.status}</td>${r.cells.map((c, j) => `<td class="cell num" title="${sc.models[j]} · ${r.test} · ${r.status}">${minus(c)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("score-legend").textContent = sc.legend;
  $("score-filter").innerHTML = ["all", "axioms", "map", "shapes"].map((g) => `<button data-g="${g}" aria-pressed="${g === filter}">${g[0].toUpperCase() + g.slice(1)}</button>`).join("");
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

// ---------------------------------------------------------------- field map, matrices, depth
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
function mapTraces3d(xyz, highlight, keys) {  // 3-D MDS: edges as 3-D lines, pure / applied as two series (the legend)
  const p = P(), F = S.I.fields, [X, Y, Z] = cols(xyz), pin = S.pin.map, ex = [], ey = [], ez = [];
  if (S.edges) F.edges.forEach(([i, j]) => { ex.push(X[i], X[j], null); ey.push(Y[i], Y[j], null); ez.push(Z[i], Z[j], null); });
  const hl = (i) => highlight === null || highlight === i || pin === i;
  const pts = (kind, color, name) => {
    const idx = F.names.map((_, i) => i).filter((i) => F.pa[i] === kind);
    return { type: "scatter3d", mode: "markers+text", name, showlegend: keys, x: idx.map((i) => X[i]), y: idx.map((i) => Y[i]), z: idx.map((i) => Z[i]),
      text: idx.map((i) => F.labels[i]), textposition: "top center", textfont: { size: 10, color: idx.map((i) => (hl(i) ? p.ink2 : p.grid)) },
      marker: { size: idx.map((i) => (i === pin ? 12 : i === highlight ? 10 : 7)), color, line: { color: idx.map((i) => (i === pin ? p.ink : p.surface)), width: 1 } },
      customdata: idx, hovertemplate: idx.map((i) => `<b>${F.labels[i]}</b><br>${F.branch_labels[F.branch[i]] || F.branch[i]} · ${F.pa[i]}<br>click to pin<extra></extra>`) };
  };
  return [{ type: "scatter3d", mode: "lines", x: ex, y: ey, z: ez, line: { color: p.axis, width: 2 }, hoverinfo: "skip", name: "2 nearest arXiv neighbours", showlegend: keys },
    pts("pure", p.blue, "pure field"), pts("applied", p.orange, "applied field")];
}
function pinInfoMap() {  // a pinned field's three nearest fields in arXiv, in the model and in the twin (full-space distances)
  const i = S.pin.map, F = S.I.fields, el = $("map-pin");
  if (i === null) { el.textContent = "Click a field to pin it and list its nearest fields in each map."; return; }
  const L = S.layer - 1, M = S.M[S.key].fields;
  const near = (Rk) => Rk[i].map((d, j) => [d, j]).filter(([, j]) => j !== i).sort((a, b) => a[0] - b[0]).slice(0, 3).map(([, j]) => F.labels[j]).join(", ");
  el.innerHTML = `<b>${F.labels[i]}</b> · nearest in arXiv: ${near(F.arxiv_rank)} · in ${model().name} (layer ${S.layer}): ${near(M.trained.rank[L])} · in the twin: ${near(M.random_init.rank[L])}`;
}
function renderMap() {
  const M = S.M[S.key], F = M.fields, L = S.layer - 1, tr = { transition: TRANSITION }, d3 = is3d("map"), k = d3 ? 1 : 0;
  const hi = S.hover !== null ? S.hover : S.pin.map;
  const st = (side) => ` · stress ${fmt(F[side].stress[L][k])}`, ast = ` · stress ${fmt(S.I.fields.arxiv_stress[k])}`;
  if (d3) {
    const lay = (t, keys) => layout(t, { margin: { l: 0, r: 0, t: 40, b: 0 }, scene: scene3d("map", ["MDS 1", "MDS 2", "MDS 3"]), ...(keys ? legend(4) : {}) });
    plot3d("map-arxiv", mapTraces3d(S.I.fields.arxiv_layout3, hi, true), lay("arXiv cross-listing (ground truth)" + ast, true));
    plot3d("map-trained", mapTraces3d(F[main()].layout3[L], hi, false), lay(fieldTitle(main(), S.twin) + st(main()), false));
    plot3d("map-twin", mapTraces3d(F.random_init.layout3[L], hi, false), lay(fieldTitle("random_init") + st("random_init"), false));
    linkCameras("map", SPIN_IDS.map);
  } else {
    plot("map-arxiv", mapTraces(S.I.fields.arxiv_layout, hi, true, $("map-arxiv")), layout("arXiv cross-listing (ground truth)" + ast, { ...padded(S.I.fields.arxiv_layout), ...legend() }));
    plot("map-trained", mapTraces(F[main()].layout[L], hi, false, $("map-trained")), layout(fieldTitle(main(), S.twin) + st(main()), { ...padded(F[main()].layout[L]), ...tr }));
    plot("map-twin", mapTraces(F.random_init.layout[L], hi, false, $("map-twin")), layout(fieldTitle("random_init") + st("random_init"), { ...padded(F.random_init.layout[L]), ...tr }));
  }
  ["map-arxiv", "map-trained", "map-twin"].forEach((id) => {
    const el = $(id);
    if (el._linked) return;
    el._linked = true;
    el.on("plotly_hover", (e) => { const i = e.points[0].customdata; if (i !== undefined && S.hover !== i) hoverField(i); });
    el.on("plotly_unhover", () => hoverField(null));
    el.on("plotly_click", (e) => { const i = e.points[0].customdata; if (i === undefined) return; S.pin.map = S.pin.map === i ? null : i; rerender(["fig-map"]); });
  });
  pinInfoMap();
  const lab = S.I.fields.labels;
  table($("map-table"), ["Field", "Pure/applied", "Branch", "Nearest arXiv neighbours"], lab.map((l, i) => [l, S.I.fields.pa[i], S.I.fields.branch_labels[S.I.fields.branch[i]],
    S.I.fields.edges.filter(([a, b]) => a === i || b === i).map(([a, b]) => lab[a === i ? b : a]).join(", ")]));
  pinRow("map-table", S.pin.map);
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

// ---------------------------------------------------------------- concept cloud + card
function renderCloud() {
  const p = P(), C = S.I.concepts, side = S.cloudTwin ? "random_init" : "trained", Mc = S.M[S.key].concepts, ax = S.axes.cloud;
  const xyz = pick3(Mc[side].pc5, ax), v = Mc[side].var;
  axesPicker("cloud");
  scree("cloud", Mc.trained.var, Mc.random_init.var);
  const cloud = (idx, marker, name) => ({ type: "scatter3d", mode: "markers", name, x: idx.map((i) => xyz[i][0]), y: idx.map((i) => xyz[i][1]), z: idx.map((i) => xyz[i][2]), marker,
    customdata: idx, hovertemplate: idx.map((i) => `<b>${C.concept[i]}</b><br>${label(C.field[i])}<extra></extra>`) });
  const all = C.concept.map((_, i) => i);
  const traces = S.colorMode === "abs"   // one series coloured by rating, or the highlighted field against the rest (two series)
    ? [cloud(all, { size: 3.5, color: C.abstraction, colorscale: [[0, p.lo], [1, p.hi]], colorbar: { title: { text: "abstraction", side: "right" }, thickness: 10, len: 0.6, tickfont: { color: p.muted } } }, "concepts")]
    : [cloud(all.filter((i) => C.field[i] !== S.field), { size: 2.6, color: p.twin, opacity: 0.7 }, "other fields"),
       cloud(all.filter((i) => C.field[i] === S.field), { size: 5, color: p.blue, opacity: 0.95 }, label(S.field))];
  if (S.picked !== null) traces.push({ type: "scatter3d", mode: "markers+text", name: "selected concept", x: [xyz[S.picked][0]], y: [xyz[S.picked][1]], z: [xyz[S.picked][2]], text: [C.concept[S.picked]],
    marker: { size: 8, color: p.orange }, textfont: { color: p.ink }, hoverinfo: "skip" });
  plot3d("cloud", traces, layout(`${model().name}${S.cloudTwin ? " · random-init twin" : ""} · layer ${model().primary_layer} · ${ax.map((k) => `PC${k + 1}`).join("/")} hold ${held(ax, v)}% of variance`,
    { margin: { l: 0, r: 0, t: 40, b: 0 }, ...(S.colorMode === "abs" ? {} : legend(4)), scene: scene3d("cloud", pcTitles(ax, v)) }));
  linkCameras("cloud", SPIN_IDS.cloud);
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
  $("concept-card").innerHTML = `<h4>${C.concept[i]}</h4><div class="muted">${label(C.field[i])} · flags: none set (primary set: the field-leak, eponym and everyday-word flags are all off; the eponym flag misses a few names, e.g. “tate curve”) · judge abstraction ${fmt(C.abstraction[i])} / 7${ml ? ` · Mathlib <code>${ml.decl}</code>, import depth ${ml.depth}` : " · no Mathlib match"}</div>
    <div class="nn"><div><b>Nearest in ${model().name}</b><ol>${nn("trained")}</ol></div><div><b>Nearest in the twin</b><ol>${nn("random_init")}</ol></div></div>`;
  rerender(["fig-concepts"]);
}

// ---------------------------------------------------------------- families
const FAM_TAKE = {
  days: "Positive control: in the trained models the days keep their cyclic order at most depths; the twins mostly do not (one reaches it at a single layer by chance). Whether the loop also closes is the line under the variance chart.",
  months: "Positive control: the months keep their cyclic order at most depths in some models, not all; in the top-2 plane their loop can stay open, and it closes in a lower plane in some models.",
  angles: "Falsified prediction: angles follow the number line (line ρ > circle ρ in every model), so 330° is farther from 0° than neighbouring angles are (orange edge). Every twin meets the preregistered circle rule by chance (a structureless cloud passes it), so the evidence is the trained models' line-like order, not the failed rule.",
  ring_chain: "Pre-registered: ordered if the Robinson score and the leave-one-out order beat their nulls (Holm) and the twin; the scorecard shows which models pass. In the confirmatory model the chain runs as a path and the twin is a tangle.",
  space_chain: "Pre-registered: ordered if the Robinson score and the leave-one-out order beat their nulls (Holm) and the twin; the scorecard shows which models pass. In the confirmatory model the chain runs as a path and the twin is a tangle.",
  number_systems: "Exploratory (7 items): ordered, but in most models not beyond word frequency and token count (the frequency control under the panel).",
  ngons: "Compared with the number words three … twelve; in the Qwen2.5 models the twins track them as well as the trained models do.", dimensions: "Follows the number words zero … ten, more closely than the twins do.",
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
    textposition: placeLabels(xy, items.map((s) => s.replace(" degrees", "°").replace("-dimensional", "-d")), el, 0.3, 0.1), cliponaxis: false, textfont: { size: 10, color: p.ink2 }, marker: { size: items.map((_, i) => (i === S.pin.fam ? 15 : 11)), color: ramp(xy.length), line: { color: items.map((_, i) => (i === S.pin.fam ? p.orange : p.surface)), width: 2 } },
    hovertemplate: items.map((s, i) => `<b>${s}</b><br>position ${i + 1} of ${items.length}${stat}<extra></extra>`) });
  return t;
}
const shortItem = (s) => s.replace(" degrees", "°").replace("-dimensional", "-d");
function famTraces3d(V, items, closed, isAngles, cycle, ideal) {  // one family in 3 of its top-5 PCs
  const p = P(), [X, Y, Z] = cols(V), n = V.length, seq = closed ? [...V.keys(), 0] : [...V.keys()], pin = S.pin.fam;
  const line = (A, B, C, idx) => ({ x: idx.map((i) => A[i]), y: idx.map((i) => B[i]), z: idx.map((i) => C[i]) });
  const t = [];
  if (S.ideal) {
    const [IX, IY, IZ] = cols(ideal);
    t.push({ type: "scatter3d", mode: "lines+markers", name: `ideal ${closed ? "circle" : "line"} (fitted in PC1–5)`, ...line(IX, IY, IZ, seq), line: { color: p.grid, width: 9 }, marker: { size: 3, color: p.axis }, hoverinfo: "skip" });
  }
  t.push({ type: "scatter3d", mode: "lines", name: "path in true order", ...line(X, Y, Z, seq), line: { color: p.axis, width: 3 }, hoverinfo: "skip" });
  if (isAngles) t.push({ type: "scatter3d", mode: "lines", name: "330° → 0° (predicted neighbours)", ...line(X, Y, Z, [n - 1, 0]), line: { color: p.orange, width: 6 }, hoverinfo: "skip" });
  const stat = cycle === undefined ? "" : `<br>PC1–2 cycle score at this layer ${fmt(cycle)} (post-hoc)`;
  t.push({ type: "scatter3d", mode: "markers+text", showlegend: false, x: X, y: Y, z: Z, text: items.map(shortItem), textposition: "top center",
    textfont: { size: 10, color: items.map((_, i) => (i === pin ? p.ink : p.ink2)) },
    marker: { size: items.map((_, i) => (i === pin ? 13 : 9)), color: ramp(n), line: { color: items.map((_, i) => (i === pin ? p.orange : p.surface)), width: 2 } },
    customdata: items.map((_, i) => i), hovertemplate: items.map((s, i) => `<b>${s}</b><br>position ${i + 1} of ${n}${stat}<br>click to pin<extra></extra>`) });
  return t;
}
function pinInfoFam(F, L, cyc) {  // pinned item: position, its coordinates on the displayed PCs, and the family's statistic at this layer
  const i = S.pin.fam, el = $("fam-pin"), ax = S.axes.fam;
  if (i === null || i >= F.items.length) { el.textContent = "Click an item to pin it in both panels."; return; }
  const at = (side) => ax.map((k) => `PC${k + 1} ${fmt(F.pc5[side][L][i][k])}`).join(", ");
  el.innerHTML = `<b>${F.items[i]}</b> · position ${i + 1} of ${F.items.length} · ${model().name}: ${at("trained")} · twin: ${at("random_init")}` +
    (cyc ? ` · the family's PC1–2 cycle score at this layer: ${fmt(cyc[L])} (post-hoc)` : "");
}
function renderFamilies() {
  const p = P(), f = S.fam, F = S.M[S.key].families[f], L = S.layer - 1, closed = F.expected === "circle", ang = f === "angles";
  const name = (S.I.family_labels[f] || f).split("\n")[0], tr = { transition: TRANSITION };
  const cyc = R(`posthoc.${S.key}.pc_plane_cycle.${f}`), cycT = R(`posthoc.${S.key}__random.pc_plane_cycle.${f}`);
  const mc = S.twin ? cycT : cyc;  // the main panel follows the global trained ↔ twin morph
  const who = S.twin ? "random-init twin (morphed)" : model().name, ax = S.axes.fam;
  axesPicker("fam");
  $("axes-fam").disabled = !is3d("fam");
  scree("fam", F.var.trained[L], F.var.random_init[L]);
  if (is3d("fam")) {
    const V = (s) => pick3(F.pc5[s][L], ax), I = (s) => pick3(F.ideal5[s][L], ax), v = (s) => F.var[s][L];
    const pcs = ax.map((k) => `PC${k + 1}`).join("/");
    const lay = (t, s, keys) => layout(t, { margin: { l: 0, r: 0, t: 40, b: 0 }, scene: scene3d("fam", pcTitles(ax, v(s))), ...(keys ? legend(4) : {}) });
    plot3d("fam-trained", famTraces3d(V(main()), F.items, closed, ang, mc && mc[L], I(main())), lay(`${name} · ${who} · layer ${S.layer} · ${pcs} hold ${held(ax, v(main()))}%`, main(), true));
    plot3d("fam-twin", famTraces3d(V("random_init"), F.items, closed, ang, cycT && cycT[L], I("random_init")), lay(`${name} · random-init twin · ${pcs} hold ${held(ax, v("random_init"))}%`, "random_init", false));
    linkCameras("fam", SPIN_IDS.fam);
  } else {
    plot("fam-trained", famTraces(F[main()][L], F.items, closed, ang, mc && mc[L], $("fam-trained")), layout(`${name} · ${who} · layer ${S.layer}`, { ...tr, ...legend() }));
    plot("fam-twin", famTraces(F.random_init[L], F.items, closed, ang, cycT && cycT[L], $("fam-twin")), layout(`${name} · random-init twin`, tr));
  }
  ["fam-trained", "fam-twin"].forEach((id) => {
    const el = $(id);
    if (el._linked) return;
    el._linked = true;
    el.on("plotly_click", (e) => { const i = e.points[0].customdata ?? e.points[0].pointIndex; if (e.points[0].data.mode !== "markers+text") return; S.pin.fam = S.pin.fam === i ? null : i; rerender(["fig-families"]); });
  });
  pinInfoFam(F, L, cyc);
  // post-hoc 4 (reports/prereg_addendum_3d.md): the best closed-cycle plane among PC1–5 at this layer
  const bp = (k) => { const e = R(`posthoc.${k}.pc_planes_cycle.${f}`); return e && e[L]; };
  const b = bp(S.key), bt = bp(`${S.key}__random`);
  $("fam-planes").innerHTML = b ? `<b>Closed cycle in any plane of PC1–5</b> (post-hoc, pre-specified addendum): best plane PC${b[1] + 1}–PC${b[2] + 1}, score ${fmt(b[0])}, ` +
    `p ${fmt(b[3], ".3f")} (max over the 10 planes; unadjusted per layer)${bt ? `; twin ${fmt(bt[0])}, p ${fmt(bt[3], ".3f")}` : ""}.` : "";
  table($("fam-table"), ["Item", "Position", `${model().name} PC1`, "PC2", "Twin PC1", "Twin PC2"],
    F.items.map((it, i) => [it, i + 1, fmt(F.trained[L][i][0]), fmt(F.trained[L][i][1]), fmt(F.random_init[L][i][0]), fmt(F.random_init[L][i][1])]));
  if (cyc) $("fam-table").insertAdjacentHTML("beforeend", `<table><thead><tr><th>Layer</th><th>Cycle score, trained</th><th>Cycle score, twin</th></tr></thead><tbody>${
    cyc.map((v, l) => `<tr><td>${l + 1}</td><td>${fmt(v)}</td><td>${fmt(cycT[l])}</td></tr>`).join("")}</tbody></table>`);
  pinRow("fam-table", S.pin.fam);
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

// ---------------------------------------------------------------- axiom lattice: lattice, cosines, analogy
function axiomChips() {  // one axiom choice, shared by the how-the-test-works stepper and the lattice figure
  ["how-chips", "axiom-chips"].forEach((id) => {
    $(id).innerHTML = S.M[S.key].lattice.axioms.map((a) => `<button data-a="${a}" aria-pressed="${a === S.axiom}">+ ${a}</button>`).join("");
    $(id).querySelectorAll("button").forEach((b) => b.onclick = () => { S.axiom = b.dataset.a; rerender(["fig-axiom-how", "fig-lattice"]); });
  });
}
function latTraces(side, keys = false, el = null) {  // keys: legend entries for the highlighted and the other single-axiom pairs
  const p = P(), Lt = S.M[S.key].lattice, xy = Lt[side].planes[S.layer - 1], a = Lt.axioms.indexOf(S.axiom);
  const gx = [], gy = [], ann = [], hot = side === "random_init" ? p.twin : p.blue;  // the twin is always grey (style.py)
  Lt.pairs.forEach(([i, j], k) => {
    if (Lt.group[k] === a) ann.push({ x: xy[j][0], y: xy[j][1], ax: xy[i][0], ay: xy[i][1], xref: "x", yref: "y", axref: "x", ayref: "y", showarrow: true, arrowhead: 2, arrowsize: 1.9, arrowwidth: 2.2, arrowcolor: hot, standoff: 6, startstandoff: 6 });  // big filled heads: direction at a glance
    else { gx.push(xy[i][0], xy[j][0], null); gy.push(xy[i][1], xy[j][1], null); }
  });
  const has = (b) => Lt.axioms.filter((_, q) => b[q]).join(", ") || "none";
  return { data: [{ type: "scatter", mode: "lines", x: gx, y: gy, line: { color: p.grid, width: 1 }, hoverinfo: "skip", name: "other single-axiom pairs", showlegend: keys },
    { type: "scatter", mode: "markers+text", showlegend: false, x: xy.map((d) => d[0]), y: xy.map((d) => d[1]), text: Lt.items.map((s) => s.replace("commutative ", "comm. ")), textposition: placeLabels(xy, Lt.items.map((s) => s.replace("commutative ", "comm. ")), el || {}, 0.3, 0.1), cliponaxis: false,
      textfont: { size: 10, color: p.ink2 }, marker: { size: Lt.items.map((_, i) => (i === S.pin.lat ? 14 : 9)), color: Lt.items.map((_, i) => (i === S.pin.lat ? p.orange : p.ink2)), line: { color: p.surface, width: 1.5 } },
      hovertemplate: Lt.items.map((s, i) => `<b>${s}</b><br>axioms: ${has(Lt.bits[i])}<extra></extra>`) },
    { type: "scatter", mode: "lines", x: [null], y: [null], line: { color: hot, width: 2.2 }, name: `pairs differing only in ${S.axiom} (arrows)`, showlegend: keys, hoverinfo: "skip" }], ann };
}
const ANALOGY0 = ["semigroup", "monoid", "quasigroup"];
function analogyChoice(Lt) {  // a, b, c from the analogy playground (its defaults before it has rendered)
  return ["an-a", "an-b", "an-c"].map((id, k) => ($(id).value !== "" ? +$(id).value : Lt.items.indexOf(ANALOGY0[k])));
}
// arrowheads: "raw" makes each cone exactly as long as its vector (https://plotly.com/javascript/reference/cone/); flat colour
const coneTrace = (c, color) => ({ type: "cone", ...c, anchor: "tip", sizemode: "raw", sizeref: 1, colorscale: [[0, color], [1, color]], showscale: false, hoverinfo: "skip", showlegend: false,
  lighting: { ambient: 1, diffuse: 0, specular: 0, roughness: 1, fresnel: 0 } });
function latArrows(V, { keys = false, color = P().blue, lineUp = false, pin = null, hint = "" } = {}) {
  // 13 × 3 coordinates → the selected axiom's pairs as arrows (lines + cone heads), every other single-axiom pair as a grey
  // line, the structures labelled. lineUp: move each arrow (same direction and length) so all tails meet at the mean tail,
  // with the rest faded. Trace count and lengths never depend on the options, so plot3d glides between them.
  const p = P(), Lt = S.M[S.key].lattice, [X, Y, Z] = cols(V), a = Lt.axioms.indexOf(S.axiom);
  const g = { x: [], y: [], z: [] }, bl = { x: [], y: [], z: [] }, cone = { x: [], y: [], z: [], u: [], v: [], w: [] };
  const ext = Math.max(...[X, Y, Z].map((c) => Math.max(...c) - Math.min(...c))), head = 0.13 * ext;  // arrowhead length: 13% of the view, ≤ 60% of its arrow
  const sel = Lt.pairs.filter((_, k) => Lt.group[k] === a), tail0 = [0, 1, 2].map((c) => sel.reduce((s, [i]) => s + V[i][c], 0) / sel.length);
  Lt.pairs.forEach(([i, j], k) => {
    if (Lt.group[k] !== a) { g.x.push(X[i], X[j], null); g.y.push(Y[i], Y[j], null); g.z.push(Z[i], Z[j], null); return; }
    const d = [X[j] - X[i], Y[j] - Y[i], Z[j] - Z[i]], o = lineUp ? tail0 : [X[i], Y[i], Z[i]], tip = o.map((v, c) => v + d[c]);
    const nr = Math.hypot(...d) || 1, hl = Math.min(head, 0.6 * nr);
    bl.x.push(o[0], tip[0], null); bl.y.push(o[1], tip[1], null); bl.z.push(o[2], tip[2], null);
    cone.x.push(tip[0]); cone.y.push(tip[1]); cone.z.push(tip[2]); cone.u.push((d[0] / nr) * hl); cone.v.push((d[1] / nr) * hl); cone.w.push((d[2] / nr) * hl);
  });
  const has = (b) => Lt.axioms.filter((_, q) => b[q]).join(", ") || "none", ink = lineUp ? p.axis : p.ink2;
  return [
    { type: "scatter3d", mode: "lines", ...g, line: { color: p.grid, width: 2 }, hoverinfo: "skip", name: "other single-axiom pairs", showlegend: keys },
    { type: "scatter3d", mode: "lines", ...bl, line: { color, width: 5 }, hoverinfo: "skip", name: `pairs differing only in ${S.axiom}`, showlegend: keys },
    coneTrace(cone, color),
    { type: "scatter3d", mode: "markers+text", name: "structures", showlegend: false, x: X, y: Y, z: Z, text: Lt.items.map((s) => s.replace("commutative ", "comm. ")), textposition: "top center",
      textfont: { size: 10, color: ink }, marker: { size: Lt.items.map((_, i) => (i === pin ? 12 : 7)), color: Lt.items.map((_, i) => (i === pin ? p.orange : ink)) }, customdata: Lt.items.map((_, i) => i),
      hovertemplate: Lt.items.map((s, i) => `<b>${s}</b><br>axioms: ${has(Lt.bits[i])}${hint}<extra></extra>`) }];
}
function latTraces3d(side, keys) {  // 3 of the top-5 PCs: arrows (lines + cone heads) for the selected axiom, the analogy point
  const p = P(), Lt = S.M[S.key].lattice, L = S.layer - 1, ax = S.axes.lat, V = pick3(Lt[side].pc5[L], ax), [X, Y, Z] = cols(V);
  const data = latArrows(V, { keys, color: side === "random_init" ? p.twin : p.blue, pin: S.pin.lat, hint: "<br>click for its axioms" });
  if (S.layer === model().primary_layer) {
    // the analogy query b − a + c from the analogy playground, placed exactly: exact (PC-basis) coordinates → this view via to_view
    const [ia, ib, ic] = analogyChoice(Lt), E = Lt[side].exact, M = Lt[side].to_view;
    const q = E[ib].map((v, k) => v - E[ia][k] + E[ic][k]).slice(0, 5), q5 = [0, 1, 2, 3, 4].map((c) => q.reduce((s, v, r) => s + v * M[r][c], 0)), Q = ax.map((k) => q5[k]);
    data.push({ type: "scatter3d", mode: "lines", name: "analogy: a → b and c → b − a + c", showlegend: keys, hoverinfo: "skip",
      x: [X[ia], X[ib], null, X[ic], Q[0]], y: [Y[ia], Y[ib], null, Y[ic], Q[1]], z: [Z[ia], Z[ib], null, Z[ic], Q[2]], line: { color: p.orange, width: 4, dash: "dash" } });
    data.push({ type: "scatter3d", mode: "markers+text", name: "b − a + c", showlegend: false, x: [Q[0]], y: [Q[1]], z: [Q[2]], text: ["b − a + c"], textposition: "bottom center",
      textfont: { size: 10, color: p.orange }, marker: { size: 7, color: p.orange, symbol: "diamond" },
      hovertemplate: `${Lt.items[ib]} − ${Lt.items[ia]} + ${Lt.items[ic]}<br>exact projection onto the shown PCs; nearness here is not nearness in full space<extra></extra>` });
  }
  return data;
}
function renderLattice() {
  const Lt = S.M[S.key].lattice, L = S.layer - 1, ax = S.axes.lat;
  if (!S.axiom) S.axiom = Lt.axioms[0];
  axesPicker("lat");
  $("axes-lat").disabled = !is3d("lat");
  scree("lat", Lt.trained.var[L], Lt.random_init.var[L]);
  [["lat-trained", main()], ["lat-twin", "random_init"]].forEach(([id, side]) => {  // the main panel follows the global morph
    const who = side === "trained" ? model().name : `random-init twin${id === "lat-trained" ? " (morphed)" : ""}`;
    if (is3d("lat")) {
      const v = Lt[side].var[L];
      plot3d(id, latTraces3d(side, id === "lat-trained"), layout(`${who} · layer ${S.layer} · + ${S.axiom} · ${ax.map((k) => `PC${k + 1}`).join("/")} hold ${held(ax, v)}%`,
        { margin: { l: 0, r: 0, t: 40, b: 0 }, scene: scene3d("lat", pcTitles(ax, v)), ...(id === "lat-trained" ? legend(4) : {}) }));
    } else {
      const t = latTraces(side, id === "lat-trained", $(id));
      plot(id, t.data, layout(`${who} · PCA plane at layer ${S.layer} · + ${S.axiom}`, { annotations: t.ann, transition: TRANSITION, ...(id === "lat-trained" ? legend() : {}) }));
    }
  });
  if (is3d("lat")) linkCameras("lat", SPIN_IDS.lat);
  table($("lat-table"), ["Structure", ...Lt.axioms, `+ ${S.axiom} leads to`], Lt.items.map((it, i) => [it, ...Lt.bits[i].map((b) => (b ? "✓" : "")),
    Lt.pairs.filter(([a], k) => a === i && Lt.group[k] === Lt.axioms.indexOf(S.axiom)).map(([, b]) => Lt.items[b]).join(", ") || "—"]));
  const o = R(`lattice.${S.key}.group_like.offsets.${S.axiom}`), ot = R(`lattice.${S.key}__random.group_like.offsets.${S.axiom}`);
  ["lat-trained", "lat-twin"].forEach((id) => {
    const el = $(id);
    if (el._linked) return;
    el._linked = true;
    el.on("plotly_click", (e) => {
      const pt = e.points[0], Ltc = S.M[S.key].lattice;
      if (pt.data.mode !== "markers+text" || pt.data.name === "b − a + c") return;  // structures only (2-D: trace 1; 3-D: "structures")
      const i = pt.customdata ?? pt.pointIndex;
      S.pin.lat = S.pin.lat === i ? null : i;
      rerender(["fig-lattice"]);
    });
  });
  $("lat-note").innerHTML = `<b>+ ${S.axiom}:</b> offset consistency ${fmt(o.consistency)} (random-pair null 95th percentile ${fmt(o.null95)}, p ${fmt(o.p, ".3f")}); twin ${fmt(ot.consistency)}. Values at the preregistered layer.` +
    (is3d("lat") ? (S.layer === model().primary_layer ? ` The orange diamond is the ${figref("fig-analogy")} analogy query, projected exactly.` : " The analogy query (orange diamond) is drawn at the preregistered layer only.") : "");
  const pl = S.pin.lat;
  $("lat-bits").innerHTML = pl === null ? "Click a structure to pin it and see its axioms." : `<b>${Lt.items[pl]}</b>: ` + Lt.axioms.map((a, q) => `${a} ${Lt.bits[pl][q] ? "✓" : "✗"}`).join(" · ");
  pinRow("lat-table", pl);
  axiomChips();
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
    sel.forEach((s, k) => { s.innerHTML = Lt.items.map((it, i) => `<option value="${i}">${it}</option>`).join(""); s.value = Lt.items.indexOf(ANALOGY0[k]); s.onchange = () => { renderAnalogy(); rerender(["fig-lattice"]); }; });
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

// ---------------------------------------------------------------- how the axiom test works: a 4-step explainer
// Step 1 is a schematic of the definitions; steps 2–3 the top-PC view at the preregistered layer (a projection); step 4 is
// full-space, computed here from lattice.<side>.exact, which keeps every inner product of the centred residual vectors.
// step 1: one display direction per axiom. The 4th is not an integer mix of the others, so no two structures coincide (a 4-cube
// in 3-D); it was picked by a search that maximizes the smallest label separation in the default camera view
const AXIOM_DIRS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.75, -0.75, 0]];
function schematic(bits, ref) {  // Σ (axioms present) × their directions, centred and scaled onto `ref` (same mean and RMS radius), so step 1 → 2 glides
  const stats = (A) => { const m = [0, 1, 2].map((c) => A.reduce((s, r) => s + r[c], 0) / A.length); return [m, Math.sqrt(A.reduce((s, r) => s + r.reduce((t, v, c) => t + (v - m[c]) ** 2, 0), 0) / A.length)]; };
  const P0 = bits.map((b) => [0, 1, 2].map((c) => b.reduce((s, on, a) => s + on * AXIOM_DIRS[a][c], 0)));
  const [m0, r0] = stats(P0), [m1, r1] = stats(ref);
  return P0.map((r) => r.map((v, c) => m1[c] + ((v - m0[c]) * r1) / r0));
}
function compass(E, pairs) {
  // Leave-one-out angle (degrees) between each same-axiom difference vector and the mean of the *other* unit difference
  // vectors, plus the mean pairwise cosine of the unit vectors, which is offset consistency (geometry/offsets.py). Differences
  // do not depend on centring and E keeps every inner product, so both are the full-space values (tests/test_site.py).
  const dot = (u, v) => u.reduce((s, x, k) => s + x * v[k], 0), unit = (v) => { const n = Math.sqrt(dot(v, v)) || 1; return v.map((x) => x / n); };
  const U = pairs.map(([i, j]) => unit(E[j].map((v, k) => v - E[i][k])));
  const others = (k) => unit(U.reduce((s, w, q) => (q === k ? s : s.map((x, r) => x + w[r])), U[k].map(() => 0)));
  const cs = U.flatMap((u, k) => U.slice(k + 1).map((w) => dot(u, w)));
  return { deg: U.map((u, k) => (180 / Math.PI) * Math.acos(Math.max(-1, Math.min(1, dot(u, others(k)))))), cons: cs.reduce((s, c) => s + c, 0) / cs.length };
}
function compassTraces(side) {  // step 4: one compass per axiom (2 × 2); dashed vertical = mean of the other arrows; each arrow tilted by its angle
  const p = P(), Lt = S.M[S.key].lattice, E = Lt[side].exact, color = side === "trained" ? p.blue : p.twin, key = side === "trained" ? S.key : `${S.key}__random`;
  const T = (extra = {}) => ({ x: [], y: [], z: [], ...extra }), ring = T(), up = T(), arrows = [T(), T()], tips = T({ text: [] }), lab = T({ text: [], hover: [] }), cone = T({ u: [], v: [], w: [] });
  const push = (t, ...pts) => pts.forEach((q) => { t.x.push(q && q[0]); t.y.push(q && q[1]); t.z.push(q && q[2]); });  // null = gap between lines
  Lt.axioms.forEach((name, a) => {
    const r = (2.8 * (a % 2)) / Math.SQRT2, o = [r, r, -3.2 * Math.floor(a / 2)];  // columns along (1, 1, 0), the default camera's screen-right
    const pairs = Lt.pairs.filter((_, k) => Lt.group[k] === a), { deg, cons } = compass(E, pairs), off = R(`lattice.${key}.group_like.offsets.${name}`), on = name === S.axiom;
    for (let t = 0; t <= 48; t++) push(ring, [o[0] + Math.cos((Math.PI * t) / 24), o[1] + Math.sin((Math.PI * t) / 24), o[2]]);
    push(ring, null);
    push(up, o, [o[0], o[1], o[2] + 1.2], null);
    deg.forEach((d, k) => {  // only the tilt θ is data; the azimuth spreads the arrows evenly around the vertical
      const th = (d * Math.PI) / 180, ph = (2 * Math.PI * k) / deg.length, u = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)], tip = o.map((v, c) => v + u[c]);
      push(arrows[on ? 1 : 0], o, tip, null);
      push(cone, tip); cone.u.push(0.22 * u[0]); cone.v.push(0.22 * u[1]); cone.w.push(0.22 * u[2]);
      push(tips, tip); tips.text.push(`${Lt.items[pairs[k][0]]} → ${Lt.items[pairs[k][1]]}: ${fmt(d, ".0f")}° from the others’ mean`);
    });
    push(lab, [o[0], o[1], o[2] - 0.5]);  // under the ring, in three short lines, so neighbouring labels do not collide on a phone
    lab.text.push(`${on ? "<b>" : ""}+ ${name}: ${fmt(off.consistency)}${on ? "</b>" : ""}<br>null 95% ${fmt(off.null95)}<br>p ${pval(off.p)}`);
    lab.hover.push(`+ ${name}: offset consistency ${fmt(off.consistency)} (recomputed in this page: ${fmt(cons)})<br>random-pair null 95th percentile ${fmt(off.null95)}, p ${pval(off.p)} before Holm<extra></extra>`);
  });
  return [
    { type: "scatter3d", mode: "lines", ...ring, line: { color: p.axis, width: 2 }, hoverinfo: "skip", name: "90°: unrelated" },
    { type: "scatter3d", mode: "lines", ...up, line: { color: p.ink2, width: 3, dash: "dash" }, hoverinfo: "skip", name: "mean of the other arrows" },
    { type: "scatter3d", mode: "lines", ...arrows[0], line: { color, width: 4 }, hoverinfo: "skip" },
    { type: "scatter3d", mode: "lines", ...arrows[1], line: { color, width: 8 }, hoverinfo: "skip" },
    coneTrace(cone, color),
    { type: "scatter3d", mode: "markers", ...tips, marker: { size: 4, color }, hovertemplate: "%{text}<extra></extra>" },
    { type: "scatter3d", mode: "text", x: lab.x, y: lab.y, z: lab.z, text: lab.text, textposition: "bottom center", textfont: { size: 11, color: p.ink }, hovertemplate: lab.hover },
  ];
}
const flat = (ts) => ts.filter((t) => t.type !== "cone").map(({ z, ...t }) => ({ ...t, type: "scatter", y: z }));  // no WebGL: the x–z side view, no arrowheads
const HOW_TITLES = ["Structures are sets of axioms", "Adding an axiom is an arrow", "Line the arrows up", "Measure in the full space"];
function howText(Lt, sel) {
  const a = S.axiom, n = sel.length, ex = sel.slice(0, 2).map(([i, j]) => `${Lt.items[i]} → ${Lt.items[j]}`).join(" and ");
  const o = R(`lattice.${S.key}.group_like.offsets.${a}`), ot = R(`lattice.${S.key}__random.group_like.offsets.${a}`);
  return `<b>${S.how} · ${HOW_TITLES[S.how - 1]}.</b> ` + [
    `Each structure is the set of axioms it satisfies: a magma has none, a group has associativity, identity and divisibility, an abelian group all four. Left: the 13 structures drawn from their definitions alone, one fixed direction per axiom; right: the same as a table. The ${n} blue arrows join the pairs that differ only in ${a}, such as ${ex}. In this picture they are parallel by construction; the test asks whether they are in the model.`,
    `The model’s own vectors for the 13 names at the preregistered layer ${model().primary_layer}, in three of their top-5 principal components: ${model().name} on the left (blue), its random-init twin on the right (grey). Each arrow runs from a structure to the one that adds ${a}.`,
    `The same ${n} arrows, each moved so that the tails meet at one point; direction and length are unchanged. A tight bundle suggests one shared direction, a wide fan none. This is still a projection, so it can mislead either way.`,
    `Each compass is one axiom, measured over every dimension of the residual stream (computed in your browser). The dashed vertical is the mean direction of the other arrows of that axiom, and each arrow is tilted from it by its true angle: 0° points straight up, 90° lies on the ring (unrelated). The test statistic, offset consistency, is the mean cosine between all pairs of same-axiom arrows: for + ${a}, ${fmt(o.consistency)} in ${model().name} against a random-pair null with 95th percentile ${fmt(o.null95)} (p ${pval(o.p)}), and ${fmt(ot.consistency)} in the twin.`,
  ][S.how - 1];
}
function howCard(Lt, sel) {  // step 1, right: the definitions as a table, with the selected axiom's column and its pairs
  const a = Lt.axioms.indexOf(S.axiom), c = (q) => (q === a ? ' class="sel"' : "");
  $("how-card").innerHTML = `<h4>Which axioms each structure has</h4><table><thead><tr><th>Structure</th>${Lt.axioms.map((x, q) => `<th${c(q)}>${x}</th>`).join("")}</tr></thead><tbody>${
    Lt.items.map((it, i) => `<tr><td>${it}</td>${Lt.bits[i].map((b, q) => `<td${c(q)}>${b ? "✓" : ""}</td>`).join("")}</tr>`).join("")}</tbody></table>` +
    `<p><b>Pairs that differ only in ${S.axiom}</b> (the blue arrows): ${sel.map(([i, j]) => `${Lt.items[i]} → ${Lt.items[j]}`).join(", ")}.</p>`;
}
function setHowStep(s) {
  if ((s === 4) !== (S.how === 4)) delete S.cam.how;  // the compass is a different scene: start it from the default camera
  S.how = s;
  rerender(["fig-axiom-how"]);
}
function renderHow() {
  const p = P(), Lt = S.M[S.key].lattice, L = model().primary_layer - 1, ax = S.axes.how, st = S.how;
  if (!S.axiom) S.axiom = Lt.axioms[0];
  axesPicker("how");
  $("axes-how").disabled = st === 1 || st === 4;
  document.querySelectorAll("#how-steps [data-step]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.step === st)));
  $("how-twin").hidden = st === 1;
  $("how-card").hidden = st !== 1;
  const sel = Lt.pairs.filter((_, k) => Lt.group[k] === Lt.axioms.indexOf(S.axiom)), V = (side) => pick3(Lt[side].pc5[L], ax);
  const bare = () => {  // schematic, compass: the axes mean nothing, so no titles or grid (not visible: false, which gl3d
    const s = scene3d("how", ["", "", ""]);  // does not undo when a later step shows the PC axes again)
    ["xaxis", "yaxis", "zaxis"].forEach((k) => { s[k] = { ...s[k], showgrid: false }; });
    return s;
  };
  const draw = (id, traces, title, scene, keys) => (HAS_GL
    ? plot3d(id, traces, layout(title, { margin: { l: 0, r: 0, t: 40, b: 0 }, scene, ...(keys ? legend(4) : {}) }))
    : plot(id, flat(traces), layout(title, keys ? legend() : {})));
  const who = (side) => (side === "trained" ? model().name : "random-init twin");
  if (st === 1) {
    draw("how-trained", latArrows(schematic(Lt.bits, V("trained")), { keys: true }), `13 structures placed by their axioms (schematic) · + ${S.axiom}`, bare(), true);
    howCard(Lt, sel);
  } else {
    [["how-trained", "trained"], ["how-twin", "random_init"]].forEach(([id, side]) => {
      const v = Lt[side].var[L];
      if (st === 4) draw(id, compassTraces(side), `${who(side)} · full space · layer ${L + 1}: each arrow’s angle to the mean of the others`, bare(), false);
      else draw(id, latArrows(V(side), { keys: id === "how-trained", color: side === "trained" ? p.blue : p.twin, lineUp: st === 3 }),
        `${who(side)} · layer ${L + 1} · + ${S.axiom}${st === 3 ? " · tails at one point" : ""} · ${ax.map((k) => `PC${k + 1}`).join("/")} hold ${held(ax, v)}%`, scene3d("how", pcTitles(ax, v)), id === "how-trained");
    });
  }
  if (HAS_GL) linkCameras("how", ["how-trained", "how-twin"]);
  $("how-text").innerHTML = howText(Lt, sel);
  axiomChips();
}

// ---------------------------------------------------------------- Mathlib
function renderMathlib() {
  const p = P(), pts = S.I.mathlib_points, ml = R("mathlib");
  plot("mathlib", [{ type: "scatter", mode: "markers", x: pts.map((d) => d.depth), y: pts.map((d) => d.rating), marker: { size: 10, color: p.blue, line: { color: p.surface, width: 1.5 } },
    hovertemplate: pts.map((d) => `<b>${d.concept}</b> · ${d.field}<br>Mathlib <code>${d.decl}</code><br>import depth ${d.depth} · judge ${fmt(d.rating)}<extra></extra>`) }],
    layout(`Judges vs Mathlib: ρ = ${fmt(ml.rho_mean_rating_depth)}, 95% CI [${fmt(ml.ci95[0])}, ${fmt(ml.ci95[1])}], n = ${ml.n_matched}`,
      { height: 380, margin: { l: 54, r: 16, t: 40, b: 46 }, xaxis: { title: "Mathlib import depth", gridcolor: p.grid, color: p.muted },
        yaxis: { title: "mean judge abstraction (1–7)", gridcolor: p.grid, color: p.muted, scaleanchor: null } }));
  table($("ml-table"), ["Concept", "Field", "Mathlib declaration", "Import depth", "Judge rating"], pts.map((d) => [d.concept, d.field, `<code>${d.decl}</code>`, d.depth, fmt(d.rating)]));
}

// ---------------------------------------------------------------- group objects
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

// ---------------------------------------------------------------- partial-operation lattice (negative result)
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

// ---------------------------------------------------------------- figure decorations: numbers, paper links, table views
const FIGNUM = {};
function numberFigures() {  // figures are numbered in page order, so moving a section never leaves a stale number behind
  document.querySelectorAll("figure.fig").forEach((f, i) => { FIGNUM[f.id] = i + 1; f.querySelector(".fig-num").textContent = `Figure ${i + 1}`; });
  document.querySelectorAll("a.figref").forEach((a) => { a.textContent = `Figure ${FIGNUM[a.hash.slice(1)]}`; });
}
const figref = (id) => `<a class="figref" href="#${id}">Figure ${FIGNUM[id]}</a>`;  // the same link, for text the script writes
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
  if (!HAS_GL) $("gl-note").hidden = false;
  document.querySelectorAll("[data-view]").forEach((b) => {  // 2-D ↔ 3-D per figure (3-D by default; 2-D without WebGL)
    const fig = b.dataset.view;
    b.setAttribute("aria-pressed", String(is3d(fig)));
    if (!HAS_GL) { b.disabled = true; b.title = "3-D views need WebGL; showing the 2-D view"; }
    b.onclick = () => {
      if (S.spin[fig]) toggleSpin(fig);
      S.view[fig] = is3d(fig) ? "2d" : "3d";
      b.setAttribute("aria-pressed", String(is3d(fig)));
      document.querySelector(`[data-spin="${fig}"]`).disabled = !is3d(fig);
      purge(SPIN_IDS[fig]);
      rerender([FIG_OF[fig]]);
    };
  });
  document.querySelectorAll("[data-spin]").forEach((b) => {
    b.disabled = REDUCED || !(b.dataset.spin === "cloud" ? HAS_GL : is3d(b.dataset.spin));
    b.onclick = () => toggleSpin(b.dataset.spin);
  });
  $("link").onclick = (e) => { S.link = !S.link; e.target.setAttribute("aria-pressed", String(S.link)); };
  $("edges-toggle").onclick = (e) => { S.edges = !S.edges; e.target.setAttribute("aria-pressed", String(S.edges)); rerender(["fig-map"]); };
  $("all-models").onclick = (e) => { S.allModels = !S.allModels; e.target.setAttribute("aria-pressed", String(S.allModels)); rerender(["fig-depth"]); };
  $("ideal").onclick = (e) => { S.ideal = !S.ideal; e.target.setAttribute("aria-pressed", String(S.ideal)); rerender(["fig-families"]); };
  const stopHow = () => { clearInterval(S.howTimer); S.howTimer = null; $("how-play").setAttribute("aria-pressed", "false"); $("how-play").textContent = "▶ Play"; };
  document.querySelectorAll("#how-steps [data-step]").forEach((b) => b.onclick = () => { stopHow(); setHowStep(+b.dataset.step); });
  $("how-play").onclick = () => {  // advance now, then every few seconds; stop on the last step
    if (S.howTimer) return stopHow();
    $("how-play").setAttribute("aria-pressed", "true"); $("how-play").textContent = "❚❚ Pause";
    setHowStep(S.how === 4 ? 1 : S.how + 1);
    S.howTimer = setInterval(() => (S.how >= 4 ? stopHow() : setHowStep(S.how + 1)), 4500);
  };
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
  const q = new URLSearchParams(location.search);
  if (q.get("view") === "2d" || !HAS_GL) Object.keys(S.view).forEach((f) => { S.view[f] = "2d"; });  // ?view=2d deep-links the 2-D views
  fillNums();
  numberFigures();
  decorate();
  wire();
  renderTiles();
  renderScorecard();
  register("fig-forest", renderForest);  // in page order (the fallback below renders unseen figures in this order)
  register("fig-axiom-how", renderHow); register("fig-lattice", renderLattice); register("fig-cosines", renderCosines); register("fig-analogy", renderAnalogy); register("fig-groups", renderGroups);
  register("fig-map", renderMap); register("fig-rdm", renderRdm); register("fig-depth", renderDepth);
  register("fig-concepts", renderCloud); register("fig-families", renderFamilies); register("fig-abstraction", renderMathlib); register("fig-partial", renderPartial);
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
