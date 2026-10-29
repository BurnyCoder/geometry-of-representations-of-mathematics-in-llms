"""Build paper/paper.md (+ PDF via pandoc) from paper/template.md and results JSON.

Global context: the paper's prose is hand-written, but every number in it is a placeholder
resolved here from results/<cfg>/*.json, and every table is rendered by report/render.py:
  {{num:<cfg>.<file>.<dotted.path>[:fmt]}}  e.g. {{num:mvp.map_fields.qwen3.5-2b.math_primary.rsa.trained:.3f}}
  {{table:<name>:<cfg>}}                    e.g. {{table:field_map:mvp}}
  {{fig:<file>}}                            figure in reports/figures/
Unresolvable placeholders raise, so a stale paper cannot be built silently.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess

from mathgeom.config import ROOT, load_yaml

log = logging.getLogger(__name__)


def _results(cfg_name: str) -> dict:
    d = ROOT / "results" / cfg_name
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json") if p.stem != "numbers"}


def render_template(tpl: str, fig_prefix: str = "../reports/figures/") -> str:
    """Resolve {{num:..}}, {{table:..}} and {{fig:..}} placeholders in a Markdown template."""
    from mathgeom.report import render

    cache: dict[str, dict] = {}
    fams = load_yaml("families.yaml")

    def num(m: re.Match) -> str:
        spec = m.group(1)
        path, _, fmt = spec.partition(":")
        # model keys contain dots (qwen3.5-2b), so split cfg.file first, then match the longest key prefix
        cfg_name, file, rest = path.split(".", 2)
        res = cache.setdefault(cfg_name, _results(cfg_name))[file]
        obj, parts = res, rest.split(".")
        while parts:
            for j in range(len(parts), 0, -1):  # greedy: try the longest dotted key first
                key = ".".join(parts[:j])
                if isinstance(obj, dict) and key in obj:
                    obj, parts = obj[key], parts[j:]
                    break
                if isinstance(obj, list) and j == 1 and key.lstrip("-").isdigit():
                    obj, parts = obj[int(key)], parts[1:]
                    break
            else:
                raise KeyError(f"cannot resolve {spec!r} at {parts}")
        return format(obj, fmt) if fmt else str(obj)

    def table(m: re.Match) -> str:
        name, cfg_name = m.group(1), m.group(2)
        r = cache.setdefault(cfg_name, _results(cfg_name))
        fn = {
            "field_map": lambda: render.table_field_map(r["map_fields"]),
            "clustering": lambda: render.table_clustering(r["map_fields"]),
            "shapes": lambda: render.table_shapes(r["shapes"], fams),
            "negative_control": lambda: render.table_negative_control(r["shapes"]),
            "beyond_reference": lambda: render.table_beyond_reference(r["shapes"], fams),
            "lattice": lambda: render.table_lattice(r["lattice"]),
            "calibration": lambda: render.table_calibration(r["calibration"]),
            "holm": lambda: render.table_holm(r["map_fields"], r["shapes"], r["lattice"]),
            "structures": lambda: render.table_structures(r["structures"]),
            "hierarchy": lambda: render.table_hierarchy(r["hierarchy"]),
            "baselines": lambda: render.table_baselines(r["baselines"]),
            "smds": lambda: render.table_smds(r["smds"]),
            "posthoc": lambda: render.table_posthoc(r["posthoc"]),
            "cycles": lambda: render.table_cycles(r["posthoc"], {k: v["primary_layer"] for k, v in r["shapes"].items()}),
        }[name]
        return fn()

    out = re.sub(r"\{\{num:([^}]+)\}\}", num, tpl)
    out = re.sub(r"\{\{table:(\w+):(\w+)\}\}", table, out)
    out = re.sub(r"\{\{fig:([^}]+)\}\}", lambda m: f"{fig_prefix}{m.group(1)}", out)
    if "{{" in out:
        raise ValueError("unresolved placeholder: " + out[out.index("{{") : out.index("{{") + 80])
    return out


def build_summary() -> None:
    """reports/summary.md from reports/summary_template.md (layman summary + tables)."""
    tpl = (ROOT / "reports" / "summary_template.md").read_text(encoding="utf-8")
    (ROOT / "reports" / "summary.md").write_text(render_template(tpl, fig_prefix="figures/"), encoding="utf-8")


def build_readme_block() -> None:
    """Replace README's <!-- RESULTS:START/END --> block with the rendered cross-model template."""
    block = render_template((ROOT / "reports" / "readme_results_template.md").read_text(encoding="utf-8"), fig_prefix="reports/figures/")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda _: f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", readme, flags=re.S)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


def build_paper() -> None:
    out = render_template((ROOT / "paper" / "template.md").read_text(encoding="utf-8"))
    md = ROOT / "paper" / "paper.md"
    md.write_text(out, encoding="utf-8")
    pdf = ROOT / "paper" / "paper.pdf"
    cmd = ["pandoc", str(md), "-f", "markdown+lists_without_preceding_blankline", "-o", str(pdf), "--pdf-engine=xelatex",
           "--resource-path", str(ROOT / "paper"), "-V", "colorlinks=true", "--include-in-header", str(ROOT / "paper" / "header.tex"),
           "-V", "geometry:margin=2cm", "-V", "mainfont=Cambria Math", "-V", "monofont=Consolas", "-V", "fontsize=10pt"]  # Cambria Math covers ρ Δ ⊃ ℕ ↔ …
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT / "paper")
    if r.returncode != 0:
        log.warning("pandoc failed (paper.md still written): %s", r.stderr[-2000:])
    else:
        log.info("paper built: %s", pdf)
