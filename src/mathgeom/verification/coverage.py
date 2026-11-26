"""Trace the scientific files in the exact starting revision to evidence or historical provenance."""
from collections import Counter
import json
from pathlib import PurePosixPath
import subprocess

from mathgeom.config import ROOT

BASE = "43dcda70a7f6093e946fbc975c88cd65a7d12e5f"


def inventory():
    names = subprocess.check_output(["git","ls-tree","-r","--name-only",BASE], cwd=ROOT, text=True).splitlines()
    out = {}
    for name in names:
        p = PurePosixPath(name)
        area, stem = p.parts[0], p.stem
        if area == "results":
            # Byte-level comparison against Git would depend on working-tree newline conversion.
            raw = subprocess.check_output(["git","show",BASE+":"+name], cwd=ROOT)
            if json.loads(raw) != json.loads((ROOT/name).read_text(encoding="utf-8")):
                raise ValueError(f"Original numerical record changed: {name}")
            check, evidence = "historical numerical record, parsed and compared with starting revision", "results/scientific_audit/verification.json"
        elif area == "data":
            check, evidence = "archived study input", "data/wiki_revisions.json; data/field_distance.json; data/judge_ratings.csv; configs/mathlib_review.tsv"
        elif area == "configs":
            check, evidence = "design, literal mappings or run configuration", "docs/prereg.md; results/mvp/manifest.json; configs/mathlib_review.tsv"
        elif area == "reports" and len(p.parts)>1 and p.parts[1]=="figures":
            if stem.startswith(("field","map","depth")): source = "maps.py; overview.py"
            elif stem.startswith(("axiom","lattice")): source = "lattice.py"
            elif stem.startswith("abstraction"): source = "axes.py"
            elif stem in ("forest","hero","group_objects","partial_operation"): source = "overview.py"
            else: source = "families.py"
            check, evidence = "display projection or plotted statistic; generator and paired numeric record", "src/mathgeom/report/figures/"+source+"; results/"
        elif area in {"reports","docs","paper","README.md"}:
            if "prereg" in stem or stem in {"deviations","experiments","audit_lexical","audit_template"} or "2026-11-22" in stem:
                check, evidence = "historical specification or retrospective report; revision and superseding review identified", "git:"+BASE+"; reports/reviews/2026-11-26_scientific_audit.md"
            else:
                check, evidence = "publication source or rendered result", "src/mathgeom/report/paper.py; src/mathgeom/report/render.py; reports/reviews/2026-11-26_scientific_audit.md"
        elif area == "site":
            check, evidence = "explorer labels, statistics or saved display coordinates", "src/mathgeom/report/site.py; tests/test_site.py; results/"
        elif area in {"src","tests","run.py"}:
            check, evidence = "implementation, model interface or regression specification; static inspection and applicable numerical checks", "docs/prereg.md; src/mathgeom/verification/; reports/reviews/2026-11-26_scientific_audit.md"
        else:
            check, evidence = "environment, deployment or project metadata", "uv.lock; .github/workflows/pages.yml"
        out[name] = {"disposition": check, "evidence": evidence}
    from mathgeom.config import load_config
    from mathgeom.verification.records import extraction_order
    import pandas as pd
    traces = extraction_order(load_config(ROOT / "configs/mvp.yaml").artifacts, pd.read_csv(ROOT / "data/concepts.csv"))
    return {"field_extraction_logs": traces, "examined_revision": BASE, "counts": dict(Counter(v["disposition"] for v in out.values())), "files":out}
