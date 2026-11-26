"""Cross-check the current rendered claims, coordinates, matches and paper copies."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from mathgeom.config import ROOT
from mathgeom.report import paper


def check(results):
    def text(path):
        return (ROOT/path).read_text(encoding="utf-8")
    site = json.loads(text("site/data/index.json"))
    ml = results["mathlib"]
    assert site["results"]["mathlib"] == ml["reviewed"]
    assert site["audit"]["multiplicity"] == results["multiplicity"]
    retained = {v["concept"]:v["checked_depth"] for v in ml["dispositions"] if v["decision"] == "retain"}
    assert {v["concept"]:v["depth"] for v in site["mathlib_points"]} == retained
    rows = {r["test"]:r for r in site["scorecard"]["rows"]}
    assert any("†" in r["cells"][0] for k,r in rows.items() if k.startswith("Ring chain ordered"))
    for name in ("summary","readme_top","readme_results","data_card_results"):
        prefix = "figures/" if name == "summary" else "reports/figures/"
        expected = paper.render_template(text(f"reports/{name}_template.md"), fig_prefix=prefix).strip()
        dest = "reports/summary.md" if name == "summary" else "docs/data_card.md" if name == "data_card_results" else "README.md"
        assert expected in text(dest), name
    review = "reports/reviews/2026-11-26_scientific_audit"
    expected = paper.render_template(text(review+"_template.md"), fig_prefix="../figures/")
    assert expected == text(review+".md")
    assert paper.render_template(text("paper/template.md"), fig_ext=".pdf") == text("paper/paper.md")
    for link in re.findall(r"\]\((\.\.?/[^)#]+)", expected):
        assert (ROOT/Path(review).parent/link).exists(), link
    # Inspect the distributed PDF itself. Two identical copies could both be stale.
    plain = subprocess.check_output(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(ROOT / "paper/paper.pdf"), "-"]
    ).decode("utf-8").replace("‑", "-")
    plain = " ".join(plain.split())
    ring = results["multiplicity"]["ring_chain"]
    assert f"Holm value is {ring['published']:.4f} in the published calculation and {ring['reserved']:.4f}" in plain
    assert "two incomparable minimal upper bounds" in plain
    assert "also failed their full-space positive control" in plain
    assert f"Retaining {ml['reviewed']['n_matched']} sense matches gave ρ = {ml['reviewed']['rho_mean_rating_depth']:.3f}" in plain
    for name in ("paper/paper.pdf","reports/paper.pdf"):
        pdf = (ROOT/name).read_bytes()
        assert pdf.startswith(b"%PDF-")
        if name == "paper/paper.pdf":
            digest = hashlib.sha256(pdf).digest()
        else:
            assert hashlib.sha256(pdf).digest() == digest
