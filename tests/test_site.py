"""The explorable site reads only generated data: schema of site/data/*.json, every data-num path in
index.html resolves, and the page text contains no hand-typed decimal numbers."""

import json
import re

import numpy as np
import pytest

from mathgeom.config import ROOT

SITE = ROOT / "site"


def _get(obj, path):
    """Dotted lookup where keys may contain dots (model names): longest key first, as site/app.js does."""
    parts = path.split(".")
    while parts:
        for j in range(len(parts), 0, -1):
            k = ".".join(parts[:j])
            if isinstance(obj, dict) and k in obj:
                obj, parts = obj[k], parts[j:]
                break
            if isinstance(obj, list) and j == 1 and k.isdigit():
                obj, parts = obj[int(k)], parts[1:]
                break
        else:
            raise KeyError(path)
    return obj


@pytest.mark.skipif(not (SITE / "data" / "index.json").exists(), reason="site data not generated")
def test_site_data_schema_and_page_numbers():
    idx = json.loads((SITE / "data" / "index.json").read_text(encoding="utf-8"))
    assert {"models", "fields", "concepts", "scorecard", "forest", "results", "mathlib_points"} <= set(idx)
    n_fields = len(idx["fields"]["names"])
    for m in idx["models"]:
        path = SITE / "data" / f"{m['key']}.json"
        assert path.stat().st_size < 1.5e6
        d = json.loads(path.read_text())
        assert len(d["fields"]["trained"]["layout"]) == m["n_layers"] and len(d["fields"]["trained"]["layout"][0]) == n_fields
        L = m["n_layers"]
        for side in ("trained", "random_init"):  # 3-D views: one k-column layout per layer, plus how much each shows
            F = d["fields"][side]
            assert len(F["layout3"]) == L and np.array(F["layout3"][0]).shape == (n_fields, 3) and len(F["stress"]) == L
            C = d["concepts"][side]
            assert np.array(C["pc5"]).shape == (len(idx["concepts"]["concept"]), 5) and len(C["var"]) == 5
            for f, fam in d["families"].items():
                assert np.array(fam["pc5"][side]).shape == (L, len(fam["items"]), 5), f
                assert np.array(fam["ideal5"][side]).shape == (L, len(fam["items"]), 5) and np.array(fam["var"][side]).shape == (L, 5), f
        lt = d["lattice"]
        assert len(lt["trained"]["exact"]) == len(lt["items"]) and len(lt["trained"]["cos"]) == len(lt["pairs"])
        for side in ("trained", "random_init"):
            # the analogy point is exact: exported PC-basis coordinates map onto the view at the preregistered layer
            E, M, V = np.array(lt[side]["exact"]), np.array(lt[side]["to_view"]), np.array(lt[side]["pc5"][m["primary_layer"] - 1])
            np.testing.assert_allclose(E[:, :5] @ M, V, atol=2e-3)  # 3-decimal rounding of the exported view
    assert np.array(idx["fields"]["arxiv_layout3"]).shape == (n_fields, 3)
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for path in re.findall(r'data-num="([^"]+)"', html):
        assert isinstance(_get(idx, path), (int, float)), path
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"Qwen\d\.\d", "Qwen", text)  # model names, not results
    assert not re.findall(r"\d\.\d", text), "numbers in the page must come from data/index.json"


def test_links_into_the_site_resolve():
    """Every #anchor that README, summary or paper uses to point into the site exists in index.html."""
    html = (SITE / "index.html").read_text(encoding="utf-8")
    sources = ["README.md", "reports/summary_template.md", "reports/readme_top_template.md", "reports/readme_results_template.md", "paper/template.md"]
    anchors = {a for p in sources for a in re.findall(r"mathematics-in-llms/#([\w-]+)", (ROOT / p).read_text(encoding="utf-8"))}
    assert anchors and not [a for a in anchors if f'id="{a}"' not in html]


def test_script_targets_exist_in_page():
    """Every figure app.js registers and every element it looks up by id exists in index.html (ids the
    script creates itself, such as the per-facet forest charts and table views, are excluded)."""
    html, js = (SITE / "index.html").read_text(encoding="utf-8"), (SITE / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([\w-]+)"', html))
    created = set(re.findall(r'id="([\w-]+)"', js)) | set(re.findall(r'data-tableid="([\w-]+)"', html))
    registered = set(re.findall(r'register\("([\w-]+)"', js))
    looked_up = set(re.findall(r'\$\("([\w-]+)"\)', js))
    assert registered and registered <= ids
    assert not (looked_up - ids - created), looked_up - ids - created
