"""Tests for prompt rendering, leakage flags, glossary parsing and family config integrity."""

import numpy as np

from mathgeom.config import load_yaml
from mathgeom.data.flags import head_noun, is_eponym, leaks_field
from mathgeom.data.wiki_glossary import clean_term, extract_terms
from mathgeom.pipeline import family_items
from mathgeom.prompts.templates import FIELD_NEUTRAL, TEMPLATE_SETS, render


def test_templates_end_with_concept_and_never_name_a_subfield():
    fields = load_yaml("fields.yaml")
    roots = {r for f in fields["math"] for r in f["root_words"]} - {"set", "real", "complex", "number", "random", "algebra"}
    for t in FIELD_NEUTRAL:
        prefix, full = render(t, "compact space", "mathematics")
        assert full.endswith(" compact space") and full.startswith(prefix)
        assert not any(r in prefix.lower() for r in roots), (t, prefix)
    assert all(t.endswith("{X}") for ts in TEMPLATE_SETS.values() for t in ts)


def test_leak_and_eponym_flags():
    assert leaks_field("topological space", ["topolog"])
    assert leaks_field("group action", ["group"])
    assert not leaks_field("compact", ["topolog"])
    assert not leaks_field("string", ["ring"])  # 'ring' must match at a word start
    assert is_eponym("Hausdorff space") and not is_eponym("compact space")
    assert head_noun("Banach spaces") == "space"


def test_clean_term_and_glossary_parsing():
    assert clean_term("Compact (topology)[1]") == "compact"
    assert clean_term("T1") is None and clean_term("x") is None
    html = "<div class='mw-parser-output'><dl><dt>Open set</dt><dd>..</dd><dt>CW</dt></dl></div>"
    assert extract_terms(html, is_list_page=False) == ["open set"]


def test_family_config_integrity():
    fams = load_yaml("families.yaml")
    for name, f in fams["chains"].items():
        if "reference" in f:
            assert len(fams["references"][f["reference"]]["items"]) == len(f["items"]), name
    for lat in fams["lattices"].values():
        bits = np.array(list(lat["items"].values()))
        assert bits.shape[1] == len(lat["axioms"])
        assert len({tuple(b) for b in bits}) == len(bits)  # every structure has a distinct axiom vector
    assert len(family_items(fams)) > 100


def test_causal_mentions_are_located_after_their_anchors():
    """A bare search for " ring" would hit the "ring" inside "commutative ring" (the first mention)."""
    from mathgeom.analyses.causal import TEMPLATE, _mention_starts

    for a, b in (("commutative ring", "ring"), ("algebraically closed field", "field"), ("ring", "commutative ring")):
        prompt = TEMPLATE.format(D="abstract algebra", A=a, B=b)
        sa, sb = _mention_starts(prompt, a, b)
        assert prompt[sa + 1: sa + 1 + len(a)] == a and prompt[sb + 1: sb + 1 + len(b)] == b and sb > sa + len(a)
