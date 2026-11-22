"""Leakage / confound flags for concepts in the field map (H-A1).

Global context: a field map is trivial if concept names reveal their field lexically
("topological space" in topology, "group action" in group theory) or cluster by eponym
(Hausdorff, Banach). Flagged concepts are excluded from the PRIMARY analysis and kept for a
sensitivity analysis. Head nouns feed a nuisance RDM (all "...space" terms share a last token).
"""

from __future__ import annotations

import re


def leaks_field(term: str, root_words: list[str]) -> bool:
    """True if a word of the name STARTS with a root word of the concept's own field (case-insensitive):
    "topological space" and "group action" match "topolog" and "group", but compounds such as "subgroup" or
    "submodule" do not, because the root must follow the start of the name or a non-letter. This is the frozen
    `leak_own` rule; the preregistration describes it more broadly as "the name contains its own field's root word"
    (second independent audit, docs/deviations.md 2026-11-22; see leaks_field_anywhere)."""
    t = term.lower()
    return any(re.search(rf"(^|[^a-z]){re.escape(r)}", t) for r in root_words)


def leaks_field_anywhere(term: str, root_words: list[str]) -> bool:
    """True if a root word of the concept's own field occurs anywhere in its name, inside compounds too
    ("subgroup" contains "group"). Used only by the post-hoc `recheck` sensitivity analysis; it also matches
    unrelated words ("symmetric" contains "metric"), which configs/flag_audit.yaml lists as false positives."""
    t = term.lower()
    return any(r in t for r in root_words)


def leaks_any_math_field(term: str, all_roots: list[str]) -> bool:
    """True if the name contains ANY field's root word (stricter variant for sensitivity)."""
    return leaks_field(term, all_roots)


def is_eponym(term: str) -> bool:
    """Heuristic: a capitalized token anywhere (terms are stored lowercase-initial) ⇒ proper name."""
    return any(w[:1].isupper() for w in term.split())


def head_noun(term: str) -> str:
    """Last word, lowercased, crude plural stripping ('spaces' → 'space')."""
    w = term.split()[-1].lower()
    return w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w
