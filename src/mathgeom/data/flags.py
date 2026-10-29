"""Leakage / confound flags for concepts in the field map (H-A1).

Global context: a field map is trivial if concept names reveal their field lexically
("topological space" in topology, "group action" in group theory) or cluster by eponym
(Hausdorff, Banach). Flagged concepts are excluded from the PRIMARY analysis and kept for a
sensitivity analysis. Head nouns feed a nuisance RDM (all "...space" terms share a last token).
"""

from __future__ import annotations

import re


def leaks_field(term: str, root_words: list[str]) -> bool:
    """True if any root word of the concept's own field occurs in its name (case-insensitive)."""
    t = term.lower()
    return any(re.search(rf"(^|[^a-z]){re.escape(r)}", t) for r in root_words)


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
