"""Prompt templates. The concept always comes LAST so the final token is the concept's last token.

Global context: representations are template-averaged residual-stream states at the
concept's last token (and, as a variant, mean-pooled over the concept's tokens). Templates
contain a domain slot {D} that is CONSTANT across all compared items ("mathematics" for all
math concepts, "science" for the non-math control, a fixed family domain for H-B/H-B2), so
the domain word cannot create differences between items. Templates never name a sub-field,
which would leak the label H-A1 tries to predict; a separate, explicitly labelled
"field_revealing" condition measures how much such leakage would inflate the effect.
Templates 0–3 and 4–7 form the two halves for split-half reliability (noise ceiling).
"""

FIELD_NEUTRAL = [
    "In {D}, the notion of {X}",
    "A {D} textbook defines {X}",
    "In {D}, the term {X}",
    "A lecture on {D} introduced {X}",
    "In {D}, one important concept is {X}",
    "Every student of {D} learns about {X}",
    "In the language of {D}, {X}",
    "A {D} glossary entry: {X}",
]

# Secondary condition only (H-A1 leakage check): {F} = human-readable field name.
FIELD_REVEALING = [t.replace("{D}", "{F}") for t in FIELD_NEUTRAL]

FAMILY = [
    "In {D}, consider {X}",
    "A lecture on {D} discussed {X}",
    "In {D}, the term {X}",
    "An example from {D}: {X}",
    "In the context of {D}, {X}",
    "A note on {D} mentions {X}",
    "Within {D}, we study {X}",
    "A textbook on {D} explains {X}",
]

# Robustness variant for families: no domain context at all.
NO_DOMAIN = [
    "Consider {X}",
    "Here is a word: {X}",
    "The term {X}",
    "An example: {X}",
    "Think about {X}",
    "A note mentions {X}",
    "We study {X}",
    "A book explains {X}",
]

TEMPLATE_SETS = {"field_neutral": FIELD_NEUTRAL, "field_revealing": FIELD_REVEALING, "family": FAMILY, "no_domain": NO_DOMAIN}


def render(template: str, concept: str, domain: str = "", field: str = "") -> tuple[str, str]:
    """Return (prefix, full_prompt). prefix is everything before the concept (no trailing space).

    The concept is joined with a single space so that BPE tokenizers attach the space to the
    concept's first token, making the prefix tokenization a prefix of the full tokenization.
    """
    head = template.split("{X}")[0].format(D=domain, F=field).rstrip()
    return head, f"{head} {concept}"
