"""Build the concept table for the field map (H-A1) from pinned Wikipedia pages.

Global context: one row per concept with its field, kind (math/control), pure/applied label,
leakage flags, head noun, frequency covariate and pilot/confirmatory split. Concepts that
appear in more than one field's pages are dropped (ambiguous field membership). Each field is
capped at `max_per_field` concepts sampled with a fixed seed AFTER flagging, so the primary
(unflagged) set stays balanced across fields.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from mathgeom.config import Config, load_yaml
from mathgeom.data.flags import head_noun, is_eponym, leaks_field
from mathgeom.data.wiki_glossary import extract_terms, fetch_page

log = logging.getLogger(__name__)


def build_concepts(cfg: Config) -> pd.DataFrame:
    """Fetch pages (pinned revisions), extract terms, flag, dedupe, sample, split; save CSV."""
    spec = load_yaml("fields.yaml")
    raw_dir, lock = cfg.data_dir / "raw" / "wiki", cfg.data_dir / "wiki_revisions.json"
    rows, owners = [], defaultdict(set)
    all_roots = sorted({r for f in spec["math"] for r in f["root_words"]})
    for kind in ("math", "control"):
        for f in spec[kind]:
            for page in f["pages"]:
                revid, html = fetch_page(page, raw_dir, lock)
                for term in extract_terms(html, is_list_page=page.startswith("List of")):
                    owners[term.lower()].add(f["id"])
                    rows.append({"concept": term, "field": f["id"], "kind": kind, "pa": f.get("pa", "control"), "page": page, "revid": revid,
                                 "leak_own": leaks_field(term, f.get("root_words", [])),
                                 "leak_any": leaks_field(term, all_roots) if kind == "math" else False,
                                 "eponym": is_eponym(term)})
    df = pd.DataFrame(rows).drop_duplicates(["concept", "field"])
    df["multi_field"] = df["concept"].str.lower().map(lambda c: len(owners[c]) > 1)
    df = df[~df["multi_field"]].copy()  # ambiguous membership → dropped entirely
    df["head"] = df["concept"].map(head_noun)
    df["zipf"] = df["concept"].map(lambda c: zipf_frequency(c, "en"))  # Zipf scale: log10(freq per billion words)
    # polysemy proxy: a single everyday word (Zipf ≥ 4, e.g. "mean", "walk", "state") is dominated by
    # its non-math senses; flagged and kept out of the primary set (pre-registered rule, set before
    # any activation was analyzed)
    df["polysemous"] = (df["concept"].str.split().str.len() == 1) & (df["zipf"] >= 4.0)
    df["primary"] = ~(df["leak_own"] | df["eponym"] | df["polysemous"])

    rng = np.random.default_rng(cfg.seed)
    keep = []
    for fid, g in df.groupby("field", sort=False):  # cap per field: primary first, then flagged extras
        prim, flag = g[g.primary], g[~g.primary]
        keep.append(prim.sample(min(len(prim), cfg.max_concepts_per_field), random_state=int(rng.integers(1 << 31))))
        keep.append(flag.sample(min(len(flag), cfg.max_concepts_per_field // 3), random_state=int(rng.integers(1 << 31))))
    out = pd.concat(keep).reset_index(drop=True)
    out["split"] = np.where(rng.random(len(out)) < cfg.pilot_fraction, "pilot", "confirmatory")
    path = cfg.data_dir / "concepts.csv"
    out.to_csv(path, index=False)
    log.info("concepts: %d rows (%d primary) across %d fields → %s", len(out), out.primary.sum(), out.field.nunique(), path)
    log.info("per-field primary counts:\n%s", out[out.primary].groupby("field").size().to_string())
    return out
