"""Evaluate the fixed, sense-reviewed crosswalk against pinned glossary and Lean sources."""
from pathlib import Path
import re
from functools import lru_cache
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from mathgeom.config import ROOT


def reviewed_rows():
    return pd.read_csv(ROOT / "configs/mathlib_review.tsv", sep="\t").set_index("concept", verify_integrity=True)


def reviewed_frame():
    """Original matched depths remain in the source table; the reviewed view masks excluded candidates."""
    full = pd.read_csv(ROOT / "data/mathlib_depth.csv")
    mask = full.depth.notna()
    rows = reviewed_rows()
    if set(full.loc[mask, "concept"]) != set(rows.index) or not set(rows.decision) <= {"retain", "exclude"}:
        raise ValueError("The semantic review must cover exactly the matched concepts")
    excluded = full.concept.isin(rows.index[rows.decision == "exclude"])
    full.loc[excluded, ["decl", "module", "depth", "source"]] = np.nan
    return full


def module_depths(root, modules):
    """Read the import closure of the candidate modules. Skip nested Lean header comments."""
    @lru_cache(maxsize=None)
    def depth(module):
        path = root.joinpath(*module.split(".")).with_suffix(".lean")
        imports, nesting = [], 0
        with path.open(encoding="utf-8") as source:
            for line in source:
                visible = []
                for token in re.split(r"(/-|-\/|--)", line):
                    if token == "/-":
                        nesting += 1
                    elif token == "-/" and nesting:
                        nesting -= 1
                    elif token == "--" and not nesting:
                        break
                    elif not nesting:
                        visible.append(token)
                command = "".join(visible).strip()
                if not command or command in {"module", "prelude"}:
                    continue
                match = re.fullmatch(r"(?:(?:public|meta)\s+)*import\s+(.+)", command)
                if not match:
                    break
                imports.extend(m for m in match[1].split() if m.startswith("Mathlib."))
        return max((1 + depth(dep) for dep in imports), default=0)

    return {m: depth(m) for m in modules}


def assess(root, original):
    rev = original["mathlib_rev"]
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == rev
    full = pd.read_csv(ROOT / "data/mathlib_depth.csv").merge(pd.read_csv(ROOT / "data/concepts.csv")[["concept", "field", "revid", "zipf"]], on=["concept", "field"], validate="one_to_one")
    candidates = full.dropna(subset="depth").set_index("concept").join(reviewed_rows(), validate="one_to_one")
    assert len(candidates) == original["n_matched"]
    depths = module_depths(root, candidates.module.unique())
    links = []
    for name, row in candidates.iterrows():
        path = Path(*row.module.split(".")).with_suffix(".lean")
        lines = (root / path).read_text(encoding="utf-8").splitlines()
        short = re.escape(row.decl.split(".")[-1])
        decl = re.compile(r"\b(?:class|def|structure|abbrev|theorem|lemma|inductive|irreducible_def)\s+(?:[\w.]+\.)?" + short + r"(?=[\s:({\[]|$)")
        line = next(i for i, s in enumerate(lines, 1) if decl.search(s))
        links.append({"concept": name, "decision": row.decision, "reason": row.reason, "field": row.field,
                      "original_depth": int(row.depth), "checked_depth": depths[row.module],
                      "glossary": f"https://en.wikipedia.org/w/index.php?oldid={row.revid}",
                      "declaration": f"https://github.com/leanprover-community/mathlib4/blob/{rev}/{path.as_posix()}#L{line}"})
    valid = reviewed_frame().merge(full[["concept", "zipf"]], on="concept", validate="one_to_one")
    m = valid.dropna(subset="depth").merge(pd.read_csv(ROOT / "data/judge_ratings.csv"), on=["concept", "field"], validate="one_to_one")
    def rho(a, b):
        return float(spearmanr(a, b).statistic)
    m["depth"] = m.module.map(depths)
    observed = rho(m.mean_rating, m.depth)
    rng = np.random.default_rng(0)
    indices = rng.integers(len(m), size=(2000, len(m)))
    a = rankdata(m.mean_rating.to_numpy()[indices], axis=1)
    b = rankdata(m.depth.to_numpy()[indices], axis=1)
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    boot = (a*b).sum(1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
    z = np.c_[np.ones(len(m)), m.zipf]
    residual = lambda y: y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    reviewed = dict(original, n_matched=len(m), n_by_source=m.source.value_counts().to_dict(), depth_median=float(m.depth.median()),
                    rho_mean_rating_depth=observed, p=float(spearmanr(m.mean_rating, m.depth).pvalue),
                    ci95=np.quantile(boot, [.025,.975]).tolist(), rho_partial_zipf=rho(residual(m.mean_rating), residual(m.depth)),
                    rho_per_judge={j: rho(m[j], m.depth) for j in original["rho_per_judge"]},
                    coverage_by_field=valid.assign(hit=valid.depth.notna()).groupby("field").hit.mean().round(3).to_dict())
    reproduction = rho(candidates.mean_rating, candidates.depth)
    np.testing.assert_allclose(reproduction, original["rho_mean_rating_depth"], atol=1e-12)
    return {"original_n": len(candidates), "original_rho": reproduction, "original_ci": original["ci95"],
            "reviewed": reviewed, "change_in_rho": observed-reproduction, "retained_fields": int(m.field.nunique()),
            "depth_disagreements": sum(d["checked_depth"] != d["original_depth"] for d in links), "dispositions": links}
