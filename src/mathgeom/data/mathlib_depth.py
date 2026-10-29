"""Mathlib definitional depth: a non-LLM abstraction proxy used to validate the judge ratings (H-A4).

Global context: the plan validates LLM-judge abstraction ratings against Mathlib, the Lean 4
mathematics library (https://github.com/leanprover-community/mathlib4, Apache-2.0). A concept is
matched to a Lean declaration, the declaration to the module (file) that defines it, and the
module to its *import depth*: the length of the longest chain of Mathlib imports below it
(0 = imports nothing from Mathlib). A deeper module is built on a longer tower of prior Mathlib
definitions. This is a module-level proxy; declaration-level dependency graphs (Li et al. 2026,
"The Network Structure of Mathlib", arXiv:2604.24797) need a compiled Mathlib.

Matching, in order (docs/deviations.md, second audit):
  1. Mathlib's curated English-name maps docs/overview.yaml and docs/undergrad.yaml
     (e.g. "normal subgroup: 'Subgroup.Normal'");
  2. multi-word names only: an exact CamelCase match to a structure/class/def/abbrev/inductive
     name, with or without an "Is" prefix ("compact space" → CompactSpace), kept only if unique.
"""

from __future__ import annotations

import re
from graphlib import TopologicalSorter  # stdlib topological order (Python ≥ 3.9)
from pathlib import Path

import yaml

# a header line `[public] [meta] import Mathlib.X.Y` (Lean 4 module system, https://lean-lang.org/doc/reference/latest/Source-Files-and-Modules/)
IMPORT = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(.+)$")
# column-0 declaration, optionally preceded by attributes and modifiers
DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)*(?:(?:noncomputable|private|protected|public|unsafe|partial|nonrec)\s+)*"
                  r"(structure|class inductive|class|def|abbrev|inductive|theorem|lemma|opaque|irreducible_def)\s+([^\s:({\[]+)")
TYPE_KINDS = {"structure", "class", "class inductive", "def", "abbrev", "inductive", "irreducible_def"}


def _module(root: Path, f: Path) -> str:
    return ".".join(f.relative_to(root).with_suffix("").parts)


def module_imports(root: Path) -> dict[str, list[str]]:
    """{module: [imported Mathlib modules]} from the header of every .lean file under root/Mathlib."""
    out = {}
    for f in (root / "Mathlib").rglob("*.lean"):
        text = re.sub(r"^\s*/-.*?-/", "", f.read_text(encoding="utf-8"), count=1, flags=re.S)  # leading copyright comment
        imps = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("--") or s in ("module", "prelude"):
                continue
            m = IMPORT.match(s)
            if not m:
                break  # header ends at the first non-import line
            imps += [t for t in m.group(1).split() if t.startswith("Mathlib.")]
        out[_module(root, f)] = imps
    return out


def import_depth(imports: dict[str, list[str]]) -> dict[str, int]:
    """Longest chain of Mathlib imports below each module (dependencies are visited first)."""
    depth: dict[str, int] = {}
    for mod in TopologicalSorter(imports).static_order():
        deps = [depth[d] for d in imports.get(mod, []) if d in depth]
        depth[mod] = 1 + max(deps) if deps else 0
    return depth


def declarations(root: Path) -> list[tuple[str, str, str]]:
    """(full name, kind, module) for every column-0 declaration, tracking `namespace … end`."""
    out = []
    for f in (root / "Mathlib").rglob("*.lean"):
        mod, stack = _module(root, f), []  # stack of (kind, name) for namespace/section scopes
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("namespace "):
                stack.append(("namespace", line.split()[1]))
            elif re.match(r"^(?:noncomputable\s+)?section\b", line):
                parts = line.split()
                stack.append(("section", parts[-1] if parts[-1] != "section" else ""))
            elif line.startswith("end") and (len(line) == 3 or line[3] == " "):
                name = line[3:].strip()
                while stack:  # `end X` closes the scope named X; bare `end` closes the innermost one
                    k, n = stack.pop()
                    if not name or n == name:
                        break
            elif m := DECL.match(line):
                kind, name = m.groups()
                if name.startswith("_root_."):
                    full = name[len("_root_."):]
                else:
                    full = ".".join([n for k, n in stack if k == "namespace"] + [name])
                out.append((full, kind, mod))
    return out


def _flatten(d, out: dict) -> dict:
    """English concept name → declaration name from the nested Mathlib YAML maps ('~' = not formalized)."""
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            _flatten(v, out)
        elif isinstance(v, str) and v.strip() not in ("~", ""):
            out.setdefault(_norm(str(k)), v.strip().split()[0])
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[$'\"`]", "", s.lower().replace("-", " "))).strip()


def camel(concept: str) -> str:
    """'compact space' → 'CompactSpace' (keeps inner capitals such as Noetherian)."""
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[\s\-]+", concept) if w)


def match_concepts(concepts: list[str], yaml_files: list[Path], decls: list[tuple[str, str, str]]) -> dict[str, dict]:
    """concept → {decl, module, source} for every concept that can be matched (see module docstring)."""
    names = {}
    for p in yaml_files:
        _flatten(yaml.safe_load(p.read_text(encoding="utf-8")), names)
    by_full = {}
    by_short: dict[str, set] = {}
    for full, kind, mod in decls:
        by_full.setdefault(full, mod)
        if kind in TYPE_KINDS:
            by_short.setdefault(full.split(".")[-1], set()).add((full, mod))
    out = {}
    for c in concepts:
        n = _norm(c)
        decl = next((names[k] for k in (n, n + "s", n.removesuffix("s")) if k in names), None)
        if decl and decl in by_full:
            out[c] = {"decl": decl, "module": by_full[decl], "source": "mathlib_yaml"}
            continue
        if len(c.split()) < 2:  # single words ("kernel", "ideal") are too ambiguous for name matching
            continue
        cands = set().union(*(by_short.get(x, set()) for x in (camel(c), "Is" + camel(c))))
        if len(cands) == 1:
            full, mod = cands.pop()
            out[c] = {"decl": full, "module": mod, "source": "camelcase"}
    return out
