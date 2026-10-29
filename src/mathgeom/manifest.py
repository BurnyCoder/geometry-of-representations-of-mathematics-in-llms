"""Run manifest: which code, config, seed and library versions produced each results file.

Global context: every `run.py <phase>` appends an entry to results/<cfg>/manifest.json, so any
number in the reports can be traced to a git commit (and whether the tree was dirty), the exact
config contents (SHA-256), the seed, and the versions of the numerical libraries.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mathgeom.config import ROOT, Config

_PKGS = ("torch", "transformers", "numpy", "scipy", "scikit-learn", "pandas", "statsmodels")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def record(cfg: Config, phase: str, config_path: str) -> None:
    """Append one manifest entry for `phase` to results/<cfg>/manifest.json."""
    def ver(p: str) -> str:
        try:
            return version(p)
        except PackageNotFoundError:
            return "not installed"

    entry = {"phase": phase, "time": datetime.now().isoformat(timespec="seconds"), "git_sha": _git("rev-parse", "HEAD"),
             "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
             "config": config_path, "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(), "seed": cfg.seed,
             "packages": {p: ver(p) for p in _PKGS}}
    path = cfg.results_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(path.read_text()) if path.exists() else []
    runs.append(entry)
    path.write_text(json.dumps(runs, indent=1))
