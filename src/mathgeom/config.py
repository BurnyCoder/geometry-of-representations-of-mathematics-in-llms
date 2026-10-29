"""Configuration: .env (paths, tokens) + YAML run configs (models, layers, seeds).

Global context: every phase in run.py receives one `Config`; artifacts (activations,
results) go under `artifacts/` (a junction to D:\\AI on the author's machine, because C: is
nearly full) and small text outputs under `results/` and `reports/` inside the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]  # repo root (src/mathgeom/config.py → ../..)


@dataclass
class Config:
    name: str
    models: list[dict]  # [{id: hf repo, key: short name, random_init: bool, ...}]
    n_perm: int = 10_000
    n_boot: int = 2_000
    seed: int = 0
    max_concepts_per_field: int = 30
    pilot_fraction: float = 0.2
    primary_rel_depth: float = 0.5
    phases: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def artifacts(self) -> Path:
        return Path(os.environ.get("MATHGEOM_ARTIFACTS", ROOT / "artifacts"))

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def results_dir(self) -> Path:
        return ROOT / "results" / self.name


def load_config(path: str | Path) -> Config:
    """Read .env (if present) then the YAML config; unknown YAML keys are kept in `raw`."""
    load_dotenv(ROOT / ".env")
    raw = yaml.safe_load(Path(path).read_text())
    known = {k: raw[k] for k in Config.__dataclass_fields__ if k in raw and k != "raw"}
    return Config(**known, raw=raw)


def load_yaml(name: str) -> dict:
    """Load a YAML file from configs/."""
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
