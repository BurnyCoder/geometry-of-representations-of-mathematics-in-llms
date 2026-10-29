"""Timestamped logging to console + file, and an uncut JSONL log of every LLM prompt/output.

Global context: the project rule is that every prompt sent to an LLM and every output it
produces is written, in full, to a timestamped log file and to the terminal in real time.
Activation extraction logs each prompt plus the model's greedy next token (a sanity check
that the context is understood); judge models log full prompts and completions.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROMPT_FILE: Path | None = None


def setup_logging(log_dir: Path, run_name: str) -> Path:
    """Configure root logging; return the path of this run's prompt JSONL file."""
    global _PROMPT_FILE
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252; prompts contain Unicode
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"{stamp}_{run_name}.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "filelock"):  # one line per HTTP range request otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _PROMPT_FILE = log_dir / f"{stamp}_{run_name}_prompts.jsonl"
    return _PROMPT_FILE


def log_llm(record: dict, echo: bool = True) -> None:
    """Append one prompt/output record (uncut) to the JSONL log; echo it to the terminal."""
    record = {"time": datetime.now().isoformat(timespec="milliseconds"), **record}
    if _PROMPT_FILE is not None:
        with _PROMPT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if echo:
        logging.getLogger("llm").info(json.dumps(record, ensure_ascii=False))
