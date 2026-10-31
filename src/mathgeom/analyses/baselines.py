"""Non-LLM and non-generative baselines for the field map (plan: fastText, sentence embeddings) and
an infini-gram frequency covariate (plan: covariates).

Global context: Karkada et al. 2026 (arXiv:2602.15029) argue that representation geometry follows
co-occurrence statistics, so the natural question is whether a causal LM's field map is any
stronger than what static distributional vectors or a contrastively trained text embedder give.
  * fastText wiki-news-300d-1M (Mikolov et al. 2018, arXiv:1712.09405; CC BY-SA 3.0,
    https://fasttext.cc/docs/en/english-vectors.html): concept vector = mean of its word vectors
    (concepts with no in-vocabulary word are dropped and the coverage is reported).
  * Qwen3-Embedding-0.6B (Jun 2025, Apache-2.0): last-token pooling with left padding and L2
    normalization, as in the model card https://huggingface.co/Qwen/Qwen3-Embedding-0.6B; the bare
    concept name is embedded (documents need no instruction per the card).
  * infini-gram counts (Liu et al. 2024, arXiv:2401.17377; API https://infini-gram.readthedocs.io/en/latest/api.html, index v4_dolma-v1_7_llama):
    n-gram count of each concept name in Dolma v1.7, as a pretraining-scale frequency covariate.
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger(__name__)


def fasttext_vectors(zip_path: Path, concepts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean fastText word vector per concept → (vectors n×300, mask of concepts with ≥1 known word)."""
    words = {w for c in concepts for w in c.replace("-", " ").split()}
    words |= {w.lower() for w in words}
    table: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
            if i == 0:
                continue  # header: "n_words dim"
            w, rest = line.split(" ", 1)
            if w in words:
                table[w] = np.asarray(rest.split(), dtype=np.float32)
    out, ok = np.zeros((len(concepts), 300), dtype=np.float32), np.zeros(len(concepts), dtype=bool)
    for k, c in enumerate(concepts):
        vs = [table.get(w, table.get(w.lower())) for w in c.replace("-", " ").split()]
        vs = [v for v in vs if v is not None]
        if vs:
            out[k], ok[k] = np.mean(vs, 0), True
    log.info("fastText coverage: %d/%d concepts", ok.sum(), len(concepts))
    return out, ok


def qwen3_embeddings(concepts: list[str], model_id: str = "Qwen/Qwen3-Embedding-0.6B", batch: int = 32) -> np.ndarray:
    """L2-normalized last-token embeddings (model-card recipe)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    outs = []
    with torch.inference_mode():
        for i in range(0, len(concepts), batch):
            enc = tok(concepts[i : i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state[:, -1].float()  # left padding → last position is the final token
            outs.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(outs)


def infinigram_counts(concepts: list[str], cache: Path, index: str = "v4_dolma-v1_7_llama") -> dict[str, int]:
    """Exact n-gram counts from the public infini-gram API (cached JSON)."""
    counts = json.loads(cache.read_text()) if cache.exists() else {}
    for c in concepts:
        if c in counts:
            continue
        for attempt in range(5):
            try:
                r = requests.post("https://api.infini-gram.io/", json={"index": index, "query_type": "count", "query": c}, timeout=30)
                js = r.json()
                if "count" not in js:  # e.g. {"error": ...}: retry, then give up loudly
                    raise ValueError(js)
                counts[c] = int(js["count"])
                break
            except Exception as e:  # transient API errors: back off and retry
                log.warning("infini-gram retry %d for %r: %s", attempt, c, e)
                time.sleep(2 * (attempt + 1))
        cache.write_text(json.dumps(counts))
    return counts
