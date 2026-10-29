"""Extract template-averaged residual-stream representations of concepts at every layer.

Global context: for each concept we run K templates (prompts/templates.py) that END with the
concept, and read `hidden_states` (HF `output_hidden_states=True`: index 0 = embeddings,
index l = output of block l; for most HF decoders the LAST entry is after the final norm).
Two reads are stored per (concept, layer): the concept's last token ("last", primary) and
the mean over the concept's tokens ("pool", variant). Averages over all templates and over
the two template halves (split-half reliability) are saved as float16 arrays
[n_concepts × (n_layers+1) × d_model].

Local context: batch size 1 avoids padding/position-id subtleties across architectures
(hybrid linear-attention Qwen3.5, per-layer-embedding Gemma 4); the whole MVP is ~5k short
prompts, a few minutes on one GPU. Each prompt and the model's greedy next token are logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from mathgeom.logio import log_llm
from mathgeom.prompts.templates import render

log = logging.getLogger(__name__)


@dataclass
class Item:
    concept: str
    domain: str  # fills {D}
    field: str = ""  # fills {F} (field-revealing condition only)


def concept_token_span(tok, full: str, prefix: str) -> list[int]:
    """Indices of tokens overlapping the concept's characters (robust to boundary merges)."""
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=True)
    start = len(prefix) + 1  # concept begins after "prefix" + one space
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and b > a]
    if not span:
        raise ValueError(f"no concept tokens found in {full!r}")
    return span


@torch.inference_mode()
def extract(model, tok, items: list[Item], templates: list[str], tag: str, echo: bool = False) -> dict[str, np.ndarray]:
    """Run all items × templates; return dict of float16 arrays and per-item token counts."""
    dev = next(model.parameters()).device
    half = len(templates) // 2
    sums = {k: None for k in ("last", "last_a", "last_b", "pool")}
    n_tok = np.zeros(len(items), dtype=np.int32)
    for i, it in enumerate(tqdm(items, desc=f"extract[{tag}]")):
        acc = {k: 0.0 for k in sums}
        for t_idx, tpl in enumerate(templates):
            prefix, full = render(tpl, it.concept, it.domain, it.field)
            span = concept_token_span(tok, full, prefix)
            ids = tok(full, return_tensors="pt").input_ids.to(dev)
            out = model(input_ids=ids, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 0)[:, 0].float()  # [L+1, seq, d]
            last = hs[:, span[-1]].cpu().numpy()  # [L+1, d] concept's final token
            pool = hs[:, span].mean(1).cpu().numpy()  # [L+1, d] mean over concept tokens
            acc["last"] = acc["last"] + last
            acc["pool"] = acc["pool"] + pool
            acc["last_a" if t_idx < half else "last_b"] = acc["last_a" if t_idx < half else "last_b"] + last
            nxt = tok.decode(out.logits[0, -1].argmax().item())
            log_llm({"tag": tag, "concept": it.concept, "template": t_idx, "prompt": full, "concept_tokens": len(span), "greedy_next_token": nxt}, echo=echo)
            n_tok[i] = len(span)
        vals = {"last": acc["last"] / len(templates), "pool": acc["pool"] / len(templates), "last_a": acc["last_a"] / half, "last_b": acc["last_b"] / (len(templates) - half)}
        for k, v in vals.items():
            if sums[k] is None:
                sums[k] = np.zeros((len(items),) + v.shape, dtype=np.float16)
            sums[k][i] = v.astype(np.float16)
    sums["n_tokens"] = n_tok
    return sums
