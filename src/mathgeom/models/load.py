"""Load trained or randomly initialized causal LMs (text-only) with HF transformers v5.

Global context: every trained model is paired with a random-init copy of the same
architecture and tokenizer — the control Golechha et al. (ICLR 2025) showed is necessary,
because high-dimensional geometry can look "structured" even without training. Both see the
same prompts, so any effect also present in the random-init model is attributable to
tokenization/architecture, not learned knowledge.

Local context: multimodal checkpoints (Qwen3.5, Gemma 4) are loaded via
AutoModelForCausalLM, which instantiates the text-only causal-LM class (as in the Qwen-Scope
model card example: https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50).
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def load_model(model_id: str, random_init: bool = False, seed: int = 0, dtype=torch.bfloat16, device_map="cuda"):
    """Return (model, tokenizer) in eval mode. `random_init` keeps the config but re-draws weights."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if random_init:
        cfg = AutoConfig.from_pretrained(model_id)
        torch.manual_seed(seed)  # deterministic random weights
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        model = model.to("cuda" if device_map == "cuda" else "cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("loaded %s (random_init=%s) class=%s params=%.2fB", model_id, random_init, type(model).__name__, n_params / 1e9)
    return model, tok
