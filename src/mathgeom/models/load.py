"""Load trained or randomly initialized causal LMs (text-only) with HF transformers v5.

Global context: every trained model is paired with a random-init copy of the same
architecture and tokenizer — the control Golechha et al. (ICLR 2025 blog track,
https://iclr.cc/virtual/2025/poster/31349) showed is necessary,
because high-dimensional geometry can look "structured" even without training. Both see the
same prompts, so any effect also present in the random-init model is attributable to
tokenization/architecture, not learned knowledge.

Local context: AutoModelForCausalLM builds the text-only Qwen3_5ForCausalLM for Qwen3.5
(as in the Qwen-Scope card: https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50),
but the full multimodal Gemma4ForConditionalGeneration for Gemma 4, whose 2.35B per-layer
embedding (PLE) table plus audio/vision towers do not fit next to the decoder in 8 GB. The
"offload_ple" mode keeps that lookup table (and the unused towers) on CPU and the decoder on GPU,
with forward hooks moving the PLE lookup's input to CPU and its output back to GPU.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def _place_ple_on_cpu(model) -> None:
    """Keep Gemma-4's per-layer-embedding table (and unused audio/vision towers) on CPU, move the
    rest of the text decoder to GPU, and bridge devices with hooks around the PLE lookup.

    accelerate's "cpu" device_map entries mean *offload* (weights paged to the GPU on use), which
    fails here because the PLE weights are read directly; explicit placement avoids that.
    """
    lm = model.model.language_model
    ple = lm.embed_tokens_per_layer
    for name, child in lm.named_children():
        if child is not ple:
            child.to("cuda")
    model.lm_head.to("cuda")
    ple.register_forward_pre_hook(lambda _m, args: tuple(a.to("cpu") if hasattr(a, "to") else a for a in args))
    ple.register_forward_hook(lambda _m, _args, out: out.to("cuda"))


def load_model(model_id: str, random_init: bool = False, seed: int = 0, dtype=torch.bfloat16, device_map="cuda"):
    """Return (model, tokenizer) in eval mode. `random_init` keeps the config but re-draws weights."""
    tok = AutoTokenizer.from_pretrained(model_id)
    cfg = AutoConfig.from_pretrained(model_id)
    ple = device_map == "offload_ple"
    if random_init:
        torch.manual_seed(seed)  # deterministic random weights
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        if not ple:
            model = model.to(device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cpu" if ple else device_map)
    if ple:
        _place_ple_on_cpu(model)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("loaded %s (random_init=%s) class=%s params=%.2fB", model_id, random_init, type(model).__name__, n_params / 1e9)
    return model, tok
