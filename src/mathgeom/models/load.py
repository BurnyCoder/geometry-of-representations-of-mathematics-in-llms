"""Load trained or randomly initialized causal LMs (text-only) with HF transformers v5.

Global context: every trained model is paired with a random-init copy of the same
architecture and tokenizer — the control Golechha et al. (ICLR 2025) showed is necessary,
because high-dimensional geometry can look "structured" even without training. Both see the
same prompts, so any effect also present in the random-init model is attributable to
tokenization/architecture, not learned knowledge.

Local context: AutoModelForCausalLM builds the text-only Qwen3_5ForCausalLM for Qwen3.5
(as in the Qwen-Scope card: https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50),
but the full multimodal Gemma4ForConditionalGeneration for Gemma 4, whose 2.35B per-layer
embedding (PLE) table plus audio/vision towers do not fit next to the decoder in 8 GB. The
"offload_ple" device map keeps those lookup tables/towers on CPU and the decoder on GPU,
using accelerate's dispatch hooks (https://huggingface.co/docs/accelerate/usage_guides/big_modeling).
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)
_CPU_KEYS = ("per_layer", "vision", "audio")  # module-name fragments kept on CPU in "offload_ple" mode


def _offload_map(model) -> dict[str, str]:
    """Module-level device map: embedding tables / multimodal towers → cpu, everything else → cuda:0."""
    dm = {}

    def walk(mod, prefix, depth):
        for name, child in mod.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if any(k in name for k in _CPU_KEYS) and "proj" not in name and "norm" not in name and "gate" not in name:
                dm[full] = "cpu"
            elif depth < 3 and any(any(k in n for k in _CPU_KEYS) for n, _ in child.named_modules()):
                walk(child, full, depth + 1)  # descend only where a CPU-bound module lives below
            else:
                dm[full] = 0
        for pname, _ in mod.named_parameters(recurse=False):
            dm[f"{prefix}.{pname}" if prefix else pname] = 0

    walk(model, "", 0)
    return dm


def load_model(model_id: str, random_init: bool = False, seed: int = 0, dtype=torch.bfloat16, device_map="cuda"):
    """Return (model, tokenizer) in eval mode. `random_init` keeps the config but re-draws weights."""
    tok = AutoTokenizer.from_pretrained(model_id)
    cfg = AutoConfig.from_pretrained(model_id)
    if device_map == "offload_ple":
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(cfg)
        device_map = _offload_map(skeleton)
        del skeleton
    if random_init:
        torch.manual_seed(seed)  # deterministic random weights
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        if isinstance(device_map, dict):
            from accelerate import dispatch_model

            model = dispatch_model(model, device_map=device_map)
        else:
            model = model.to(device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("loaded %s (random_init=%s) class=%s params=%.2fB", model_id, random_init, type(model).__name__, n_params / 1e9)
    return model, tok
