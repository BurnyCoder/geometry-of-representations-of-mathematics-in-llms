"""Exploratory SAE check with official Qwen-Scope residual SAEs for Qwen3.5-2B-Base.

Global context: Li et al. 2024 (arXiv:2410.19750) found a "math/code lobe" among SAE
features. Here we ask, at the pre-registered layer: (1) is the field map also visible in the
sparse SAE code (field-RSA, LOO clustering on SAE activations); (2) are there field-selective
features, and how many math-field features are shared across fields vs with control fields.

Local context: Qwen-Scope SAEs (https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50,
arXiv:2605.11887) are TopK SAEs stored as `layer{n}.sae.pt` dicts {W_enc (32768×2048), b_enc,
W_dec, b_dec}; the card's example hooks the OUTPUT of decoder block n, i.e. our
hidden_states[n+1]. Encoding follows the card: pre = x @ W_enc.T + b_enc, keep top-k.
Caveat: we encode template-AVERAGED residuals (exploratory approximation).
"""

from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def load_sae(repo: str, block: int) -> dict:
    path = hf_hub_download(repo, f"layer{block}.sae.pt")
    return torch.load(path, map_location="cpu")


def encode(X: np.ndarray, sae: dict, k: int) -> np.ndarray:
    """TopK SAE encoding of residual vectors X (n×d) → sparse codes (n×d_sae)."""
    x = torch.from_numpy(X.astype(np.float32))
    pre = x @ sae["W_enc"].float().T + sae["b_enc"].float()
    vals, idx = pre.topk(k, dim=-1)
    codes = torch.zeros_like(pre)
    codes.scatter_(-1, idx, vals.clamp_min(0))
    return codes.numpy()


def field_selectivity(codes: np.ndarray, labels: np.ndarray, top: int = 3) -> dict:
    """Per field: features with the largest (mean in field − mean elsewhere) and their firing rates."""
    out = {}
    for f in np.unique(labels):
        inn, oth = codes[labels == f], codes[labels != f]
        score = inn.mean(0) - oth.mean(0)
        best = np.argsort(-score)[:top]
        out[str(f)] = [{"feature": int(j), "score": float(score[j]), "fire_rate_in": float((inn[:, j] > 0).mean()), "fire_rate_out": float((oth[:, j] > 0).mean())} for j in best]
    return out
