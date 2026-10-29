"""LLM-judge abstraction ratings (covariate for H-A4), from model families we do NOT probe.

Global context: H-A4 asks whether the main axes of the math-concept cloud align with
concrete↔abstract. Human norms (e.g. Brysbaert et al. 2014 concreteness ratings,
https://doi.org/10.3758/s13428-013-0403-5) barely cover technical math terms, so ratings
come from a panel of small instruct models from other families (IBM Granite, HF SmolLM,
Microsoft Phi), never Qwen or Gemma, to avoid a model rating its own representations.

Local context: each judge sees one chat prompt per concept; instead of parsing free text we
read the next-token distribution over the digits "1".."7" right after the assistant turn
starts and store its expectation (deterministic, no sampling). Every prompt and the full
digit distribution are logged (logio.log_llm). Inter-judge agreement is reported.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from tqdm import tqdm

from mathgeom.logio import log_llm

log = logging.getLogger(__name__)

PROMPT = (
    "Rate how ABSTRACT the mathematical concept \"{X}\" is, on a scale from 1 to 7.\n"
    "1 = very concrete: elementary, tangible, can be drawn or counted (e.g. 'triangle', 'fraction').\n"
    "4 = intermediate (e.g. 'derivative', 'matrix').\n"
    "7 = very abstract: defined only through layers of other abstractions (e.g. 'topos', 'derived functor').\n"
    "Answer with a single digit only."
)


@torch.inference_mode()
def rate(model_id: str, concepts: list[str], device: str = "cuda") -> np.ndarray:
    """Expected 1–7 abstraction rating per concept from one judge model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device).eval()
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 8)]
    out = np.empty(len(concepts))
    for i, c in enumerate(tqdm(concepts, desc=f"judge[{model_id}]")):
        msgs = [{"role": "user", "content": PROMPT.format(X=c)}]
        kw = {"enable_thinking": False} if "SmolLM3" in model_id else {}  # SmolLM3 reasoning off (/no_think)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        logits = model(input_ids=ids).logits[0, -1].float()
        p = torch.softmax(logits[digit_ids], 0).cpu().numpy()
        out[i] = float((p * np.arange(1, 8)).sum())
        log_llm({"judge": model_id, "concept": c, "prompt": text, "digit_probs": p.round(4).tolist(), "expected_rating": out[i],
                 "argmax_rating": int(p.argmax()) + 1, "p_mass_on_digits": float(torch.softmax(logits, 0)[digit_ids].sum())}, echo=True)
    del model
    torch.cuda.empty_cache()
    return out


def icc2(ratings: np.ndarray) -> tuple[float, float]:
    """ICC(2,1) and ICC(2,k) (two-way random, absolute agreement; Shrout & Fleiss 1979).

    ratings: n_items × k_raters.
    """
    n, k = ratings.shape
    gm = ratings.mean()
    ms_r = k * ((ratings.mean(1) - gm) ** 2).sum() / (n - 1)  # between items
    ms_c = n * ((ratings.mean(0) - gm) ** 2).sum() / (k - 1)  # between raters
    resid = ratings - ratings.mean(1, keepdims=True) - ratings.mean(0, keepdims=True) + gm
    ms_e = (resid**2).sum() / ((n - 1) * (k - 1))
    icc1 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    icck = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)
    return float(icc1), float(icck)
