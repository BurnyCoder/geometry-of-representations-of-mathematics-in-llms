"""Causal test for ordered chains (H-B), adapted from SMDS noise injection (Tiblias et al. 2025,
arXiv:2510.01025): perturb the chain's low-dimensional subspace and measure task damage.

Global context: correlational order in activations does not show the model USES that
geometry. Task: two-alternative "which is more general, a A or a B?" for every ordered
pair of a chain, both answer orders (counterbalanced), scored by comparing the
log-likelihood of the two continuations. Gate (pre-registered, docs/deviations.md): only if
baseline accuracy ≥ 0.70 do we intervene. Intervention: at the pre-registered layer, at the
last token of each concept mention, add Gaussian noise inside the top-k PCA subspace of the
chain — fitted on the OTHER chain items (held-out fit, avoiding the subspace illusion of
Makelov et al. 2023, arXiv:2311.17030) — versus inside random k-dim subspaces with the same
noise norm. Prediction: chain-subspace noise hurts accuracy more than random-subspace noise.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from mathgeom.extract import concept_token_span
from mathgeom.logio import log_llm

TEMPLATE = "Question: In {D}, which is more general, a {A} or a {B}?\nAnswer: The more general one is a"


def _spans(tok, prompt: str, a: str, b: str) -> list[int]:
    """Last-token positions of both concept mentions inside the question."""
    pos = []
    for c in (a, b):
        start = prompt.index(f" {c}")
        pos.append(concept_token_span(tok, prompt[: start + 1 + len(c)], prompt[:start])[-1])
    return pos


@torch.inference_mode()
def _score(model, tok, prompt: str, cont: str, hook_fn=None, layer_module=None) -> float:
    """Sum log p(cont tokens | prompt), optionally with a forward hook on one decoder block."""
    ids_p = tok(prompt, return_tensors="pt").input_ids
    ids_c = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([ids_p, ids_c], 1).to(model.get_input_embeddings().weight.device)
    h = layer_module.register_forward_hook(hook_fn) if hook_fn else None
    try:
        logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
    finally:
        if h:
            h.remove()
    n = ids_p.shape[1]
    return float(sum(logp[n - 1 + t, ids_c[0, t]] for t in range(ids_c.shape[1])))


def _make_hook(positions: list[int], basis: torch.Tensor, sigma: float, gen: torch.Generator):
    """Hook adding noise z = basis.T @ ε (ε ~ N(0, σ² I_k)) at the given token positions."""
    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        for p in positions:
            eps = torch.randn(basis.shape[0], generator=gen, device="cpu") * sigma
            hs[0, p] += (eps.to(basis) @ basis).to(hs.dtype)
        return out
    return hook


def chain_causal_test(model, tok, layer_module, items: list[str], domain: str, family_reps: np.ndarray, k: int = 2,
                      scales=(1.0, 2.0, 4.0), n_random: int = 5, seed: int = 0, gate: float = 0.70) -> dict:
    """Run baseline accuracy, then chain-subspace vs random-subspace noise at each scale."""
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = family_reps.shape[1]
    qs = []
    for i, j in itertools.combinations(range(len(items)), 2):  # i more general than j (ground-truth order)
        for a, b in ((i, j), (j, i)):
            qs.append((a, b, i))
    def run(hook_for=None):
        correct = []
        for a, b, gen_idx in qs:
            prompt = TEMPLATE.format(D=domain, A=items[a], B=items[b])
            hook = hook_for(a, b, prompt) if hook_for else None
            sa = _score(model, tok, prompt, f" {items[a]}", hook, layer_module)
            sb = _score(model, tok, prompt, f" {items[b]}", hook, layer_module)
            correct.append((sa > sb) == (a == gen_idx))
        return float(np.mean(correct))

    base = run()
    log_llm({"causal": "baseline", "domain": domain, "items": items, "accuracy": base, "n_questions": len(qs), "example_prompt": TEMPLATE.format(D=domain, A=items[0], B=items[1])})
    out = {"baseline_acc": base, "n_questions": len(qs), "gate": gate, "passed_gate": base >= gate, "scales": {}}
    if base < gate:
        return out
    spread = float(np.sqrt(((family_reps - family_reps.mean(0)) ** 2).sum(1).mean()))  # RMS distance to centroid

    def chain_basis(a, b):
        others = [t for t in range(len(items)) if t not in (a, b)]
        X = family_reps[others] - family_reps[others].mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return torch.tensor(Vt[:k], dtype=torch.float32)

    dev = model.get_input_embeddings().weight.device
    for s in scales:
        sigma = s * spread / np.sqrt(k)
        acc_chain = run(lambda a, b, p: _make_hook(_spans(tok, p, items[a], items[b]), chain_basis(a, b).to(dev), sigma, gen))
        acc_rand = []
        for _ in range(n_random):
            Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
            B = torch.tensor(Q.T, dtype=torch.float32).to(dev)
            acc_rand.append(run(lambda a, b, p, B=B: _make_hook(_spans(tok, p, items[a], items[b]), B, sigma, gen)))
        out["scales"][str(s)] = {"acc_chain_subspace": acc_chain, "acc_random_subspace_mean": float(np.mean(acc_rand)), "acc_random_subspace": acc_rand}
        log_llm({"causal": "noise", "domain": domain, "scale": s, "acc_chain_subspace": acc_chain, "acc_random_mean": float(np.mean(acc_rand))})
    return out
