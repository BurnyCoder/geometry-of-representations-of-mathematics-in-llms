"""Extraction correctness on a tiny random HF model (no download of real weights).

Checks: (1) concept token span is found for prefix+concept prompts; (2) the stored last-token
read equals hidden_states at the final position of a direct forward pass.
"""

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:  # offline
        pytest.skip("gpt2 tokenizer unavailable")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=tok.vocab_size)).eval()
    return model, tok


def test_span_and_last_token_match_direct_forward(tiny):
    from mathgeom.extract import Item, concept_token_span, extract
    from mathgeom.prompts.templates import render

    model, tok = tiny
    tpl = ["In {D}, the notion of {X}", "The term {X}"]
    prefix, full = render(tpl[0], "principal ideal domain", "mathematics")
    span = concept_token_span(tok, full, prefix)
    assert tok.decode(tok(full).input_ids[span[0] : span[-1] + 1]).strip() == "principal ideal domain"
    out = extract(model, tok, [Item("principal ideal domain", "mathematics")], tpl, tag="test")
    with torch.inference_mode():
        hs = [model(input_ids=tok(render(t, "principal ideal domain", "mathematics")[1], return_tensors="pt").input_ids, output_hidden_states=True).hidden_states for t in tpl]
    direct = np.mean([torch.stack(h)[:, 0, -1].numpy() for h in hs], 0)
    assert np.allclose(out["last"][0].astype(np.float32), direct, atol=2e-2)
    assert out["last"].shape == (1, 3, 32) and out["n_tokens"][0] == len(span)


def test_noise_hook_zero_is_identity_and_only_touches_target_positions(tiny):
    """Causal-test hook: σ=0 must not change logits; σ>0 must leave positions before the target unchanged."""
    from mathgeom.analyses.causal import _make_hook

    model, tok = tiny
    ids = tok("In topology, a Hausdorff space is", return_tensors="pt").input_ids
    layer = model.transformer.h[0]
    basis = torch.nn.functional.normalize(torch.randn(2, 32), dim=-1)
    gen = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        ref = model(input_ids=ids, output_hidden_states=True)
        h = layer.register_forward_hook(_make_hook([3], basis, 0.0, gen))
        same = model(input_ids=ids).logits
        h.remove()
        h = layer.register_forward_hook(_make_hook([3], basis, 5.0, gen))
        noisy = model(input_ids=ids, output_hidden_states=True)
        h.remove()
    assert torch.allclose(ref.logits, same)
    assert torch.allclose(ref.hidden_states[1][0, :3], noisy.hidden_states[1][0, :3])  # causal: earlier positions untouched
    assert not torch.allclose(ref.hidden_states[1][0, 3], noisy.hidden_states[1][0, 3])
