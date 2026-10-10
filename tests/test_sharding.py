"""Layer sharding: CPU loader/hooks checks and an offline two-GPU parity test with a tiny hybrid checkpoint.

Run on the target pair: CUDA_VISIBLE_DEVICES=2,3 uv run --extra serve pytest tests/test_sharding.py -q
No model downloads; the CUDA cases skip unless two GPUs are visible.
"""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from accelerate import dispatch_model

from kev.checkpoint import Checkpoint, LoadOptions, Meta, write_meta
from kev.model import PointerHead
from kev import sharding


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """Four small Qwen3.5 layers with both recurrent and attention caches, saved as full weights or a LoRA."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                              num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                              linear_num_value_heads=2, linear_num_key_heads=1,
                              linear_key_head_dim=8, linear_value_head_dim=8,
                              layer_types=["linear_attention", "full_attention"] * 2, pad_token_id=0)
    torch.manual_seed(12)
    generation = Qwen3_5ForCausalLM(config).eval()
    generation.save_pretrained(tmp_path / "base")
    base = generation.model
    head = PointerHead(config.hidden_size, dp=16).state_dict()
    monkeypatch.setattr("kev.checkpoint.load_tokenizer", lambda *a, **kw: SimpleNamespace(pad_token_id=0))

    def make(kind):
        path = tmp_path / kind
        lm = copy.deepcopy(base)
        if kind == "lora":
            from peft import LoraConfig, get_peft_model
            lm = get_peft_model(lm, LoraConfig(task_type="FEATURE_EXTRACTION", r=2, lora_alpha=2,
                                              target_modules=["q_proj", "v_proj", "in_proj_qkv"]))
            with torch.no_grad():
                for name, parameter in lm.named_parameters():
                    if "lora_B" in name: parameter.normal_(std=0.02)
        lm.save_pretrained(path)
        write_meta(path, Meta(base=str(tmp_path / "base"), head=head, head_dim=16, weights=kind, lora=2 if kind == "lora" else 0))
        return Checkpoint(path)
    return make


def encoded():
    """A shared state and two independent question branches, with two and three options."""
    return {"ids": list(range(1, 17)), "pos": [0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 9, 10],
            "seg": [0] * 4 + [1] * 5 + [2] * 7, "decide_idx": [8, 15], "opt_idx": [[5, 7], [10, 12, 14]]}


@pytest.mark.parametrize("device,options,count,error", [
    ("cpu", {}, 2, "requires the torch backend on CUDA"),
    ("cuda", {"backend": "mlx"}, 2, "requires the torch backend on CUDA"),
    ("cuda", {"device_map": "auto"}, 2, "device_map must be"),
    ("cuda", {"cuda_graphs": True}, 2, "KEV_CUDA_GRAPHS=0"),
    ("cuda", {"fused": True}, 2, "KEV_FUSED=0"),
    ("cuda", {}, 1, "at least two visible"),
])
def test_invalid_sharding_fails_before_loading(checkpoint, monkeypatch, device, options, count, error):
    ck = checkpoint("full")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(ck, "_load_torch", lambda *args: pytest.fail("loaded incompatible model"))
    with pytest.raises(ValueError, match=error):
        ck.load(device, replace(LoadOptions(device_map="balanced"), **options))


def test_environment_option():
    assert LoadOptions.from_env({"KEV_DEVICE_MAP": "balanced"}).device_map == "balanced"
    assert LoadOptions.from_env({}).device_map is None


def test_full_checkpoint_local_base_supplies_tokenizer_not_weights(checkpoint, monkeypatch, tmp_path):
    ck = checkpoint("full")
    local_base = tmp_path / "tokenizer-only"
    local_base.mkdir()
    loaded = []
    def tokenizer(name, revision=None):
        loaded.append(name)
        return SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr("kev.checkpoint.load_tokenizer", tokenizer)
    _, reference = ck.load("cpu")
    override = Checkpoint(ck.path, base=str(local_base))
    _, actual = override.load("cpu")
    assert loaded == [ck.meta.base, str(local_base)]
    assert override.meta.base == ck.meta.base
    assert all(torch.equal(value, actual.lm.state_dict()[key]) for key, value in reference.lm.state_dict().items())
    check_probabilities(reference, actual, atol=0, rtol=0)


def test_balanced_plan_keeps_layers_whole_and_refuses_offload(checkpoint, monkeypatch):
    _, model = checkpoint("full").load("cpu")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    placement = sharding.balanced_device_map(model.lm, {0: 10**9, 1: 10**9})
    assert set(placement.values()) == {0, 1}
    assert all(not name.startswith("layers.") or len(name.split(".")) == 2 for name in placement)
    assert {"layers.0", "layers.1", "layers.2", "layers.3"} <= placement.keys()
    with pytest.raises(ValueError, match="do not fit"):
        sharding.balanced_device_map(model.lm, {0: 1, 1: 1})


@pytest.mark.parametrize("kind,merge", [("full", True), ("lora", True), ("lora", False)])
def test_cpu_staging_and_dispatch_hooks_preserve_probabilities(checkpoint, monkeypatch, kind, merge):
    """Exercise the real loader and Accelerate hooks on CPU; GPU transfer parity is checked separately below."""
    ck = checkpoint(kind)
    opts = LoadOptions(attn="sdpa", merge=merge)
    _, reference = ck.load("cpu", opts)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    called = []

    def dispatch_on_cpu(model):
        assert {p.device.type for p in model.parameters()} == {"cpu"}
        assert model.lm.config._attn_implementation == "sdpa"
        assert model.graphs is None
        called.append(True)
        model.lm = dispatch_model(model.lm, {"": "cpu"}, main_device="cpu", force_hooks=True,
                                  skip_keys=["past_key_values", "cache_params"])
        return model

    monkeypatch.setattr(sharding, "shard", dispatch_on_cpu)
    _, actual = ck.load("cuda", replace(opts, attn=None, device_map="balanced"))
    assert called == [True]
    check_probabilities(reference, actual, atol=0, rtol=0)


def check_probabilities(reference, actual, **tolerance):
    enc = encoded()
    expected = torch.cat(reference.probs(enc))
    fresh, prefix = actual.probs_and_prefix(enc)
    torch.testing.assert_close(torch.cat(fresh), expected, **tolerance)
    # Each branch forks the prefix. A second use must see the original recurrent and KV states.
    for _ in range(2):
        torch.testing.assert_close(torch.cat(actual.probs_with_prefix(enc, prefix)), expected, **tolerance)
    answers, _ = actual.probs_batch([enc, enc], [None, prefix], [True, True])
    for answer in answers:
        torch.testing.assert_close(torch.cat(answer), expected, **tolerance)


@pytest.mark.skipif(torch.cuda.device_count() != 2, reason="requires exactly two visible CUDA GPUs")
@pytest.mark.parametrize("kind,merge", [("full", True), ("lora", True), ("lora", False)])
def test_two_gpu_probabilities_and_cache_placement(checkpoint, monkeypatch, kind, merge):
    """A real multi-device test, including both cache types and nonzero adapters. No external checkpoint needed."""
    ck = checkpoint(kind)
    opts = LoadOptions(dtype=torch.bfloat16, attn="sdpa", merge=merge, cuda_graphs=False, fused=False)
    _, reference = ck.load("cuda", opts)
    _, actual = ck.load("cuda", replace(opts, device_map="balanced"))
    assert {p.device.index for p in actual.lm.parameters()} == {0, 1}
    assert actual.head.q.weight.device == torch.device("cuda:0") and actual.graphs is None
    _, cache, _ = actual.prefix(encoded())
    backbone = actual.lm.get_base_model() if hasattr(actual.lm, "get_base_model") else actual.lm
    for layer, cached in zip(backbone.layers, cache.layers):
        device = next(layer.parameters()).device
        if hasattr(cached, "keys"):
            assert cached.keys.device == device
        else:
            assert cached.conv_states[0].device == cached.recurrent_states[0].device == device
    check_probabilities(reference, actual, atol=0.005, rtol=0.005)
