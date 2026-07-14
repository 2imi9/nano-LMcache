"""Model-aware KV geometry — why prefix caching has a model-dependent angle.

The KV tensor's shape and dtype come from the model config (layers, kv-heads,
head-dim, kv dtype), and that sets bytes-per-token — which is exactly what a cache
stores and moves. FP8 KV halves the bytes-per-token of the same geometry vs bf16.
Compare actual `kv_bytes_per_token()` values across models.

Presets below are illustrative configs.
"""
from __future__ import annotations
import torch

MODEL_CONFIGS = {
    # dense, GQA 32:8, bf16 KV — a small public model to test with.
    "qwen3-8b": dict(num_layers=36, num_kv_heads=8, head_dim=128,
                     dtype=torch.bfloat16, attn="full"),
    # a large MoE with FP8 KV (illustrative) — shows FP8 halves KV bytes/token.
    "fp8-moe": dict(num_layers=60, num_kv_heads=4, head_dim=128,
                    dtype=torch.float8_e4m3fn, attn="sparse"),
}


def cfg(model: str) -> dict:
    if model not in MODEL_CONFIGS:
        raise KeyError("unknown model %r; known: %s" % (model, list(MODEL_CONFIGS)))
    return MODEL_CONFIGS[model]


def kv_bytes_per_token(model: str) -> int:
    c = cfg(model)
    elt = torch.empty(0, dtype=c["dtype"]).element_size()
    return c["num_layers"] * 2 * c["num_kv_heads"] * c["head_dim"] * elt


def make_kv(model: str, num_tokens: int):
    """A KV tensor of the right shape/dtype for `model`, [L,2,T,KVH,D].

    Values are dummy (this project measures cache *mechanics*, not model math);
    float8 can't randn, so it's zero-filled — shape/dtype/bytes are what matter.
    """
    c = cfg(model)
    shape = (c["num_layers"], 2, num_tokens, c["num_kv_heads"], c["head_dim"])
    if c["dtype"] == torch.float8_e4m3fn:
        return torch.zeros(shape, dtype=c["dtype"])
    return torch.randn(shape, dtype=torch.float32).to(c["dtype"])
