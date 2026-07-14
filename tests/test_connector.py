"""Mock-vLLM test for the KV connector's logic (roadmap: 'real prefix-cache hit
through the vLLM v1 connector'). We can't run a real vLLM on a laptop, so we drive
the connector's framework-agnostic core (matched_prefix_tokens / save / load) with
a fake request object and real tensors — proving the connector would report the
right reuse count and round-trip KV, independent of the vLLM binding.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from vllm_connector.mini_connector import MiniPrefixConnector
from nanolmcache.kv_shape import make_kv as model_kv

CS = 16  # connector chunk size


class Req:  # a stand-in for vLLM's Request (only prompt_token_ids is used)
    def __init__(self, ids): self.prompt_token_ids = ids


def kv(n):  # tiny KV: [layers, 2, tokens, kv_heads, head_dim]
    return torch.randn(2, 2, n, 4, 8)


def test_cold_then_warm_full_hit():
    c = MiniPrefixConnector()
    ids = list(range(10 * CS))                       # 10 full chunks
    assert c.get_num_new_matched_tokens(Req(ids), 0) == (0, False)   # cold: nothing cached
    assert c.save(ids, kv(len(ids))) == 10           # worker stores 10 chunks
    ext, load_async = c.get_num_new_matched_tokens(Req(ids), 0)      # warm: same prompt
    assert ext == 10 * CS and load_async is False
    hit, chunks = c.load(ids)
    assert hit == 10 * CS and len(chunks) == 10


def test_partial_prefix_and_num_computed():
    c = MiniPrefixConnector()
    base = list(range(10 * CS))
    c.save(base, kv(len(base)))
    shares = base[:5 * CS] + [999] * (5 * CS)         # shares first 5 chunks, then diverges
    assert c.matched_prefix_tokens(shares, 0) == 5 * CS
    # vLLM already has 3 chunks resident -> connector only adds the other 2
    assert c.matched_prefix_tokens(shares, 3 * CS) == 2 * CS


def test_no_false_hit_on_unrelated_prompt():
    c = MiniPrefixConnector()
    c.save(list(range(10 * CS)), kv(10 * CS))
    assert c.matched_prefix_tokens(list(range(5000, 5000 + 10 * CS))) == 0


def test_fp8_kv_roundtrips():
    # model-specific: the cache must handle FP8 KV, not just bf16 (roadmap item 3, partial)
    c = MiniPrefixConnector()
    ids = list(range(10 * CS))
    fp8 = model_kv("fp8-moe", len(ids))               # float8_e4m3fn tensor
    assert c.save(ids, fp8) == 10
    hit, chunks = c.load(ids)
    assert hit == 10 * CS and chunks[0].dtype == fp8.dtype


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print("\n%d/%d connector tests passed" % (len(fns), len(fns)))
