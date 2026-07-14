"""Simulate a shared-prefix request stream and show the prefix cache working.

Scenario (the classic prefix-caching win): a big fixed SYSTEM prompt shared by
every request (think RAG context / agent system prompt / chat history), plus a
unique per-request suffix. After the first request, the system prefix is served
from cache and only the unique suffix is "computed".

Prints hit rate, prefill tokens saved, store size, and the model-specific
bytes-per-token (bf16 vs FP8) that determines transfer cost.

Run: python3 bench/simulate.py [model] [num_requests]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from nanolmcache import PrefixCache
from nanolmcache.kv_shape import make_kv, kv_bytes_per_token, cfg

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
N_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 20
CHUNK = 16
SYS_LEN = 512          # shared system prefix (multiple of CHUNK)
SUFFIX_LEN = 128       # unique per request

SYSTEM = list(range(SYS_LEN))                      # identical across requests
pc = PrefixCache(chunk_size=CHUNK, max_chunks=100_000)

computed_with_cache = 0
computed_no_cache = 0
t0 = time.time()
for r in range(N_REQ):
    suffix = [1_000_000 + r * 10_000 + j for j in range(SUFFIX_LEN)]   # unique
    tokens = SYSTEM + suffix
    hit, _ = pc.lookup(tokens)
    computed_with_cache += (len(tokens) - hit)     # only the miss is recomputed
    computed_no_cache += len(tokens)
    # store the full KV so the shared prefix is available to later requests
    pc.insert(tokens, make_kv(MODEL, len(tokens)))
elapsed = time.time() - t0

bpt = kv_bytes_per_token(MODEL)
c = cfg(MODEL)
print("model: %s  (%d layers, %d kv-heads, head_dim %d, kv dtype %s, attn=%s)"
      % (MODEL, c["num_layers"], c["num_kv_heads"], c["head_dim"],
         str(c["dtype"]).replace("torch.", ""), c["attn"]))
print("requests: %d   system prefix: %d tok (shared)   suffix: %d tok (unique)"
      % (N_REQ, SYS_LEN, SUFFIX_LEN))
print("-" * 62)
print("prefill tokens WITHOUT cache: %d" % computed_no_cache)
print("prefill tokens WITH cache:    %d" % computed_with_cache)
saved = 1 - computed_with_cache / computed_no_cache
print("prefill SAVED: %.1f%%   (steady-state per request: %d/%d = %.1f%%)"
      % (saved * 100, SYS_LEN, SYS_LEN + SUFFIX_LEN, 100 * SYS_LEN / (SYS_LEN + SUFFIX_LEN)))
st = pc.stats()
print("cache: %d chunks stored, %.1f MB, hit-token-rate %.1f%%, %d evictions"
      % (st["stored_chunks"], st["store_mb"], st["hit_token_rate"] * 100, st["evictions"]))
print("model-specific KV: %d bytes/token  (=> %.2f MB per 1K-token prefix moved)"
      % (bpt, bpt * 1000 / 1e6))
# The FP8 angle: compare against a bf16 equivalent of the same geometry.
bf16_bpt = c["num_layers"] * 2 * c["num_kv_heads"] * c["head_dim"] * 2
if bpt < bf16_bpt:
    print("  FP8 KV moves %.1fx fewer bytes than bf16 (%d vs %d) -> that much less"
          " to store and transfer per cached token." % (bf16_bpt / bpt, bpt, bf16_bpt))
print("ran in %.3fs on CPU" % elapsed)
