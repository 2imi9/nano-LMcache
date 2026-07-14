# mini-prefix-cache

A minimal, readable **prefix cache** for LLM serving — a "mini-LMCache" built to
understand (and then extend) KV-cache reuse from first principles. Personal side
project; runs locally on CPU, no GPU required.

## Why

Prefix caching reuses the KV cache for shared token prefixes (system prompts, RAG
context, chat history) so you skip recomputing prefill. This repo implements the
core mechanism in ~200 lines of dependency-light Python, plus the hooks for the
two things that make it interesting on AMD:

1. **Model-specific KV geometry** (`kv_shape.py`) — KV shape/dtype come from the
   model config, which sets bytes-per-token. FP8 KV halves the bytes-per-token of
   the same geometry vs bf16 (compare `kv_bytes_per_token()` across models) —
   directly relevant to the ROCm KV-transfer bottleneck.
2. **A vLLM connector adapter** (`vllm_connector/`) — plugs into the same v1 API
   LMCache uses, so the core can drive real serving later.

## Layout

```
miniprefixcache/
  hashing.py    chained per-chunk prefix hashing (blake2b, stdlib)
  store.py      LRU CPU KV store (the offload tier)
  cache.py      PrefixCache: lookup longest cached prefix / insert new chunks
  kv_shape.py   model-aware KV geometry (qwen3-8b, minimax-m3 presets)  <- model-specific hook
tests/test_cache.py   self-contained tests (no pytest needed)
bench/simulate.py     shared-prefix request stream -> hit rate + prefill saved
vllm_connector/       KVConnectorBase_V1 adapter (skeleton; needs vLLM to run)
```

## Run

```bash
pip install torch                      # only hard dep (numpy optional)
python3 tests/test_cache.py            # unit tests
python3 bench/simulate.py qwen3-8b 20  # sim on Qwen3-8B geometry
python3 bench/simulate.py minimax-m3 20 # sim on M3 geometry (FP8 KV -> fewer bytes)
```

## Roadmap

- [x] Core: chunk hashing, LRU store, prefix lookup/insert, model-aware KV
- [x] Simulation harness (CPU, real tensors) — proves the mechanics
- [ ] Complete the vLLM v1 connector against a target version; run on Qwen (MI250)
- [ ] **ROCm-native transfer** — write the KV copy in torch/HIP from the start,
      avoiding LMCache's CUDA-only `c_ops` path (the ~2 GB/s ceiling on ROCm)
- [ ] **MSA-aware caching for M3** — reuse only the blocks MiniMax Sparse Attention
      actually attends to, instead of the whole prefix

## Relation to LMCache

This is a teaching/clean-room implementation of the same idea. LMCache is the
mature system; the goal here is to own the mechanics end-to-end so the
model-specific (M3) and ROCm-native pieces are ours to write.
