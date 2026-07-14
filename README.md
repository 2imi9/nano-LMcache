# nano-LMcache

**Prefix caching for LLM serving, in ~200 lines of readable Python.** A tiny,
from-scratch take on the idea behind [LMCache](https://github.com/LMCache/LMCache):
reuse the KV cache for shared prompt prefixes so you skip recomputing prefill.
Runs on a laptop CPU — no GPU required.

> Reference / inspiration: **[vLLM + LMCache: A Starter Guide, No GPU Required](https://blog.lmcache.ai/en/2026/06/23/vllm-lmcache-a-starter-guide-no-gpu-required/)** —
> the LMCache team's walkthrough of running the *real* vLLM + LMCache on a Mac.
> This repo is the nano version of the cache layer that guide sets up.

`nanoGPT` teaches the model. `nano-vllm` teaches the serving loop. This teaches the
**cache layer** that sits underneath them.

---

## Architecture

<p align="center">
  <img src="docs/figure.png" alt="nano-LMcache architecture" width="640">
</p>

A request that shares a system prompt / RAG context / chat history with an earlier
one gets that prefix's KV for free — only the divergent suffix is recomputed. Figure
drawn in TikZ ([`docs/figure.tex`](docs/figure.tex)); rebuild with `pdflatex docs/figure.tex`
(or on Overleaf).

## What's inside (~200 LOC)

| file | what it does |
|---|---|
| `nanolmcache/hashing.py` | chained per-chunk prefix hash (blake2b, stdlib) |
| `nanolmcache/store.py` | LRU CPU KV store — the offload tier |
| `nanolmcache/cache.py` | `PrefixCache`: look up longest cached prefix / insert new chunks |
| `nanolmcache/kv_shape.py` | **model-aware KV geometry** — shape & dtype per model config |
| `bench/simulate.py` | shared-prefix request stream → hit rate + prefill saved |
| `tests/test_cache.py` | 8 self-contained tests (no pytest needed) |
| `vllm_connector/` | adapter for vLLM's KV-connector API (the same hook LMCache uses) |

## Quickstart

```bash
pip install torch          # the only hard dependency
python3 tests/test_cache.py
python3 bench/simulate.py qwen3-8b 20
```

```text
$ python3 bench/simulate.py qwen3-8b 20
model: qwen3-8b  (36 layers, 8 kv-heads, head_dim 128, kv dtype bfloat16, attn=full)
requests: 20   system prefix: 512 tok (shared)   suffix: 128 tok (unique)
--------------------------------------------------------------
prefill tokens WITHOUT cache: 12800
prefill tokens WITH cache:    3072
prefill SAVED: 76.0%   (steady-state per request: 512/640 = 80.0%)
```

```python
from nanolmcache import PrefixCache
cache = PrefixCache(chunk_size=16, namespace="qwen3-8b")   # namespace = one cache per model
hit, chunks = cache.lookup(prompt_token_ids)               # how many leading tokens are cached
cache.insert(prompt_token_ids, kv_tensor)                  # store [L, 2, T, kv_heads, head_dim]
```

## Why FP8 KV matters (the AMD angle)

KV bytes-per-token come from the model config, and that's what a cache actually moves:

```
$ python3 bench/simulate.py minimax-m3 20
model-specific KV: 61440 bytes/token   (MiniMax-M3, FP8 KV)
  FP8 KV moves 2.0x fewer bytes than bf16 -> that much more effective transfer bandwidth.
```

On ROCm, KV transfer bandwidth is the bottleneck for CPU-offloaded caching — so
moving fewer bytes (FP8) and a native transfer path are the real levers.

## How it maps to real LMCache

| this repo | LMCache |
|---|---|
| chained chunk hash | blake3 over 256-token chunks (`TokenHasher`) |
| `KVStore` (CPU, LRU) | L1 CPU backend (+ L2 disk / Redis / remote) |
| `PrefixCache.lookup/insert` | the cache engine's store/retrieve |
| `vllm_connector/` | `LMCacheConnectorV1` (same vLLM v1 KV-connector API) |

## Roadmap

Only what's actually planned — the *essence*, not LMCache parity.

- [x] **Core** — chunk hashing, LRU CPU store, prefix lookup/insert, model-aware KV
- [x] **CPU simulation** with real tensors — proves the mechanics
- [ ] **One real cache hit** — finish the vLLM v1 connector; land a genuine prefix-cache hit on Qwen
- [ ] **ROCm-native transfer** (torch/HIP) — avoid LMCache's CUDA-only `c_ops` (~2 GB/s ceiling)
- [ ] **Model-specific** — FP8-KV-layout-aware transfer + sparse-attention (MSA)-aware reuse for MiniMax M3

## Not a replacement for LMCache

This is a clean-room teaching implementation of the same idea. LMCache is the
production system — use it for real. This exists to make the mechanics readable
end-to-end. MIT licensed.
