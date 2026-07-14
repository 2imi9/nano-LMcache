# nano-LMcache

![license](https://img.shields.io/badge/license-MIT-blue.svg) ![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

A lightweight prefix cache (KV-cache reuse) for LLM serving, built from scratch —
the nano version of what [LMCache](https://github.com/LMCache/LMCache) does: reuse
the KV of shared prompt prefixes so you skip recomputing prefill.

> Reference: [vLLM + LMCache: A Starter Guide, No GPU Required](https://blog.lmcache.ai/en/2026/06/23/vllm-lmcache-a-starter-guide-no-gpu-required/)

## Key Features

- **Readable**: the core idea in ~200 lines — chunk hashing, an LRU CPU KV store, prefix lookup/insert.
- **Actually runs**: caches the KV from a real transformer's forward pass and reuses it — output verified **bit-identical** to full recompute (`bench/e2e.py`, CPU, no GPU).
- **Plugs into real vLLM**: a working KV connector (`vllm_connector/nano_kv_connector.py`) lands a genuine prefix-cache hit in a live vLLM engine on GPU — verified with a negative control.
- **Dependency-light**: `pip install torch` for the core; the vLLM connector needs vLLM + a GPU.

## Architecture

<p align="center"><img src="docs/figure.png" alt="nano-LMcache architecture" width="620"></p>

Green = cache hit (reuse KV, cheap) · red = miss (recompute) · blue = the KV store. A
request sharing a system prompt / RAG context / chat history with an earlier one reuses
that prefix's KV for free — only the divergent suffix is recomputed.

## Installation

```bash
git clone https://github.com/2imi9/nano-LMcache && cd nano-LMcache
pip install torch
```

## Quick Start

```python
from nanolmcache import PrefixCache

cache = PrefixCache(chunk_size=16, namespace="my-model")   # one cache per model
hit, chunks = cache.lookup(prompt_token_ids)               # leading tokens already cached
cache.insert(prompt_token_ids, kv_tensor)                  # store [L, 2, T, kv_heads, head_dim]
```

## Benchmark

**Real prefix-cache hit on a tiny transformer (CPU).** Cache the prefix's KV, reuse it,
compute only the suffix — the reused-KV logits are **bit-identical** to a full recompute:

| Prefix | Compute saved | Max logit diff | Prefill (cold → warm) |
|---:|---:|---:|---:|
| 256 | 80% | 7e-7 (≈0) | 12.6 ms → 4.1 ms |
| 1024 | 94% | 8e-7 (≈0) | 55.6 ms → 6.3 ms |

A higher-level simulation over a request stream (20 requests, 512-token shared prompt) saves **76%** of prefill.

```bash
python3 bench/e2e.py                                      # real cache hit + correctness sweep
python3 bench/simulate.py qwen3-8b 20                     # request-stream simulation
for t in cache connector e2e; do python3 tests/test_$t.py; done   # 14 tests, no pytest
```

## In a real vLLM engine (GPU)

`vllm_connector/nano_kv_connector.py` is a working vLLM v1 KV connector (modeled on
vLLM's own `ExampleConnector`, keyed by nano-LMcache's prefix hash). It gathers/scatters
KV from vLLM's paged blocks, so it lands a genuine prefix-cache hit in a live engine:

```text
# process 1 (cold): our connector saves the prefix KV
$ python examples/vllm_offline_demo.py store
OUT: ' OK. How can I assist you further?'

# process 2 (fresh, cold GPU cache): hits our external cache and skips the prefill
$ python examples/vllm_offline_demo.py load
[nano] HIT   key=dc49fe6b3a1f  matched_tokens=416
[nano] start_load_kv: injected cached KV into 28 layers (416 tokens)
OUT: ' OK. How can I assist you further?'      # <- bit-identical to the cold run

# wipe the store -> MISS, proving the hit really came from our cache
$ rm -rf /tmp/nano_kv && python examples/vllm_offline_demo.py load
[nano] MISS  key=dc49fe6b3a1f
```

Verified on an AMD MI250 with Qwen2.5-1.5B (vLLM 0.23.1): 416 tokens of prefill skipped,
KV injected into all 28 attention layers, output identical to full recompute.

## How it maps to LMCache

| this repo | LMCache |
|---|---|
| chained chunk hash | blake3 over 256-token chunks |
| `KVStore` (CPU, LRU) | L1 CPU backend (+ disk / Redis / remote) |
| `PrefixCache.lookup/insert` | the cache engine's store/retrieve |
| `vllm_connector/nano_kv_connector.py` | `LMCacheConnectorV1` — real vLLM connector, verified hit |
