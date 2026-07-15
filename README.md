# nano-LMcache

![license](https://img.shields.io/badge/license-MIT-blue.svg) ![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

Prefix caching (KV reuse) for LLM serving, from scratch — the nano version of
[LMCache](https://github.com/LMCache/LMCache): reuse the KV of shared prompt prefixes
so you skip recomputing prefill.

> References: [LMCache paper (arXiv:2510.09665)](https://arxiv.org/abs/2510.09665) ·
> [vLLM + LMCache: A Starter Guide, No GPU Required](https://blog.lmcache.ai/en/2026/06/23/vllm-lmcache-a-starter-guide-no-gpu-required/)

## Features

- **~200 readable lines** — chunk hashing, an LRU CPU KV store, prefix lookup/insert.
- **Runs for real, verified correct** — caches KV from a real transformer and reuses it; output **bit-identical** to full recompute.
- **Plugs into vLLM** — a working KV connector lands a genuine prefix-cache hit in a live vLLM engine on GPU.
- **torch-only core** (the vLLM connector needs vLLM + a GPU).

## Architecture

<p align="center"><img src="docs/figure.png" alt="nano-LMcache architecture" width="600"></p>

Green = hit (reuse KV) · red = miss (recompute) · blue = the KV store.

## Install & use

```bash
git clone https://github.com/2imi9/nano-LMcache && cd nano-LMcache && pip install torch
```
```python
from nanolmcache import PrefixCache
cache = PrefixCache(chunk_size=16, namespace="my-model")
hit, chunks = cache.lookup(prompt_token_ids)   # leading tokens already cached
cache.insert(prompt_token_ids, kv_tensor)       # [L, 2, T, kv_heads, head_dim]
```

## Results

**Correct** — `bench/e2e.py` runs a real (tiny) transformer, caches the prefix KV, reuses
it, and the reused-KV logits are **bit-identical** to full recompute (~8e-7) across a
128–1024 prefix sweep; 67–94% of prefill skipped.

**Real vLLM hit** — `examples/vllm_offline_demo.py` (MI250 / Qwen2.5-1.5B): a fresh
process with cold VRAM hits our external cache, injects KV into all 28 layers, output
bit-identical; wipe the store → MISS (negative control).

**Speedup where offload caches win** — `examples/vllm_capacity_bench.py` (Qwen3-8B,
working set > VRAM, so vLLM evicts reused prefixes):

| revisit of **evicted** prefixes | vLLM only (recompute) | + nano CPU tier |
|---|---:|---:|
| latency | 0.671 s | **0.383 s (1.75×)** |

Same tradeoffs as real LMCache: a cold-store cost, and the win is capacity-bound (overall
mean not yet better — a known `num_computed_tokens` overlap).

```bash
python3 bench/e2e.py                                              # correctness sweep
for t in cache connector e2e; do python3 tests/test_$t.py; done   # 14 tests, no pytest
```

## Maps to LMCache

| this repo | LMCache |
|---|---|
| chained chunk hash | blake3 over 256-tok chunks |
| `KVStore` (CPU, LRU) | L1 CPU (+ disk / Redis / remote) |
| `PrefixCache.lookup/insert` | the cache engine's store/retrieve |
| `vllm_connector/nano_kv_connector.py` | `LMCacheConnectorV1` — real vLLM connector, verified hit |

## Citation

This is an educational reimplementation. For the real system and its design, cite the
LMCache paper:

```bibtex
@article{liu2025lmcache,
  title   = {LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference},
  author  = {Liu, Yuhan and Cheng, Yihua and Yao, Jiayi and An, Yuwei and
             Chen, Xiaokun and Feng, Shaoting and Huang, Yuyang and Shen, Samuel and
             Zhang, Rui and Du, Kuntai and Jiang, Junchen},
  journal = {arXiv preprint arXiv:2510.09665},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.09665}
}
```
