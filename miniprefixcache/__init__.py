"""mini-prefix-cache: a minimal, readable prefix cache (a "mini-LMCache").

Core = chunk hashing + LRU CPU store + a reuse cache. Model-aware KV geometry
lives in kv_shape. See vllm_connector/ for the vLLM integration adapter.
"""
from .hashing import chunk_prefix_hashes
from .store import KVStore
from .cache import PrefixCache

__all__ = ["chunk_prefix_hashes", "KVStore", "PrefixCache"]
__version__ = "0.1.0"
