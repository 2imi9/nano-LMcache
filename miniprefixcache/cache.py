"""PrefixCache — ties hashing + store into a reuse cache.

lookup(tokens) -> how many leading tokens' KV we already have (longest cached
prefix) plus the cached chunk tensors. insert(tokens, kv) stores KV per chunk.

KV tensor layout (matches paged attention): [num_layers, 2(k/v), num_tokens,
num_kv_heads, head_dim]. Chunks slice along the token dim (dim=2).
"""
from __future__ import annotations
import hashlib
from typing import List, Tuple, Any
from .hashing import chunk_prefix_hashes
from .store import KVStore


class PrefixCache:
    def __init__(self, chunk_size: int = 16, max_chunks: int = 4096,
                 namespace: str = ""):
        self.chunk_size = chunk_size
        self.store = KVStore(max_chunks)
        # namespace seeds the hash so caching for two different models / KV dtypes
        # can't collide (same tokens, incompatible KV). Use one cache per model,
        # or distinct namespaces on a shared instance.
        self._seed = hashlib.blake2b(namespace.encode("utf-8"), digest_size=16).digest()
        self.lookups = 0
        self.hit_tokens_total = 0
        self.request_tokens_total = 0

    def lookup(self, token_ids: List[int]) -> Tuple[int, List[Any]]:
        """Return (num_hit_tokens, cached_chunks) for the longest cached prefix.

        Walk chunk hashes in order; each present hash extends the hit; stop at the
        first miss (prefix caching only reuses a *contiguous leading* run).
        """
        self.lookups += 1
        self.request_tokens_total += len(token_ids)
        hashes = chunk_prefix_hashes(token_ids, self.chunk_size, self._seed)
        cached: List[Any] = []
        for h in hashes:
            kv = self.store.get(h)
            if kv is None:
                break
            cached.append(kv)
        hit = len(cached) * self.chunk_size
        self.hit_tokens_total += hit
        return hit, cached

    def insert(self, token_ids: List[int], kv: Any) -> int:
        """Store KV for each full chunk of token_ids. Returns #chunks newly stored.

        kv must have num_tokens == len(token_ids) along dim 2. Chunks already
        present are not re-stored (idempotent, saves copies).
        """
        hashes = chunk_prefix_hashes(token_ids, self.chunk_size, self._seed)
        stored = 0
        for i, h in enumerate(hashes):
            if h in self.store:
                continue
            t0, t1 = i * self.chunk_size, (i + 1) * self.chunk_size
            chunk_kv = kv[:, :, t0:t1]
            # .contiguous().clone() = an independent copy, i.e. the "offload".
            self.store.put(h, chunk_kv.contiguous().clone() if hasattr(chunk_kv, "contiguous") else chunk_kv.copy())
            stored += 1
        return stored

    def stats(self) -> dict:
        hr = (self.hit_tokens_total / self.request_tokens_total) if self.request_tokens_total else 0.0
        return {
            "lookups": self.lookups,
            "hit_token_rate": hr,
            "stored_chunks": len(self.store),
            "store_mb": self.store.nbytes() / 1e6,
            "evictions": self.store.evictions,
        }
