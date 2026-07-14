"""LRU CPU KV store — the "offload tier" of the prefix cache.

Maps chunk_hash -> KV tensor for one chunk. Bounded by a chunk count; least-
recently-used chunks are evicted first (the eviction that makes the capacity
story interesting: under pressure, old prefixes fall out). This is the piece a
real system would back with CPU RAM / disk / a remote KV server.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Optional, Any


class KVStore:
    def __init__(self, max_chunks: int = 4096):
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.max_chunks = max_chunks
        self._d: "OrderedDict[str, Any]" = OrderedDict()
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)  # mark most-recently-used
        return v

    def put(self, key: str, value: Any) -> None:
        if key in self._d:
            self._d.move_to_end(key)
            self._d[key] = value
            return
        self._d[key] = value
        while len(self._d) > self.max_chunks:
            self._d.popitem(last=False)  # evict LRU
            self.evictions += 1

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)

    def nbytes(self) -> int:
        total = 0
        for v in self._d.values():
            # torch tensor or numpy array
            if hasattr(v, "numel") and hasattr(v, "element_size"):
                total += v.numel() * v.element_size()
            elif hasattr(v, "nbytes"):
                total += int(v.nbytes)
        return total
