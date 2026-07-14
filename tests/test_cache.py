"""Self-contained tests (run: `python3 tests/test_cache.py`; no pytest needed)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from miniprefixcache import chunk_prefix_hashes, KVStore, PrefixCache


def test_hash_determinism_and_prefix_sharing():
    a = list(range(100))
    b = list(range(100))
    assert chunk_prefix_hashes(a, 16) == chunk_prefix_hashes(b, 16), "same tokens -> same hashes"
    # shares first 48 tokens (3 chunks), then diverges
    c = list(range(48)) + [999] * 52
    ha, hc = chunk_prefix_hashes(a, 16), chunk_prefix_hashes(c, 16)
    assert ha[:3] == hc[:3], "shared prefix -> shared leading hashes"
    assert ha[3] != hc[3], "divergence -> different hash after the shared prefix"


def test_partial_chunk_ignored():
    assert len(chunk_prefix_hashes(list(range(70)), 16)) == 4  # 70//16 = 4 full chunks


def test_store_lru_eviction():
    s = KVStore(max_chunks=2)
    s.put("a", torch.zeros(1)); s.put("b", torch.zeros(1))
    s.get("a")                       # touch a -> b is now LRU
    s.put("c", torch.zeros(1))       # evicts b
    assert "a" in s and "c" in s and "b" not in s
    assert s.evictions == 1


def test_cache_full_partial_and_miss():
    pc = PrefixCache(chunk_size=16, max_chunks=1024)
    toks = list(range(160))          # 10 chunks
    pc.insert(toks, torch.randn(2, 2, 160, 4, 8))
    # exact re-request -> full hit
    hit, chunks = pc.lookup(toks)
    assert hit == 160 and len(chunks) == 10, (hit, len(chunks))
    # shares first 4 chunks (64 tokens), then diverges -> partial hit
    other = list(range(64)) + [7] * 96
    hit2, _ = pc.lookup(other)
    assert hit2 == 64, hit2
    # unrelated -> zero hit
    hit3, _ = pc.lookup([123456] * 160)
    assert hit3 == 0, hit3


def test_insert_idempotent():
    pc = PrefixCache(chunk_size=16)
    toks = list(range(160))
    n1 = pc.insert(toks, torch.randn(2, 2, 160, 4, 8))
    n2 = pc.insert(toks, torch.randn(2, 2, 160, 4, 8))   # same prefix -> nothing new
    assert n1 == 10 and n2 == 0, (n1, n2)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t(); print("PASS", t.__name__); passed += 1
    print("\n%d/%d tests passed" % (passed, len(tests)))
