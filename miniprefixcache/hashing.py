"""Chained per-chunk prefix hashing — the core of prefix caching.

A prompt is split into fixed-size chunks. Each chunk gets a hash that binds *all*
preceding tokens (chained), so hash[i] matches only if the entire prefix up to
chunk i is identical. That's exactly the property prefix caching needs: two
requests that share the first k chunks produce the same first k hashes, so the
KV for that shared prefix can be reused.

This mirrors what LMCache/vLLM do (they use blake3 over 256-token chunks); we use
blake2b from the stdlib so the project has zero native dependencies.
"""
from __future__ import annotations
import hashlib
import struct
from typing import List

_SEED = b"\x00" * 16


def _tokens_to_bytes(tokens: List[int]) -> bytes:
    # Pack token ids as little-endian uint32. Reject out-of-range ids rather than
    # masking them: masking would alias distinct ids to the same hash -> false hit
    # -> wrong KV returned. (Real vocabs are small; this just makes it explicit.)
    for t in tokens:
        if t < 0 or t > 0xFFFFFFFF:
            raise ValueError("token id %d out of range [0, 2**32)" % t)
    return struct.pack("<%dI" % len(tokens), *tokens)


def chunk_prefix_hashes(token_ids: List[int], chunk_size: int,
                        seed: bytes = _SEED) -> List[str]:
    """Return one chained hash per FULL chunk of `token_ids`.

    Only whole chunks are cacheable (a partial trailing chunk is ignored), which
    matches how block/paged KV caches work. len(result) == len(token_ids)//chunk_size.
    `seed` namespaces the hashes (e.g. per model) so incompatible KV can't collide.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    hashes: List[str] = []
    prev = seed
    n_full = len(token_ids) // chunk_size
    for i in range(n_full):
        chunk = token_ids[i * chunk_size:(i + 1) * chunk_size]
        digest = hashlib.blake2b(prev + _tokens_to_bytes(chunk), digest_size=16).digest()
        hashes.append(digest.hex())
        prev = digest
    return hashes
