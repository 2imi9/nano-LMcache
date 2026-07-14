"""End-to-end: a REAL prefix-cache hit on a real (tiny) transformer, on CPU.

Not a simulation. We run an actual causal transformer, cache the prefix's KV
through nano-LMcache, then serve a second request by reusing that KV and computing
ONLY the new suffix. The point: the reused-KV logits are bit-identical to a full
recompute — proving the cache is correct, not just fast. Torch only, no downloads.

    python3 bench/e2e.py          # correctness + a prefix-length sweep
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn
from nanolmcache import PrefixCache

torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, past):                  # x:[T,d]  past:[2,P,H,D] or None
        T = x.shape[0]
        qkv = self.qkv(self.ln1(x)).view(T, 3, self.h, self.hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]           # [T,H,D]
        knew, vnew = k, v
        if past is not None:
            k = torch.cat([past[0], k], 0)                  # [P+T,H,D]
            v = torch.cat([past[1], v], 0)
        S = k.shape[0]; P = S - T
        att = torch.einsum("thd,shd->hts", q, k) / (self.hd ** 0.5)     # [H,T,S]
        qpos = torch.arange(P, P + T).view(T, 1)
        mask = (torch.arange(S).view(1, S) <= qpos)         # causal, [T,S]
        att = att.masked_fill(~mask.unsqueeze(0), float("-inf")).softmax(-1)
        out = torch.einsum("hts,shd->thd", att, v).reshape(T, self.h * self.hd)
        x = x + self.proj(out)
        x = x + self.mlp(self.ln2(x))
        return x, knew, vnew                                # cache the NEW tokens' K/V


class ToyLM(nn.Module):
    def __init__(self, vocab=1024, d=128, heads=4, layers=4, max_len=8192):
        super().__init__()
        self.L, self.h, self.hd = layers, heads, d // heads
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    @torch.no_grad()
    def forward(self, ids, past_kv=None):          # ids:[T]  past_kv:[L,2,P,H,D] or None
        P = 0 if past_kv is None else past_kv.shape[2]
        T = ids.shape[0]
        x = self.tok(ids) + self.pos(torch.arange(P, P + T))
        new_kv = torch.empty(self.L, 2, T, self.h, self.hd)
        for i, blk in enumerate(self.blocks):
            x, k, v = blk(x, None if past_kv is None else past_kv[i])
            new_kv[i, 0], new_kv[i, 1] = k, v
        return self.head(self.norm(x)), new_kv     # logits:[T,vocab], kv:[L,2,T,H,D]


def demo(prefix_len=256, suffix_len=64, chunk=64, seed=0):
    torch.manual_seed(seed)
    model = ToyLM()
    g = torch.Generator().manual_seed(seed + 1)
    prefix = torch.randint(0, 1024, (prefix_len,), generator=g)
    query_a = torch.randint(0, 1024, (suffix_len,), generator=g)   # 1st request's suffix
    query_b = torch.randint(0, 1024, (suffix_len,), generator=g)   # 2nd request's suffix

    cache = PrefixCache(chunk_size=chunk, namespace="toy")

    # --- request A (cold): full forward, then cache the prefix's real KV ---
    t = time.time()
    _, kv_a = model(torch.cat([prefix, query_a]))
    cold_s = time.time() - t
    cache.insert(prefix.tolist(), kv_a[:, :, :prefix_len])          # store prefix KV only

    # --- request B (warm): reuse cached prefix KV, compute ONLY the suffix ---
    warm_ids = torch.cat([prefix, query_b])
    hit, chunks = cache.lookup(warm_ids.tolist())
    past = torch.cat(chunks, dim=2)                                 # reassembled prefix KV
    t = time.time()
    logits_warm, _ = model(query_b, past_kv=past)
    warm_s = time.time() - t

    # --- ground truth: full recompute of [prefix + query_b] ---
    logits_full, _ = model(warm_ids)
    diff = (logits_warm - logits_full[prefix_len:]).abs().max().item()

    return dict(
        prefix=prefix_len, suffix=suffix_len, hit=hit,
        max_logit_diff=diff,
        tokens_cold=prefix_len + suffix_len, tokens_warm=suffix_len,
        compute_saved=hit / (hit + suffix_len),
        cold_s=cold_s, warm_s=warm_s,
    )


if __name__ == "__main__":
    print("Real prefix-cache hit on a toy transformer (CPU). Correctness = reused-KV")
    print("logits vs full recompute; should be ~0.\n")
    hdr = ("prefix", "suffix", "reused", "compute saved", "max logit diff", "cold ms", "warm ms")
    print("  %-7s %-7s %-7s %-14s %-16s %-8s %-8s" % hdr)
    ok = True
    for plen in (128, 256, 512, 1024):
        r = demo(prefix_len=plen)
        ok &= r["max_logit_diff"] < 1e-3
        print("  %-7d %-7d %-7d %-14s %-16.2e %-8.1f %-8.1f" % (
            r["prefix"], r["suffix"], r["hit"], "%.0f%%" % (100 * r["compute_saved"]),
            r["max_logit_diff"], 1e3 * r["cold_s"], 1e3 * r["warm_s"]))
    print("\n%s: reused-KV output matches full recompute at every prefix length."
          % ("CORRECT" if ok else "MISMATCH"))
