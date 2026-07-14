"""End-to-end correctness: reusing cached KV must produce the SAME output as a full
recompute — that's what makes a prefix cache *correct*, not just fast. Drives a real
(tiny) transformer on CPU via bench/e2e.py."""
import os, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
from e2e import demo


def test_reused_kv_bit_identical_to_full_recompute():
    for plen in (64, 128, 256, 512):
        r = demo(prefix_len=plen, suffix_len=48)
        assert r["hit"] == plen, ("prefix not fully cached", plen, r["hit"])
        assert r["max_logit_diff"] < 1e-3, ("cache changed the output!", plen, r["max_logit_diff"])


def test_only_the_suffix_is_recomputed():
    r = demo(prefix_len=256, suffix_len=64)
    assert r["tokens_cold"] == 320 and r["tokens_warm"] == 64   # warm computes only the suffix
    assert r["compute_saved"] > 0.7


if __name__ == "__main__":
    test_reused_kv_bit_identical_to_full_recompute(); print("PASS reused-KV == full recompute")
    test_only_the_suffix_is_recomputed(); print("PASS only-suffix-recomputed")
    print("\n2/2 e2e tests passed")
