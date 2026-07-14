"""Does nano-LMcache's CPU tier actually ACCELERATE serving? Measure it.

The scenario where a KV-offload cache helps (and vLLM's VRAM-only prefix cache
can't): the working set exceeds VRAM, so reused prefixes get evicted. We serve K
distinct long prefixes (populate), then revisit them. The ones evicted from VRAM
must be recomputed — unless nano-LMcache served them from its CPU tier.

Run twice in one process each:
    USE_CONN=0 python examples/vllm_capacity_bench.py     # baseline: evicted -> recompute
    NANO_KV_MEM=1 USE_CONN=1 python examples/vllm_capacity_bench.py  # + nano CPU tier

Compare the "revisit early(evicted)" means. Needs a GPU + vLLM; keep the VRAM pool
small (gpu_memory_utilization) so K prefixes overflow it.
"""
import os, time


def main():
    os.environ.setdefault("VLLM_USE_V1", "1")
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    use_conn = os.environ.get("USE_CONN") == "1"
    MODEL = os.environ.get("MODEL", "/models/Qwen/Qwen3-8B")
    K = int(os.environ.get("K", "40"))
    PREF = int(os.environ.get("PREF", "3500"))          # approx prefix tokens
    util = float(os.environ.get("UTIL", "0.30"))

    kwargs = dict(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=util,
                  max_model_len=int(os.environ.get("MAXLEN", "8192")), enforce_eager=True)
    if use_conn:
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="NanoKVConnector", kv_role="kv_both",
            kv_connector_module_path="vllm_connector.nano_kv_connector")
    os.environ["NANO_KV_QUIET"] = "1"                    # silence per-req prints during timing
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    body = "The quick brown fox jumps over the lazy dog. " * (PREF // 8)
    prompts = ["SESSION-%d UNIQUE-%d :: %s" % (i, i * 7919 + 13, body) for i in range(K)]

    def phase(ps):
        lat = []
        for p in ps:
            t = time.time(); llm.generate([p], sp, use_tqdm=False); lat.append(time.time() - t)
        return lat

    pop = phase(prompts)          # cold: recompute (+ store to CPU tier if connector on)
    rev = phase(prompts)          # revisit: early ones evicted from VRAM by now
    h = K // 2
    me, ml = sum(rev[:h]) / h, sum(rev[h:]) / h
    print("=== CONN=%s  K=%d  PREF~%d  util=%.2f ===" % (use_conn, K, PREF, util))
    print("  populate mean=%.3fs   revisit mean=%.3fs" % (sum(pop) / K, sum(rev) / K))
    print("  revisit early(evicted first) mean=%.3fs   late(resident) mean=%.3fs   early/late=%.2fx"
          % (me, ml, me / ml if ml else 0))


if __name__ == "__main__":
    main()
