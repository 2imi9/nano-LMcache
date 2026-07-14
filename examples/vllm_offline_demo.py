"""A REAL prefix-cache hit through nano-LMcache, inside a live vLLM engine.

Run it TWICE, in two separate processes that share the on-disk store:

    # 1) cold: computes prefill, our connector saves the prefix KV to disk
    python examples/vllm_offline_demo.py store

    # 2) fresh process, cold GPU cache: our connector finds the saved KV,
    #    injects it into vLLM's paged blocks, and skips the prefill
    python examples/vllm_offline_demo.py load

The `load` run prints `[nano] HIT ... matched_tokens=N` and
`start_load_kv: injected cached KV into L layers`, and its output is
bit-identical to the `store` run. Wipe the store (`rm -rf /tmp/nano_kv`) and the
`load` run prints `[nano] MISS` instead — proof the hit came from our cache.

Needs a GPU + vLLM, and `--enable-prefix-caching` (the connector reuses whole
KV blocks). PYTHONPATH must include the repo root so vLLM can import the module.
"""
import sys, os, time


def main():
    os.environ.setdefault("VLLM_USE_V1", "1")
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    mode = sys.argv[1] if len(sys.argv) > 1 else "store"
    model = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    # A long shared prefix (a system prompt / RAG-style context) + a tiny question.
    prompt = ("System: You are a helpful assistant.\n"
              + "The quick brown fox jumps over the lazy dog. " * 40
              + "\nUser: reply with OK.\nAssistant:")

    kv = KVTransferConfig(
        kv_connector="NanoKVConnector", kv_role="kv_both",
        kv_connector_module_path="vllm_connector.nano_kv_connector",
    )
    llm = LLM(model=model, enable_prefix_caching=True, kv_transfer_config=kv,
              gpu_memory_utilization=0.35, max_model_len=2048, enforce_eager=True)
    t = time.time()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=16))
    print("=== mode=%s  time=%.3fs ===" % (mode, time.time() - t))
    print("OUT: %r" % out[0].outputs[0].text)


if __name__ == "__main__":
    main()
