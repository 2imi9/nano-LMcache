"""vLLM KV-connector adapter (reference skeleton).

This is the integration layer that plugs our PrefixCache into vLLM's v1 KV
connector API — the SAME hook LMCache uses (KVConnectorBase_V1). The core cache
in ../miniprefixcache is fully tested standalone; this file needs a vLLM runtime
to exercise, so it's a documented skeleton, not a validated path yet.

Wire-up when running against vLLM (on the MI250 / MI308):
    --kv-transfer-config '{"kv_connector":"MiniPrefixConnector",
      "kv_role":"kv_both",
      "kv_connector_module_path":"vllm_connector.mini_connector"}'

The two sides of the v1 API:
  Scheduler side  -> how many external tokens can we reuse for this request?
  Worker side     -> actually load cached KV into paged blocks / save new KV out.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from miniprefixcache import PrefixCache

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1, KVConnectorRole,
    )
    _HAVE_VLLM = True
except Exception:                      # vLLM not installed locally -> skeleton mode
    KVConnectorBase_V1 = object
    KVConnectorRole = None
    _HAVE_VLLM = False


class MiniPrefixConnector(KVConnectorBase_V1):
    """Minimal prefix-reuse connector backed by an in-process CPU PrefixCache."""

    def __init__(self, vllm_config=None, role=None):
        if _HAVE_VLLM:
            super().__init__(vllm_config, role)
        # chunk_size should match the model's block size; 16 is a placeholder.
        self.cache = PrefixCache(chunk_size=16, max_chunks=200_000)

    # ---- scheduler side -------------------------------------------------
    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        """Return (num_external_hit_tokens, load_async).

        We hash the request's prompt and report the longest cached prefix beyond
        what vLLM already has in its own (VRAM) cache. vLLM then skips prefill for
        those tokens and asks the worker side to load them.
        """
        token_ids = list(getattr(request, "prompt_token_ids", []) or [])
        hit, _ = self.cache.lookup(token_ids)
        external = max(0, hit - num_computed_tokens)
        return external, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """Record which blocks were allocated for the external (cached) tokens so
        the worker side knows where to write the loaded KV. (impl per vLLM version)"""
        ...

    def build_connector_meta(self, scheduler_output):
        """Package per-request load/save instructions for the workers."""
        ...

    # ---- worker side ----------------------------------------------------
    def start_load_kv(self, forward_context, **kw):
        """Copy cached chunk tensors (store.get) into the paged KV blocks."""
        ...

    def wait_for_layer_load(self, layer_name: str) -> None:
        ...

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kw):
        """Offload freshly computed KV for full chunks into the store (cache.insert)."""
        ...

    def wait_for_save(self) -> None:
        ...


# Register under this name so vLLM's KVConnectorFactory can find it.
CONNECTOR_NAME = "MiniPrefixConnector"
