"""vLLM KV-connector adapter (reference skeleton).

This is the integration layer that plugs our PrefixCache into vLLM's v1 KV
connector API — the SAME hook LMCache uses (KVConnectorBase_V1). The core cache
in ../nanolmcache is fully tested standalone; this file needs a vLLM runtime
to exercise, so it's a documented skeleton, not a validated path yet.

Wire-up when running against vLLM (any GPU):
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
from nanolmcache import PrefixCache

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

    # ---- framework-agnostic core --------------------------------------------
    # The vLLM hooks below delegate to these three; they're also what the
    # mock-vLLM tests (tests/test_connector.py) drive without a real vLLM.
    def matched_prefix_tokens(self, token_ids, num_computed_tokens: int = 0) -> int:
        """Additional leading tokens servable from cache, beyond the ones vLLM
        already has resident (num_computed_tokens). This is the reuse count."""
        hit, _ = self.cache.lookup(list(token_ids))
        return max(0, hit - num_computed_tokens)

    def save(self, token_ids, kv) -> int:
        """Store freshly-computed KV for the request's full chunks. -> #chunks stored."""
        return self.cache.insert(list(token_ids), kv)

    def load(self, token_ids):
        """Fetch cached KV for the longest cached prefix. -> (hit_tokens, chunks)."""
        return self.cache.lookup(list(token_ids))

    # ---- scheduler side (vLLM v1 API) ---------------------------------------
    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        """(num_external_hit_tokens, load_async) — vLLM skips prefill for the hits."""
        token_ids = list(getattr(request, "prompt_token_ids", []) or [])
        return self.matched_prefix_tokens(token_ids, num_computed_tokens), False

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
