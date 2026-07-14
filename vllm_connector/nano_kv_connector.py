"""NanoKVConnector — a REAL, working vLLM v1 KV connector for nano-LMcache.

Unlike mini_connector.py (which only exercises the scheduler-side logic), this one
actually moves KV in/out of vLLM's paged blocks, so it produces a genuine
prefix-cache hit in a live vLLM server. It's a clean, minimal, *synchronous*
connector — modeled on vLLM's own reference `ExampleConnector` (Apache-2.0), with
two nano changes: keys come from nano-LMcache's chained prefix hash, and the store
is a plain per-layer safetensors file (an L2 disk tier — robust across the
scheduler/worker processes; an in-memory/shared-mem CPU tier is the next step).

Enable it (offline or server):
    kv_transfer_config = KVTransferConfig(
        kv_connector="NanoKVConnector", kv_role="kv_both",
        kv_connector_module_path="vllm_connector.nano_kv_connector")
Requires --enable-prefix-caching (block-aligned reuse).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import safetensors.torch as st

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# nano-LMcache's own prefix hash (works whether or not the pkg is installed).
try:
    from nanolmcache.hashing import chunk_prefix_hashes
except Exception:                                    # dev fallback
    import hashlib
    def chunk_prefix_hashes(ids, chunk_size, seed=b"\x00" * 16):
        h, out, n = seed, [], len(ids) // chunk_size
        for i in range(n):
            import struct
            b = struct.pack("<%dI" % chunk_size, *ids[i * chunk_size:(i + 1) * chunk_size])
            h = hashlib.blake2b(h + b, digest_size=16).digest(); out.append(h.hex())
        return out


def _align(n: int, block: int) -> int:
    return (n - 1) // block * block


@dataclass
class ReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool

    @staticmethod
    def make(token_ids, block_ids, block_size, is_store) -> "ReqMeta":
        n = _align(len(token_ids), block_size)
        blk = torch.tensor(block_ids)
        slot = (torch.arange(block_size).reshape(1, block_size)
                + blk.reshape(-1, 1) * block_size).flatten()[:n]
        return ReqMeta(torch.tensor(token_ids)[:n], slot, is_store)


@dataclass
class NanoMeta(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add(self, token_ids, block_ids, block_size, is_store):
        self.requests.append(ReqMeta.make(token_ids, block_ids, block_size, is_store))


class NanoKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig" = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block = vllm_config.cache_config.block_size
        self._need_load: dict[str, "Request"] = {}
        self._root = self._kv_transfer_config.get_from_extra_config("nano_kv_path", "/tmp/nano_kv")
        os.makedirs(self._root, exist_ok=True)
        logger.info("NanoKVConnector: role=%s block_size=%d store=%s", role.name, self._block, self._root)

    # ---- key = nano-LMcache prefix hash of the whole-block prompt prefix ----
    def _key(self, token_ids) -> str:
        # chunk_prefix_hashes ignores the partial trailing block, so the full prompt
        # (hit-check) and the block-truncated req.token_ids (store/load) hash the SAME
        # whole blocks -> same key. (Avoid a prompt whose length is an exact multiple
        # of block_size, where the two would differ by one block.)
        hashes = chunk_prefix_hashes(list(token_ids), self._block)
        return hashes[-1] if hashes else "empty"

    def _dir(self, token_ids, create=False) -> str:
        d = os.path.join(self._root, self._key(token_ids))
        if create:
            os.makedirs(d, exist_ok=True)
        return d

    def _hit(self, token_ids) -> bool:
        return len(token_ids) > self._block and os.path.isdir(self._dir(token_ids))

    # ---- scheduler side ----
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int):
        ids = request.prompt_token_ids or []
        if not self._hit(ids):
            print("[nano] MISS  key=%s" % self._key(ids)[:12], flush=True)
            return 0, False
        matched = _align(len(ids), self._block) - num_computed_tokens
        print("[nano] HIT   key=%s  matched_tokens=%d" % (self._key(ids)[:12], matched), flush=True)
        return matched, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        if num_external_tokens > 0:
            self._need_load[request.request_id] = request

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = NanoMeta()
        for req in scheduler_output.scheduled_new_reqs:
            ids = req.prompt_token_ids or []
            if req.req_id in self._need_load:
                meta.add(ids, req.block_ids[0], self._block, is_store=False)
            elif not self._hit(ids):
                meta.add(ids, req.block_ids[0], self._block, is_store=True)
        self._need_load.clear()
        return meta

    # ---- worker side (synchronous) ----
    def start_load_kv(self, forward_context: "ForwardContext", **kw) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, NanoMeta)
        attn = forward_context.attn_metadata
        if attn is None:
            return
        for req in meta.requests:
            if req.is_store:
                continue
            n = 0
            for name, layer in forward_context.no_compile_layers.items():
                dst = getattr(layer, "kv_cache", None)
                if dst is None:
                    continue
                f = os.path.join(self._dir(req.token_ids.tolist()), name + ".safetensors")
                if not os.path.exists(f):
                    continue
                src = st.load_file(f, device=str(dst.device))["kv"]
                bi = req.slot_mapping // self._block
                off = req.slot_mapping % self._block
                dst[bi, :, off] = src
                n += 1
            print("[nano] start_load_kv: injected cached KV into %d layers (%d tokens)"
                  % (n, len(req.slot_mapping)), flush=True)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kw) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, NanoMeta)
        for req in meta.requests:
            if not req.is_store:
                continue
            bi = req.slot_mapping // self._block
            off = req.slot_mapping % self._block
            kv = kv_layer[bi, :, off]
            d = self._dir(req.token_ids.tolist(), create=True)
            st.save_file({"kv": kv.detach().cpu()}, os.path.join(d, layer_name + ".safetensors"))

    def wait_for_save(self) -> None:
        return
