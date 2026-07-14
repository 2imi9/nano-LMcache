"""NanoKVConnector — a REAL, working vLLM v1 KV connector for nano-LMcache.

Unlike mini_connector.py (scheduler-side only), this moves KV in/out of vLLM's
paged blocks, so it lands a genuine prefix-cache hit in a live vLLM engine.
Modeled on vLLM's own reference `ExampleConnector` (Apache-2.0); keys come from
nano-LMcache's chained prefix hash.

Two storage tiers (env `NANO_KV_MEM`):
  * default (disk): per-layer safetensors under /tmp/nano_kv — works ACROSS the
    scheduler/worker processes, so it survives a fresh vLLM process (the two-process
    demo in examples/vllm_offline_demo.py).
  * NANO_KV_MEM=1 (CPU RAM): a module-level dict shared within one EngineCore process
    (scheduler + worker are the same process for TP=1) — the fast in-memory tier, for
    the capacity benchmark where reused prefixes get evicted from VRAM.

Enable it:
    kv_transfer_config = KVTransferConfig(
        kv_connector="NanoKVConnector", kv_role="kv_both",
        kv_connector_module_path="vllm_connector.nano_kv_connector")
Requires --enable-prefix-caching. Non-MLA models only (GQA/MHA, e.g. Qwen).
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
_VERBOSE = os.environ.get("NANO_KV_QUIET") != "1"     # print HIT/MISS lines (the demo signal)

try:
    from nanolmcache.hashing import chunk_prefix_hashes
except Exception:                                     # dev fallback
    import hashlib, struct
    def chunk_prefix_hashes(ids, chunk_size, seed=b"\x00" * 16):
        h, out, n = seed, [], len(ids) // chunk_size
        for i in range(n):
            b = struct.pack("<%dI" % chunk_size, *ids[i * chunk_size:(i + 1) * chunk_size])
            h = hashlib.blake2b(h + b, digest_size=16).digest(); out.append(h.hex())
        return out

# In-process CPU KV tier: key -> {layer_name: cpu_tensor}. Shared between the
# scheduler- and worker-role connector instances (same EngineCore process, TP=1).
_MEM: dict[str, dict[str, torch.Tensor]] = {}


def _align(n: int, block: int) -> int:
    return (n - 1) // block * block


def _log(msg):
    if _VERBOSE:
        print("[nano] " + msg, flush=True)


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
        self._mem = os.environ.get("NANO_KV_MEM") == "1"
        self._root = self._kv_transfer_config.get_from_extra_config("nano_kv_path", "/tmp/nano_kv")
        if not self._mem:
            os.makedirs(self._root, exist_ok=True)
        logger.info("NanoKVConnector role=%s block=%d tier=%s", role.name, self._block,
                    "cpu-mem" if self._mem else "disk")

    # ---- key = nano-LMcache prefix hash of the whole-block prompt prefix ----
    def _key(self, token_ids) -> str:
        # chunk_prefix_hashes ignores the partial trailing block, so the full prompt
        # (hit-check) and the block-truncated req.token_ids (store/load) hash the SAME
        # whole blocks -> same key. (Avoid a prompt length that is an exact multiple
        # of block_size, where the two differ by one block.)
        hashes = chunk_prefix_hashes(list(token_ids), self._block)
        return hashes[-1] if hashes else "empty"

    def _dir(self, token_ids, create=False) -> str:
        d = os.path.join(self._root, self._key(token_ids))
        if create:
            os.makedirs(d, exist_ok=True)
        return d

    def _hit(self, token_ids) -> bool:
        if len(token_ids) <= self._block:
            return False
        if self._mem:
            return bool(_MEM.get(self._key(token_ids)))
        return os.path.isdir(self._dir(token_ids))

    # ---- scheduler side ----
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int):
        ids = request.prompt_token_ids or []
        if not self._hit(ids):
            _log("MISS  key=%s" % self._key(ids)[:12])
            return 0, False
        matched = _align(len(ids), self._block) - num_computed_tokens
        _log("HIT   key=%s  matched_tokens=%d" % (self._key(ids)[:12], matched))
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
        if forward_context.attn_metadata is None:
            return
        for req in meta.requests:
            if req.is_store:
                continue
            key = self._key(req.token_ids.tolist())
            layers = _MEM.get(key, {}) if self._mem else None
            bi = req.slot_mapping // self._block
            off = req.slot_mapping % self._block
            n = 0
            for name, layer in forward_context.no_compile_layers.items():
                dst = getattr(layer, "kv_cache", None)
                if dst is None:
                    continue
                if self._mem:
                    if name not in layers:
                        continue
                    src = layers[name].to(dst.device)
                else:
                    f = os.path.join(self._root, key, name + ".safetensors")
                    if not os.path.exists(f):
                        continue
                    src = st.load_file(f, device=str(dst.device))["kv"]
                dst[bi, :, off] = src
                n += 1
            _log("start_load_kv: injected cached KV into %d layers (%d tokens)" % (n, len(req.slot_mapping)))

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
            kv = kv_layer[bi, :, off].detach().cpu()
            key = self._key(req.token_ids.tolist())
            if self._mem:
                _MEM.setdefault(key, {})[layer_name] = kv
            else:
                d = self._dir(req.token_ids.tolist(), create=True)
                st.save_file({"kv": kv}, os.path.join(d, layer_name + ".safetensors"))

    def wait_for_save(self) -> None:
        return
