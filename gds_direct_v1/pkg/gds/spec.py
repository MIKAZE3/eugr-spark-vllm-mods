# GdsDirectOffloadingSpec: GPU -> SSD direct KV offloading via cuFile (GDS).
#
# Replaces the CPU primary tier entirely: the SSD filesystem is the ONLY
# persistent layer. Stores gather GPU KV rows into registered device ring
# slots (D2D) and issue cuFileWrite straight from device memory; loads do
# cuFileRead into slots and scatter back to GPU. There is no CPU residency,
# no slot-exhaustion deadlock ("KVOFFFAIL"), and no promotion path.
#
# Multi-group models (e.g. DeepSeek-V4-Flash: 5 KV cache groups) are fully
# supported: one transfer job spans one chunk per group; each chunk maps to
# its own file + ring slot, exactly mirroring the upstream CPUOffloading
# layout contract (GPULoadStoreSpec.group_sizes / block_indices semantics,
# see SingleDirectionOffloadingHandler.transfer_async).
#
# Reused upstream machinery (unchanged files):
#   - FileMapper          : deterministic key->path layout (PYTHONHASHSEED=0)
#   - FsGCManager         : LRU on-disk budget sweep (gc_max_size_gb)
#   - AsyncLookupManager  : batched existence lookups w/ caching

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
from typing_extensions import override

from vllm.distributed.kv_events import MEDIUM_FS
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    Locality,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TransferResult,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager

logger = init_logger(__name__)

READY = 0
STORE_INFLIGHT = 1
LOAD_INFLIGHT = 2

# Process-wide fs GC singleton: the scheduler-side manager and the
# worker-side IO threads live in the SAME process (EngineCore), and on
# headless nodes only get_worker() runs. Sharing one instance keeps every
# node's SSD tree under the same budget without double sweeping on node1.
#
# NOTE: upstream FsGCManager's sweep thread proved dead on these engines
# (frozen in a long-timeout futex wait, 0 CPU, no sweep ever ran, on all
# four nodes). GdsGcManager reimplements the same LRU logic on the same
# short-timeout wait pattern as gds_load_io, which is known to work here.
_GC_SINGLETON: "GdsGcManager | None" = None
_GC_LOCK = threading.Lock()

_GC_INTERVAL_S = 300.0
_GC_LOW_WATERMARK = 0.9
_GC_GRACE_S = 300.0
_BLOCK_SUFFIX = ".bin"


class GdsGcManager:
    """LRU on-disk budget manager for the GDS chunk tree.

    API-compatible with the upstream FsGCManager (protect/release/touch are
    used by the scheduler-side manager; headless worker processes only run
    the sweep thread). The sweep thread uses a 50ms wait tick like
    gds_load_io, so it cannot be frozen the way the upstream one was.
    """

    def __init__(
        self, file_mapper: FileMapper, gc_max_size_gb: float
    ) -> None:
        # The scheduler builds chunk paths from ITS FileMapper (rank-0 model
        # fingerprint) and broadcasts them; every node persists a local
        # <base>_r0 tree under that fingerprint. Worker-side FileMappers
        # carry a different fingerprint (per-rank fields), so do not trust
        # base_path here: probe the actual tree instead.
        root_dir = os.path.dirname(file_mapper.base_path)
        try:
            cands = [
                d for d in os.listdir(root_dir)
                if d.endswith("_r0") and d.startswith("_models_")
            ]
            newest = max(
                cands,
                key=lambda d: os.path.getmtime(os.path.join(root_dir, d)),
            )
            self.block_root = os.path.join(root_dir, newest)
        except OSError:
            self.block_root = f"{file_mapper.base_path}_r{file_mapper.rank}"
        self._to_path = file_mapper.get_file_name
        self.max_bytes = int(gc_max_size_gb * 2**30)
        self.target_bytes = int(self.max_bytes * _GC_LOW_WATERMARK)
        self._lock = threading.Lock()
        self._protected: dict[str, int] = {}
        self._stop = threading.Event()
        self._last_sweep = 0.0
        self._thread = threading.Thread(
            target=self._loop, name="gds_gc", daemon=True
        )
        self._thread.start()
        logger.info(
            "GDS GC enabled: cap %.1f GiB, sweeping to %.1f GiB every %.0fs "
            "root=%s",
            self.max_bytes / 2**30,
            self.target_bytes / 2**30,
            _GC_INTERVAL_S,
            self.block_root,
        )

    def protect(self, keys) -> None:
        with self._lock:
            for k in keys:
                try:
                    p = self._to_path(k)
                except Exception:
                    continue
                self._protected[p] = self._protected.get(p, 0) + 1

    def release(self, keys) -> None:
        with self._lock:
            for k in keys:
                try:
                    p = self._to_path(k)
                except Exception:
                    continue
                c = self._protected.get(p)
                if c is None:
                    continue
                if c <= 1:
                    del self._protected[p]
                else:
                    self._protected[p] = c - 1

    def touch(self, keys) -> None:
        pass

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(0.05)
            now = time.time()
            if now - self._last_sweep < _GC_INTERVAL_S:
                continue
            self._last_sweep = now
            try:
                self.sweep()
            except Exception as e:
                logger.warning("GDS GC sweep failed: %s", e)

    def sweep(self) -> int:
        t0 = time.time()
        blocks = []
        total = 0
        for dp, _dn, fn in os.walk(self.block_root):
            for f in fn:
                if not f.endswith(_BLOCK_SUFFIX):
                    continue
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                blocks.append((st.st_mtime, p, st.st_size))
                total += st.st_size
        if total <= self.max_bytes:
            logger.info(
                "GDS GC: %.1f GiB in %d blocks, under the %.1f GiB cap "
                "(walk %.1fs)",
                total / 2**30, len(blocks), self.max_bytes / 2**30,
                time.time() - t0,
            )
            return 0
        cutoff = time.time() - _GC_GRACE_S
        with self._lock:
            protected = set(self._protected)
        blocks.sort()
        freed = 0
        removed = 0
        for mtime, path, size in blocks:
            if total - freed <= self.target_bytes:
                break
            if path in protected or mtime > cutoff:
                continue
            try:
                os.unlink(path)
            except OSError:
                continue
            freed += size
            removed += 1
            if removed % 50000 == 0:
                logger.info(
                    "GDS GC: freed so far %.1f GiB in %d blocks (%.1fs)",
                    freed / 2**30, removed, time.time() - t0,
                )
        logger.info(
            "GDS GC: %.1f GiB over the %.1f GiB cap, freed %.1f GiB in %d "
            "blocks, now %.1f GiB (%.1fs)",
            total / 2**30, self.max_bytes / 2**30, freed / 2**30,
            removed, (total - freed) / 2**30, time.time() - t0,
        )
        return freed


def get_gc(
    file_mapper: FileMapper, gc_max_size_gb: float | None
) -> "GdsGcManager | None":
    """Return the per-process GDS GC (None if gc_max_size_gb unset)."""
    global _GC_SINGLETON
    if gc_max_size_gb is None:
        return None
    with _GC_LOCK:
        if _GC_SINGLETON is None:
            _GC_SINGLETON = GdsGcManager(file_mapper, gc_max_size_gb)
        return _GC_SINGLETON


def _pad4k(n: int) -> int:
    return (n + 4095) // 4096 * 4096


class GdsChunkSpec(BlockIDsLoadStoreSpec):
    """Paths of the chunk files backing one transfer job (key order)."""

    def __init__(self, paths: list[str]):
        super().__init__(list(range(len(paths))))
        self.paths = paths


class FileFinderShim:
    """Duck-typed stand-in exposing only get_file_name."""

    def __init__(self, fm: FileMapper):
        self._fm = fm

    def get_file_name(self, key: OffloadKey) -> str:
        return self._fm.get_file_name(key)


class GdsAsyncLookupManager(AsyncLookupManager):
    def __init__(self, shim: FileFinderShim):
        super().__init__(tier_type="gds_fs")
        self._shim = shim

    def batch_lookup(self, keys, req_context):
        return (os.path.exists(self._shim.get_file_name(k)) for k in keys)


class GdsOffloadingManager(OffloadingManager):
    """Scheduler-side state for the SSD-only layer."""

    def __init__(
        self,
        file_mapper: FileMapper,
        max_inflight_stores: int,
        gc_max_size_gb: float | None,
        enable_events: bool,
        sync_lookup: bool = True,
    ):
        self.file_mapper = file_mapper
        self._shim = FileFinderShim(file_mapper)
        self._lookup_mgr = GdsAsyncLookupManager(self._shim)
        # Synchronous existence checks in lookup(): eliminates RETRY/deferred
        # scheduling storms entirely (one exists() syscall per unknown key).
        self._sync_lookup = sync_lookup
        self._entries: dict[OffloadKey, int] = {}
        self._inflight_stores = 0
        # Loads must also count as pending work: while a request's prefix is
        # being read back from SSD the engine has nothing to compute, and if
        # has_pending_work() returned False the engine would idle-sleep,
        # starving the load pipeline (observed ~0.5s/path).
        self._inflight_loads = 0
        self._dbg = os.environ.get("VLLM_GDS_DEBUG") == "1"
        self._max_inflight_stores = max_inflight_stores
        self._gc = get_gc(file_mapper, gc_max_size_gb)
        config_path = file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(file_mapper.get_run_config(), f, indent=2, sort_keys=True)
        self._events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )

    def _paths(self, keys) -> list[str]:
        return [self.file_mapper.get_file_name(k) for k in keys]

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        st = self._entries.get(key)
        if st == READY:
            if self._dbg and (getattr(req_context, "_dlk", 0) or 0) % 1000 == 0:
                logger.warning(
                    "GDSDBG lookup READY g=%d %s req=%s",
                    get_offload_group_idx(key), key.hex()[:16],
                    req_context.req_id,
                )
            return LookupResult.HIT
        if st is not None:  # STORE_INFLIGHT / LOAD_INFLIGHT
            return LookupResult.HIT_PENDING
        res = self._lookup_mgr.lookup(key, req_context)
        if res is None:
            if self._sync_lookup:
                # Fast path: resolve inline; never return RETRY so the
                # scheduler never defers on our account.
                hit = os.path.exists(self._shim.get_file_name(key))
                if hit:
                    self._lookup_mgr.mark_present([key])
                    if self._dbg:
                        logger.warning(
                            "GDSDBG lookup EXISTHIT g=%d %s req=%s",
                            get_offload_group_idx(key), key.hex()[:16],
                            req_context.req_id,
                        )
                    return LookupResult.HIT
                self._lookup_mgr.invalidate([key])
                if self._dbg:
                    logger.warning(
                        "GDSDBG lookup MISS g=%d %s req=%s",
                        get_offload_group_idx(key), key.hex()[:16],
                        req_context.req_id,
                    )
                return LookupResult.MISS
            return LookupResult.RETRY
        return LookupResult.HIT if res else LookupResult.MISS

    @override
    def prepare_load(self, keys, req_context) -> LoadStoreSpec:
        # Keys absent from _entries were HIT via on-disk existence lookup
        # (cross-restart reuse); they are implicitly READY.
        ready = [
            k for k in keys if self._entries.get(k, READY) == READY
        ]
        if self._gc is not None:
            self._gc.protect(ready)
        for k in ready:
            self._entries[k] = LOAD_INFLIGHT
        if ready:
            self._inflight_loads += len(ready)
        if self._dbg:
            logger.warning(
                "GDSDBG prepare_load keys=%d req=%s", len(ready), req_context.req_id
            )
        return GdsChunkSpec(self._paths(ready))

    @override
    def complete_load(self, keys, req_context):
        self._inflight_loads = max(0, self._inflight_loads - len(keys))
        for k in keys:
            if self._entries.get(k) == LOAD_INFLIGHT:
                self._entries[k] = READY
        if self._gc is not None:
            self._gc.release(list(keys))

    @override
    def touch(self, keys, req_context):
        if self._gc is not None:
            self._gc.touch(keys)

    @override
    def prepare_store(self, keys, req_context) -> PrepareStoreOutput | None:
        # NOTE: no admission cap here. The connector advances
        # next_stored_chunk_idx past every chunk it offers in a call, so any
        # key we decline here is permanently skipped -> permanent holes in
        # the stored prefix (observed: only ~200 of ~10k chunks of a P128
        # request ever landed on disk; replay prefix-lookup then broke at the
        # first hole and external hits converged to 0). The device ring is
        # the natural backpressure point: jobs queue per-path and slots
        # recycle, so admitting everything is safe.
        fresh = [k for k in keys if self._entries.get(k) is None]
        if not fresh:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=GdsChunkSpec([]),
                evicted_keys=[],
            )
        if self._dbg:
            logger.warning(
                "GDSDBG prepare_store fresh=%d req=%s",
                len(fresh), req_context.req_id,
            )
        for k in fresh:
            self._entries[k] = STORE_INFLIGHT
        self._inflight_stores += len(fresh)
        if self._gc is not None:
            self._gc.protect(fresh)
        paths = self._paths(fresh)
        # Create hash-subdirectories up front so worker O_CREAT never fails.
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
        return PrepareStoreOutput(
            keys_to_store=fresh,
            store_spec=GdsChunkSpec(paths),
            evicted_keys=[],
        )

    @override
    def complete_store(self, keys, req_context, success: bool = True):
        self._inflight_stores = max(0, self._inflight_stores - len(keys))
        for k in keys:
            if self._entries.get(k) == STORE_INFLIGHT:
                if success:
                    self._entries[k] = READY
                    if self._events is not None:
                        self._events.append(
                            OffloadingEvent(
                                keys=[k],
                                medium=MEDIUM_FS,
                                removed=False,
                                locality=Locality.LOCAL,
                            )
                        )
                else:
                    del self._entries[k]
        if success:
            self._lookup_mgr.mark_present(list(keys))
        if self._gc is not None:
            self._gc.release(list(keys))

    @override
    def on_new_request(self, req_context) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context) -> None:
        self._lookup_mgr.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_mgr.flush()

    @override
    def has_pending_work(self) -> bool:
        return self._inflight_stores > 0 or self._inflight_loads > 0

    @override
    def take_events(self):
        if self._events is not None:
            yield from self._events
            self._events.clear()

    @override
    def reset_cache(self) -> None:
        # Files persist on disk; existence lookups rediscover them.
        self._entries.clear()
        self._inflight_stores = 0

    @override
    def get_stats(self):
        return None

    @override
    def shutdown(self) -> None:
        self._lookup_mgr.shutdown()
        if self._gc is not None:
            self._gc.shutdown()


@dataclass
class _PathItem:
    """One chunk-file within a job: static descriptor plan + runtime state."""

    path_idx: int
    group: int
    gpu_ptrs: np.ndarray   # int64 device pointers (fixed for the job)
    offs: np.ndarray       # int64 byte offsets inside the chunk file/slot
    sizes: np.ndarray      # int64 copy sizes
    nbytes: int            # total logical bytes of this chunk
    slot: int = -1
    stage: str = ""        # "", "gather", "gathered", "io", "read_done", "scatter", "scattered"
    ev_start: object = None
    ev_end: object = None
    future: object = None
    ok: bool = False
    retries: int = 0


@dataclass
class _Job:
    job_id: int
    is_store: bool
    paths: list[str]
    gpu_spec: GPULoadStoreSpec
    num_bytes: int = 0
    items: list = field(default_factory=list)   # index == path_idx
    next_path: int = 0                          # next item to start
    done_count: int = 0
    ok_all: bool = True
    t0: float = 0.0


class GdsOffloadingWorker(OffloadingWorker):
    """Worker-side GPU<->SSD direct transfers over a registered device ring.

    Threading model (GB10 constraints):
      - cuFile IO dispatch/harvest runs on a dedicated background thread, so
        load progress never depends on engine step cadence.
      - ALL CUDA work (batched D2D gather/scatter via swap_blocks_batch and
        event record/query) stays on the engine thread — cuMemcpyBatchAsync
        is thread-context sensitive here.
      - The scheduler-side manager reports has_pending_work() while loads are
        in flight, keeping the engine busy-stepping so scatter issuance and
        completion polling run at full rate.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        ring_depth: int,
        max_io_threads: int = 8,
        gc: GdsGcManager | None = None,
    ):
        self.refs = kv_caches.group_data_refs
        self.views = [
            t.tensor.view(torch.int8).view((-1, t.page_size_bytes))
            for t in kv_caches.tensors
        ]
        self.blocks_per_chunk = blocks_per_chunk
        self._dbg = os.environ.get("VLLM_GDS_DEBUG") == "1"

        # Per-group full-chunk byte size (fixed file/slot size per group).
        self._group_chunk_bytes = []
        for group_refs in self.refs:
            total = sum(r.page_size_bytes for r in group_refs)
            self._group_chunk_bytes.append(total * blocks_per_chunk)
        self.slot_bytes = _pad4k(max(self._group_chunk_bytes))

        # (base_ptr, extent_bytes) of every tensor region per group, used to
        # validate planned copy ops before handing them to the driver.
        self._group_ranges = []
        for group_refs in self.refs:
            rng = []
            for r in group_refs:
                t = self.views[r.tensor_idx]
                rng.append((int(t.data_ptr()), int(t.numel())))
            self._group_ranges.append(rng)

        from vllm.platforms import current_platform
        from vllm.v1.kv_offload.gds.cufile_io import get_pool

        self.ring_depth = max(4, ring_depth)
        self.pool, self.ring = get_pool(self.slot_bytes, self.ring_depth)
        self._gc = gc
        logger.info(
            "GDS worker: groups=%d chunk_bytes=%s max_slot=%.2f MiB depth=%d",
            len(self.refs),
            self._group_chunk_bytes,
            self.slot_bytes / 2**20,
            self.ring_depth,
        )
        self._executor = ThreadPoolExecutor(max_workers=max_io_threads)

        self._free_slots = list(range(self.ring_depth))
        self._jobs_store: deque[_Job] = deque()   # engine-thread driven
        self._jobs_load: deque[_Job] = deque()    # background-thread driven
        self._inflight_gather: deque[_PathItem] = deque()
        self._pending_scatter: deque[_PathItem] = deque()
        self._io_store: dict[tuple, tuple] = {}
        self._io_load: dict[tuple, tuple] = {}
        self._results: list[TransferResult] = []
        self._done_results: dict[int, TransferResult] = {}
        self._lock = threading.Lock()

        self._cur_stream = current_platform.current_stream
        self._stream_ctx = current_platform.stream
        self.store_stream = current_platform.Stream()
        self.load_stream = current_platform.Stream()
        self._last_store_ev = None
        self._last_load_ev = None
        self._stop_evt = threading.Event()
        self._ld_thread = threading.Thread(
            target=self._load_loop, name="gds_load_io", daemon=True
        )
        self._ld_thread.start()

    def _slot_pad(self, group: int) -> int:
        return _pad4k(self._group_chunk_bytes[group])

    # ---- planning ----------------------------------------------------------

    def _plan_job(self, job: _Job) -> None:
        """Split GPULoadStoreSpec into per-path static descriptors."""
        spec = job.gpu_spec
        gs = spec.group_sizes
        bis = spec.block_indices
        assert len(gs) == len(self.refs) == len(bis)
        bids_all = spec.block_ids
        off = 0
        first_path = 0
        for g, (n_g, bi_g) in enumerate(zip(gs, bis)):
            n_g = int(n_g)
            skip = int(bi_g) % self.blocks_per_chunk
            n_chunks = -(-((n_g + skip) // self.blocks_per_chunk))
            assert first_path + n_chunks <= len(job.paths)
            refs = self.refs[g]
            bids_g = bids_all[off : off + n_g]
            for c in range(n_chunks):
                lo = max(0, c * self.blocks_per_chunk - skip)
                hi = min(n_g, (c + 1) * self.blocks_per_chunk - skip)
                cnt = hi - lo
                if cnt <= 0:
                    # Degenerate slice (shouldn't happen; keep path indexing
                    # intact by emitting an empty item).
                    job.items.append(
                        _PathItem(
                            path_idx=first_path + c,
                            group=g,
                            gpu_ptrs=np.empty(0, dtype=np.int64),
                            offs=np.empty(0, dtype=np.int64),
                            sizes=np.empty(0, dtype=np.int64),
                            nbytes=0,
                        )
                    )
                    first_path += 1
                    continue
                n_ops = cnt * len(refs)
                gpu = np.empty(n_ops, dtype=np.int64)
                offs = np.empty(n_ops, dtype=np.int64)
                sizes = np.empty(n_ops, dtype=np.int64)
                k = 0
                nbytes = 0
                for j in range(lo, hi):
                    bid = int(bids_g[j])
                    # Chunk-file-relative block index: each file holds
                    # blocks_per_chunk logical blocks starting at its own
                    # offset 0 (unpadded head chunks begin at `skip`).
                    rel = skip + j - c * self.blocks_per_chunk
                    for r in refs:
                        t = self.views[r.tensor_idx]
                        page = r.page_size_bytes
                        gpu[k] = int(t.data_ptr()) + bid * int(t.stride(0))
                        offs[k] = rel * page
                        sizes[k] = page
                        nbytes += page
                        k += 1
                assert k == n_ops
                job.items.append(
                    _PathItem(
                        path_idx=first_path + c,
                        group=g,
                        gpu_ptrs=gpu,
                        offs=offs,
                        sizes=sizes,
                        nbytes=nbytes,
                    )
                )
            first_path += n_chunks
            off += n_g
        assert off == len(bids_all)
        assert first_path == len(job.paths), (
            f"path count {len(job.paths)} != planned chunks {first_path}"
        )

    def _validate_item(self, it: _PathItem) -> str | None:
        """Return a description of the first invalid copy op, or None."""
        rng = self._group_ranges[it.group]
        nrefs = len(rng)
        slot_lim = self.slot_bytes
        for k in range(len(it.gpu_ptrs)):
            b, e = rng[k % nrefs]
            p = int(it.gpu_ptrs[k])
            o = int(it.offs[k])
            s = int(it.sizes[k])
            if s <= 0 or o < 0 or o + s > slot_lim:
                return f"op{k}: slot_off={o} size={s} slot_lim={slot_lim}"
            if not (b <= p and p + s <= b + e):
                return (
                    f"op{k}: gpu_ptr={p:#x} size={s} outside region "
                    f"[{b:#x},{b + e:#x})"
                )
        return None

    def _issue_item_copy(self, item: _PathItem, is_store: bool):
        """Engine-thread only: enqueue the D2D batch for one path."""
        from vllm import _custom_ops as ops

        bad = self._validate_item(item)
        if bad is not None:
            # Scheduler/worker inconsistency (should not happen once offs are
            # chunk-relative). Skip copy+IO but keep the job alive: upstream
            # connector asserts success on every finished job, so a failed
            # path must not poison it. The skipped file simply never appears
            # -> lookup MISS -> tokens recomputed (graceful hole).
            logger.error(
                "GDSDBG invalid %s item job=%d path=%d group=%d: %s",
                "store" if is_store else "load",
                item.job_ref.job_id,
                item.path_idx,
                item.group,
                bad,
            )
            item.stage = "bad"
            return
        s = self.store_stream if is_store else self.load_stream
        last = self._last_store_ev if is_store else self._last_load_ev
        if is_store:
            s.wait_stream(self._cur_stream())
        if last is not None:
            s.wait_event(last)
        slot_base = int(self.ring.data_ptr()) + item.slot * self.slot_bytes
        pos = slot_base + item.offs.astype(np.int64)
        b_gpu = torch.from_numpy(item.gpu_ptrs.copy())
        b_pos = torch.from_numpy(pos)
        b_sizes = torch.from_numpy(item.sizes.copy())
        with self._stream_ctx(s):
            item.ev_start.record(s)
            if len(item.gpu_ptrs) > 0:
                ops.swap_blocks_batch(
                    b_gpu if is_store else b_pos,
                    b_pos if is_store else b_gpu,
                    b_sizes,
                    is_src_access_order_any=not is_store,
                )
            item.ev_end.record(s)
        if is_store:
            self._last_store_ev = item.ev_end
        else:
            self._last_load_ev = item.ev_end

    # ---- store pipeline (engine thread) ------------------------------------

    def _pump_store(self):
        alive = deque()
        while self._jobs_store:
            job = self._jobs_store.popleft()
            while job.next_path < len(job.items) and self._free_slots:
                it = job.items[job.next_path]
                job.next_path += 1
                it.slot = self._free_slots.pop()
                it.job_ref = job
                it.is_store_item = True
                it.ev_start = torch.Event(enable_timing=True)
                it.ev_end = torch.Event(enable_timing=True)
                self._issue_item_copy(it, True)
                if it.stage == "bad":
                    # Skip copy+IO; recycle slot and count the path done.
                    self._path_done(job, it)
                    continue
                it.stage = "gather"
                self._inflight_gather.append(it)
            if job.next_path < len(job.items):
                alive.append(job)
        for j in reversed(alive):
            self._jobs_store.appendleft(j)

        remaining = deque()
        while self._inflight_gather:
            it = self._inflight_gather.popleft()
            if it.ev_end.query():
                job = it.job_ref
                it.t0_read = time.monotonic()
                it.future = self._executor.submit(
                    self.pool.write_slot,
                    job.paths[it.path_idx],
                    it.slot,
                    self._slot_pad(it.group),
                )
                it.stage = "io"
                self._io_store[(job.job_id, it.path_idx)] = (job, it)
            else:
                remaining.append(it)
        self._inflight_gather.extend(remaining)

        for key, (job, it) in list(self._io_store.items()):
            fut = it.future
            if fut is not None and fut.done():
                del self._io_store[key]
                it.ok = bool(fut.result())
                if not it.ok and it.retries < 3:
                    # Slot data is still valid; just redo the cuFileWrite.
                    it.retries += 1
                    it.future = self._executor.submit(
                        self.pool.write_slot,
                        job.paths[it.path_idx],
                        it.slot,
                        self._slot_pad(it.group),
                    )
                    self._io_store[key] = (job, it)
                    continue
                self._path_done(job, it)

    # ---- load pipeline (background thread + engine scatter) ----------------

    def _load_loop(self):
        while not self._stop_evt.is_set():
            worked = False
            with self._lock:
                worked = self._pump_load_reads()
            if not worked:
                self._stop_evt.wait(0.0005)

    def _pump_load_reads(self) -> bool:
        """Bg-thread: dispatch cuFileReads and harvest completions.
        Lock must be held. Returns whether any work was done."""
        worked = False
        alive = deque()
        while self._jobs_load:
            job = self._jobs_load.popleft()
            while job.next_path < len(job.items) and self._free_slots:
                it = job.items[job.next_path]
                job.next_path += 1
                it.slot = self._free_slots.pop()
                it.job_ref = job
                it.is_store_item = False
                it.t0_read = time.monotonic()
                it.future = self._executor.submit(
                    self.pool.read_slot,
                    job.paths[it.path_idx],
                    it.slot,
                    self._slot_pad(it.group),
                )
                it.stage = "io"
                self._io_load[(job.job_id, it.path_idx)] = (job, it)
                worked = True
            if job.next_path < len(job.items):
                alive.append(job)
        for j in alive:
            self._jobs_load.append(j)

        for key, (job, it) in list(self._io_load.items()):
            fut = it.future
            if fut is not None and fut.done():
                del self._io_load[key]
                it.ok = bool(fut.result())
                worked = True
                if not it.ok and it.retries < 3:
                    # Read failed (e.g. transient fd pressure). The source
                    # file may have been unlinked by the self-heal path, in
                    # which case retries will keep failing and the job ends
                    # unsuccessful; the scheduler-side hole logic applies.
                    it.retries += 1
                    time.sleep(0.05 * it.retries)
                    it.future = self._executor.submit(
                        self.pool.read_slot,
                        job.paths[it.path_idx],
                        it.slot,
                        self._slot_pad(it.group),
                    )
                    self._io_load[key] = (job, it)
                    continue
                if it.ok:
                    it.ev_start = torch.Event(enable_timing=True)
                    it.ev_end = torch.Event(enable_timing=True)
                    it.stage = "read_done"
                    self._pending_scatter.append(it)
                else:
                    self._path_done(job, it)
        return worked

    def _pump_scatter(self):
        """Engine-thread: issue scatters and harvest their completion."""
        rem = deque()
        while self._pending_scatter:
            it = self._pending_scatter.popleft()
            if it.stage == "read_done":
                self._issue_item_copy(it, False)
                if it.stage == "bad":
                    # Skip scatter; recycle slot and count the path done.
                    self._path_done(it.job_ref, it)
                    continue
                it.stage = "scattered"
            if it.stage == "scattered":
                if it.ev_end.query():
                    self._path_done(it.job_ref, it)
                else:
                    rem.append(it)
        self._pending_scatter.extend(rem)

    def _path_done(self, job: _Job, it: _PathItem):
        """Lock must be held by caller."""
        self._free_slots.append(it.slot)
        job.done_count += 1
        if not it.ok:
            job.ok_all = False
        if job.done_count >= len(job.items):
            dt = max(time.monotonic() - job._t0, 1e-6)
            result = TransferResult(
                job_id=job.job_id,
                success=job.ok_all,
                transfer_size=job.num_bytes,
                transfer_time=dt,
            )
            if self._dbg:
                logger.warning(
                    "GDSDBG job done id=%d store=%s paths=%d ok=%s "
                    "bytes=%d dt=%.3f",
                    job.job_id, job.is_store, len(job.items),
                    job.ok_all, job.num_bytes, dt,
                )
            self._results.append(result)
            self._done_results[job.job_id] = result

    def _pump(self):
        self._pump_store()
        self._pump_scatter()

    # ---- OffloadingWorker API ----------------------------------------------

    def _submit(self, job: _Job) -> bool:
        if not job.paths:
            with self._lock:
                res = TransferResult(job_id=job.job_id, success=True)
                self._results.append(res)
                self._done_results[job.job_id] = res
            return True
        self._plan_job(job)
        job.num_bytes = sum(
            int(g) * sum(r.page_size_bytes for r in self.refs[g])
            for g, gsz in enumerate(job.gpu_spec.group_sizes)
        )
        job._t0 = time.monotonic()
        if self._dbg:
            logger.warning(
                "GDSDBG job submit id=%d store=%s paths=%d",
                job.job_id, job.is_store, len(job.paths),
            )
        q = self._jobs_store if job.is_store else self._jobs_load
        with self._lock:
            q.append(job)
            if job.is_store:
                self._pump()
        return True

    @override
    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        paths = list(dst_spec.paths)
        return self._submit(
            _Job(
                job_id=job_id,
                is_store=True,
                paths=paths,
                gpu_spec=src_spec,
            )
        )

    @override
    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        paths = list(src_spec.paths)
        if not paths:
            with self._lock:
                res = TransferResult(job_id=job.job_id, success=True)
                self._results.append(res)
                self._done_results[job.job_id] = res
            return True
        return self._submit(
            _Job(
                job_id=job_id,
                is_store=False,
                paths=paths,
                gpu_spec=dst_spec,
            )
        )

    @override
    def get_finished(self) -> list[TransferResult]:
        with self._lock:
            self._pump()
            out = list(self._results)
            self._results.clear()
        return out

    @override
    def wait(self, job_ids: set[int]) -> None:
        remaining = set(job_ids)
        while remaining:
            with self._lock:
                self._pump()
                done = {
                    jid for jid in remaining if jid in self._done_results
                }
            remaining -= done
            if remaining:
                time.sleep(0.0002)

    @override
    def shutdown(self) -> None:
        self._stop_evt.set()
        self._ld_thread.join(timeout=5)
        self._executor.shutdown(wait=True)
        self.pool.shutdown()
        if self._gc is not None:
            self._gc.shutdown()

class GdsDirectOffloadingSpec(OffloadingSpec):
    """
    kv_connector_extra_config:
      root_dir:            SSD dir for chunk files
                           (default /root/.cache/kv_offload_fs)
      gc_max_size_gb:      on-disk LRU budget (default None = unbounded)
      ring_depth:          device ring slots (default env VLLM_GDS_RING_DEPTH
                           or 32); each slot ~= largest padded group chunk
      max_inflight_stores: DEPRECATED/no-op (admission is uncapped; see
                           prepare_store)
    """

    def _ring_depth(self) -> int:
        return int(
            self.extra_config.get(
                "ring_depth", os.environ.get("VLLM_GDS_RING_DEPTH", 64)
            )
        )

    @override
    def get_manager(self) -> OffloadingManager:
        fm = FileMapper.from_offloading_spec(
            root_dir=self.extra_config.get(
                "root_dir", "/root/.cache/kv_offload_fs"
            ),
            offloading_spec=self,
            blocks_per_file=self.blocks_per_chunk,
            parallel_agnostic=True,
        )
        depth = self._ring_depth()
        self._file_mapper = fm
        self._manager = GdsOffloadingManager(
            file_mapper=fm,
            max_inflight_stores=int(
                self.extra_config.get("max_inflight_stores", depth)
            ),
            gc_max_size_gb=self.extra_config.get("gc_max_size_gb"),
            enable_events=self.kv_events_config.enable_kv_cache_events,
            sync_lookup=bool(self.extra_config.get("sync_lookup", True)),
        )
        logger.info(
            "GdsDirectOffloadingManager ready: root=%s depth=%d",
            fm.base_path,
            depth,
        )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        fm = FileMapper.from_offloading_spec(
            root_dir=self.extra_config.get(
                "root_dir", "/root/.cache/kv_offload_fs"
            ),
            offloading_spec=self,
            blocks_per_file=self.blocks_per_chunk,
            parallel_agnostic=True,
        )
        # FsGCManager sweeps <base>_r<rank>. Every node persists chunks under
        # the rank-0 path broadcast by the scheduler (each machine keeps its
        # own local <base>_r0 tree), so pin rank 0 here; with the TP rank the
        # sweep would scan a nonexistent _r1/_r2/... dir and stay silent.
        fm.rank = 0
        return GdsOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            ring_depth=self._ring_depth(),
            gc=get_gc(fm, self.extra_config.get("gc_max_size_gb")),
        )
