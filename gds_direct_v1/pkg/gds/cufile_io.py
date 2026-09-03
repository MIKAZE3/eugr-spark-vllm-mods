# GDS (cuFile) device-path IO backend for KV cache offloading on GB10.
# Validated constraints (2026-08-23 PoC):
#   - CUfileDescr_t union MUST carry both fd(int) and handle(void*)
#   - cuFileHandleRegister.restype is a packed u64 (err | cu<<32)
#   - host-registered buffers SEGV on GB10: device pointers ONLY
#
# Registration strategy (err 5048 = CU_FILE_NVFS_INTERNAL_DRIVER_ERROR seen
# in the wild on some GB10 nodes):
#   1. single full-ring registration (dev_offset addressing)
#   2. per-slot registration (slot-base addressing)
#   3. no registration: cuFile falls back to internal pinned buffers
#      (compat path; functional but slower)
# A process-wide singleton prevents double-registration when vLLM invokes
# spec.get_worker() more than once per process.

import ctypes
import collections
import os
import threading

from vllm.logger import init_logger

logger = init_logger(__name__)

_LIB = None
_LIB_LOCK = threading.RLock()
_POOL_SINGLETON = None

# Cap on simultaneously open chunk files. Each entry holds an fd plus a
# cuFile handle; uncached growth hit the fd ceiling once stores stopped
# being admission-capped (millions of chunk files over a churn phase).
HANDLE_CACHE_MAX = 4096

CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1


class CUfileDescr(ctypes.Structure):
    class U(ctypes.Union):
        _fields_ = [
            ("fd", ctypes.c_int),
            ("handle", ctypes.c_void_p),
        ]

    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", U),
        ("fs_ops", ctypes.c_void_p),
    ]


def _lib():
    global _LIB
    if _LIB is None:
        with _LIB_LOCK:
            if _LIB is None:
                lib = ctypes.CDLL("libcufile.so")
                lib.cuFileDriverOpen.restype = ctypes.c_uint64
                lib.cuFileHandleRegister.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(CUfileDescr),
                ]
                lib.cuFileHandleRegister.restype = ctypes.c_uint64
                lib.cuFileWrite.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_longlong,
                    ctypes.c_longlong,
                ]
                lib.cuFileWrite.restype = ctypes.c_ssize_t
                lib.cuFileRead.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_longlong,
                    ctypes.c_longlong,
                ]
                lib.cuFileRead.restype = ctypes.c_ssize_t
                lib.cuFileBufRegister.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                lib.cuFileBufRegister.restype = ctypes.c_uint64
                lib.cuFileBufDeregister.argtypes = [ctypes.c_void_p]
                lib.cuFileBufDeregister.restype = ctypes.c_uint64
                lib.cuFileHandleDeregister.argtypes = [ctypes.c_void_p]
                lib.cuFileHandleDeregister.restype = ctypes.c_uint64
                rc = lib.cuFileDriverOpen()
                if rc != 0:
                    raise RuntimeError(
                        f"cuFileDriverOpen failed: err={rc & 0xFFFFFFFF} "
                        f"cu={(rc >> 32) & 0xFFFFFFFF}"
                    )
                try:
                    # Headroom for compat-mode (unregistered) IO staging,
                    # used e.g. on nodes whose nvfs driver rejects
                    # registrations with err 5048. Value in KiB.
                    lib.cuFileDriverSetMaxPinnedMemSize.argtypes = [
                        ctypes.c_long
                    ]
                    lib.cuFileDriverSetMaxPinnedMemSize.restype = (
                        ctypes.c_uint64
                    )
                    rc2 = lib.cuFileDriverSetMaxPinnedMemSize(2 * 1024 * 1024)
                    if rc2 != 0:
                        logger.info(
                            "cuFileDriverSetMaxPinnedMemSize not applied "
                            "(err=%d); using driver default", rc2 & 0xFFFFFFFF
                        )
                except Exception as e:
                    logger.info("MaxPinnedMemSize setter unavailable: %s", e)
                _LIB = lib
    return _LIB


def pad4k(n: int) -> int:
    return (n + 4095) // 4096 * 4096


def get_pool(slot_bytes: int, depth: int) -> tuple["CuFilePool", "object"]:
    """Process-wide singleton owning BOTH the ring tensor and registration:
    vLLM may invoke spec.get_worker() more than once per process; a second
    registration is at best wasted work, at worst err 5048. Reusing the same
    ring tensor also prevents writes into a freed first-instance buffer."""
    global _POOL_SINGLETON
    import torch

    with _LIB_LOCK:
        if _POOL_SINGLETON is None:
            ring = torch.empty(
                depth * slot_bytes, dtype=torch.uint8, device="cuda"
            )
            _POOL_SINGLETON = (
                CuFilePool(ring.data_ptr(), ring.numel(), depth),
                ring,
            )
        else:
            logger.info("GDS pool singleton reused (skip re-registration)")
    return _POOL_SINGLETON


class CuFilePool:
    """Per-process cuFile context: handle cache + registered device ring."""

    def __init__(self, ring_ptr: int, ring_bytes: int, depth: int):
        self.lib = _lib()
        self.ring_ptr = ring_ptr
        self.ring_bytes = ring_bytes
        self.depth = depth
        # LRU-bounded handle cache: path -> (fd, cufile handle).
        self._handles: collections.OrderedDict[
            str, tuple[int, object]
        ] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.mode = self._register_ring()
        logger.info(
            "GDS ring ready ptr=%#x size=%.2f MiB mode=%s",
            self.ring_ptr,
            ring_bytes / 2**20,
            self.mode,
        )

    def _register_ring(self) -> str:
        # 1) full-range registration
        r = self.lib.cuFileBufRegister(
            ctypes.c_void_p(self.ring_ptr), self.ring_bytes, 0
        )
        if r & 0xFFFFFFFF == 0:
            return "full"
        logger.warning(
            "GDS full-ring register failed err=%d cu=%d; trying per-slot",
            r & 0xFFFFFFFF,
            (r >> 32) & 0xFFFFFFFF,
        )
        self._try_deregister(self.ring_ptr)

        # 2) per-slot registration
        slot = self.ring_bytes // self.depth
        ok_all = True
        for i in range(self.depth):
            p = ctypes.c_void_p(self.ring_ptr + i * slot)
            r = self.lib.cuFileBufRegister(p, slot, 0)
            if r & 0xFFFFFFFF != 0:
                ok_all = False
                logger.warning(
                    "GDS slot %d/%d register failed err=%d",
                    i + 1,
                    self.depth,
                    r & 0xFFFFFFFF,
                )
                self._try_deregister(self.ring_ptr + i * slot)
        if ok_all:
            return "per_slot"

        # 3) compat mode: unregistered buffers go through internal pinned
        # staging inside libcufile (slower, but keeps the engine alive).
        logger.warning(
            "GDS registration unavailable; falling back to compat mode "
            "(unregistered buffers)"
        )
        return "compat"

    def _try_deregister(self, ptr: int) -> None:
        try:
            self.lib.cuFileBufDeregister(ctypes.c_void_p(ptr))
        except Exception:
            pass

    def _buf_args(self, slot_idx: int):
        """(buf_base, dev_offset) honoring the registration mode."""
        off = slot_idx * (self.ring_bytes // self.depth)
        if self.mode == "full":
            return self.ring_ptr, off
        return self.ring_ptr + off, 0

    def _close_entry(self, entry: tuple[int, object]) -> None:
        fd, ch = entry
        try:
            self.lib.cuFileHandleDeregister(ch)
        except Exception:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def _get_handle(self, path: str):
        """Return (entry, fd, ch) with a hold on the entry.

        The caller MUST call _release_handle(entry) when done. The hold keeps
        the LRU eviction from closing the fd mid-IO (observed: fd closed by an
        eviction while another IO thread was reading -> cuFileRead returned 0
        -> self-heal unlinked a perfectly good file).
        """
        with self._lock:
            h = self._handles.pop(path, None)
            if h is not None:
                self._handles[path] = h  # move to MRU end
                h[3] += 1
                return h
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
        d = CUfileDescr()
        d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD
        d.handle.fd = fd
        d.fs_ops = None
        ch = ctypes.c_void_p()
        r = self.lib.cuFileHandleRegister(ctypes.byref(ch), ctypes.byref(d))
        err = r & 0xFFFFFFFF
        cu = (r >> 32) & 0xFFFFFFFF
        if err != 0 or cu != 0:
            os.close(fd)
            raise RuntimeError(
                f"cuFileHandleRegister({path}) failed err={err} cu={cu}"
            )
        entry = [fd, ch, path, 1]  # [fd, cufile_handle, path, refs]
        with self._lock:
            while len(self._handles) >= HANDLE_CACHE_MAX:
                # Evict the LRU entry that is not currently in use.
                victim = None
                for p in self._handles:
                    if self._handles[p][3] == 0:
                        victim = self._handles.pop(p)
                        break
                if victim is None:
                    break  # all in use; cache may exceed the soft cap
                self._close_entry((victim[0], victim[1]))
            self._handles[path] = entry
        return entry

    def _release_handle(self, entry) -> None:
        with self._lock:
            entry[3] -= 1

    def write_slot(self, path: str, slot_idx: int, nbytes: int) -> bool:
        """Write ring[slot_idx][:nbytes] -> path."""
        entry = None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            base, doff = self._buf_args(slot_idx)
            entry = self._get_handle(path)
            fd, ch = entry[0], entry[1]
            n = self.lib.cuFileWrite(
                ch,
                ctypes.c_void_p(base),
                ctypes.c_size_t(nbytes),
                ctypes.c_longlong(0),
                ctypes.c_longlong(doff),
            )
            if n != nbytes:
                self._drop_handle(path, entry)
                entry = None
                raise OSError(f"cuFileWrite short: {n}/{nbytes}")
            return True
        except Exception as e:
            logger.warning("GDS write %s failed: %s", path, e)
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
            return False
        finally:
            if entry is not None:
                self._release_handle(entry)

    def read_slot(self, path: str, slot_idx: int, nbytes: int) -> bool:
        """Read path -> ring[slot_idx][:nbytes].

        A failing read first refreshes the cached handle (a stale handle can
        read 0 after the fd was closed by an LRU eviction), retries once, and
        only then self-heals by unlinking the file.
        """
        for attempt in range(2):
            entry = None
            try:
                base, doff = self._buf_args(slot_idx)
                entry = self._get_handle(path)
                fd, ch = entry[0], entry[1]
                n = self.lib.cuFileRead(
                    ch,
                    ctypes.c_void_p(base),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_longlong(0),
                    ctypes.c_longlong(doff),
                )
                if n != nbytes:
                    self._drop_handle(path, entry)
                    entry = None
                    raise OSError(f"cuFileRead short: {n}/{nbytes}")
                return True
            except Exception as e:
                if attempt == 0:
                    logger.warning(
                        "GDS read %s failed (retrying with fresh handle): %s",
                        path, e,
                    )
                else:
                    logger.warning("GDS read %s failed: %s", path, e)
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                    return False
            finally:
                if entry is not None:
                    self._release_handle(entry)
        return False

    def _drop_handle(self, path: str, entry) -> None:
        """Remove `entry` from the cache and close its resources."""
        with self._lock:
            cur = self._handles.pop(path, None)
        if cur is not None and cur is entry:
            self._close_entry((entry[0], entry[1]))

    def shutdown(self):
        if self.mode == "full":
            self._try_deregister(self.ring_ptr)
        elif self.mode == "per_slot":
            slot = self.ring_bytes // self.depth
            for i in range(self.depth):
                self._try_deregister(self.ring_ptr + i * slot)
        with self._lock:
            entries = list(self._handles.items())
            self._handles.clear()
        for _, e in entries:
            self._close_entry((e[0], e[1]))
