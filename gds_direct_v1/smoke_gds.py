import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1) import chain: pulls FileMapper, FsGCManager, AsyncLookupManager, ops
from vllm.v1.kv_offload.gds.spec import (
    GdsChunkSpec,
    GdsDirectOffloadingSpec,
)
from vllm.v1.kv_offload.gds.cufile_io import CuFilePool, pad4k
print("[1] imports OK", flush=True)

# 2) cuFile ring roundtrip through the real pool API
SLOT = pad4k(5 * 1024 * 1024)
DEPTH = 3
ring = torch.zeros(DEPTH * SLOT, dtype=torch.uint8, device="cuda")
pool = CuFilePool(ring.data_ptr(), ring.numel(), DEPTH)
print(f"[1b] registration mode = {pool.mode}", flush=True)

pattern = torch.arange(0, SLOT, dtype=torch.int32, device="cuda") % 251
pattern = pattern.to(torch.uint8)
ring[:SLOT] = pattern

path = "/root/.cache/kv_offload_fs/__gds_smoke__/t.bin"
t0 = time.time()
ok_w = pool.write_slot(path, 0, SLOT)
t1 = time.time()
ring[SLOT : 2 * SLOT] = 0
ok_r = pool.read_slot(path, 1, SLOT)
torch.cuda.synchronize()
t2 = time.time()
same = torch.equal(ring[:SLOT], ring[SLOT : 2 * SLOT])
mb = SLOT / 1e6
print(
    f"[2] write={ok_w}({mb/(t1-t0)/1e3:.2f} GB/s) read={ok_r}({mb/(t2-t1)/1e3:.2f} GB/s) "
    f"data_equal={same}",
    flush=True,
)

# 3) spec picklability (connector meta crosses processes)
spec = GdsChunkSpec(["/tmp/a.bin", "/tmp/b.bin"])
import pickle

spec2 = pickle.loads(pickle.dumps(spec))
print(f"[3] pickle OK paths={spec2.paths} block_ids={spec2.block_ids.tolist()}", flush=True)

pool.shutdown()
os.unlink(path)
os.rmdir(os.path.dirname(path))
ok = ok_w and ok_r and same
print("GDS-SMOKE-PASS" if ok else "GDS-SMOKE-FAIL", flush=True)
sys.exit(0 if ok else 1)
