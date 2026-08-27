"""F6: relax the success assert in upstream offloading/worker.py (idempotent).

Run INSIDE the vllm container:
    python3 patch_offloading_worker_assert.py

Upstream get_finished() asserts every TransferResult succeeded ("we currently
do not support job failures"). The GDS backend now retries IO internally and
degrades unrecoverable store failures to holes (file absent -> lookup MISS ->
tokens recomputed), so a failed job must log-and-continue instead of killing
the worker process.
"""
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py"
OLD = """            job_id = transfer_result.job_id
            assert transfer_result.success
"""
NEW = """            job_id = transfer_result.job_id
            if not transfer_result.success:
                logger.error(
                    "Offloading transfer job %s reported failure; "
                    "degrading (missing chunk -> lookup MISS / recompute) "
                    "instead of crashing the engine",
                    job_id,
                )
"""

src = open(PATH).read()
if "degrading (missing chunk" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)
assert OLD in src, "pattern not found - upstream version drift"
open(PATH, "w").write(src.replace(OLD, NEW))
print("PATCHED:", PATH)
