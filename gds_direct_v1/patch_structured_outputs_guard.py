"""Idempotent guard: tolerate grammar mask/mapping size mismatch.

vLLM's StructuredOutputsWorker.apply_grammar_bitmask asserts
num_masks == len(mapping). With DSpark spec decode the scheduler fills one
mask row per scheduled draft token (+bonus) while the worker's logits rows
can differ (dynamic spec window / padding / aborts), so the assert kills the
worker on real traffic (observed 09-02 02:04 & 03:06, EngineDeadError).
Instead of crashing, align to the shorter side and log the mismatch once.
"""
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/structured_outputs.py"
src = open(PATH).read()

if "mask/mapping mismatch" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# 1) logger import after the existing imports
old_import = "from vllm.v1.worker.gpu.input_batch import InputBatch\n"
new_import = old_import + "\nfrom vllm.logger import init_logger\n\nlogger = init_logger(__name__)\n"
assert old_import in src, "import anchor not found"
src = src.replace(old_import, new_import)

# 2) tolerant alignment instead of the assert
old_assert = (
    "        num_masks = bitmask.shape[0]\n"
    "        assert num_masks == len(mapping)\n"
)
new_assert = (
    "        num_masks = bitmask.shape[0]\n"
    "        if num_masks != len(mapping):\n"
    "            logger.error(\n"
    "                \"structured output mask/mapping mismatch: \"\n"
    "                \"num_masks=%d len(mapping)=%d; truncating to min\",\n"
    "                num_masks, len(mapping),\n"
    "            )\n"
    "            k = min(num_masks, len(mapping))\n"
    "            mapping = mapping[:k]\n"
    "            bitmask = bitmask[:k]\n"
    "            num_masks = k\n"
)
assert old_assert in src, "assert anchor not found"
src = src.replace(old_assert, new_assert)

open(PATH, "w").write(src)
print("PATCHED:", PATH)