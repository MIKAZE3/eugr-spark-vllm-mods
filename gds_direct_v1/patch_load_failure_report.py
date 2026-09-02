"""Idempotent patch: report load failures to the scheduler.

OffloadingConnector never implemented get_block_ids_with_load_errors(), so
the scheduler's invalid-block handling never fired: a failed load left the
request running on missing KV -> garbled output on long-context replays
(observed 09-02, 872 read failures, GC-evicted chunks). This patch:

  1. offloading/worker.py: track dst block ids per load job; collect ids of
     failed loads; expose get_block_ids_with_load_errors().
  2. offloading_connector.py: forward get_block_ids_with_load_errors().
  3. Recommend kv_load_failure_policy=recompute in the start script so a
     failed chunk is recomputed instead of aborting the request.
"""
import sys

W = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py"
C = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py"

# ---------- 1) worker.py ----------
w = open(W).read()
if "get_block_ids_with_load_errors" not in w:
    old_init = "        self._load_jobs: dict[int, ReqId] = {}"
    new_init = (
        "        self._load_jobs: dict[int, tuple[ReqId, list[int]]] = {}\n"
        "        # Block ids of async loads that failed; reported to the\n"
        "        # scheduler so the request recomputes instead of decoding\n"
        "        # on missing KV.\n"
        "        self._load_error_block_ids: set[int] = set()"
    )
    assert old_init in w, "init anchor not found"
    w = w.replace(old_init, new_init)

    old_sub = "            self._load_jobs[job_id] = entry.req_id"
    new_sub = (
        "            self._load_jobs[job_id] = (\n"
        "                entry.req_id, list(entry.dst_spec.block_ids)\n"
        "            )"
    )
    assert old_sub in w, "submit anchor not found"
    w = w.replace(old_sub, new_sub)

    old_pop = (
        "            self._connector_worker_meta.mark_completed(job_id)\n"
        "            req_id = self._load_jobs.pop(job_id, None)\n"
        "            if req_id is not None:\n"
        "                finished_recving.add(req_id)"
    )
    new_pop = (
        "            self._connector_worker_meta.mark_completed(job_id)\n"
        "            entry = self._load_jobs.pop(job_id, None)\n"
        "            if entry is not None:\n"
        "                req_id, block_ids = entry\n"
        "                if not transfer_result.success and block_ids:\n"
        "                    self._load_error_block_ids.update(block_ids)\n"
        "                finished_recving.add(req_id)"
    )
    assert old_pop in w, "pop anchor not found"
    w = w.replace(old_pop, new_pop)

    old_meta = (
        "    def shutdown(self) -> None:\n"
        "        self._unsubmitted_store_jobs.clear()"
    )
    new_meta = (
        "    def get_block_ids_with_load_errors(self) -> set[int]:\n"
        "        ids = self._load_error_block_ids\n"
        "        self._load_error_block_ids = set()\n"
        "        return ids\n"
        "\n"
        "    def shutdown(self) -> None:\n"
        "        self._unsubmitted_store_jobs.clear()"
    )
    assert old_meta in w, "method anchor not found"
    w = w.replace(old_meta, new_meta)
    open(W, "w").write(w)
    print("PATCHED:", W)
else:
    print("ALREADY_PATCHED:", W)

# ---------- 2) offloading_connector.py ----------
c = open(C).read()
if "get_block_ids_with_load_errors" not in c:
    anchor = (
        "    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:\n"
        "        if self.connector_worker is not None:\n"
        "            return self.connector_worker.build_connector_worker_meta()\n"
        "        return None"
    )
    add = anchor + (
        "\n"
        "\n"
        "    def get_block_ids_with_load_errors(self) -> set[int]:\n"
        "        if self.connector_worker is None:\n"
        "            return set()\n"
        "        return self.connector_worker.get_block_ids_with_load_errors()"
    )
    assert anchor in c, "connector anchor not found"
    c = c.replace(anchor, add)
    open(C, "w").write(c)
    print("PATCHED:", C)
else:
    print("ALREADY_PATCHED:", C)

# ---------- 3) start script: recompute policy ----------
S = "/home/yztai/eugr-b12x-logs/start_dsv4f_4tp_nolmc.sh"
import re

s = open(S).read()
if "kv_load_failure_policy" not in s:
    # inject into the kv-transfer-config JSON
    new_cfg = s.replace(
        '"kv_connector_extra_config":{',
        '"kv_load_failure_policy":"recompute","kv_connector_extra_config":{',
    )
    assert new_cfg != s, "start-script config anchor not found"
    open(S, "w").write(new_cfg)
    print("PATCHED:", S)
elif '"recompute"' not in s:
    s = s.replace('"fail"', '"recompute"')
    open(S, "w").write(s)
    print("SET_RECOMPUTE:", S)
else:
    print("ALREADY_RECOMPUTE:", S)