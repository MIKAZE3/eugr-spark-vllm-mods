#!/bin/bash
# gds_direct_v1 deploy: install the GDS-direct KV offload package + patches
# into the running vllm container on THIS node. Run on the GPU node host:
#   bash /home/yztai/gds_direct_v1/run.sh
#
# Expects (any ONE of) the package sources to exist:
#   /home/yztai/gds_bundle/           (tarball extracted: contains pkg/gds)
#   <this dir>/pkg/gds                (mod dir pushed verbatim)
set -e
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
CTR=eugr-b12x-node-fp8-dspark
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm

SRC=""
for d in /home/yztai/gds_bundle "$MOD_DIR"; do
  if [ -d "$d/pkg/gds" ]; then SRC="$d"; break; fi
done
[ -n "$SRC" ] || { echo "ERROR: no pkg/gds found (looked in /home/yztai/gds_bundle, $MOD_DIR)" >&2; exit 1; }
echo "[gds_direct_v1] package source: $SRC"

# 1) install python package
docker cp "$SRC/pkg/gds" "$CTR:$VLLM_PKG/v1/kv_offload/"

# 2) F6 patch: relax upstream success assert (idempotent)
PATCH="$SRC/patch_offloading_worker_assert.py"
[ -f "$PATCH" ] || PATCH="$MOD_DIR/patch_offloading_worker_assert.py"
docker cp "$PATCH" "$CTR:/tmp/gds_patch_assert.py"
docker exec "$CTR" python3 /tmp/gds_patch_assert.py

# 3) import self-check (catches NameError/syntax issues BEFORE engine start)
docker exec "$CTR" python3 - <<'PY'
from vllm.v1.kv_offload.gds import spec, cufile_io
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker as ow
assert hasattr(spec, "GdsDirectOffloadingSpec")
print("IMPORT_OK: GdsDirectOffloadingSpec ready")
PY

echo "[gds_direct_v1] deployed. Restart the cluster to load the new code."
