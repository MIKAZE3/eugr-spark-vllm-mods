"""Idempotent start-script patcher for the GDS-direct KV offload setup.

Run on a GPU node (host):  python3 patch_start_script.py [path-to-start-sh]

Ensures start_dsv4f_4tp_nolmc.sh has:
  - export VLLM_SERVER_DEV_MODE=1        (enables POST /reset_prefix_cache)
  - --gpu-memory-utilization 0.78
  - kv-transfer-config -> GdsDirectOffloadingSpec @ kv_offload_gds root
"""
import re
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "/home/yztai/eugr-b12x-logs/start_dsv4f_4tp_nolmc.sh"
NEW_CFG = (
    "  --kv-transfer-config "
    "'{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\"," 
    "\"kv_connector_extra_config\":{\"spec_name\":\"GdsDirectOffloadingSpec\","
    "\"spec_module_path\":\"vllm.v1.kv_offload.gds.spec\","
    "\"root_dir\":\"/root/.cache/kv_offload_gds\",\"gc_max_size_gb\":1024}}' \\"
)
DEV_LINE = "export VLLM_SERVER_DEV_MODE=1"

src = open(P).read()
lines = src.splitlines(keepends=True)
out = []
hit_util = hit_cfg = hit_dev = False
for ln in lines:
    s = ln.strip()
    if s.startswith("--gpu-memory-utilization"):
        out.append(re.sub(r"0\.\d+", "0.78", ln))
        hit_util = True
    elif s.startswith("--kv-transfer-config"):
        out.append(NEW_CFG + "\n")
        hit_cfg = True
    else:
        out.append(ln)
        if s == DEV_LINE:
            hit_dev = True

if not hit_dev:
    # insert after the ring-depth env line, else after PYTHONFAULTHANDLER
    for i, ln in enumerate(out):
        if "VLLM_GDS_RING_DEPTH" in ln or "PYTHONFAULTHANDLER" in ln:
            out.insert(i + 1, DEV_LINE + "\n")
            hit_dev = True
            break

assert hit_util and hit_cfg and hit_dev, (
    f"markers not found: util={hit_util} cfg={hit_cfg} dev={hit_dev}"
)
open(P, "w").write("".join(out))
print("PATCHED: util=0.78 spec=GdsDirectOffloadingSpec(root=kv_offload_gds) DEV_MODE=1")
