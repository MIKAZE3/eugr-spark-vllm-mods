# eugr-spark-vllm-mods

**Mods / patches for the [`eugr/spark-vllm-b12x`](https://github.com/eugr/spark-vllm-b12x) container image**, developed and validated on a 4-node DGX Spark (GB10) cluster (DeepSeek-V4-Flash-0731-CRACK-NVFP4, TP4, CUDA 13).

All mods are drop-in patches applied inside the container (`docker exec`) or as bundled Python packages installed into the site-packages tree. They were written against `eugr/spark-vllm-b12x:current-prepped` (vLLM `0.1.dev19043+gfa033bd4e.d20260812`); re-verify anchors before applying to other versions.

## Mods

| Mod | Target | What it does |
|---|---|---|
| `gds_direct_v1/` | `vllm/v1/kv_offload/gds/` (+1 upstream line) | **GPU↔SSD direct KV offload** via cuFile (GDS) — replaces the CPU primary tier entirely. Replay of long contexts goes from ~1.00× (full recompute) to **51×/118×/163×** for P128/P512/P800. Includes: `GdsDirectOffloadingSpec` (manager+worker), cuFile pool with LRU handle cache & 3-level registration fallback, chunk-relative D2D planning, and a relaxation of the upstream `assert transfer_result.success`. |
| `fix-nvfp4-kv-sm121/` | NVFP4 KV cache (build-time `.so` + runtime) | Makes `--kv-cache-dtype nvfp4` correct and fast on sm120/sm121 (DGX Spark GB10): interleaved-vs-separated NVFP4 layout fix, XQA decode disable for NVFP4, empty-architectures guard, Eagle3 spec-decode warmup hang fix. |

## Deploying & using the mods

Each mod ships its own manual — follow it top to bottom:

| Mod | Manual | Highlights |
|---|---|---|
| `gds_direct_v1/` | [`gds_direct_v1/README.md`](gds_direct_v1/README.md) | Host `nvidia_fs` module + container device node + `/run/udev` → `docker cp pkg/gds` + F6 patch (`patch_offloading_worker_assert.py`) + import self-check → set env (`PYTHONHASHSEED=0`, `VLLM_GDS_RING_DEPTH=32`, `VLLM_SERVER_DEV_MODE=1`) & `--kv-transfer-config` → restart cluster (node1 → 25s → node2~4) → verify via `POST /reset_prefix_cache` replay. Rollback = switch config string. |
| `fix-nvfp4-kv-sm121/` | [`fix-nvfp4-kv-sm121/DEPLOY.md`](fix-nvfp4-kv-sm121/DEPLOY.md) | Build-time first: `vllm-nvfp4-layout-fix.patch` is auto-applied by the Dockerfile when building the wheel; then per-container: install wheel + flashinfer 0.6.18 + run `run.sh` (validates build-time fixes present). Use `cudagraph_mode=PIECEWISE` with NVFP4+Eagle3 — FULL mode corrupts spec-decode sampling. |

## Layout conventions

- `run.sh` / `apply_mod.sh` — idempotent apply entrypoint, run on the GPU node host (or inside the container where noted).
- `patch_*.py` — idempotent string-anchored patch scripts (fail loudly on upstream version drift).
- `gds_direct_v1/pkg/` — installable package: `docker cp pkg/gds <container>:/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/`.
- `smoke_gds.py` / `replay_smoke.py` — pre/post-deployment smoke checks.

## Platform prerequisites (GDS mod only)

- Host kernel module `nvidia_fs` (`modprobe nvidia_fs`; major number differs per node — read `/proc/devices`).
- Container device node `/dev/nvidia-fs` (`mknod`) and a copy of the host's `/run/udev` (libudev attribute probe).
- `PYTHONHASHSEED=0` for deterministic offload-key hashing across requests/restarts.

## Notes

- Configuration hook: vLLM's `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"spec_name":"GdsDirectOffloadingSpec","spec_module_path":"vllm.v1.kv_offload.gds.spec",...}}'` — zero edits to upstream registration code; rollback = switch the config string back.
- These mods contain no credentials; internal validation data lives in the project's operational runbooks, not here.