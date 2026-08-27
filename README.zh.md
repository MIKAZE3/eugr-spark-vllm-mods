# eugr-spark-vllm-mods（中文说明）

**面向 [`eugr/spark-vllm-b12x`](https://github.com/eugr/spark-vllm-b12x) 容器镜像的 mod / 补丁集合**，在 4 节点 DGX Spark（GB10）集群上开发并验证（DeepSeek-V4-Flash-0731-CRACK-NVFP4，TP4，CUDA 13）。

所有 mod 均为容器内直接应用的补丁（`docker exec`），或安装进 site-packages 的独立 Python 包。编写基准为 `eugr/spark-vllm-b12x:current-prepped`（vLLM `0.1.dev19043+gfa033bd4e.d20260812`）；应用于其他版本前请先核对锚点。

## Mod 清单

| Mod | 作用对象 | 说明 |
|---|---|---|
| `gds_direct_v1/` | `vllm/v1/kv_offload/gds/`（另改动上游 1 行） | **GPU↔SSD 直通 KV 卸载**（cuFile/GDS）——彻底移除 CPU 主层。长上下文重放由 ~1.00×（全量重算）提升至 **P128/P512/P800 的 51×/118×/163×**。包含：`GdsDirectOffloadingSpec`（manager+worker）、带 LRU 句柄缓存与三级注册降级的 cuFile 池、chunk 相对偏移的 D2D 规划，以及上游 `assert transfer_result.success` 的放宽补丁。 |
| `fix-nvfp4-kv-sm121/` | NVFP4 KV 缓存（构建期编译进 `.so` + 运行期） | 使 `--kv-cache-dtype nvfp4` 在 sm120/sm121（DGX Spark GB10）上输出正确且吞吐达标：修复 interleaved/separated 布局不匹配、NVFP4 KV 禁用 XQA decode、空 architectures 守卫、Eagle3 投机解码 warmup 挂起修复。 |

## 部署与使用

每个 mod 自带完整手册，按序执行即可：

| Mod | 手册 | 要点 |
|---|---|---|
| `gds_direct_v1/` | [`gds_direct_v1/README.md`](gds_direct_v1/README.md) | 宿主机 `nvidia_fs` 模块 + 容器设备节点 + `/run/udev` → `docker cp pkg/gds` + F6 补丁（`patch_offloading_worker_assert.py`）+ import 自检 → 设置环境变量（`PYTHONHASHSEED=0`、`VLLM_GDS_RING_DEPTH=32`、`VLLM_SERVER_DEV_MODE=1`）与 `--kv-transfer-config` → 重启集群（node1 → 25s → node2~4）→ 用 `POST /reset_prefix_cache` 重放验证。回滚 = 换回配置字符串。 |
| `fix-nvfp4-kv-sm121/` | [`fix-nvfp4-kv-sm121/DEPLOY.md`](fix-nvfp4-kv-sm121/DEPLOY.md) | 先构建期：`vllm-nvfp4-layout-fix.patch` 由 Dockerfile 构建 wheel 时自动应用；再运行期（每次容器）：装 wheel + flashinfer 0.6.18 + 跑 `run.sh`（校验构建期修复已就位）。NVFP4+Eagle3 必须 `cudagraph_mode=PIECEWISE`——FULL 模式会损坏投机解码采样。 |

## 目录约定

- `run.sh` / `apply_mod.sh` — 幂等应用入口，在 GPU 节点宿主上执行（注明需在容器内执行的情况）。
- `patch_*.py` — 幂等的字符串锚点补丁脚本（上游版本漂移时显式报错）。
- `gds_direct_v1/pkg/` — 可安装包：`docker cp pkg/gds <容器>:/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/`。
- `smoke_gds.py` / `replay_smoke.py` — 部署前后冒烟检查。

## 平台前置条件（仅 GDS mod）

- 宿主机内核模块 `nvidia_fs`（`modprobe nvidia_fs`；各节点 major 号不同——需读 `/proc/devices`）。
- 容器设备节点 `/dev/nvidia-fs`（`mknod`）及宿主 `/run/udev` 的拷贝（libudev 属性探测）。
- `PYTHONHASHSEED=0`，保证 offload key 哈希在跨请求/跨重启间确定。

## 说明

- 接入方式：vLLM 的 `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"spec_name":"GdsDirectOffloadingSpec","spec_module_path":"vllm.v1.kv_offload.gds.spec",...}}'` —— 不改动上游任何注册代码；回滚 = 换回原配置字符串。
- 本仓库不含任何凭据；内部验证数据存放于项目运维手册，不在本仓库。