# gds_direct_v1：GPU↔SSD 直通 KV 卸载 mod（部署手册）

让 vLLM 的 KV 卸载走 **GDS（cuFile）直通 SSD**，彻底移除 CPU 内存缓存层。基于 `eugr/spark-vllm-b12x:current-prepped`（vLLM `0.1.dev19043+gfa033bd4e.d20260812`）在 4 节点 DGX Spark GB10（DeepSeek-V4-Flash-0731-CRACK-NVFP4，TP4，util 0.78）上开发验证。

**性能结论**：冷存 P800=564.9s；重放 P128=0.9s（51.8×）、P512=2.5s（118.3×）、P800=3.5s（162.7×）；内容语义正确；SSD 缓存跨重启复用有效；全程零丢写（cannot_store=0）。

---

## 目录

- [一、平台前置条件（容器重建后必做）](#一平台前置条件容器重建后必做)
- [二、安装部署](#二安装部署)
- [三、启动脚本配置](#三启动脚本配置)
- [四、重启集群流程](#四重启集群流程)
- [五、功能验证](#五功能验证)
- [六、正确使用与注意事项](#六正确使用与注意事项)
- [七、回滚](#七回滚)
- [八、故障排查速查](#八故障排查速查)

---

## 一、平台前置条件（容器重建后必做）

GDS 依赖 cuFile 驱动，宿主机/容器缺一不可。**注意各节点 major 号不同，不能硬编码。**

```bash
# 1) 宿主机（每节点）：加载 nvidia_fs 内核模块
sudo modprobe nvidia_fs
grep nvidia_fs /proc/devices   # 取本机 major，如 496 / 497 / 502

# 2) 容器（每节点）：创建设备节点
docker exec eugr-b12x-node-fp8-dspark bash -c \
  'mknod /dev/nvidia-fs c <本机major> 0 && chmod 666 /dev/nvidia-fs'

# 3) 容器（每节点）：移植宿主 /run/udev（libudev 属性探测需要，缺了 handle_register 报错）
tar -C / -xf udev_run.tar        # udev_run.tar 取自宿主 /run/udev

# 4) 验证（四节点容器内）
docker exec eugr-b12x-node-fp8-dspark python3 /path/to/smoke_gds.py
# 预期：cuFile 设备缓冲写/读 256MB 往返一致；"GDS ring ready ... mode=full/per_slot/compat"
```

> 兼容说明：node4 偶发 `cuFileBufRegister err=5048`（nvfs 驱动问题），自动降级 `compat` 模式仍可运行（功能一致，吞吐略低），无需干预。

## 二、安装部署

### 方式 A：手动部署（推荐，四节点重复执行）

```bash
# 1) 传输包到各节点（可用仓库 tools/dgx_jump_put.py 经跳板，或直接 scp）
# 2) 解包后 docker cp 进容器
docker cp gds_direct_v1/pkg/gds <容器>:/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/

# 3) F6 上游补丁（容器内执行，幂等）：
#    放宽 offloading/worker.py 的 assert transfer_result.success，
#    防止偶发 IO 失败直接杀死 worker。
docker cp gds_direct_v1/patch_offloading_worker_assert.py <容器>:/tmp/patch.py
docker exec <容器> python3 /tmp/patch.py            # 输出 PATCHED 或 ALREADY_PATCHED

# 4) structured output 容错补丁（容器内执行，幂等）：
#    DSpark 投机解码下 grammar 掩码与 logits 行数可能错位，
#    上游 assert 会杀死 worker（EngineDeadError，09-02 实测复现）；
#    此补丁改为截断对齐 + ERROR 日志（语义轻微降级但不崩）。
docker cp gds_direct_v1/patch_structured_outputs_guard.py <容器>:/tmp/patch2.py
docker exec <容器> python3 /tmp/patch2.py           # 输出 PATCHED 或 ALREADY_PATCHED

# 5) load 失败上报补丁（容器内执行，幂等）：
#    OffloadingConnector 未实现 get_block_ids_with_load_errors()，
#    load 失败时调度器毫不知情，请求带残缺 KV 继续 decode → 长上下文乱输出
#    （09-02 实测：872 次 read 失败 + GC 淘汰 chunk 命中）。此补丁让 worker
#    记录失败块 id 并上报 invalid_block_ids，调度器走重算路径。
#    配套：启动脚本 kv_load_failure_policy 改为 "recompute"（失败块自动重算，
#    而非请求报错）。
docker cp gds_direct_v1/patch_load_failure_report.py <容器>:/tmp/patch3.py
docker exec <容器> python3 /tmp/patch3.py           # 输出 PATCHED 或 ALREADY_PATCHED

# 4) import 自检（必须成功，否则启动必挂）
docker exec <容器> python3 -c \
  "from vllm.v1.kv_offload.gds import spec; print('IMPORT_OK')"
```

### 方式 B：run.sh 一键部署（节点宿主上执行）

```bash
# 前提：仓库 gds_direct_v1/ 目录已在节点上（如 /home/yztai/gds_direct_v1/），
# 或 /home/yztai/gds_bundle/ 存在解包后的 pkg/gds
bash /home/yztai/gds_direct_v1/run.sh
# 自动完成：docker cp → F6 补丁 → import 自检，然后提示重启集群
```

### 可选：启动脚本自动注入（patch_start_script.py）

在 GPU 节点宿主执行，把启动脚本改到正确配置（幂等，重复执行安全）：

```bash
python3 patch_start_script.py /home/yztai/eugr-b12x-logs/start_dsv4f_4tp_nolmc.sh
# 效果：util→0.78、kv-transfer-config→GdsDirectOffloadingSpec、
#       注入 VLLM_SERVER_DEV_MODE=1
```

## 三、启动脚本配置

以下为**最低必需配置**（已含于 patch_start_script.py 结果中，其余沿用原脚本）：

```bash
export PYTHONHASHSEED=0            # 必需：跨进程/跨重启 offload key 确定性
export VLLM_GDS_RING_DEPTH=32      # 必需：深度 64 触发 cuMemcpyBatchAsync error 1
export VLLM_SERVER_DEV_MODE=1      # 建议：启用 POST /reset_prefix_cache 验证端点
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800   # 建议：防长上下文 RPC 超时
# 可选：export VLLM_GDS_DEBUG=1    # GDSDBG 插桩日志（量大，稳定后关闭）

--gpu-memory-utilization 0.78
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"spec_name":"GdsDirectOffloadingSpec",
  "spec_module_path":"vllm.v1.kv_offload.gds.spec",
  "root_dir":"/root/.cache/kv_offload_gds","gc_max_size_gb":1024}}'
```

> 根目录 `kv_offload_gds` 与旧 CPU 层方案 `kv_offload_fs` 隔离，切勿混用。

## 四、重启集群流程

```bash
# 1) 四节点清理（kill4.sh 已固化到各节点 /home/yztai/kill4.sh）
bash /home/yztai/kill4.sh          # pkill [v]llm serve + [V]LLM:: + mmap 清理

# 2) 【仅做冷基线时】四节点容器内清 SSD 缓存
docker exec <容器> rm -rf /root/.cache/kv_offload_gds

# 3) 按序启动：node1 → 等 25s → node2~4 并行
docker exec -d <node1容器> bash /logs/start_dsv4f_4tp_nolmc.sh
# ……25 秒后……
docker exec -d <node2/3/4容器> bash /logs/start_dsv4f_4tp_nolmc.sh

# 4) 等待 READY（约 8~9 分钟），并验证
grep -c "startup complete" /logs/vllm-dsv4f.log    # ≥1
curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" \
  http://192.168.88.11:8000/v1/models               # 200

# 5) 暖机：发一个短请求触发 JIT 编译，避免首个大请求撞编译
```

## 五、功能验证

```bash
KEY=$(cat /home/yztai/eugr-b12x-logs/vllm-api-key)

# 1) 基础冒烟：小请求能正常返回
curl -s -X POST http://192.168.88.11:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'

# 2) 外部命中验证（关键）：先清本地前缀缓存，再重发同文本请求
curl -s -X POST -H "Authorization: Bearer $KEY" http://192.168.88.11:8000/reset_prefix_cache
#    → {"success":true}
# 然后重放一个之前冷存过的长文本（同文本），秒级返回 = 外部命中生效
# （毫秒~秒级 vs 冷存几十~几百秒；响应 reasoning 字段应正确复述 prompt）

# 3) 磁盘确认：/root/.cache/kv_offload_gds/_models_*/ 下文件数与组分布
docker exec <容器> find /root/.cache/kv_offload_gds -type f | wc -l
```

## 六、正确使用与注意事项

1. **ring_depth 钉死 32**：64 会触发 `cuMemcpyBatchAsync error 1`（GB10 驱动对 >34MiB 注册环的限制），勿改。
2. **长上下文重放才有收益**：P16 级小请求外部加载开销大于 prefill 节省（实测 0.82×），属预期。
3. **缓存清理只动 `kv_offload_gds`**：`kv_offload_fs` 是已废弃旧方案的数据，勿误删也不要让新 spec 指向它。
4. **冷基线顺序**：先 kill 再清缓存；清缓存后必须重启集群才生效（manager 的 _entries 在内存中）。
5. **SSD 预算（1TiB/节点，自动 LRU）**：`gc_max_size_gb=1024` 为上限，自研 `GdsGcManager` 每 300s 巡检，超限按 mtime 压回 921.6GiB。⚠️ **勿回退上游 FsGCManager**——其 sweep 线程在引擎进程内冻结（长超时 futex 不醒、0 CPU），且 headless 节点 FileMapper 指纹/rank 与实树不符会导致空转（曾致 node2~4 缓存无上限暴涨打满磁盘）。异常征兆：日志无 `GDS GC:` 行或树越界增长。
6. **句柄/fd**：cuFile 句柄 LRU 上限 4096（cufile_io.py `HANDLE_CACHE_MAX`），正常无需干预；若日志出现 EMFILE 检查是否改动过该值。
7. **GDSDBG 插桩**：`VLLM_GDS_DEBUG=1` 时 GDSDBG 日志量大，稳定运行后可移除该行。
8. **b12x JIT 偶发挂死**（环境级）：worker 栈停在 `cutlass/base_dsl/jit_executor.py` 超 10 分钟不动 → `docker restart` 容器后按 §四 重启。
9. **API 响应字段**：本 fork 的 thinking 输出字段是 `reasoning`（非 `reasoning_content`）；max_tokens 过小会被 reasoning 耗尽导致 content 为空（正常现象）。
10. **跨重启复用**依赖 `PYTHONHASHSEED=0` 与相同模型路径/指纹；换模型目录后旧缓存自然失效（路径含模型指纹）。
11. **GDS 设备节点自愈**：docker restart 会清空容器 /dev（mknod 节点丢失、悄然降级 compat）；启动脚本已内置自愈（`MAJOR=$(grep -i nvidia-fs /proc/devices ...)` + mknod），保持启动脚本含该段即可。

## 七、回滚

```bash
# 1) 换回原 kv-transfer-config（TieringOffloadingSpec + CPU 层）并重启集群
#    即：把启动脚本中 --kv-transfer-config 一行改回旧值（见 08-14 运行手册 §8.1）
# 2) 可选：卸载包 + 还原上游文件
docker exec <容器> rm -rf .../vllm/v1/kv_offload/gds
#    还原 offloading/worker.py：从原 wheel 重新解出该文件替换即可
```

## 八、故障排查速查

| 症状 | 原因 | 处理 |
|---|---|---|
| 引擎启动即死：`name 'OffloadingSpec' is not defined` | 部署包不完整/import 失败 | 重新 docker cp 并跑 import 自检 |
| worker 死：`cuMemcpyBatchAsync error 1 at index 0` | ring 深度 64 或 D2D 规划越界（旧版 offs bug） | 确认 RING_DEPTH=32 + 已用含 `_validate_item` 的版本 |
| 重放仍 ≈ 冷存（1.00×） | 存储空洞（准入帽旧版）或文件未落盘 | 确认 prepare_store 无 `[:cap]`；检查磁盘文件数与 §五-2 验证 |
| `[Errno 24] Too many open files` | 句柄缓存被改/泄漏 | 确认 cufile_io.py LRU 上限存在；重启 |
| `POST /reset_prefix_cache` 404 | 未开 DEV_MODE | 注入 `VLLM_SERVER_DEV_MODE=1` 后重启 |
| 重放内容为空/乱码 | max_tokens 被 reasoning 耗尽（正常） | 加大 max_tokens；用 `reasoning` 字段校验语义 |
| 首请求挂死 10+ 分钟（JIT 栈） | b12x 编译器偶发 | docker restart + §四 重启 |

---

*部署/使用基准：2026-08-25 四节点 ms11 全量验证（零错误、零丢写）。*