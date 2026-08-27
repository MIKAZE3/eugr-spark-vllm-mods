# fix-nvfp4-kv-sm121 部署流程与使用手册

> 原理与修复项详见同目录 `README.md`。本文是**从零到可用的操作流程**。
> 目标：DGX Spark（GB10，sm120/sm121）+ `--kv-cache-dtype nvfp4` + 可选 Eagle3 投机解码。
> 验证基准：Qwen3-VL-4B（bill OCR）+ Qwen3-VL-4B-Eagle3-Bill，w2n1，2026-08-22。

---

## 目录

- [一、总览](#一总览)
- [二、构建期（一次性，每目标 vLLM ref 一次）](#二构建期一次性每目标-vllm-ref-一次)
- [三、运行期部署（每次容器重建）](#三运行期部署每次容器重建)
- [四、启动引擎（正确用法）](#四启动引擎正确用法)
- [五、验证清单](#五验证清单)
- [六、注意事项与已知限制](#六注意事项与已知限制)
- [七、回滚](#七回滚)

---

## 一、总览

本 mod 分**两层**，缺一不可：

| 层 | 内容 | 时机 |
|---|---|---|
| 构建期 | `vllm-nvfp4-layout-fix.patch`（C++ 内核布局修复 + Python 侧修复，编译进 `.so`） | 构建 vLLM wheel 时（Dockerfile 自动应用） |
| 运行期 | `run.sh`：校验构建期修复是否在 wheel 中 + 应用 `model.py` 空 architectures 守卫 + 清 `__pycache__` | 每次容器启动后 |

> 只跑 run.sh 不重新构建 → 输出仍然乱码（C++ 修复不在 wheel 里）。只构建不跑 run.sh → 多模态模型可能 IndexError 崩溃。

## 二、构建期（一次性，每目标 vLLM ref 一次）

在构建机（spark-vllm-docker 仓库目录）：

```bash
cd spark-vllm-docker
# mods/fix-nvfp4-kv-sm121/vllm-nvfp4-layout-fix.patch 需存在于构建上下文，
# Dockerfile 会在 checkout 后自动应用
BUILD_PROXY=http://192.168.101.150:8888 ./build-and-copy.sh \
    --exp-b12x \
    --vllm-repo https://github.com/vllm-project/vllm.git \
    --vllm-ref refs/pull/50288/head \
    --rebuild-vllm
```

前提：
- GitHub 代理（`BUILD_PROXY`，节点直连 GitHub 不稳时需要）
- 预构建 flashinfer 0.6.18 wheel 于 `.wheel-cache/flashinfer/regular/`：
  `flashinfer_cubin`、`flashinfer_jit_cache`、`flashinfer_python` 三件套（均为合法 zip，否则经代理重下）

产物：`.wheel-cache/vllm/b12x/vllm-0.26.1rc1.dev922+g8275b36b9.d20260822-*.whl`

## 三、运行期部署（每次容器重建）

```bash
CONTAINER=eugr-b12x-qwen3vl4b

# 1) 安装构建期产物（vLLM wheel）
docker cp .wheel-cache/vllm/b12x/vllm-0.26.1rc1.dev922+g8275b36b9.d20260822-*.whl $CONTAINER:/tmp/
docker exec $CONTAINER pip install --break-system-packages --no-deps \
        --force-reinstall /tmp/vllm-0.26.1rc1.dev922+g8275b36b9.d20260822-*.whl

# 2) 安装 flashinfer 0.6.18（0.6.17 缺 sm120 NVFP4/XQA 内核）
docker cp .wheel-cache/flashinfer/regular/flashinfer_*.whl $CONTAINER:/tmp/
docker exec $CONTAINER bash -lc 'pip install --break-system-packages --no-deps \
  /tmp/flashinfer_python-0.6.18-py3-none-any.whl \
  /tmp/flashinfer_jit_cache-0.6.18-*.whl \
  /tmp/flashinfer_cubin-0.6.18-py3-none-any.whl'

# 3) 运行期 mod（幂等）
#    方式 a：随启动命令注入
./launch-cluster.sh ... --apply-mod mods/fix-nvfp4-kv-sm121 ...
#    方式 b：已有运行容器手动执行
docker exec $CONTAINER bash mods/fix-nvfp4-kv-sm121/run.sh
# 输出 "OK" 说明 wheel 已含构建期修复；非零退出=需按 §二 重新构建
```

## 四、启动引擎（正确用法）

**普通 NVFP4（无投机解码，推荐）**：

```bash
docker exec -d $CONTAINER bash -lc 'exec vllm serve /models/Qwen3-VL-4B-Instruct-NVFP4 \
  --kv-cache-dtype nvfp4 ...'
```

**NVFP4 + Eagle3（票据场景）**：

```bash
docker exec -d $CONTAINER bash -lc 'exec vllm serve /models/Qwen3-VL-4B-Instruct-NVFP4 \
  --kv-cache-dtype nvfp4 \
  --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\"}" \
  --speculative-config "{\"method\":\"eagle3\",\"model\":\"/models/Qwen3-VL-4B-Instruct-Eagle3-Bill\",\"num_speculative_tokens\":4}" ...'
```

> **cudagraph_mode 必须 PIECEWISE**：FULL 模式在 NVFP4 下会损坏投机解码采样（token 1 后草稿/验证输出 NUL）。

## 五、验证清单

1. **正确性**：文本生成正常；票据 OCR 提取结果（如 `SXCK202605060559`）字符完全正确。
2. **吞吐**（`bill_bench.py`，64 张非训练图，max_tokens=128）：

   | 配置 | decode(c1) | 峰值聚合 |
   |---|---|---|
   | NVFP4（修复后，无 draft） | ~65 tok/s | 128.56 tok/s (c4) |
   | FP8 + Eagle3-Bill（同 wheel） | ~71 tok/s | 115.44 tok/s (c16) |
   | NVFP4 修复前（M9 python swizzle 兜底） | ~10 tok/s | — |
3. **回归**：FP8 配置在新 wheel 下吞吐不劣化（NVFP4 修复不破坏 FP8 路径）。

## 六、注意事项与已知限制

1. **必须 PIECEWISE**：NVFP4 + FULL cudagraph → 投机解码采样损坏（静默错误）。
2. **Eagle3 文本 draft 对图像 OCR 几乎零接受**（~0%），OCR 场景不加速度；纯文本有 ~10% 接受率。
3. **XQA decode 对 NVFP4 保持禁用**（构建期已禁），FP8/bf16 不受影响。
4. **warmup 挂起已由构建期修复**（Eagle3 多 token 批量），若仍挂检查 wheel 是否真的 rebuild 过（run.sh 会校验）。
5. **架构**：flashinfer wheel 为 aarch64（`cp39-abi3-manylinux_2_28_aarch64`），本 mod 只支持 DGX Spark 类 aarch64 平台。
6. **构建机代理**：`BUILD_PROXY` 内网地址在无代理环境需替换或跳过；wheel-cache 里 flashinfer 三件套缺失/损坏会导致构建失败。
7. **运行期 run.sh 幂等**：可反复执行；`model.py` 守卫在 wheel 重装（pip --force-reinstall）后会被覆盖，**必须重跑 run.sh**。

## 七、回滚

```bash
# 运行时：换回旧 wheel + 旧 flashinfer 版本
docker exec $CONTAINER pip install --break-system-packages --no-deps \
  --force-reinstall /tmp/vllm-<旧版本>.whl
# 构建期：删除/改名 mods/fix-nvfp4-kv-sm121/vllm-nvfp4-layout-fix.patch 后重新构建
```