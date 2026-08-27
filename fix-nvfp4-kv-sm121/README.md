# NVFP4 KV Cache Fix for DGX Spark (sm121)

**Last updated:** `2026-08-22`

Makes `--kv-cache-dtype nvfp4` produce correct output **and** competitive
throughput on consumer Blackwell (sm120/sm121, e.g. DGX Spark GB10). Without
this mod, NVFP4 KV on the upstream vLLM PR #50288 head garbles every output
(first token already wrong) because the PR's C++ kernel and Python cache
slicing use a *separated* `[data | scale]` layout while current vLLM
allocates the HND NVFP4 cache *interleaved per token*.

## What the mod fixes

| # | Problem | Fix | Layer |
|---|---|---|---|
| 1 | NVFP4 KV write/slice mismatch (interleaved vs separated layout) → garbled output | `nvfp4_kv_cache_kernels.cu` dispatch uses the tensor's physical strides; `nvfp4_split_data_scale` splits per-token `[data_dim | scale_dim]` | **build-time** (compiled into `.so`) |
| 2 | flashinfer 0.6.18 XQA decode reads nvfp4 KV with a compact/separated layout → garbled decode past token 1 | Disable XQA decode for NVFP4 KV on family(120); native decode is used (fp8/bf16 keep XQA) | **build-time** |
| 3 | Upstream regression: `model.py:992` crashes on multimodal models with empty `architectures` (`IndexError`) | Guard `try_match_architecture_defaults(self.architectures[0])` with `if self.architectures` | **runtime** (wheel re-install wipes it) |
| 4 | NVFP4 KV + spec decode (Eagle3) init hang: flashinfer 0.6.18's NVFP4 FA2 decode kernel hangs for multi-token decode batches (`q_len_per_req>1`, >=2 reqs) during warmup | Warm up spec decode as plain 1-token decode for NVFP4 KV (`warmup.py`); runtime multi-token batches are reordered to the prefill path (`reorder_batch_threshold=1`) which the prefill warmup covers | **build-time** (nvfp4 only; fp8/bf16 spec warmup unchanged) |

## Layout

```
mods/fix-nvfp4-kv-sm121/
├── README.md
├── run.sh                              # runtime: verify + model.py guard
└── vllm-nvfp4-layout-fix.patch         # build-time patch (C++ + Python)
```

## Usage

### 1. Build the fixed wheel (required once per target vLLM ref)

The C++ kernel fix is compiled into the wheel, so it must be applied at build
time. The Dockerfile auto-applies `mods/fix-nvfp4-kv-sm121/vllm-nvfp4-layout-fix.patch`
after checkout whenever the patch file is present in the build context:

```bash
cd spark-vllm-docker
BUILD_PROXY=http://192.168.101.150:8888 ./build-and-copy.sh \
    --exp-b12x \
    --vllm-repo https://github.com/vllm-project/vllm.git \
    --vllm-ref refs/pull/50288/head \
    --rebuild-vllm
```

- Requires a GitHub proxy on nodes that cannot reach GitHub reliably
  (`BUILD_PROXY`), and prebuilt flashinfer wheels in
  `.wheel-cache/flashinfer/regular/` (0.6.18, includes `flashinfer_cubin`,
  `flashinfer_jit_cache`, `flashinfer_python` — all three must be valid
  zips; re-download via the proxy if not).
- The wheel lands in `.wheel-cache/vllm/b12x/`:
  `vllm-0.26.1rc1.dev922+g8275b36b9.d20260822-cp312-cp312-linux_aarch64.whl`.

### 2. Deploy the wheel and flashinfer 0.6.18

```bash
CONTAINER=eugr-b12x-qwen3vl4b
docker cp .wheel-cache/vllm/b12x/vllm-0.26.1rc1.dev922+g8275b36b9.d20260822-*.whl \
        $CONTAINER:/tmp/
docker exec $CONTAINER pip install --break-system-packages --no-deps \
        --force-reinstall /tmp/vllm-0.26.1rc1.dev922+g8275b36b9.d20260822-*.whl
# flashinfer must be 0.6.18 (0.6.17 lacks the sm120 NVFP4/XQA kernels):
docker cp .wheel-cache/flashinfer/regular/flashinfer_*.whl $CONTAINER:/tmp/
docker exec $CONTAINER bash -lc 'pip install --break-system-packages --no-deps /tmp/flashinfer_python-0.6.18-py3-none-any.whl /tmp/flashinfer_jit_cache-0.6.18-*.whl /tmp/flashinfer_cubin-0.6.18-py3-none-any.whl'
```

### 3. Start the engine with the runtime mod

```bash
./launch-cluster.sh ... --apply-mod mods/fix-nvfp4-kv-sm121 ...
# or, on a running container:
docker exec $CONTAINER bash mods/fix-nvfp4-kv-sm121/run.sh
docker exec -d $CONTAINER bash -lc 'exec vllm serve /models/Qwen3-VL-4B-Instruct-NVFP4 \
  --kv-cache-dtype nvfp4 ...'
```

`run.sh` is idempotent: it verifies both build-time fixes are present in the
installed wheel (exits non-zero with rebuild instructions otherwise) and
(re)applies the model.py guard + clears `__pycache__`.

### 4. NVFP4 KV + Eagle3 spec decode

The warmup hang is fixed by the build-time patch (fix #4). Full cudagraph
(`FULL` mode, default) additionally corrupts spec-decode sampling under NVFP4
(draft/verify tokens decode to NUL after token 1); run with
`cudagraph_mode=PIECEWISE` so sampling stays eager:

```bash
docker exec -d $CONTAINER bash -lc 'exec vllm serve /models/Qwen3-VL-4B-Instruct-NVFP4 \
  --kv-cache-dtype nvfp4 \
  --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\"}" \
  --speculative-config "{\"method\":\"eagle3\",\"model\":\"/models/Qwen3-VL-4B-Instruct-Eagle3-Bill\",\"num_speculative_tokens\":4}" ...'
```

Notes:
- Text prompts get real draft acceptance (~10%); image/OCR prompts accept
  ~0% because the Eagle3 drafter is text-only, so spec adds no speed there.
- Measured on w2n1: NVFP4+Eagle3 (PIECEWISE) decode ≈ 56 tok/s on image OCR
  (draft rejected); NVFP4 without draft ≈ 65 tok/s; FP8+Eagle3 ≈ 71 tok/s.

## Verified results (DGX Spark w2n1, 2026-08-22)

- Correctness: text generation OK; bill OCR extracts `SXCK202605060559` (JSON).
- Throughput (`bill_bench.py`, 64 non-train images, `max_tokens=128`):

| Config | decode (c1) | peak aggregate |
| --- | ---: | ---: |
| NVFP4 (fixed, no draft) | 64.78 tok/s | 128.56 tok/s (c4) |
| FP8 + Eagle3-Bill (same wheel) | 71.17 tok/s | 115.44 tok/s (c16) |
| NVFP4 before fix (M9 python swizzle workaround) | 10.35 tok/s | — |

## Known limitations

- `nvfp4 + speculative (Eagle3)` hangs at engine init on this upstream ref;
  use NVFP4 without a draft model, or FP8 + Eagle3.
- XQA decode stays disabled for NVFP4 KV until flashinfer's XQA nvfp4 reader
  matches the interleaved layout (or vLLM switches to a separated layout).
- The flashinfer wheels are aarch64 (`cp39-abi3-manylinux_2_28_aarch64`);
  this mod targets DGX Spark (aarch64).

## Changelog

- 2026-08-22: extracted from the DGXSpark hands-on session into a reusable
  mod; build-time patch wired into Dockerfile (applied after checkout,
  both custom-repo and repo-cache branches).