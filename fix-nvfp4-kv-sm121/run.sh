#!/bin/bash
#
# fix-nvfp4-kv-sm121 runtime mod
#
# Verifies the build-time NVFP4 KV fixes are present in the installed vLLM
# wheel (interleaved [data|scale] layout + XQA disabled for NVFP4 KV) and
# (re)applies the runtime-only model.py multimodal guard that wheel
# re-installs wipe out. Exits non-zero on any missing build-time fix so a
# broken wheel is never silently served.
set -euo pipefail

SITE_PACKAGES="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
PREFIX="[fix-nvfp4-kv-sm121]"

echo "=== NVFP4 KV sm121 fix mod ==="

# ---------------------------------------------------------------------------
# 1. Wheel must contain the build-time interleaved-layout fix
#    (vllm/utils/torch_utils.py: nvfp4_split_data_scale uses physical
#    strides and scale offset = data_dim).
# ---------------------------------------------------------------------------
CHECK_PY="$SITE_PACKAGES/vllm/utils/torch_utils.py"
if [ -f "$CHECK_PY" ] && grep -q 'storage_offset=base + data_dim' "$CHECK_PY"; then
    echo "$PREFIX OK: NVFP4 interleaved [data|scale] layout fix present in wheel."
else
    echo "$PREFIX ERROR: installed vLLM wheel lacks the NVFP4 layout fix." >&2
    echo "$PREFIX   The C++ kernel fix lives in the compiled .so and cannot be" >&2
    echo "$PREFIX   applied at runtime; rebuild the wheel with the build-time" >&2
    echo "$PREFIX   patch (see README.md):" >&2
    echo "$PREFIX     ./build-and-copy.sh --exp-b12x \\" >&2
    echo "$PREFIX       --vllm-repo https://github.com/vllm-project/vllm.git \\" >&2
    echo "$PREFIX       --vllm-ref refs/pull/50288/head --rebuild-vllm" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Wheel must disable the XQA decode kernel for NVFP4 KV on family(120)
#    (flashinfer 0.6.18 XQA reads a compact/separated layout and garbles
#    the interleaved cache beyond the first decoded token).
# ---------------------------------------------------------------------------
FI_PY="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"
if [ -f "$FI_PY" ] && grep -q 'and not self.is_kvcache_nvfp4' "$FI_PY"; then
    echo "$PREFIX OK: XQA decode disabled for NVFP4 KV."
else
    echo "$PREFIX ERROR: XQA-disable guard not found in flashinfer backend." >&2
    echo "$PREFIX   This is part of the same build-time patch; rebuild with" >&2
    echo "$PREFIX   the instructions above." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. model.py multimodal architectures guard (upstream regression on
#    Qwen3-VL etc.: self.architectures is empty for nested multimodal
#    configs). Wheel re-installs wipe this out, so (re)apply it here.
# ---------------------------------------------------------------------------
MPY="$SITE_PACKAGES/vllm/config/model.py"
if grep -q 'if self.architectures and (defaults := try_match_architecture_defaults' "$MPY" 2>/dev/null; then
    echo "$PREFIX OK: model.py multimodal architectures guard already applied."
else
    python3 - "$MPY" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = 'if defaults := try_match_architecture_defaults(self.architectures[0]):'
new = 'if self.architectures and (defaults := try_match_architecture_defaults(self.architectures[0])):'
assert src.count(old) == 1, f'model.py pattern count: {src.count(old)}'
open(p, 'w').write(src.replace(old, new))
print('model.py multimodal architectures guard applied')
PY
    echo "$PREFIX Applied model.py multimodal architectures guard."
fi

# Clear stale bytecode so the patched sources take effect.
find "$SITE_PACKAGES/vllm" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "=====> NVFP4 KV sm121 fixes verified/applied; start vllm with --kv-cache-dtype nvfp4."