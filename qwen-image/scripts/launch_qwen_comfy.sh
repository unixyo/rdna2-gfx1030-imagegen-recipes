#!/usr/bin/env bash
# Start a localhost-only ComfyUI v0.37.2 for Qwen-Image-2.1 INT8 on gfx1030 with the
# settings used in the measured runs. Stop it with Ctrl+C when done to free VRAM.
#
#   COMFY=<path to ComfyUI checkout>  VENV=<path to venv>  bash launch_qwen_comfy.sh
#
# Extra arguments are passed through to ComfyUI.
set -euo pipefail
: "${COMFY:?set COMFY to the ComfyUI v0.37.2 checkout}"
: "${VENV:?set VENV to the Python 3.12 venv with the ROCm 10.2 wheels}"
port="${PORT:-8188}"
headroom="${VRAM_HEADROOM:-3}"   # GiB, soft DynamicVRAM target (see README)

export HF_HUB_OFFLINE=1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export Q21_RDNA2_INT8_PATCH=1     # patientx Triton INT8 GEMM (required on gfx1030)

exec "$VENV/bin/python" -u "$COMFY/main.py" \
  --listen 127.0.0.1 --port "$port" \
  --extra-model-paths-config "${EXTRA_MODEL_PATHS:-$COMFY/extra_model_paths.yaml}" \
  --disable-auto-launch \
  --disable-triton-backend \
  --use-split-cross-attention \
  --reserve-vram 1 \
  --vram-headroom "$headroom" \
  "$@"
