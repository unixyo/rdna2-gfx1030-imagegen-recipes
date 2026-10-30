#!/usr/bin/env bash
# Start a localhost-only ComfyUI v0.43.1 for MOSS-Image INT8 on gfx1030 with the flags
# used in the measured runs. Stop it with Ctrl+C when done to free VRAM.
#
#   COMFY=<ComfyUI v0.43.1 checkout>  VENV=<venv>  MODELS=<models>  bash start_moss_comfy.sh [--reserve-vram 1.0]
#
# The edit runs (armV5, DMD8, CharRef) were started with --reserve-vram 1.0;
# the 4-step T2I runs used no extra flag.
set -euo pipefail
: "${COMFY:?set COMFY to the ComfyUI v0.43.1 checkout}"
: "${VENV:?set VENV to the Python 3.12 venv with the ROCm 10.2 wheels}"
: "${MODELS:?set MODELS to the models directory}"
port="${PORT:-8299}"
work="${WORKDIR:-$PWD/moss-work}"

# hipBLASLt has no gfx1030 INT8 kernels; the Triton replacement is required.
# Set to 0 only to reproduce the HIPBLAS_STATUS_INVALID_VALUE failure.
export MOSS_RDNA2_INT8_PATCH="${MOSS_RDNA2_INT8_PATCH:-1}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
mkdir -p -- "$work/input" "$work/temp" "$work/outputs" "$work/user"

exec "$VENV/bin/python" "$COMFY/main.py" \
  --listen 127.0.0.1 --port "$port" \
  --models-directory "$MODELS" \
  --output-directory "$work/outputs" \
  --input-directory "$work/input" \
  --temp-directory "$work/temp" \
  --user-directory "$work/user" \
  --use-pytorch-cross-attention \
  --disable-comfy-compiler \
  "$@"
