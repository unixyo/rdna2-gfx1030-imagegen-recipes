#!/usr/bin/env bash
# Start a localhost-only ComfyUI v0.37.2 for MiniMax H3 GGUF on gfx1030 with the flags
# used in the measured runs. Stop it with Ctrl+C when done to free VRAM.
#
#   COMFY=<ComfyUI checkout>  VENV=<venv>  RECIPE=<this repo>/minimax-h3  bash launch_h3_comfy.sh
#
# RESERVE_VRAM: the measured runs used 2 (T2V 608x352), 3 (I2V 512x768) and 4 (640x640).
set -euo pipefail
: "${COMFY:?set COMFY to the patched ComfyUI v0.37.2 checkout}"
: "${VENV:?set VENV to the Python 3.12 venv with the ROCm 10.2 wheels}"
: "${RECIPE:?set RECIPE to the minimax-h3 folder of this repository}"
port="${PORT:-8188}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
# Text encoder on the GPU; needs patches/comfyui-v0.37.2-text-encoder-gpu-optin.patch.
# Set to 0 to keep the text encoder on the CPU (the slower first baseline).
export H3_LAB_TE_GPU="${H3_LAB_TE_GPU:-1}"
# Per-node timing log lines; needs patches/comfyui-v0.37.2-node-timing-optin.patch.
export H3_LAB_PROFILE_NODES="${H3_LAB_PROFILE_NODES:-0}"
# Video-only torchaudio stub; avoids the broken torchaudio ROCm wheel. No audio output.
export PYTHONPATH="$RECIPE/runtime_shims${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

exec "$VENV/bin/python" -u "$COMFY/main.py" \
  --listen 127.0.0.1 --port "$port" \
  --disable-auto-launch \
  --lowvram --disable-dynamic-vram \
  --bf16-unet \
  --disable-async-offload \
  --reserve-vram "${RESERVE_VRAM:-3}" \
  --use-split-cross-attention \
  --database-url "sqlite:///:memory:" \
  "$@"
