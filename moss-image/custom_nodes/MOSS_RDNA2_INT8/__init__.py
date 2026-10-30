"""Opt-in replacement for comfy-kitchen's INT8 GEMM, for MOSS-Image INT8 on gfx1030.

Copy this folder into <ComfyUI>/custom_nodes/ and start ComfyUI with
MOSS_RDNA2_INT8_PATCH=1. Only comfy_kitchen.backends.eager.quantization
._int8_matmul_accumulate is replaced (by a Triton INT8xINT8->INT32 kernel);
Kitchen's quantizer, ConvRot, scaling and bias handling stay as they are.
Refuses to load on any GPU other than gfx1030.
"""
import logging
import os
from pathlib import Path
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if os.environ.get("MOSS_RDNA2_INT8_PATCH") == "1":
    import torch
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("MOSS RDNA2 patch requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx1030":
        raise RuntimeError("This selective INT8 patch is only validated on gfx1030, not " + arch)
    import comfy_kitchen.backends.eager.quantization as ck_quant
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rdna2_int8_matmul import int8_matmul
    ck_quant._int8_matmul_accumulate = int8_matmul
    logging.getLogger("moss-rdna2").warning(
        "MOSS_RDNA2_INT8_PATCH active on %s: replacing ONLY Kitchen _int8_matmul_accumulate "
        "with gfx1030 Triton INT32 GEMM (keeping Kitchen quantizer/epilogue)", arch
    )
