"""Opt-in bridge that loads the patientx INT8 Triton patch for comfy-kitchen.

Copy this folder into <ComfyUI>/custom_nodes/ and put the two patientx files
(rocm_int8_kitchen_patch.py, int8_fused_kernel.py) next to this file with
../scripts/fetch_patientx_int8.sh. Those two files are AGPL-3.0 and are NOT part
of this repository; this bridge only imports them.

Nothing happens unless Q21_RDNA2_INT8_PATCH=1 is set in the ComfyUI process.
"""
import logging
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if os.environ.get("Q21_RDNA2_INT8_PATCH", "0") == "1":
    from . import rocm_int8_kitchen_patch  # noqa: F401  (patientx, AGPL-3.0)
    logging.getLogger(__name__).info("Qwen21 RDNA2 INT8 patch imported by explicit opt-in")
else:
    logging.getLogger(__name__).info("Qwen21 RDNA2 INT8 patch disabled")
