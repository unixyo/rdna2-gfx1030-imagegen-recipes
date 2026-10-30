# Patches

These are small diffs against third-party code. They are not covered by this repository's MIT license; each one keeps the license of the file it modifies.

| Patch | Applies to | Upstream license |
|---|---|---|
| `comfyui-v0.37.2-attention-split-nondivisible.patch` | ComfyUI `830232b8`, `comfy/ldm/modules/attention.py` | GPL-3.0 |
| `comfyui-v0.37.2-text-encoder-gpu-optin.patch` | ComfyUI `830232b8`, `comfy/model_management.py` | GPL-3.0 |
| `comfyui-v0.37.2-node-timing-optin.patch` | ComfyUI `830232b8`, `execution.py` | GPL-3.0 |
| `comfyui-gguf-6ea2651-allow-minimax_h3.patch` | city96/ComfyUI-GGUF `6ea2651e`, `loader.py` | Apache-2.0 |

Apply from the root of the target checkout with `git apply <patch>`.

The attention patch changes one line: when the query length is not divisible by the number of slices, the original code fell back to a single unsplit slice, so OOM retries with more slices never helped. The patched code uses `ceil(len / steps)` and lets the last slice be shorter. The lab checked it numerically on CPU against dense attention (71 queries × 73 keys, 16 slices) and used it in every H3 GPU run. It was not tested on other models.
