# Patches

`ostris_edit_v1_comfy0431_rope_compat.patch` modifies `nodes.py` of [ostris/ComfyUI-MOSS-Image-Ostris-Edit](https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit) at commit `8f9b50c0d2b28248ecd48a4d4905c815e726f146` (MIT, Copyright (c) 2026 Ostris, LLC). Apply it from inside that clone with `git apply`.

The plugin's `_pe()` calls `dit.pe_embedder`, which ComfyUI v0.43.1's native MOSS model no longer has. The patch keeps the old call when `pe_embedder` exists and otherwise builds the four-axis RoPE with `comfy.ldm.flux.math.rope(ids[..., i], d, 10000)` over `dit.rope_axes_dim`. Only needed for the CharRef v2 recipe.
