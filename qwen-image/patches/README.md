# Patches

`comfyui-v0.37.2-lazy-torchaudio.patch` modifies two GPL-3.0 files of ComfyUI at commit `830232b856045ca2892833212d7771078a13edd5` and keeps that license. It is not covered by this repository's MIT license.

It replaces the top-level `import torchaudio` in `comfy/ldm/lightricks/vae/audio_vae.py` and `comfy/text_encoders/gemma4.py` with imports inside the functions that use them. ComfyUI then starts even though the ROCm torchaudio wheel fails to load. Audio features still need a working torchaudio.

Apply from the ComfyUI checkout root with `git apply`.
