# Qwen-Image-2.1 INT8 ConvRot on RX 6900 XT (gfx1030)

Text-to-image and single-reference image editing with the official Comfy-Org INT8 ConvRot weights, in ComfyUI v0.37.2, on ROCm 10.2 nightly wheels. See [../hardware.md](../hardware.md) for the machine and Python stack.

## What works

- Text-to-image at 1216×640, 640×960, 512×768 and 768×1152.
- One-reference edit (Qwen3-VL vision input + VAE reference latent) at 640×960, and once at 1024×1536.
- Comfy-Org Turbo LoRA with the official 8-step sigma schedule.
- Real-ESRGAN Anime6B ×4 upscale of the result.

All runs needed the patientx Triton INT8 patch because hipBLASLt has no gfx1030 INT8 kernels.

## Model files

From [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) at revision `df94239739eef7205973a145ffa6f441f7b64e84`:

| ComfyUI folder | File | Bytes | SHA-256 |
|---|---|---:|---|
| `diffusion_models/` | `qwen_image_2.1_int8_convrot.safetensors` | 7,256,783,064 | `cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d` |
| `text_encoders/` | `qwen3vl_8b_int8_convrot.safetensors` | 9,350,798,360 | `8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f` |
| `vae/` | `qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |

Optional:

| Folder | File | Source | Bytes | SHA-256 |
|---|---|---|---:|---|
| `loras/` | `qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) `loras/`, revision `b42e6c7675686a74ac3c517eda2ebf2f2a545539` | 913,314,512 | `208dd43250e1e01467ba190572ae2e7107a7870ec1ff026bc65f791fa1e80e95` |
| `upscale_models/` | `RealESRGAN_x4plus_anime_6B.pth` | [xinntao/Real-ESRGAN release v0.2.2.4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) | 17,938,799 | `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da` |

The Turbo LoRA is a Comfy-Org extraction of the official Qwen Turbo checkpoint, not a separately trained Qwen LoRA.

**License:** Comfy-Org/Qwen-Image-2.1 and Qwen/Qwen-Image-2.1-Turbo list the **Qwen Research License** (`qwen-research`) on Hugging Face (checked 2026-10-30). It is not an open-source license; read its terms before any use. Read the model cards before any use: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo). Real-ESRGAN: see [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

## Setup

1. Create a Python 3.12 venv with the ROCm 10.2 nightly torch/Triton for gfx1030 ([../hardware.md](../hardware.md)).
2. Clone ComfyUI at v0.37.2, commit `830232b856045ca2892833212d7771078a13edd5`, and install its requirements without replacing torch. `comfy-kitchen` must be `0.2.35`.
3. Apply `patches/comfyui-v0.37.2-lazy-torchaudio.patch` inside the ComfyUI checkout (`git apply`). It makes two audio-only imports lazy so ComfyUI starts despite the broken torchaudio wheel. Some unrelated audio/video nodes still fail to import; Qwen nodes load.
4. Copy `custom_nodes/RDNA2-Qwen21-INT8/` into `<ComfyUI>/custom_nodes/`, then fetch the patientx files into it:
   ```bash
   bash scripts/fetch_patientx_int8.sh <ComfyUI>/custom_nodes/RDNA2-Qwen21-INT8
   ```
   This downloads `rocm_int8_kitchen_patch.py` and `int8_fused_kernel.py` from [patientx/ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM) at commit `a9560174bfe78cf16eb8fd21705c869d49eddf9d` and checks their SHA-256. Only these two files are used. The full patientx node pack (with its updater) was not installed.
5. Put the model files under `<models>` and point ComfyUI at them with `extra_model_paths.example.yaml`.
6. Start the server:
   ```bash
   COMFY=<ComfyUI> VENV=<venv> EXTRA_MODEL_PATHS=<path>/extra_model_paths.yaml \
     bash scripts/launch_qwen_comfy.sh
   ```
7. Put a 640×960 reference image into `<ComfyUI>/input/reference_640x960.png` and submit a workflow:
   ```bash
   python3 ../scripts/submit_workflow.py workflows/edit-640x960-25step-api.json --port 8188 --seed 7201
   ```

## Working settings

Launch flags (all measured runs): `--use-split-cross-attention --reserve-vram 1 --disable-triton-backend`, environment `Q21_RDNA2_INT8_PATCH=1`. The ComfyUI generic Triton backend was disabled; only the patientx patch was used.

`--vram-headroom 3` (GiB) is the chosen memory-saving setting for 640×960 edits. It is a soft DynamicVRAM target, not a hard limit, and did not change the output pixels.

Graph (see `workflows/`):

| Node | Setting |
|---|---|
| `UNETLoader` | `qwen_image_2.1_int8_convrot.safetensors`, weight_dtype default |
| `CLIPLoader` | `qwen3vl_8b_int8_convrot.safetensors`, type `qwen_image` |
| `TextEncodeQwenImage21` | `images.image_1` = LoadImage, VAE connected, `resolution` 0 (keep the prepared size); use its latent output for the sampler |
| `QwenImage21Cache` | device `cpu`, dtype default |
| `KSampler` (normal) | 25 steps, Euler / simple, CFG 1.0, denoise 1.0 |
| `SamplerCustom` (Turbo) | `LoraLoaderModelOnly` Turbo LoRA strength 1.0, `ManualSigmas` `1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0`, Euler, CFG 1 |
| `VAEDecodeTiled` | tile 512, overlap 64 |

The reference image was resized with Lanczos to 640×960 (2:3) before loading. The edit is a new generation guided by the reference, not a masked pixel-preserving edit.

The prompts in `workflows/` are placeholders. The timings below were measured with the lab's own prompts.

## Measured speed and VRAM

Single runs unless noted. "Prompt" = API submit to ComfyUI history success, excluding server startup. VRAM = sampled sysfs peak.

### Standalone ComfyUI runs

| Task | Resolution | Steps | Other conditions | Prompt time | VRAM peak |
|---|---|---|---|---:|---:|
| T2I (Shino benchmark graph) | 1216×640 | 25 | no headroom flag, cold first run after server start | 118.06 s (123.07 s incl. startup); ~2.29 s/step after warm-up | 15.59 GiB |
| Edit | 640×960 | 25 | no headroom flag | 114.19 s | 15.66 GiB |
| Edit | 640×960 | 25 | `--vram-headroom 2` | 82.05 s | 13.51 GiB |
| Edit | 640×960 | 25 | `--vram-headroom 2.5` | 80.04 s | 12.91 GiB |
| Edit | 640×960 | 25 | `--vram-headroom 3` | 80.05 s | 12.41 GiB |
| Edit | 1024×1536 | 25 | `--vram-headroom 3`, ~8.6–9 s/step | 332.18 s | 15.95 GiB |
| Edit, Turbo LoRA | 640×960 | 8 (official sigmas) | headroom 3 | 48.03 s | 12.42 GiB |
| Edit, no LoRA, same 8 sigmas | 640×960 | 8 | headroom 3, control | 48.18 s | 12.41 GiB |
| Edit, Turbo LoRA, CFG1 positive-only encode | 640×960 | 8 | headroom 3, experimental node, see below | 42.08 s (vs 48.05 s baseline in the same A/B) | 12.42 GiB |
| Anime6B ×4 upscale only | 640×960 → 2560×3840 | – | separate server | ComfyUI 1.65 s, API 2.00 s, 7.01 s incl. startup | not measurable at 2 s polling |

Notes from the lab:

- The 114 s → 80 s drop between no-headroom and headroom runs happened across different launches and cache states. The lab did not establish that headroom itself speeds things up. What it did establish: decoded RGB output was pixel-identical across the 2, 2.5 and 3 GiB runs and the no-headroom run.
- Turbo8 vs normal25 saved about 40 % of the time, but the 8-step run without the LoRA was equally fast. The saving comes from fewer steps.
- The 1024×1536 edit left about 31 MiB of sampled spare VRAM. It is not a safe default.

### Runs through the lab's web UI (separate ComfyUI child per job, headroom 3)

"Qwen stage" is the generation part of the job as the UI reported it.

| Task | Resolution | Steps / LoRAs | Qwen stage | VRAM peak |
|---|---|---|---:|---:|
| T2I | 640×960 | 25 | 57.30 s | 12.36 GiB |
| Edit | 640×960 | 25 | 81.08 s; a later identical-graph job 91.19 s | 13.05 / 12.41 GiB |
| Edit, Turbo LoRA | 640×960 | 8 | 58.90 s | 13.47 GiB |
| T2I, Turbo LoRA | 512×768 | 8 | 23.02 s | 12.41 GiB |
| T2I, Turbo + two style LoRAs (0.6 / 0.4) | 512×768 | 8 | 41.19 s | 12.38 GiB |
| T2I | 768×1152 | 16 | 77.07 s | 12.39 GiB |
| Edit + one style LoRA | 640×960 | 25 | 88.09–90.14 s | 12.42–13.25 GiB |

The web UI itself is not included in this repository.

## Options that were tested and left off

- **CFG=1 negative-encode skip.** At CFG exactly 1.0 ComfyUI does not use the negative prediction, so encoding the negative prompt is wasted work. A lab node that reuses the positive conditioning gave pixel-identical output and 48.05 s → 42.08 s in one matched A/B (Turbo8 edit). Only valid at CFG 1.0. Not included here; it is a small custom node in the lab and was opt-in there.
- **Text encoder on CPU.** The 8.9 GB INT8 encoder loaded on CPU but was still encoding after more than 240 s (host load average ~34). The run was stopped. Keep the encoder on the GPU.
- **Conv3D exact-tile workaround** for the Qwen3-VL patch embedding ([ROCm/TheRock#8729](https://github.com/ROCm/TheRock/issues/8729)). Not needed: all runs succeeded with it off, with only a non-fatal MIOpen warning.
- **Operator research (2026-10-23).** On captured Qwen attention inputs, PyTorch automatic SDPA used the math backend and took 2.16× / 1.93× the time of the existing split attention. A Linux port of the 2kiss FP32 HIP attention kernel was correct but 5.06–5.55× slower than split attention. Neither was adopted.

## Known limits

- VRAM is nearly full at 1216×640 T2I without headroom (15.59 GiB) and at 1024×1536 edit (15.95 GiB). Close other GPU programs first.
- Console warnings that did not stop any run: missing `MIOpenCKGroupedConv_gfx1030.so`, rocBLASLt gfx1030 `TensileLibrary` not found.
- The torchaudio patch is a startup workaround only. Audio nodes in this ComfyUI do not work.
- Reference edits change more than requested: in the lab's clothing edit, the clothing changes succeeded, with minor drift in face, hair and proportions.
- Some LoRAs on Hugging Face use Diffusers/PEFT key layouts that this ComfyUI does not load. The lab kept those off and did not rename keys.

## Not verified

- Repeated warm runs on one server, and whether output stays deterministic across them.
- Statistical speed comparisons. Every row above is one run.
- Multi-reference editing (for example LoRAs designed for a person + clothing reference pair).
- Q8_0 GGUF, other 4/6-step accelerator LoRAs (downloaded by the lab, not run).
- A newer patientx commit than `a9560174`.
- The exact package install commands.
- Visual quality. The lab's pixel metrics show that settings changed the output; they do not rank quality, and the person running the lab did not see a clear difference between the 25-step, Turbo8 and no-LoRA 8-step results.

## Credits

- [Shino's RX 6800 post](https://shino.dev/posts/comfyui-rx6800/) and [vallicgrr/rx6800-comfyui](https://github.com/vallicgrr/rx6800-comfyui) (GPL-3.0) for the original 1216×640 benchmark graph and the Windows rocBLAS approach. The graph is not copied here; it is at `workflows/rx6800/qwen_image_21_og_s7201_1216x640_api.json` in that repository at commit `e73928e2d2cf5e3b722a4e4f20f3afb2d05e4366`. The Windows patch was not used on Linux.
- [patientx/ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM) (AGPL-3.0) for the Triton INT8 kernels.
