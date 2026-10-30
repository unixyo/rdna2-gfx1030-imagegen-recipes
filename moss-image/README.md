# MOSS-Image INT8 ConvRot on RX 6900 XT (gfx1030)

Fast 4-step text-to-image, reference editing with Edit-HQ armV5 (50-step and 8-step DMD), and a few style LoRAs, in ComfyUI v0.43.1 on ROCm 10.2 nightly wheels. See [../hardware.md](../hardware.md) for the machine and Python stack.

## What works

- SFT INT8 + 4-step LoRA text-to-image at 512×512 and 768×1152.
- Single-reference editing with the armV5 Edit-HQ LoRA through ComfyUI's built-in `TextEncodeMOSSImageEdit` (edit format `ai-toolkit`), 50 steps.
- The same edit with the DMD 8-step speed LoRA stacked after armV5.
- Anime style LoRA (alone at 50 steps, and stacked on the 4-step LoRA), CharRef v2 on the Base model, PixelArt16 on the Base model.

All runs needed a small Triton INT8 GEMM (`custom_nodes/MOSS_RDNA2_INT8/`) because hipBLASLt has no gfx1030 INT8 kernels. Without it, the first diffusion step fails with `HIPBLAS_STATUS_INVALID_VALUE`.

## Model files

From [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) at revision `530a4c94f92c35ef6818b1b035a983fe9dcf5f54`:

| ComfyUI folder | File | Bytes | SHA-256 |
|---|---|---:|---|
| `diffusion_models/` | `moss_image_sft_int8_convrot.safetensors` | 5,413,698,958 | `b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f` |
| `diffusion_models/` | `moss_image_base_int8_convrot.safetensors` (CharRef, PixelArt only) | 5,413,698,958 | `f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f` |
| `text_encoders/` | `qwen3.5_9b_int8_convrot.safetensors` | 9,748,121,532 | `32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a` |
| `vae/` | `flux2-vae_small.safetensors` | 249,521,340 | `d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b` |

LoRAs (all into `loras/`):

| Role | Source (pinned revision) | File | Bytes | SHA-256 |
|---|---|---|---:|---|
| 4-step T2I (v2, Euler-native) | [akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) `2ded95c37a029360245926fe72e475e30039bbfb` | `moss_image_sft_4step_v2.0.safetensors` | 1,189,176,752 | `154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f` |
| Edit-HQ armV5 | [akhaliq/MOSS-Image-Edit-HQ](https://huggingface.co/akhaliq/MOSS-Image-Edit-HQ) `3b1f2a27ddf2cbe38074531be80ba137b165ee01` | `weights/armV5/armV5.safetensors` → `armV5.safetensors` | 495,514,936 | `1fbfae4eaaafc62c66a298e63e16fc5211566e1e3de1c58f83c35c6bc226c9f9` |
| DMD 8-step speed | [lodestones/MOSS-Image-SFT-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) `f95df83b5e90ae91ea412434b3f9b7f1c8c6314f` | `lora.safetensors` → `moss_sft_8step_dmd_speed.safetensors` | 1,189,176,800 | `4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe` |
| Anime style | [axiomofmind/Moss-image-anime-lora](https://huggingface.co/axiomofmind/Moss-image-anime-lora) `38fa56fa567b70412828ac67c050a4c2c1a5c40a` | `MOSS-Image-Anime-LoRA-ComfyUI.safetensors` | 74,379,584 | `6661f88183757c7be9044578d3907335d6abe83a736995bd502b78db2c5b83cf` |
| CharRef v2 | [7err4/MOSS-Image-CharRef](https://huggingface.co/7err4/MOSS-Image-CharRef) `70db7d88c478bff21c86a3709673d17655f6e162` | `moss_image_charref_v2.safetensors` | 148,697,656 | `1a77e98228242d7ca79dd01df737d751f211f2ae676f16cce664987dbf0bac05` |
| PixelArt 16×16 | [Angais/MOSS-Image-PixelArt-16x16](https://huggingface.co/Angais/MOSS-Image-PixelArt-16x16) `c3062e0999f33c59d985457dd290ffb00c0e28b7` | `moss-image-pixelart-16x16.safetensors` | 148,689,216 | `955900cd37741e51b475c26383b1af6bb6be27d656559d0e3f6d59ec790a69b7` |

`scripts/fetch_moss_asset.sh <id>` downloads any one of these and checks size and SHA-256. For the DMD LoRA, the correct SHA-256 is the Hugging Face tree's `lfs.oid`; the CDN `etag` is a Xet chunk hash and does not match the file.

**Licenses:** the lab's catalog lists Apache-2.0 for Comfy-Org/MOSS-Image and for every LoRA above. The original models are [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) and [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image); both list **Apache-2.0** on Hugging Face (OpenMOSS-Team/MOSS-Image-SFT and Comfy-Org/MOSS-Image, checked 2026-10-30).

## Setup

1. Create a Python 3.12 venv with the ROCm 10.2 nightly torch/Triton for gfx1030 ([../hardware.md](../hardware.md)). The lab used `comfy-kitchen 0.2.40`, `comfy-aimdo 0.6.0`, `transformers 5.21.0`. Torchaudio was not installed.
2. Clone ComfyUI at v0.43.1, commit `086099f5d8886163ef77dc354f8382cb7431582a`. No core patches are needed for MOSS.
3. Copy `custom_nodes/MOSS_RDNA2_INT8/` into `<ComfyUI>/custom_nodes/`.
4. Check the INT8 kernel on the GPU before the first generation:
   ```bash
   <venv>/bin/python scripts/smoke_int8_dot.py
   ```
   It must print `MOSS_RDNA2_INT8_MATMUL_SMOKE_PASS` (exact equality against a CPU int32 matmul for five shapes plus a non-contiguous case).
5. Download the models with `MODELS=<models> bash scripts/fetch_moss_asset.sh sft_int8` (and `te_int8`, `vae_small`, `sft4_lora`, ...).
6. Start the server and submit a workflow from another terminal:
   ```bash
   COMFY=<ComfyUI> VENV=<venv> MODELS=<models> bash scripts/start_moss_comfy.sh
   python3 ../scripts/submit_workflow.py workflows/sft_fast4_int8_768x1152.json --port 8299 --seed 44
   ```
   Change the seed for every timing run. An identical graph returns ComfyUI's cached result (about 2 s, no sampling).
7. For CharRef v2 only: clone [ostris/ComfyUI-MOSS-Image-Ostris-Edit](https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit) (MIT) at commit `8f9b50c0d2b28248ecd48a4d4905c815e726f146` into `custom_nodes/` and apply `patches/ostris_edit_v1_comfy0431_rope_compat.patch` inside that clone. The plugin calls `dit.pe_embedder`, which v0.43.1 no longer has; the patch computes the same four-axis RoPE from `dit.rope_axes_dim`. The lab checked it against the native RoPE on CPU (exact match). armV5 editing does not need this plugin.

## Working settings

Launch flags: `--use-pytorch-cross-attention --disable-comfy-compiler`, DynamicVRAM left at its default (on). Edit runs added `--reserve-vram 1.0`. Environment `MOSS_RDNA2_INT8_PATCH=1`.

| Recipe | Model chain | Sampler |
|---|---|---|
| Fast T2I | SFT INT8 → 4-step LoRA 1.0 | 4 steps, Euler / simple, CFG 1.0, empty negative |
| Edit, quality | SFT INT8 → armV5 1.0 → `SkipLayerGuidanceDiTSimple` single layer 22 | 50 steps, Euler / normal, CFG 3.0 |
| Edit, fast | SFT INT8 → armV5 1.0 → DMD8 1.0 → SLG layer 22 | 8 steps, Euler / simple, CFG 1.0 |
| Anime style | SFT INT8 → Anime 1.0 | 50 steps, Euler / sgm_uniform, CFG 3, SLG 22 |
| Anime on fast T2I | SFT INT8 → 4-step 1.0 → Anime 0.7 | 4 steps, Euler / simple, CFG 1 |
| CharRef v2 | Base INT8 → CharRef v2, Ostris edit encode | 50 steps, Euler / sgm_uniform, CFG 3 |
| PixelArt16 | Base INT8 → PixelArt16 1.0 | 50 steps, Euler / sgm_uniform, CFG 3, SLG 22, 512×512 |

Edit graphs encode both prompts with `TextEncodeMOSSImageEdit`, VAE connected, reference on `images.image_0`, `image_strength` 1.0, and use an `EmptyFlux2LatentImage` of the target size. The order armV5 then DMD8, 8 steps, CFG 1, Euler/simple follows the Edit-HQ model card.

The edit prompts in `workflows/` are placeholders; the storefront T2I and storefront edit prompts are the ones the lab used.

## Measured speed and VRAM

Single runs. Time = HTTP queue to ComfyUI history, excluding server startup. VRAM = sampled sysfs peak (polled every 0.4 to 1 s, monitor started after submission).

| Task | Resolution | Steps / CFG | Cache state | Time | VRAM peak |
|---|---|---|---|---:|---:|
| T2I SFT + 4-step | 512×512 | 4 / 1 | first image after start | 22.338 s | not recorded |
| T2I SFT + 4-step | 768×1152 | 4 / 1 | first run at this size (after the 512² run) | 36.028 s | not recorded |
| T2I SFT + 4-step | 768×1152 | 4 / 1 | warm, new seed (text encoder, model, latent cached; sampler and decode re-ran) | **24.021 s** (~16.9 s sampling) | 14.55 GiB |
| T2I SFT + 4-step, anime character prompt | 768×1152 | 4 / 1 | cold | 36.235 s | not recorded |
| T2I SFT + 4-step + Anime 0.7 | 768×1152 | 4 / 1 | warm / partly cached | 28.026 s | 15.57 GiB |
| T2I SFT + Anime | 512×768 | 50 / 3 | not cached | 278.237 s | 15.76 GiB |
| Edit armV5 (storefront sign) | 512×768 | 50 / 3 | not cached; ~6.25 s/step | 340.212 s | 15.59 GiB |
| Edit armV5 (photo → anime portrait) | 528×704 | 50 / 3 | not cached / loaders and reference cached, new prompt | 348.372 s / 338.210 s | 15.43 / 15.29 GiB |
| Edit armV5 + DMD8 | 768×576 | 8 / 1 | **cold**, nothing cached | **56.036 s** | 15.44 GiB |
| Edit armV5 only, same seed and reference | 768×576 | 50 / 3 | encoders and reference cached | 358.178 s | 15.27 GiB |
| CharRef v2 on Base | 512×768 | 50 / 3 | ~7.68 s/step | 400.186 s | 15.46 GiB |
| PixelArt16 on Base | 512×512 | 50 / 3 | no cached nodes | 196.335 s (ComfyUI 195.96 s) | 15.30 GiB |

The DMD8 vs 50-step comparison (358.178 / 56.036 ≈ 6.39×) compares whole recipes: steps, CFG and scheduler all differ, and the cache state favoured the 50-step run. It is not a per-step speedup.

VRAM stays close to the 16 GiB limit in every 50-step and edit run (15.27 to 15.76 GiB sampled). The static weights alone (DiT 5.41 GB + encoder 9.75 GB + VAE + 1.19 GB FP32 LoRA) exceed 16 GB, so ComfyUI loads and offloads them in stages.

## Known limits

- Edits are regenerations. The storefront edit produced the requested sign text but regenerated reflections and moved the lettering. The photo → anime edits drifted in hair colour and clothing detail; CharRef v2 kept broad identity with costume and face drift.
- One 50-step armV5 result at seed 2026102901 looked washed out compared with the DMD8 result. This is one image and one seed.
- The INT8 kernel was checked for exact integer results on the tested GEMM shapes only. It was not tuned (fixed tiles BM32/BN64/BK32).
- The 4-step T2I LoRA is not an edit accelerator. Only the DMD8 LoRA was tested for fast editing.

## Not verified

- BF16 control runs (the lab built the graphs but did not download the BF16 weights).
- Repeated timings or cold/warm distributions.
- 1024×1024 or larger outputs.
- The chinoll edit LoRA family and other style LoRAs that the lab only catalogued.
- Training on gfx1030.
- Identity preservation across prompts and seeds.
