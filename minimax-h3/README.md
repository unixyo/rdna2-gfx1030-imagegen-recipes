# MiniMax H3 (FL2VA) GGUF video on RX 6900 XT (gfx1030)

Text-to-video and first-frame image-to-video, 124 frames at 24 fps, with GGUF-quantized H3 weights and the LightX2V Turbo LoRAs, in ComfyUI v0.37.2 with ComfyUI-GGUF, on ROCm 10.2 nightly wheels. See [../hardware.md](../hardware.md) for the machine and Python stack.

This is the slowest of the three recipes. A 5-second 512×768 I2V clip took 462.56 s with 4 steps and 855.00 s with 8 steps. **These are early-experiment settings: a starting point for speed, not a quality reference.** Quality was only judged by eye on a few clips, and those clips looked clearly behind MiniMax's own web examples.

## What works

- T2V 608×352, 124 frames, Turbo 4 steps.
- I2V from a first frame at 384×576, 512×768, 576×864 and 640×640, 124 frames.
- DiT Q4_0 or Q5_K_M, text encoder (Qwen3-VL 32B H3) Q2_K or Q4_K_M, with CPU offload.
- Text encoder on the GPU (faster) with a 3-line ComfyUI patch.

Audio is not generated: the torchaudio stub used to start ComfyUI disables it.

## Model files

| Role | Source (pinned revision) | File → ComfyUI folder | Bytes | SHA-256 |
|---|---|---|---:|---|
| DiT, Q4_0 (baseline) | [molbal/MiniMax-H3-GGUF](https://huggingface.co/molbal/MiniMax-H3-GGUF) `b45874371c61c49bf04096602f5527aac71b360b` | `minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` → `unet/` | 11,377,542,880 | `50891b806d6d700f4f20931791ca42a083dd9148609838268ccdc782bf899c1c` |
| DiT, Q5_K_M (optional) | [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) `76061e7fe4866dbc68903bfdf852430f0f1b2487` | `MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf` → `unet/` | 14,068,966,496 | `abe739ac98bc14d6e4e56987198765b02899da632ad254c820f559b6dc4d1171` |
| Text encoder, Q2_K (baseline) | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) `daf03b4ca652cce16dfd4fcf91e79c52ffa5c1e7` | `qwen3vl-32B-MiniMax-H3-Q2_K.gguf` → `clip/` | 8,487,968,160 | `5bbc11d0b3ef197c98df2ce8f05de8fbb8eb5917cd91c33d0b59f93759b34914` |
| Text encoder, Q4_K_M (optional) | same repo and revision | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` → `clip/` | 14,576,977,888 | `1bf75e038c5895b97b6ea16cc1e3d32076254b06ec3df10657650d86dc82279e` |
| Video VAE FP16 | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) `e5eb578a89295337b8ff433a035929ce0279e0b6` | `vae/minimax_h3_video_vae_fp16.safetensors` → `vae/` | 5,207,808,496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| Turbo 4-step LoRA | same Comfy-Org revision | `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` → `loras/` | 1,956,192,992 | `c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc` |
| Turbo 8-step 768p LoRA (optional) | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) `3ec17a324ced54151364f24f8b5fb6bf7e26414f` | `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` → `loras/` | 1,956,193,000 | `08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980` |

The Q4_0 and Q5_K_M DiT files come from different publishers. Same tensor names and shapes (532/532), but the lab could not prove they were converted from the same checkpoint. Q5_K_M stores some 1-D FP32 tensors with `comfy.gguf.orig_shape` metadata, which the pinned ComfyUI-GGUF loader already handles.

**License:** the Hugging Face model cards of Comfy-Org/MiniMax-H3, molbal/MiniMax-H3-GGUF and Abiray/MiniMax-H3-Pruned-GGUF list the **MiniMax H3 Community License Agreement** (checked 2026-10-30); realrebelai/MiniMax-H3_GGUFs and lightx2v/Minimax-h3-Turbo list no license. Read the agreement in [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3), [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) and each GGUF/LoRA repository above before use.

## Setup

1. Create a Python 3.12 venv with the ROCm 10.2 nightly torch/Triton for gfx1030 ([../hardware.md](../hardware.md)), plus `gguf` (lab: 0.19.0) and `protobuf` (lab: 7.36.2).
2. Clone ComfyUI v0.37.2 at commit `830232b856045ca2892833212d7771078a13edd5` and apply, inside the checkout:
   - `patches/comfyui-v0.37.2-attention-split-nondivisible.patch` (**required**). Without it the first sampling step tried to allocate a 13.94 GiB attention tensor and failed.
   - `patches/comfyui-v0.37.2-text-encoder-gpu-optin.patch` (recommended). Adds `H3_LAB_TE_GPU=1` to put the text encoder on the GPU while keeping `--lowvram` for the DiT.
   - `patches/comfyui-v0.37.2-node-timing-optin.patch` (optional). Logs per-node seconds when `H3_LAB_PROFILE_NODES=1`.
3. Clone [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) at commit `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` into `custom_nodes/` and apply `patches/comfyui-gguf-6ea2651-allow-minimax_h3.patch` inside it (adds `minimax_h3` to the accepted architectures; see [city96/ComfyUI-GGUF#476](https://github.com/city96/ComfyUI-GGUF/pull/476)).
4. Download the model files into `<ComfyUI>/models/`.
5. Start the server (sets the torchaudio stub, the GPU text encoder flag and the measured launch flags):
   ```bash
   COMFY=<ComfyUI> VENV=<venv> RECIPE=<this repo>/minimax-h3 RESERVE_VRAM=3 \
     bash scripts/launch_h3_comfy.sh
   ```
6. For I2V, put a 512×768 first frame at `<ComfyUI>/input/first_frame_512x768.png` and submit:
   ```bash
   python3 ../scripts/submit_workflow.py workflows/h3_i2v_turbo4_lora085_shift6_512x768_124f_api.json --port 8188 --max-wait 1800
   ```

## Working settings

Launch flags (all measured runs): `--lowvram --disable-dynamic-vram --bf16-unet --disable-async-offload --reserve-vram <2|3|4> --use-split-cross-attention`, environment `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, the torchaudio stub on `PYTHONPATH`. `--reserve-vram`, `--disable-async-offload` and `--bf16-unet` came from a public RX 6700 XT example ([GabrielSalerno03/minimax-h3-rdna2](https://github.com/GabrielSalerno03/minimax-h3-rdna2)); the lab added `--lowvram --disable-dynamic-vram`. No other flag combination was measured for this recipe.

Graph (see `workflows/`):

| Node | Setting |
|---|---|
| `UnetLoaderGGUFAdvanced` | DiT GGUF, `dequant_dtype` bfloat16 |
| `CLIPLoaderGGUF` | H3 Qwen3-VL GGUF, type `minimax` |
| `LoraLoaderModelOnly` | Turbo 4-step LoRA; strength 1.0 for T2V, **0.85** for I2V |
| `MiniMaxH3SigmaShift` | **video 6, audio 3**, placed after the LoRA and before scheduler and guider (I2V graph) |
| `MiniMaxH3ImageToVideo` | width/height, `length` 124, `first_frame` = LoadImage for I2V |
| Sampling | `RandomNoise` → `BasicGuider` → `KSamplerSelect` euler → `BasicScheduler` simple, 4 steps → `SamplerCustomAdvanced` |
| `VAEDecodeTiled` | tile 128, overlap 32, temporal 8, temporal overlap 4 |
| `CreateVideo` / `SaveVideo` | 24 fps; I2V graph saves H.264 CRF 14 |

Two settings mattered for quality in the lab's tests:

- **Sigma shift 6/3.** The LightX2V 4-step LoRAs expect video shift 6 and audio shift 3. This ComfyUI's H3 default video shift was 12. At 384×576, with the SLA-trained 4-step LoRA and dense attention, the first decoded frame's SSIM against the reference was 0.725 with shift 12 and 0.933 with shift 6 (same image and seed).
- **LoRA strength 0.85 for I2V.** At 512×768 it removed most of the broad gray horizontal bands that strength 1.0 produced on a white background, and first-frame SSIM went 0.943 → 0.946. One clip, one seed.

The 8-step 768p LoRA uses shift 6/3 as well; the older 544p 8-step variant uses 12/3.

`length` must be 17·n + 5 frames (22, 39, 124, ...). 124 frames is 5.17 s at 24 fps.

## Measured speed and VRAM

Single runs. "Total" is the lab runner's wall time, which includes ComfyUI startup, model loading, encoding, sampling, VAE decode and saving. VRAM = sysfs peak sampled every 2 s.

### T2V, 608×352, 124 frames, DiT Q4_0, TE Q2_K, Turbo4 LoRA 1.0, `--reserve-vram 2`

| Variant | Total | Sampling | VAE decode | VRAM peak |
|---|---:|---:|---:|---:|
| Text encoder on CPU (first success) | 377.90 s (ComfyUI prompt 370.98 s) | 171.4 s, 42.85 s/step | not timed | 12.35 GiB |
| Text encoder on GPU (`H3_LAB_TE_GPU=1`) | **228.32 s** | sampler node 189.46 s | 20.10 s | 12.84 GiB |

The two outputs are not pixel-identical (frame-aligned SSIM 0.9809).

### I2V, 124 frames, DiT Q4_0, TE Q2_K, Turbo4, shift 6/3, text encoder on GPU

| Resolution | LoRA strength / prompt | `--reserve-vram` | Total | VRAM peak |
|---|---|---:|---:|---:|
| 384×576 | 1.0, dense | not recorded | 226.32 s | 13.07 GiB |
| 512×768 | 1.0, dense | 3 | 481.65 s (sampler 418.43 s) | 12.70 GiB |
| 512×768 | 1.0, timed-motion prompt | 3 | 550.93 s | not recorded |
| 512×768 | **0.85**, timed-motion prompt | 3 | **462.56 s** (sampling ~387 s) | 12.76 GiB |
| 576×864 | 0.85, timed-motion prompt | 3 | 724.78 s | 12.80 GiB |
| 512×768 | 1.0, 8-step 768p LoRA | 3 | 855.00 s | ~12.7 GiB |

576×864 has 26.6 % more pixels than 512×768 and took about 57 % longer; the lab found it slightly sharper and kept 512×768 as the draft size.

### I2V square 640×640, 124 frames, LoRA 0.85, shift 6/3, `--reserve-vram 4`

| DiT / TE / steps | Total | VRAM peak |
|---|---:|---:|
| Q4_0 / Q2_K / 4 | 475.51 s | 13.36 GiB |
| Q4_0 / Q4_K_M / 4 | 530.35 s | 12.95 GiB |
| Q5_K_M / Q4_K_M / 4 (~99–100 s per step) | 497.74 s | 14.21 GiB |
| Q5_K_M / Q4_K_M / 8 | 895.17 s | 11.94 GiB |

Short 22-frame tests with Q5_K_M / Q4_K_M: 640×640 Turbo4 165.9 s (12.21 GiB), 768×768 Turbo4 266.40 s (14.21 GiB), 640×640 Turbo8 226.90 s, 640×640 without Turbo LoRA at 20 steps 367.33 s (12.21 GiB). These are loader/OOM checks; 22-frame memory does not predict 124-frame memory.

Peak VRAM values move with ComfyUI's load/offload decisions, so a higher-precision run sometimes showed a lower peak. Do not read the table as "8 steps uses less VRAM".

Quality findings from the lab, all single clips: Q4_K_M vs Q2_K text encoder gave no clear visual gain. Q5_K_M DiT gave a measurable first-frame face improvement over Q4_0 on one close-up. 8 steps vs 4 steps gave more motion, not a clear quality gain.

## Experimental paths (not packaged here)

- **Sparse attention (SLA).** [patientx/comfyui-h3-sla-attention-rocm](https://github.com/patientx/comfyui-h3-sla-attention-rocm) at commit `dc6d7785a76fc047aba3471fc10c4dce8e22df44`. T2V 608×352: 174.27 s total with 85 % sparsity and `min_seq_len` 4096 (the default 8192 never activated the sparse kernel because H3's sequence length here is 8175). With the normal Turbo4 LoRA the output diverged strongly from dense attention (SSIM 0.689). With the SLA-trained LoRA ([lightx2v/Minimax-h3-Turbo-SLA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) revision `10ade67cd15ff7a135fa35c2a0673ea96c839247`, file `minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors`, SHA-256 `5cae6df40a06ea825f85fc8876c9ea1c9692c833a9af07bb8b3bac9ce2a71bac`): T2V sparse 155.29 s vs dense 244.64 s; I2V 384×576 sparse 169.28 s vs dense 227.31 s; I2V 512×768 sparse 314.43 s (13.02 GiB, `--reserve-vram 3`). Visual quality was not assessed, so dense split attention stays the default.
- **VDN DMD 8-step (true hybrid linear attention).** Uses [xmarre/ComfyUI-VDN-H3-Plus](https://github.com/xmarre/ComfyUI-VDN-H3-Plus) at `1e19b986d1bfaa97b2ae1bb5f145264da867e705` with local, unpublished gfx1030 changes (query chunking in the attention fallback, Gram matrix guards). 39 frames, Q5_K_M DiT, Q2_K TE: 512×512 243.5 s (14.69 GiB, pinned memory off, `--reserve-vram 5`); 640×640 313.68 s with query chunk 256 (12.09 GiB) and 338.51 s with chunk 512 plus a Gram guard (12.81 GiB). Other attempts aborted on swap growth, VRAM, or non-finite Gram matrices before Cholesky. The numerical fault is not fixed, so this path is not reproducible from this repository.
- **Operator research (2026-10-23).** A native-contract INT8→INT32 GEMM ran on gfx1030 but was 62.2 % slower than H3's FP32 dequantize path, and a subquadratic attention fallback cut extra allocation by 41.72 % / 68.43 % on synthetic inputs while adding 15.47 % / 6.98 % latency. Neither was adopted in the video recipe.

## Known limits

- Output quality: judged only by eye on a few clips during early experiments, where results looked clearly behind MiniMax's web examples. Changes to resolution, TE precision, DiT precision, steps and LoRA strength each gave small or mixed differences, and the cause of the gap was not found. No systematic quality evaluation was done.
- A VAE-only round-trip test showed large losses (PSNR 24.19 dB for 17→5 frames vs 35.87 dB for codec only), but the lab warns these artificial static inputs do not represent production decoding, so the VAE is not proven to be the cause.
- No audio (torchaudio stub).
- 16 GiB is enough only with `--lowvram` offload. The DiT ran partly on CPU (for example 5,460 MB GPU + 8,250 MB CPU for Q5_K_M), so host RAM and swap matter.
- On 2026-10-12 one H3 trial hung while loading the text encoder after a kernel `sdma0` GPU reset; a reboot was needed. Not attributed to H3.

## Not verified

- Repeated runs or any speed distribution.
- Human review of temporal consistency for most clips.
- Resolutions above 576×864 at 124 frames on this card.
- Other launch flag combinations (for example DynamicVRAM instead of `--lowvram`).
- Audio generation.

Results from a cloud NVIDIA L4 session in the same lab are excluded because they are not RX 6900 XT results.
