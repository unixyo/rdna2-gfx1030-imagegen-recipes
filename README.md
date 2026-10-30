# RDNA2 (gfx1030) image and video generation recipes

Working ComfyUI setups for three recent generation models on an **AMD Radeon RX 6900 XT** (RDNA2, `gfx1030`, 16 GB VRAM) with 64 GB RAM under Ubuntu, using AMD's ROCm 10.2 nightly PyTorch wheels.

gfx1030 is missing several kernels that newer cards have (hipBLASLt INT8, Flash attention). Each recipe lists the exact files, patches and flags that got it running, the measured times with their conditions, and what was not verified.

| Recipe | Task | Headline result (single runs) |
|---|---|---|
| [qwen-image/](qwen-image/README.md) | Qwen-Image-2.1 INT8 ConvRot, T2I and one-reference edit | 640×960 edit, 25 steps: 80.05 s, 12.41 GiB peak. Turbo LoRA 8 steps: 48.03 s |
| [moss-image/](moss-image/README.md) | MOSS-Image SFT INT8, fast T2I and Edit-HQ editing | 768×1152 T2I, 4 steps, warm: 24.02 s, 14.55 GiB. 768×576 edit with DMD 8 steps, cold: 56.04 s |
| [minimax-h3/](minimax-h3/README.md) | MiniMax H3 FL2VA GGUF, 124-frame T2V and I2V | 608×352 T2V, 4 steps: 228.32 s total. 512×768 I2V, 4 steps: 462.56 s total |

Start with [hardware.md](hardware.md): it describes the machine, the ROCm/PyTorch versions, the gfx1030 problems that all three recipes work around, and how the numbers were measured.

## Read this before trusting a number

- Nearly every time and VRAM figure is from **one run**. Repeat runs of the same Qwen graph took 80.05 s, 81.08 s and 91.19 s.
- VRAM is a sysfs value polled every 0.25 to 2 s. It includes allocator cache and can miss short spikes.
- "Time" means different things in different tables (ComfyUI prompt time, API queue-to-history, or wall time including server start). Each table says which.
- Quality statements are the lab's visual notes on one or a few images or clips. Pixel and SSIM metrics in the READMEs show that a setting changed the output; they do not rank quality.

## Repository layout

```
hardware.md                 shared machine, ROCm, ComfyUI notes
scripts/gpu_preflight.py    read-only "is the GPU idle?" check (sysfs)
scripts/submit_workflow.py  submit an API workflow to localhost ComfyUI and time it
qwen-image/                 README, launch script, INT8 bridge node, patch, workflows
moss-image/                 README, start script, INT8 GEMM node, fetch script, patch, workflows
minimax-h3/                 README, launch script, patches, torchaudio stub, workflows
```

Workflows are ComfyUI **API-format** JSON (for `POST /prompt`), not UI workflows. Their prompts and input image names are placeholders.

## What this repository does not contain

- Model weights. Every README lists the Hugging Face repository, pinned revision, file size and SHA-256.
- The patientx INT8 Triton files (AGPL-3.0). `qwen-image/scripts/fetch_patientx_int8.sh` downloads them at a pinned commit.
- ComfyUI itself or any custom node repository. The READMEs give the commit to check out.
- Results from a cloud NVIDIA L4 session and other non-RX 6900 XT experiments from the same labs.

## Licenses

The code and documentation in this repository are MIT ([LICENSE](LICENSE)), with these exceptions:

- `*/patches/*.patch` modify third-party files and keep those files' licenses (ComfyUI: GPL-3.0; ComfyUI-GGUF: Apache-2.0; Ostris MOSS edit node: MIT). See each `patches/README.md`.
- The models, LoRAs and upscalers are under their own licenses, linked from each recipe README. Some are research-only. Check them before use.

## Safety notes

All servers in these recipes bind to `127.0.0.1`. Run one GPU job at a time and stop the server afterwards to release VRAM. None of the scripts download anything unless you run a fetch script.
