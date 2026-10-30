# Shared machine, ROCm and ComfyUI setup

All three recipes ran on one machine. Each model has its own ComfyUI checkout and its own Python virtual environment; nothing is shared between them at runtime except the GPU.

## Machine

| Item | Value recorded in the labs |
|---|---|
| GPU | AMD Radeon RX 6900 XT, RDNA2, `gfx1030`, PCI device id `0x73af` |
| VRAM | 17,163,091,968 bytes reported by sysfs (`mem_info_vram_total`), about 16 GiB |
| CPU | AMD Ryzen 7 7800X3D |
| RAM | 64 GB |
| OS | Ubuntu 26.04.1 LTS, kernel `7.0.0-38-generic` |
| PCIe | 16.0 GT/s x16 observed |
| Driver | `amdgpu`; the labs made no driver or system ROCm changes |

Swap was in use on this machine during the runs. The H3 lab added a guard that aborts a run if swap use grows by more than 3 GiB, because several H3 attempts failed on swap growth rather than on VRAM.

## ROCm: two separate stacks

The machine has two user-space ROCm builds. They do not interact.

1. **ROCm 10.2 nightly Python wheels (used by all three image/video recipes).** PyTorch, Triton and the ROCm runtime libraries are installed as pip packages inside each recipe's venv from AMD's TheRock `whl-next` index ([nightly.repo.amd.com/rocm/whl-next](https://nightly.repo.amd.com/rocm/whl-next/)). None of the recipe launch scripts source a system ROCm.
2. **ROCm 10.0.0 TheRock tarball `gfx103X-all` (not used by these recipes).** A user-local copy, activated per shell with an `env.sh`, used on this machine for llama.cpp. Its verified facts are still useful for gfx1030: `rocminfo` identifies the RX 6900 XT natively as `gfx1030`, and no `HSA_OVERRIDE_GFX_VERSION` is needed.

A system ROCm 7.1 is also installed and was left untouched.

### Python packages that worked

Recorded with `pip freeze` in the Qwen (2026-10-11) and MOSS (2026-10-28) venvs. The H3 venv was a copy of the Qwen venv's site-packages plus `gguf 0.19.0` and `protobuf 7.36.2`.

| Package | Version |
|---|---|
| Python | 3.12.14 (installed with `uv`) |
| `torch` (`amd-torch-device-gfx1030`) | `2.12.0+rocm10.2.0a20261011` |
| `torchvision` (`amd-torchvision-device-gfx1030`) | `0.27.0+rocm10.2.0a20261011` |
| `triton` | `3.8.0+git669b31ac.rocm10.2.0a20261011` |
| `rocm`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1030` | `10.2.0a20261011` |
| HIP runtime reported by torch | `7.17.26405` |
| `comfy-kitchen` | `0.2.35` (ComfyUI v0.37.2) / `0.2.40` (ComfyUI v0.43.1) |
| `comfy-aimdo` | `0.5.5` (v0.37.2) / `0.6.0` (v0.43.1) |
| `transformers` | `5.21.0` (MOSS venv) |

The labs did not record the exact `pip install` command line. The shape below is a suggestion, not a tested command:

```bash
uv venv --python 3.12 <venv>
# install torch/torchvision/triton for gfx1030 from the whl-next index, pinned to the
# versions above, then ComfyUI's requirements WITHOUT letting pip replace torch.
```

Do not run a plain `pip install -r requirements.txt` from ComfyUI first: its unpinned `torch` can pull a CUDA build.

### torchaudio

The matching nightly `torchaudio 2.11.0.3+rocm10.2.0a20261011` fails to load on this torch build (missing symbol `torch_exception_get_what_without_backtrace` in `libtorchaudio.abi3.so`). The recipes avoid it in two different ways:

- Qwen: two ComfyUI files import torchaudio lazily (`qwen-image/patches/comfyui-v0.37.2-lazy-torchaudio.patch`).
- MiniMax H3: a stub `torchaudio` package on `PYTHONPATH` (`minimax-h3/runtime_shims/`). It raises if any audio function is called, so H3 audio output does not work with it.
- MOSS: torchaudio is not installed in the venv.

## gfx1030 problems that every recipe hits

| Problem | Symptom | What the recipes do |
|---|---|---|
| hipBLASLt has no gfx1030 INT8 kernels | `torch._int_mm` -> `hipblasLtMatmulAlgoGetHeuristic` -> `HIPBLAS_STATUS_INVALID_VALUE`; logs mention missing `TensileLibrary_lazy_gfx1030.dat` | Qwen: patientx Triton INT8 patch. MOSS: a small Triton INT8 GEMM that replaces only Kitchen's matmul call. The main H3 recipe uses GGUF and FP16 weights; no INT8 patch was used in its runs. |
| MIOpen CK grouped-conv library missing | warning about `MIOpenCKGroupedConv_gfx1030.so` during VAE | Non-fatal in every run. No workaround enabled. |
| Flash / memory-efficient SDPA not compiled for gfx1030 | PyTorch falls back to the math backend | Qwen and H3 use `--use-split-cross-attention`; MOSS uses `--use-pytorch-cross-attention`. |
| `attention_split` in ComfyUI v0.37.2 does not chunk when the sequence length is not divisible by the split count | H3 OOM at the first step: a 13.94 GiB score tensor | `minimax-h3/patches/comfyui-v0.37.2-attention-split-nondivisible.patch` |

## ComfyUI versions

| Recipe | ComfyUI | Commit |
|---|---|---|
| Qwen-Image-2.1 | v0.37.2 | `830232b856045ca2892833212d7771078a13edd5` |
| MiniMax H3 | v0.37.2 (same commit, separate checkout) | `830232b856045ca2892833212d7771078a13edd5` |
| MOSS-Image | v0.43.1 | `086099f5d8886163ef77dc354f8382cb7431582a` |

MOSS needs the newer ComfyUI for its native nodes. The Qwen and H3 checkouts were not upgraded for it.

## How measurements were taken

- **Time** is either ComfyUI's "Prompt executed in" line, the API queue-to-history time, or a wrapper's wall time that also includes server startup. Each per-model table says which one.
- **VRAM peak** is the maximum of `/sys/class/drm/cardN/device/mem_info_vram_used` polled every 0.25 to 2 seconds. It includes allocator cache and can miss short spikes. It is not a PyTorch allocated-memory measurement.
- Almost every number is **one run**. Run-to-run spread on the same graph was visible (for example a Qwen 640×960 25-step edit took 80.05 s, 81.08 s and 91.19 s in three separate launches).
- Cold means the first request after server start, with model loading. Warm means a new seed on an already running server.

## Before starting a GPU job

`scripts/gpu_preflight.py --gate` reads sysfs only and exits 3 if the card already has more than 15 % VRAM in use. The labs ran one GPU job at a time and stopped each ComfyUI server after use. On 2026-10-12 the kernel logged an `sdma0` ring timeout and a failed GPU reset that needed a reboot; the labs could not attribute it to a specific process.
