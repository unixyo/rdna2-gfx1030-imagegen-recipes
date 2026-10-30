"""Isolated INT8×INT8->INT32 matrix multiply for gfx1030, implemented in Triton.

This replaces ONLY the backend call that normally reaches unsupported hipBLASLt.
Comfy Kitchen's activation quantization, optional ConvRot, scaling, bias and
residual operations remain untouched. No source/venv from the Qwen lab is used.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dot_i8_i32_kernel(
    a_ptr, b_ptr, c_ptr,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    asm: tl.constexpr, ask: tl.constexpr, bsk: tl.constexpr, bsn: tl.constexpr,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * bm + tl.arange(0, bm)
    cols = pn * bn + tl.arange(0, bn)
    d = tl.arange(0, bk)
    acc = tl.full((bm, bn), 0, tl.int32)
    for ki in range(tl.cdiv(k, bk)):
        kk = ki * bk + d
        a = tl.load(a_ptr + rows[:, None] * asm + kk[None, :] * ask,
                    mask=(rows[:, None] < m) & (kk[None, :] < k), other=0)
        b = tl.load(b_ptr + kk[:, None] * bsk + cols[None, :] * bsn,
                    mask=(kk[:, None] < k) & (cols[None, :] < n), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rows[:, None] * n + cols[None, :], acc,
             mask=(rows[:, None] < m) & (cols[None, :] < n))


def int8_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"Expected [M,K] @ [K,N], got {a.shape} and {b.shape}")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError(f"Expected two int8 tensors; got {a.dtype} and {b.dtype}")
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("Both tensors must be on the same HIP device")
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=torch.int32)
    if m == 0 or n == 0:
        return out
    if k == 0:
        return out.zero_()
    _dot_i8_i32_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        a, b, out, m, n, k, *a.stride(), *b.stride(),
        32, 64, 32, num_warps=4, num_stages=1
    )
    return out
