#!/usr/bin/env python3
"""Actual gfx1030 GPU correctness probe vs CPU integer oracle.

Runs a few small INT8 GEMMs on the GPU and requires exact equality with a CPU
int32 matmul. Run with the ComfyUI venv's Python before the first generation.
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "custom_nodes" / "MOSS_RDNA2_INT8"))
from rdna2_int8_matmul import int8_matmul  # noqa: E402


def run():
    assert torch.version.hip is not None and torch.cuda.is_available()
    prop = torch.cuda.get_device_properties(0)
    arch = prop.gcnArchName.split(":")[0]
    assert arch == "gfx1030", arch
    cases = [(1, 32, 8), (13, 37, 17), (37, 256, 512), (33, 3072, 256), (17, 8192, 64)]
    for m, k, n in cases:
        torch.manual_seed(m * 100000 + k * 100 + n)
        a = torch.randint(-12, 13, (m, k), dtype=torch.int8)
        b = torch.randint(-12, 13, (k, n), dtype=torch.int8)
        oracle = a.to(torch.int32) @ b.to(torch.int32)
        actual = int8_matmul(a.cuda(), b.cuda()).cpu()
        if not torch.equal(actual, oracle):
            err = (actual - oracle).abs()
            raise AssertionError(f"INT8 GEMM mismatch {m,k,n}: maxerr={err.max().item()}")
        print(f"INT8_EXACT m={m} k={k} n={n} PASS", flush=True)
    # Check non-contiguous inputs as well, because Kitchen often transposes weights.
    a = torch.randint(-8, 9, (256, 37), dtype=torch.int8).t()
    b = torch.randint(-8, 9, (64, 256), dtype=torch.int8).t()
    oracle = a.to(torch.int32) @ b.to(torch.int32)
    out = int8_matmul(a.cuda(), b.cuda()).cpu()
    assert torch.equal(out, oracle), "Strided INT8 matrix mismatch"
    print("INT8_NONCONTIGUOUS_EXACT PASS", flush=True)
    print("MOSS_RDNA2_INT8_MATMUL_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    run()
