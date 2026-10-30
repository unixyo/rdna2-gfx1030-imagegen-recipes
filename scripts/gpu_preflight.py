#!/usr/bin/env python3
"""Read-only sysfs/procfs check before a GPU job. Does not import torch or touch the GPU.

Looks for an AMD GPU (vendor 0x1002) with the given PCI device id
(default 0x73af = Navi 21, as reported by the RX 6900 XT used for these recipes),
prints VRAM use and host memory, and with --gate exits 3 when VRAM use is above
--max-ratio. This is a "is the card idle?" check, not an ROCm or inference test.
"""
import argparse
import json
from pathlib import Path


def val(path):
    try:
        return int(path.read_text().strip(), 0)
    except (OSError, ValueError):
        return None


def inspect(device_id, drm=Path("/sys/class/drm"), mem=Path("/proc/meminfo")):
    gpus = []
    for card in sorted(drm.glob("card[0-9]*")):
        d = card / "device"
        if val(d / "vendor") != 0x1002 or val(d / "device") != device_id:
            continue
        total, used = val(d / "mem_info_vram_total"), val(d / "mem_info_vram_used")
        gpus.append({"card": card.name, "pci": d.resolve().name,
                     "vram_total": total, "vram_used": used,
                     "vram_used_ratio": used / total if total and used is not None else None})
    try:
        lines = mem.read_text().splitlines()
        info = {line.split(":")[0]: int(line.split(":")[1].split()[0]) * 1024
                for line in lines if ":" in line and line.split(":")[1].split()[0].isdigit()}
    except OSError:
        info = {}
    swap = info.get("SwapTotal", 0)
    free = info.get("SwapFree", 0)
    return {"gpu": gpus, "mem_available": info.get("MemAvailable"),
            "swap_total": swap, "swap_used": swap - free}


def safe(info, max_ratio):
    if len(info["gpu"]) != 1:
        return False, "target GPU PCI identity is not unique or missing"
    ratio = info["gpu"][0]["vram_used_ratio"]
    if ratio is None or ratio > max_ratio:
        return False, f"VRAM telemetry unavailable or too full: {ratio}"
    return True, "VRAM gate passed (not an ROCm or inference validation)"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", action="store_true", help="exit 3 when the gate fails")
    p.add_argument("--max-ratio", type=float, default=0.15)
    p.add_argument("--device-id", type=lambda s: int(s, 0), default=0x73AF)
    args = p.parse_args()
    if not 0.10 <= args.max_ratio <= 0.95:
        p.error("max-ratio must be in [0.10, 0.95]")
    snapshot = inspect(args.device_id)
    ok, message = safe(snapshot, args.max_ratio)
    print(json.dumps({**snapshot, "gate_pass": ok, "reason": message}, indent=2))
    if args.gate and not ok:
        raise SystemExit(3)
