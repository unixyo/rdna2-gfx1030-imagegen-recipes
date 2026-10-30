#!/usr/bin/env python3
"""Submit one ComfyUI API-format workflow to a running localhost ComfyUI and time it.

The reported time is HTTP queue -> history entry, which excludes server startup.
Use a new --seed for each timing run: ComfyUI returns cached results for an
identical graph (a 768x1152 MOSS repeat came back in about 2 s without sampling).

Standard library only. This DOES start GPU work on the server.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

SEED_KEYS = ("seed", "noise_seed")


def req_json(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def set_seed(graph, seed):
    changed = []
    for node_id, node in graph.items():
        inputs = node.get("inputs", {})
        for key in SEED_KEYS:
            if key in inputs and not isinstance(inputs[key], list):
                inputs[key] = seed
                changed.append(f"{node_id}.{key}")
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workflow", type=Path, help="API-format workflow JSON")
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--seed", type=int, default=None, help="override every literal seed input")
    p.add_argument("--max-wait", type=int, default=1800, help="seconds")
    p.add_argument("--log", type=Path, default=None, help="optional JSON result file")
    args = p.parse_args()

    graph = json.loads(args.workflow.read_text(encoding="utf-8"))
    if args.seed is not None:
        if not 0 <= args.seed <= 0xFFFFFFFFFFFFFFFF:
            p.error("--seed must be an unsigned 64-bit integer")
        print("SEED_SET", set_seed(graph, args.seed), flush=True)

    base = f"http://127.0.0.1:{args.port}"
    t0 = time.perf_counter()
    try:
        reply = req_json(base + "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    except urllib.error.HTTPError as exc:
        print("QUEUE_HTTP_ERROR", exc.code, exc.read().decode(errors="replace")[:10000], flush=True)
        raise SystemExit(3)
    if "prompt_id" not in reply:
        print("QUEUE_REJECTED", reply, flush=True)
        raise SystemExit(4)
    prompt_id = reply["prompt_id"]
    print("QUEUED", prompt_id, flush=True)

    out = None
    while time.perf_counter() - t0 <= args.max_wait:
        history = req_json(base + f"/history/{prompt_id}")
        if prompt_id in history:
            out = history[prompt_id]
            break
        time.sleep(2)
    if out is None:
        print("TIMEOUT", args.max_wait, flush=True)
        raise SystemExit(5)

    elapsed = round(time.perf_counter() - t0, 3)
    files = []
    for node in out.get("outputs", {}).values():
        for key in ("images", "gifs", "videos"):
            for f in node.get(key, []):
                files.append({"subfolder": f.get("subfolder", ""), "filename": f.get("filename")})
    record = {"workflow": str(args.workflow), "prompt_id": prompt_id,
              "elapsed_seconds_queue_to_history": elapsed,
              "status": out.get("status"), "outputs": files}
    print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
    if args.log:
        args.log.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if (out.get("status") or {}).get("status_str") != "success":
        raise SystemExit(6)


if __name__ == "__main__":
    main()
