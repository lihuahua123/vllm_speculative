#!/usr/bin/env python3

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def make_host_tensor(num_bytes: int, pinned: bool) -> torch.Tensor:
    return torch.empty(num_bytes, dtype=torch.uint8, device="cpu", pin_memory=pinned)


def make_device_tensor(num_bytes: int, device: str) -> torch.Tensor:
    return torch.empty(num_bytes, dtype=torch.uint8, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCIe host-device transfer microbenchmark.")
    parser.add_argument("--direction",
                        choices=["h2d", "d2h"],
                        default="h2d")
    parser.add_argument("--size-mb", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    size_bytes = args.size_mb * 1024 * 1024

    host_tensors = [
        make_host_tensor(size_bytes, pinned=args.pinned)
        for _ in range(args.concurrency)
    ]
    device_tensors = [
        make_device_tensor(size_bytes, device=device)
        for _ in range(args.concurrency)
    ]
    streams = [torch.cuda.Stream(device=args.device) for _ in range(args.concurrency)]

    latencies_ms: list[float] = []
    wall_latencies_ms: list[float] = []

    total_iters = args.warmup_iters + args.iters
    for step in range(total_iters):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.concurrency)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.concurrency)]
        wall_start = time.perf_counter()
        for idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                start_events[idx].record(stream)
                if args.direction == "h2d":
                    device_tensors[idx].copy_(host_tensors[idx], non_blocking=True)
                else:
                    host_tensors[idx].copy_(device_tensors[idx], non_blocking=True)
                end_events[idx].record(stream)
        for stream in streams:
            stream.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        if step >= args.warmup_iters:
            wall_latencies_ms.append(wall_ms)
            for start_event, end_event in zip(start_events, end_events):
                latencies_ms.append(start_event.elapsed_time(end_event))

    total_bytes_per_iter = size_bytes * args.concurrency
    avg_wall_ms = statistics.mean(wall_latencies_ms) if wall_latencies_ms else 0.0
    aggregate_bandwidth_gbps = 0.0
    if avg_wall_ms > 0:
        aggregate_bandwidth_gbps = total_bytes_per_iter / (avg_wall_ms / 1000.0) / (1024**3)

    result = {
        "direction": args.direction,
        "size_mb": args.size_mb,
        "size_bytes": size_bytes,
        "concurrency": args.concurrency,
        "iters": args.iters,
        "warmup_iters": args.warmup_iters,
        "device": args.device,
        "pinned": args.pinned,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
        },
        "wall_ms": {
            "mean": avg_wall_ms,
            "p50": percentile(wall_latencies_ms, 0.50),
            "p95": percentile(wall_latencies_ms, 0.95),
            "p99": percentile(wall_latencies_ms, 0.99),
        },
        "aggregate_bandwidth_gib_s": aggregate_bandwidth_gbps,
    }

    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
