#!/usr/bin/env python3
"""Estimate elastic KV-cache resize and draft-model transfer time.

This script mirrors the main data movements in:
  - vllm/worker/cache_engine.py::increase_gpu_blocks
  - vllm/worker/cache_engine.py::decrease_gpu_blocks
  - vllm/spec_decode/spec_decode_worker.py::offload/load_neural_model_async

It benchmarks a bounded sample on the current GPU and linearly extrapolates to
the requested block count. Use --estimate-only to skip CUDA allocation and rely
on explicit bandwidth assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-Triton hosts.
    triton = None
    tl = None


DTYPE_SIZES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}

TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


if triton is not None:

    @triton.jit
    def kv_cache_block_migration_kernel(
        cache_ptr,
        old_ids_ptr,
        new_ids_ptr,
        num_blocks: tl.constexpr,
        block_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        mapping_id = tl.program_id(0)
        kv_plane = tl.program_id(1)
        chunk_id = tl.program_id(2)

        old_block_id = tl.load(old_ids_ptr + mapping_id)
        new_block_id = tl.load(new_ids_ptr + mapping_id)

        offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_elements
        plane_base = kv_plane * num_blocks * block_elements
        src = plane_base + old_block_id * block_elements + offsets
        dst = plane_base + new_block_id * block_elements + offsets
        values = tl.load(cache_ptr + src, mask=mask)
        tl.store(cache_ptr + dst, values, mask=mask)


@dataclass
class Stats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def make_stats(values: list[float]) -> Stats:
    return Stats(
        mean_ms=statistics.mean(values) if values else 0.0,
        p50_ms=percentile(values, 0.50),
        p95_ms=percentile(values, 0.95),
        min_ms=min(values) if values else 0.0,
        max_ms=max(values) if values else 0.0,
    )


def gib_per_s(num_bytes: float, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return num_bytes / (ms / 1000.0) / (1024**3)


def ms_from_bandwidth(num_bytes: float, bandwidth_gib_s: float) -> float:
    if bandwidth_gib_s <= 0:
        return 0.0
    return num_bytes / (bandwidth_gib_s * (1024**3)) * 1000.0


def kv_block_bytes_per_layer(args: argparse.Namespace) -> int:
    dtype_size = DTYPE_SIZES[args.dtype]
    # The CUDA attention cache stores both K and V for normal attention.
    # This matches the shape used by CacheEngine expansion:
    # [2, num_blocks, block_size * num_kv_heads, head_size].
    return 2 * args.block_size * args.num_kv_heads * args.head_size * dtype_size


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_cuda(fn, iters: int, warmup_iters: int) -> Stats:
    values: list[float] = []
    total = warmup_iters + iters
    for step in range(total):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if step >= warmup_iters:
            values.append(elapsed_ms)
    return make_stats(values)


def benchmark_expand(args: argparse.Namespace,
                     block_bytes: int) -> tuple[Stats, float, dict[str, int]]:
    sample_old = max(1, min(args.current_blocks, args.sample_current_blocks))
    scale = args.change_blocks / max(args.current_blocks, 1)
    sample_add = max(1, min(args.change_blocks, math.ceil(sample_old * scale)))
    sample_shape = (2, sample_old, args.block_size * args.num_kv_heads,
                    args.head_size)
    add_shape = (2, sample_add, args.block_size * args.num_kv_heads,
                 args.head_size)
    dtype = TORCH_DTYPES.get(args.dtype, torch.float16)
    cache = torch.empty(sample_shape, dtype=dtype, device=args.device)

    if args.expand_backend == "pad":

        def run_once() -> None:
            nonlocal cache
            expanded = F.pad(cache, (0, 0, 0, 0, 0, sample_add))
            cache = expanded[:, :sample_old].contiguous()

        traffic = args.num_layers * block_bytes * (2 * sample_old + sample_add)
    else:
        appended = [
            torch.empty(add_shape, dtype=dtype, device=args.device)
            for _ in range(args.sample_layers)
        ]

        def run_once() -> None:
            for slab in appended:
                if args.append_init == "zero":
                    slab.zero_()
                else:
                    slab.fill_(1)

        traffic = args.sample_layers * block_bytes * sample_add

    stats = timed_cuda(run_once, args.iters, args.warmup_iters)
    bandwidth = gib_per_s(traffic, stats.mean_ms)
    return stats, bandwidth, {
        "sample_old_blocks": sample_old,
        "sample_added_blocks": sample_add,
        "sample_layers": (args.num_layers if args.expand_backend == "pad"
                           else args.sample_layers),
        "sample_traffic_bytes": traffic,
        "backend": args.expand_backend,
        "append_init": (args.append_init
                        if args.expand_backend == "append" else None),
    }


def benchmark_shrink(args: argparse.Namespace,
                     block_bytes: int) -> tuple[Stats, float, dict[str, int]]:
    sample_old = max(2, min(args.current_blocks, args.sample_current_blocks))
    requested_migrated = 0 if args.migrated_blocks is None else args.migrated_blocks
    sample_migrated = min(requested_migrated, args.change_blocks,
                          sample_old // 2)
    if sample_migrated <= 0:
        return make_stats([0.0]), 0.0, {
            "sample_old_blocks": sample_old,
            "sample_migrated_blocks": 0,
            "sample_layers": args.sample_layers,
            "sample_traffic_bytes": 0,
            "backend": args.shrink_backend,
            "kernel": None,
        }
    sample_shape = (2, sample_old, args.block_size * args.num_kv_heads,
                    args.head_size)
    dtype = TORCH_DTYPES.get(args.dtype, torch.float16)
    caches = [
        torch.empty(sample_shape, dtype=dtype, device=args.device)
        for _ in range(args.sample_layers)
    ]
    old_ids = torch.arange(sample_old - sample_migrated,
                           sample_old,
                           dtype=torch.long,
                           device=args.device)
    new_ids = torch.arange(0,
                           sample_migrated,
                           dtype=torch.long,
                           device=args.device)

    if args.shrink_backend == "triton":
        if triton is None:
            raise SystemExit("Triton is not available. Use "
                             "--shrink-backend torch or --estimate-only.")
        old_ids_i32 = old_ids.to(torch.int32)
        new_ids_i32 = new_ids.to(torch.int32)
        block_elements = args.block_size * args.num_kv_heads * args.head_size
        triton_block = args.triton_block_size

        def run_once() -> None:
            grid = (sample_migrated, 2,
                    triton.cdiv(block_elements, triton_block))
            for cache in caches:
                kv_cache_block_migration_kernel[grid](
                    cache, old_ids_i32, new_ids_i32, sample_old,
                    block_elements, triton_block)
    else:
        def run_once() -> None:
            for cache in caches:
                src = cache.index_select(1, old_ids)
                cache.index_copy_(1, new_ids, src)

    stats = timed_cuda(run_once, args.iters, args.warmup_iters)
    traffic = args.sample_layers * block_bytes * sample_migrated * 2
    bandwidth = gib_per_s(traffic, stats.mean_ms)
    return stats, bandwidth, {
        "sample_old_blocks": sample_old,
        "sample_migrated_blocks": sample_migrated,
        "sample_layers": args.sample_layers,
        "sample_traffic_bytes": traffic,
        "backend": args.shrink_backend,
        "kernel": ("kv_cache_block_migration_kernel"
                   if args.shrink_backend == "triton" else None),
    }


def benchmark_transfer(args: argparse.Namespace,
                       direction: str) -> tuple[Stats, float, dict[str, int]]:
    sample_bytes = min(args.draft_model_bytes, args.sample_transfer_mb * 1024**2)
    sample_bytes = max(1, int(sample_bytes))
    pinned = not args.no_pinned_cpu
    host = torch.empty(sample_bytes,
                       dtype=torch.uint8,
                       device="cpu",
                       pin_memory=pinned)
    device_tensor = torch.empty(sample_bytes,
                                dtype=torch.uint8,
                                device=args.device)

    def run_once() -> None:
        if direction == "cpu_to_gpu":
            device_tensor.copy_(host, non_blocking=pinned)
        else:
            host.copy_(device_tensor, non_blocking=pinned)

    stats = timed_cuda(run_once, args.iters, args.warmup_iters)
    bandwidth = gib_per_s(sample_bytes, stats.mean_ms)
    return stats, bandwidth, {
        "sample_bytes": sample_bytes,
        "pinned_cpu": pinned,
    }


def benchmark_concurrent_transfer(
        args: argparse.Namespace,
        direction: str) -> tuple[Stats, float, dict[str, int]]:
    sample_bytes = min(args.draft_model_bytes, args.sample_transfer_mb * 1024**2)
    sample_bytes = max(1, int(sample_bytes))
    pinned = not args.no_pinned_cpu
    concurrency = max(1, args.transfer_concurrency)
    hosts = [
        torch.empty(sample_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=pinned)
        for _ in range(concurrency)
    ]
    devices = [
        torch.empty(sample_bytes, dtype=torch.uint8, device=args.device)
        for _ in range(concurrency)
    ]
    streams = [torch.cuda.Stream(device=args.device) for _ in range(concurrency)]

    def run_once() -> None:
        for idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                if direction == "cpu_to_gpu":
                    devices[idx].copy_(hosts[idx], non_blocking=pinned)
                else:
                    hosts[idx].copy_(devices[idx], non_blocking=pinned)
        for stream in streams:
            stream.synchronize()

    stats = timed_cuda(run_once, args.iters, args.warmup_iters)
    bandwidth = gib_per_s(sample_bytes * concurrency, stats.mean_ms)
    return stats, bandwidth, {
        "sample_bytes_per_transfer": sample_bytes,
        "concurrency": concurrency,
        "pinned_cpu": pinned,
    }


def build_estimate(args: argparse.Namespace) -> dict[str, Any]:
    block_bytes = kv_block_bytes_per_layer(args)
    total_change_blocks = args.change_blocks
    shrink_new_blocks = max(args.current_blocks - args.change_blocks, 0)
    migrated_blocks = args.migrated_blocks
    if migrated_blocks is None:
        migrated_blocks = args.change_blocks

    expand_pad_traffic = args.num_layers * block_bytes * (
        2 * args.current_blocks + total_change_blocks)
    expand_append_traffic = args.num_layers * block_bytes * total_change_blocks
    expand_traffic = (expand_pad_traffic if args.expand_backend == "pad"
                      else expand_append_traffic)
    shrink_traffic = args.num_layers * block_bytes * migrated_blocks * 2
    trim_copy_traffic = args.num_layers * block_bytes * shrink_new_blocks * 2

    result: dict[str, Any] = {
        "input": {
            "current_blocks": args.current_blocks,
            "change_blocks": args.change_blocks,
            "migrated_blocks_for_shrink": migrated_blocks,
            "num_layers": args.num_layers,
            "block_size": args.block_size,
            "num_kv_heads": args.num_kv_heads,
            "head_size": args.head_size,
            "dtype": args.dtype,
            "draft_model_bytes": args.draft_model_bytes,
            "transfer_concurrency": args.transfer_concurrency,
            "expand_backend": args.expand_backend,
            "model_config": str(args.model_config) if args.model_config else None,
        },
        "derived": {
            "kv_block_bytes_per_layer": block_bytes,
            "kv_block_mib_per_layer": block_bytes / 1024**2,
            "expand_backend": args.expand_backend,
            "append_init": args.append_init if args.expand_backend == "append" else None,
            "expand_traffic_bytes": expand_traffic,
            "expand_pad_traffic_bytes": expand_pad_traffic,
            "expand_append_traffic_bytes": expand_append_traffic,
            "shrink_migration_traffic_bytes": shrink_traffic,
            "shrink_full_trim_copy_traffic_bytes": trim_copy_traffic,
        },
    }

    if args.estimate_only:
        result["bandwidth_source"] = "manual"
        result["bandwidth_gib_s"] = {
            "expand_effective": args.gpu_mem_bw_gib_s,
            "shrink_migration_effective": args.gpu_mem_bw_gib_s,
            "cpu_to_gpu": args.h2d_bw_gib_s,
            "gpu_to_cpu": args.d2h_bw_gib_s,
        }
        result["estimated_ms"] = {
            "expand": ms_from_bandwidth(expand_traffic,
                                        args.gpu_mem_bw_gib_s),
            "expand_pad": ms_from_bandwidth(expand_pad_traffic,
                                            args.gpu_mem_bw_gib_s),
            "expand_append": ms_from_bandwidth(expand_append_traffic,
                                               args.gpu_mem_bw_gib_s),
            "shrink_migration": ms_from_bandwidth(shrink_traffic,
                                                  args.gpu_mem_bw_gib_s),
            "shrink_full_trim_copy": ms_from_bandwidth(
                trim_copy_traffic, args.gpu_mem_bw_gib_s),
            "draft_reload_cpu_to_gpu": ms_from_bandwidth(
                args.draft_model_bytes, args.h2d_bw_gib_s),
            "draft_offload_gpu_to_cpu": ms_from_bandwidth(
                args.draft_model_bytes, args.d2h_bw_gib_s),
            "concurrent_draft_reload_group": ms_from_bandwidth(
                args.draft_model_bytes * args.transfer_concurrency,
                args.h2d_bw_gib_s),
            "concurrent_draft_offload_group": ms_from_bandwidth(
                args.draft_model_bytes * args.transfer_concurrency,
                args.d2h_bw_gib_s),
        }
        return result

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Use --estimate-only.")
    torch.cuda.set_device(args.device)

    expand_stats, expand_bw, expand_sample = benchmark_expand(args, block_bytes)
    shrink_stats, shrink_bw, shrink_sample = benchmark_shrink(args, block_bytes)
    reload_stats, h2d_bw, reload_sample = benchmark_transfer(
        args, "cpu_to_gpu")
    offload_stats, d2h_bw, offload_sample = benchmark_transfer(
        args, "gpu_to_cpu")
    concurrent_reload_stats, concurrent_h2d_bw, concurrent_reload_sample = (
        benchmark_concurrent_transfer(args, "cpu_to_gpu"))
    concurrent_offload_stats, concurrent_d2h_bw, concurrent_offload_sample = (
        benchmark_concurrent_transfer(args, "gpu_to_cpu"))

    result["bandwidth_source"] = "sample_benchmark"
    result["sample"] = {
        "expand": expand_sample,
        "shrink": shrink_sample,
        "reload_cpu_to_gpu": reload_sample,
        "offload_gpu_to_cpu": offload_sample,
        "concurrent_reload_cpu_to_gpu": concurrent_reload_sample,
        "concurrent_offload_gpu_to_cpu": concurrent_offload_sample,
    }
    result["sample_stats"] = {
        "expand": asdict(expand_stats),
        "shrink_migration": asdict(shrink_stats),
        "draft_reload_cpu_to_gpu": asdict(reload_stats),
        "draft_offload_gpu_to_cpu": asdict(offload_stats),
        "concurrent_draft_reload_group": asdict(concurrent_reload_stats),
        "concurrent_draft_offload_group": asdict(concurrent_offload_stats),
    }
    result["bandwidth_gib_s"] = {
        "expand_effective": expand_bw,
        "shrink_migration_effective": shrink_bw,
        "cpu_to_gpu": h2d_bw,
        "gpu_to_cpu": d2h_bw,
        "concurrent_cpu_to_gpu": concurrent_h2d_bw,
        "concurrent_gpu_to_cpu": concurrent_d2h_bw,
    }
    result["estimated_ms"] = {
        "expand": ms_from_bandwidth(expand_traffic, expand_bw),
        "expand_pad": ms_from_bandwidth(expand_pad_traffic, expand_bw),
        "expand_append": ms_from_bandwidth(expand_append_traffic, expand_bw),
        "shrink_migration": ms_from_bandwidth(shrink_traffic, shrink_bw),
        "shrink_full_trim_copy": ms_from_bandwidth(trim_copy_traffic,
                                                   shrink_bw),
        "draft_reload_cpu_to_gpu": ms_from_bandwidth(args.draft_model_bytes,
                                                     h2d_bw),
        "draft_offload_gpu_to_cpu": ms_from_bandwidth(args.draft_model_bytes,
                                                      d2h_bw),
        "concurrent_draft_reload_group": ms_from_bandwidth(
            args.draft_model_bytes * args.transfer_concurrency,
            concurrent_h2d_bw),
        "concurrent_draft_offload_group": ms_from_bandwidth(
            args.draft_model_bytes * args.transfer_concurrency,
            concurrent_d2h_bw),
    }
    return result


def apply_model_config(args: argparse.Namespace) -> None:
    if args.model_config is None:
        return
    with args.model_config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    args.num_layers = int(config.get("num_hidden_layers", args.num_layers))
    args.num_kv_heads = int(
        config.get("num_key_value_heads", args.num_kv_heads))
    hidden_size = int(config.get("hidden_size", 0))
    num_heads = int(config.get("num_attention_heads", 0))
    if hidden_size > 0 and num_heads > 0:
        args.head_size = hidden_size // num_heads
    torch_dtype = str(config.get("torch_dtype", "")).lower()
    if torch_dtype in ("float16", "bfloat16", "float32"):
        args.dtype = torch_dtype


def parse_size_to_bytes(value: str) -> int:
    text = value.strip().lower()
    units = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
    }
    for suffix, scale in sorted(units.items(), key=lambda x: -len(x[0])):
        if text.endswith(suffix):
            return int(float(text[:-len(suffix)]) * scale)
    return int(float(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate elastic KV-cache resize and draft-model transfer.")
    parser.add_argument("--current-blocks", type=int, required=True)
    parser.add_argument("--change-blocks", type=int, required=True)
    parser.add_argument("--migrated-blocks", type=int, default=None,
                        help="Blocks that must move during shrink. Default: "
                        "change-blocks.")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype", choices=sorted(DTYPE_SIZES), default="float16")
    parser.add_argument("--model-config", type=Path, default=None,
                        help="Optional HF config JSON. Overrides num-layers, "
                        "num-kv-heads, head-size, and dtype.")
    parser.add_argument("--draft-model-size", default="1GiB",
                        help="Draft model weight size, e.g. 512MiB, 1.2GiB.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--sample-current-blocks", type=int, default=256)
    parser.add_argument("--sample-layers", type=int, default=4)
    parser.add_argument("--sample-transfer-mb", type=int, default=256)
    parser.add_argument("--expand-backend", choices=("pad", "append"),
                        default="pad",
                        help="Backend used for KV expansion. 'pad' mirrors "
                        "contiguous resize via F.pad; 'append' models "
                        "append-only block growth after memory is freed.")
    parser.add_argument("--append-init", choices=("zero", "touch"),
                        default="zero",
                        help="Initialization behavior for append-based "
                        "growth in benchmark mode.")
    parser.add_argument("--shrink-backend", choices=("triton", "torch"),
                        default="triton",
                        help="Backend used for live KV-block migration in "
                        "benchmark mode. Triton mirrors the production "
                        "CacheEngine migration path.")
    parser.add_argument("--triton-block-size", type=int, default=128,
                        help="Vector width used by the Triton migration "
                        "kernel.")
    parser.add_argument("--transfer-concurrency", type=int, default=1,
                        help="Number of simultaneous draft offload/reload "
                        "transfers sharing the host-GPU path.")
    parser.add_argument("--no-pinned-cpu", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--gpu-mem-bw-gib-s", type=float, default=700.0)
    parser.add_argument("--h2d-bw-gib-s", type=float, default=23.0)
    parser.add_argument("--d2h-bw-gib-s", type=float, default=23.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    apply_model_config(args)
    if args.current_blocks <= 0:
        raise SystemExit("--current-blocks must be positive.")
    if args.change_blocks <= 0:
        raise SystemExit("--change-blocks must be positive.")
    if args.migrated_blocks is not None and args.migrated_blocks < 0:
        raise SystemExit("--migrated-blocks must be non-negative.")
    if args.transfer_concurrency <= 0:
        raise SystemExit("--transfer-concurrency must be positive.")
    args.draft_model_bytes = parse_size_to_bytes(args.draft_model_size)
    return args


def print_human(result: dict[str, Any]) -> None:
    inp = result["input"]
    drv = result["derived"]
    est = result["estimated_ms"]
    bw = result["bandwidth_gib_s"]
    print("Elastic KV / draft-transfer simulation")
    print(f"  current_blocks={inp['current_blocks']} "
          f"change_blocks={inp['change_blocks']} "
          f"migrated_blocks={inp['migrated_blocks_for_shrink']}")
    print(f"  KV block/layer={drv['kv_block_mib_per_layer']:.3f} MiB, "
          f"full-model KV block="
          f"{drv['kv_block_mib_per_layer'] * inp['num_layers']:.3f} MiB, "
          f"layers={inp['num_layers']}, dtype={inp['dtype']}")
    if inp["model_config"]:
        print(f"  model_config={inp['model_config']}")
    print(f"  bandwidth_source={result['bandwidth_source']}")
    print("")
    print("Estimated time")
    if inp["expand_backend"] == "append":
        print(f"  expand(append-only):        {est['expand']:.3f} ms")
        print(f"  expand(F.pad reference):    {est['expand_pad']:.3f} ms")
    else:
        print(f"  expand(F.pad):              {est['expand']:.3f} ms")
        print(f"  expand(append-only ref):    {est['expand_append']:.3f} ms")
    print(f"  shrink migration:           {est['shrink_migration']:.3f} ms")
    print(f"  shrink full trim copy:      {est['shrink_full_trim_copy']:.3f} ms")
    print(f"  draft reload CPU -> GPU:    "
          f"{est['draft_reload_cpu_to_gpu']:.3f} ms")
    print(f"  draft offload GPU -> CPU:   "
          f"{est['draft_offload_gpu_to_cpu']:.3f} ms")
    print(f"  concurrent reload group "
          f"(N={inp['transfer_concurrency']}): "
          f"{est['concurrent_draft_reload_group']:.3f} ms")
    print(f"  concurrent offload group "
          f"(N={inp['transfer_concurrency']}): "
          f"{est['concurrent_draft_offload_group']:.3f} ms")
    print("")
    print("Effective bandwidth")
    for key, value in bw.items():
        print(f"  {key}: {value:.2f} GiB/s")


def main() -> None:
    args = parse_args()
    result = build_estimate(args)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
