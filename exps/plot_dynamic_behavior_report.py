#!/usr/bin/env python3
"""Generate Nightjar dynamic-behavior figures and summaries from JSONL logs.

This script unifies three analyses for reviewer responses:
1. Acceptance-rate distribution and acceptance/gamma traces.
2. Draft / verify / autoregressive (AR) time breakdown.
3. CPU-GPU migration overhead summary and trigger timeline.

Recommended usage with a manifest JSON file:

[
  {
    "path": "benchmark_results/run_a/nightjar_events.jsonl",
    "label": "ShareGPT low load",
    "dataset": "ShareGPT",
    "load": "2 req/s"
  },
  {
    "path": "benchmark_results/run_b/nightjar_events.jsonl",
    "label": "Alpaca high load",
    "dataset": "Alpaca",
    "load": "8 req/s"
  }
]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SPEC_EVENT = "speculative_step"
EXPAND_EVENT = "memory_expand"
CONTRACT_EVENT = "memory_contract"
MIGRATION_EVENT = "kv_block_migration"


@dataclass
class RunSpec:
    path: Path
    label: str
    dataset: str
    load: str


@dataclass
class RunData:
    spec: RunSpec
    speculative_steps: list[dict[str, Any]]
    expand_events: list[dict[str, Any]]
    contract_events: list[dict[str, Any]]
    migration_events: list[dict[str, Any]]
    all_events: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Nightjar acceptance, timing, and migration plots.")
    parser.add_argument(
        "event_logs",
        nargs="*",
        type=Path,
        help="JSONL event logs. Use together with --manifest or standalone.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON manifest describing runs: path, label, dataset, load.")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("figs/dynamic_behavior"),
                        help="Directory for plots and summaries.")
    parser.add_argument("--prefix",
                        default="nightjar",
                        help="Filename prefix for generated artifacts.")
    parser.add_argument("--smooth-window",
                        type=int,
                        default=25,
                        help="Moving-average window for traces.")
    parser.add_argument(
        "--max-trace-points",
        type=int,
        default=300,
        help="Maximum plotted points per trace after downsampling.")
    parser.add_argument(
        "--only-gamma-trace",
        action="store_true",
        help="Only generate the gamma trace PDF and skip all other outputs.")
    parser.add_argument(
        "--zero-gamma-scale",
        type=float,
        default=0.08,
        help="Relative horizontal scale for gamma=0 steps. Smaller values compress disabled-speculation regions.")
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return values[:]
    averaged: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        averaged.append(running_sum / min(index + 1, window))
    return averaged


def downsample_pairs(xs: list[float], ys: list[float],
                     max_points: int) -> tuple[list[float], list[float]]:
    if len(xs) <= max_points or max_points <= 0:
        return xs, ys
    stride = max(1, math.ceil(len(xs) / max_points))
    return xs[::stride], ys[::stride]


def compressed_step_axis(gammas: list[float], zero_gamma_scale: float) -> list[float]:
    if not gammas:
        return []
    scale = max(0.0, zero_gamma_scale)
    xs: list[float] = []
    current = 0.0
    for gamma in gammas:
        xs.append(current)
        current += 1.0 if gamma > 0 else scale
    return xs


def mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pct(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole > 0 else 0.0


def load_manifest(path: Path) -> list[RunSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a list of run objects.")
    specs: list[RunSpec] = []
    for item in payload:
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError("Each manifest entry must contain at least 'path'.")
        run_path = Path(str(item["path"])).expanduser()
        label = str(item.get("label") or run_path.stem)
        dataset = str(item.get("dataset") or "unknown")
        load = str(item.get("load") or "unknown")
        specs.append(
            RunSpec(path=run_path, label=label, dataset=dataset, load=load))
    return specs


def build_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: list[RunSpec] = []
    if args.manifest:
        specs.extend(load_manifest(args.manifest))
    for path in args.event_logs:
        specs.append(
            RunSpec(path=path.expanduser(),
                    label=path.stem,
                    dataset="unknown",
                    load="unknown"))
    if not specs:
        raise SystemExit("Provide at least one event log or a manifest.")
    return specs


def load_run_data(spec: RunSpec) -> RunData:
    speculative_steps: list[dict[str, Any]] = []
    expand_events: list[dict[str, Any]] = []
    contract_events: list[dict[str, Any]] = []
    migration_events: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    with spec.path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            all_events.append(record)
            event_type = record.get("event_type")
            if event_type == SPEC_EVENT:
                speculative_steps.append(record)
            elif event_type == EXPAND_EVENT:
                expand_events.append(record)
            elif event_type == CONTRACT_EVENT:
                contract_events.append(record)
            elif event_type == MIGRATION_EVENT:
                migration_events.append(record)
    return RunData(spec=spec,
                   speculative_steps=speculative_steps,
                   expand_events=expand_events,
                   contract_events=contract_events,
                   migration_events=migration_events,
                   all_events=all_events)


def compute_run_summary(run: RunData) -> dict[str, Any]:
    steps = run.speculative_steps
    acceptances = [
        float(step.get("acceptance_rate", 0.0)) for step in steps
        if int(step.get("proposal_length_gamma", 0)) > 0
    ]
    gammas = [int(step.get("proposal_length_gamma", 0)) for step in steps]
    speculative_gammas = [gamma for gamma in gammas if gamma > 0]
    speculative_steps = [
        step for step in steps if int(step.get("proposal_length_gamma", 0)) > 0
    ]
    ar_steps = [step for step in steps if int(step.get("proposal_length_gamma", 0)) == 0]

    draft_ms = sum(float(step.get("draft_time_ms", 0.0))
                   for step in speculative_steps)
    verify_ms = sum(
        float(step.get("scoring_time_ms", 0.0)) +
        float(step.get("verify_time_ms", 0.0)) for step in speculative_steps)
    ar_ms = sum(float(step.get("step_total_time_ms", 0.0)) for step in ar_steps)
    total_pipeline_ms = draft_ms + verify_ms + ar_ms

    expand_ms = sum(float(event.get("duration_ms", 0.0))
                    for event in run.expand_events)
    contract_ms = sum(float(event.get("duration_ms", 0.0))
                      for event in run.contract_events)
    migration_ms = sum(float(event.get("duration_ms", 0.0))
                       for event in run.migration_events)
    total_migration_ms = expand_ms + contract_ms + migration_ms

    timestamps = [
        float(event["timestamp"]) for event in run.all_events
        if "timestamp" in event
    ]
    wallclock_ms = ((max(timestamps) - min(timestamps)) * 1000.0
                    if len(timestamps) >= 2 else 0.0)

    return {
        "label": run.spec.label,
        "dataset": run.spec.dataset,
        "load": run.spec.load,
        "path": str(run.spec.path),
        "num_speculative_steps": len(speculative_steps),
        "num_ar_steps": len(ar_steps),
        "mean_acceptance_rate": mean_or_zero(acceptances),
        "median_acceptance_rate": statistics.median(acceptances)
        if acceptances else 0.0,
        "std_acceptance_rate": statistics.pstdev(acceptances)
        if len(acceptances) > 1 else 0.0,
        "mean_gamma_all_steps": mean_or_zero([float(g) for g in gammas]),
        "mean_gamma_when_enabled": mean_or_zero(
            [float(g) for g in speculative_gammas]),
        "max_gamma": max(gammas) if gammas else 0,
        "draft_time_ms": draft_ms,
        "verify_time_ms": verify_ms,
        "ar_time_ms": ar_ms,
        "total_pipeline_time_ms": total_pipeline_ms,
        "draft_share_pct": pct(draft_ms, total_pipeline_ms),
        "verify_share_pct": pct(verify_ms, total_pipeline_ms),
        "ar_share_pct": pct(ar_ms, total_pipeline_ms),
        "offload_count": len(run.expand_events),
        "reload_count": len(run.contract_events),
        "migration_count": len(run.migration_events),
        "avg_offload_ms": mean_or_zero(
            [float(event.get("duration_ms", 0.0)) for event in run.expand_events]),
        "avg_reload_ms": mean_or_zero([
            float(event.get("duration_ms", 0.0))
            for event in run.contract_events
        ]),
        "avg_migration_ms": mean_or_zero([
            float(event.get("duration_ms", 0.0))
            for event in run.migration_events
        ]),
        "migration_total_ms": total_migration_ms,
        "migration_share_of_pipeline_pct": pct(total_migration_ms,
                                               total_pipeline_ms),
        "migration_share_of_wallclock_pct": pct(total_migration_ms,
                                                wallclock_ms),
        "wallclock_span_ms": wallclock_ms,
    }


def save_summary(runs: list[RunData], output_dir: Path,
                 prefix: str) -> list[dict[str, Any]]:
    summaries = [compute_run_summary(run) for run in runs]
    json_path = output_dir / f"{prefix}_dynamic_behavior_summary.json"
    csv_path = output_dir / f"{prefix}_dynamic_behavior_summary.csv"
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    fieldnames = list(summaries[0].keys()) if summaries else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def format_run_title(run: RunData) -> str:
    parts = [run.spec.label]
    if run.spec.dataset != "unknown":
        parts.append(run.spec.dataset)
    if run.spec.load != "unknown":
        parts.append(run.spec.load)
    return " | ".join(parts)


def plot_acceptance_distribution(runs: list[RunData], output_path: Path) -> None:
    relevant = [run for run in runs if run.speculative_steps]
    if not relevant:
        return
    fig, axes = plt.subplots(len(relevant),
                             1,
                             figsize=(10, max(3.2, 2.8 * len(relevant))),
                             squeeze=False)
    bins = [index / 20.0 for index in range(21)]
    for axis, run in zip(axes[:, 0], relevant):
        values = [
            float(step.get("acceptance_rate", 0.0))
            for step in run.speculative_steps
            if int(step.get("proposal_length_gamma", 0)) > 0
        ]
        axis.hist(values,
                  bins=bins,
                  color="#4c72b0",
                  alpha=0.8,
                  edgecolor="white")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylabel("Count")
        axis.grid(alpha=0.25)
        axis.set_title(format_run_title(run), fontsize=10)
        axis.axvline(mean_or_zero(values),
                     linestyle="--",
                     linewidth=1.2,
                     color="#dd8452")
    axes[-1, 0].set_xlabel("Acceptance Rate")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_acceptance_gamma_traces(runs: list[RunData], output_path: Path,
                                 smooth_window: int,
                                 max_trace_points: int,
                                 zero_gamma_scale: float) -> None:
    relevant = [run for run in runs if run.speculative_steps]
    if not relevant:
        return
    fig, axes = plt.subplots(len(relevant),
                             1,
                             figsize=(12, max(3.4, 3.0 * len(relevant))),
                             squeeze=False)
    for axis, run in zip(axes[:, 0], relevant):
        acceptance = [
            float(step.get("acceptance_rate", 0.0))
            for step in run.speculative_steps
        ]
        gamma = [
            float(step.get("proposal_length_gamma", 0))
            for step in run.speculative_steps
        ]
        steps = compressed_step_axis(gamma, zero_gamma_scale)
        acceptance_smooth = moving_average(acceptance, smooth_window)
        x_acc, y_acc = downsample_pairs(steps, acceptance_smooth,
                                        max_trace_points)
        x_gamma, y_gamma = downsample_pairs(steps, gamma, max_trace_points)

        axis.plot(x_acc, y_acc, color="#4c72b0", linewidth=2, label="Acceptance")
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Acceptance", color="#4c72b0")
        axis.tick_params(axis="y", labelcolor="#4c72b0")
        axis.grid(alpha=0.25)
        axis.set_title(format_run_title(run), fontsize=10)

        gamma_axis = axis.twinx()
        gamma_axis.step(x_gamma,
                        y_gamma,
                        where="post",
                        color="#dd8452",
                        alpha=0.85,
                        label="Gamma")
        gamma_axis.set_ylabel("Gamma", color="#dd8452")
        gamma_axis.tick_params(axis="y", labelcolor="#dd8452")
    axes[-1, 0].set_xlabel("Compressed Decode Step")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_traces(runs: list[RunData], output_path: Path,
                      max_trace_points: int,
                      zero_gamma_scale: float) -> None:
    relevant = [run for run in runs if run.speculative_steps]
    if not relevant:
        return
    fig, axes = plt.subplots(len(relevant),
                             1,
                             figsize=(12, max(3.0, 2.8 * len(relevant))),
                             squeeze=False)
    for axis, run in zip(axes[:, 0], relevant):
        gamma = [
            float(step.get("proposal_length_gamma", 0))
            for step in run.speculative_steps
        ]
        steps = compressed_step_axis(gamma, zero_gamma_scale)
        x_gamma, y_gamma = downsample_pairs(steps, gamma, max_trace_points)
        axis.step(x_gamma,
                  y_gamma,
                  where="post",
                  color="#dd8452",
                  linewidth=1.8)
        axis.set_ylabel("Gamma")
        axis.grid(alpha=0.25)
        axis.set_title(format_run_title(run), fontsize=10)
    axes[-1, 0].set_xlabel("Compressed Decode Step")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_gamma_traces(runs: list[RunData], output_path: Path,
                                 smooth_window: int,
                                 max_trace_points: int,
                                 zero_gamma_scale: float) -> None:
    relevant = [run for run in runs if run.speculative_steps]
    if not relevant:
        return
    fig, axes = plt.subplots(len(relevant),
                             1,
                             figsize=(12, max(3.4, 3.0 * len(relevant))),
                             squeeze=False)
    for axis, run in zip(axes[:, 0], relevant):
        filtered_steps = [
            step for step in run.speculative_steps
            if float(step.get("proposal_length_gamma", 0)) > 0
        ]
        if not filtered_steps:
            continue
        gamma = [
            float(step.get("proposal_length_gamma", 0))
            for step in filtered_steps
        ]
        steps = compressed_step_axis(gamma, zero_gamma_scale)
        throughput = []
        for step in filtered_steps:
            batch_size = float(step.get("batch_size", 0.0))
            num_accepted_tokens = float(step.get("num_accepted_tokens", 0.0))
            step_total_time_ms = float(step.get("step_total_time_ms", 0.0))
            step_total_time_s = step_total_time_ms / 1000.0
            if step_total_time_s > 0:
                throughput.append((batch_size + num_accepted_tokens) /
                                  step_total_time_s)
            else:
                throughput.append(0.0)
        throughput_smooth = moving_average(throughput, smooth_window)
        x_tp, y_tp = downsample_pairs(steps, throughput_smooth,
                                      max_trace_points)
        x_gamma, y_gamma = downsample_pairs(steps, gamma, max_trace_points)

        axis.plot(x_tp,
                  y_tp,
                  color="#4c72b0",
                  linewidth=2,
                  label="Throughput")
        axis.set_ylabel("Throughput (tokens/s)", color="#4c72b0")
        axis.tick_params(axis="y", labelcolor="#4c72b0")
        axis.grid(alpha=0.25)
        axis.set_title(format_run_title(run), fontsize=10)

        gamma_axis = axis.twinx()
        gamma_axis.step(x_gamma,
                        y_gamma,
                        where="post",
                        color="#dd8452",
                        alpha=0.85,
                        label="Gamma")
        gamma_axis.set_ylabel("Gamma", color="#dd8452")
        gamma_axis.tick_params(axis="y", labelcolor="#dd8452")
    axes[-1, 0].set_xlabel("Compressed Decode Step")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_time_breakdown_bars(summaries: list[dict[str, Any]],
                             output_path: Path) -> None:
    if not summaries:
        return
    labels = [summary["label"] for summary in summaries]
    draft = [summary["draft_time_ms"] for summary in summaries]
    verify = [summary["verify_time_ms"] for summary in summaries]
    ar = [summary["ar_time_ms"] for summary in summaries]
    totals = [summary["total_pipeline_time_ms"] for summary in summaries]
    draft_pct = [pct(d, t) for d, t in zip(draft, totals)]
    verify_pct = [pct(v, t) for v, t in zip(verify, totals)]
    ar_pct = [pct(a, t) for a, t in zip(ar, totals)]

    xs = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    axes[0].bar(xs, draft, color="#4c72b0", label="Draft")
    axes[0].bar(xs, verify, bottom=draft, color="#55a868", label="Verify")
    axes[0].bar(xs,
                ar,
                bottom=[d + v for d, v in zip(draft, verify)],
                color="#c44e52",
                label="AR")
    axes[0].set_ylabel("Total Time (ms)")
    axes[0].set_title("Absolute Breakdown")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(xs, draft_pct, color="#4c72b0", label="Draft")
    axes[1].bar(xs, verify_pct, bottom=draft_pct, color="#55a868", label="Verify")
    axes[1].bar(xs,
                ar_pct,
                bottom=[d + v for d, v in zip(draft_pct, verify_pct)],
                color="#c44e52",
                label="AR")
    axes[1].set_ylabel("Share (%)")
    axes[1].set_title("Relative Breakdown")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_time_breakdown_traces(runs: list[RunData], output_path: Path,
                               smooth_window: int,
                               max_trace_points: int) -> None:
    relevant = [run for run in runs if run.speculative_steps]
    if not relevant:
        return
    fig, axes = plt.subplots(len(relevant),
                             1,
                             figsize=(12, max(3.4, 3.0 * len(relevant))),
                             squeeze=False)
    for axis, run in zip(axes[:, 0], relevant):
        steps = list(range(len(run.speculative_steps)))
        draft = []
        verify = []
        ar = []
        for step in run.speculative_steps:
            gamma = int(step.get("proposal_length_gamma", 0))
            draft_ms = float(step.get("draft_time_ms", 0.0)) if gamma > 0 else 0.0
            verify_ms = (
                float(step.get("scoring_time_ms", 0.0)) +
                float(step.get("verify_time_ms", 0.0))) if gamma > 0 else 0.0
            ar_ms = float(step.get("step_total_time_ms", 0.0)) if gamma == 0 else 0.0
            draft.append(draft_ms)
            verify.append(verify_ms)
            ar.append(ar_ms)

        draft_s = moving_average(draft, smooth_window)
        verify_s = moving_average(verify, smooth_window)
        ar_s = moving_average(ar, smooth_window)

        x_d, y_d = downsample_pairs(steps, draft_s, max_trace_points)
        x_v, y_v = downsample_pairs(steps, verify_s, max_trace_points)
        x_a, y_a = downsample_pairs(steps, ar_s, max_trace_points)

        axis.plot(x_d, y_d, color="#4c72b0", label="Draft")
        axis.plot(x_v, y_v, color="#55a868", label="Verify")
        axis.plot(x_a, y_a, color="#c44e52", label="AR")
        axis.set_ylabel("Time (ms)")
        axis.set_title(format_run_title(run), fontsize=10)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=3, fontsize=9)
    axes[-1, 0].set_xlabel("Decode Step")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def relative_trigger_times(events: list[dict[str, Any]],
                           origin: float) -> list[float]:
    points: list[float] = []
    for event in events:
        if "timestamp" in event:
            points.append((float(event["timestamp"]) - origin) * 1000.0)
    return points


def plot_migration_overhead(runs: list[RunData], summaries: list[dict[str, Any]],
                            output_path: Path) -> None:
    if not runs:
        return
    labels = [summary["label"] for summary in summaries]
    xs = list(range(len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    offloads = [summary["offload_count"] for summary in summaries]
    reloads = [summary["reload_count"] for summary in summaries]
    migrations = [summary["migration_count"] for summary in summaries]
    axes[0, 0].bar(xs, offloads, color="#4c72b0", label="Offload")
    axes[0, 0].bar(xs,
                   reloads,
                   bottom=offloads,
                   color="#55a868",
                   label="Reload")
    axes[0, 0].bar(xs,
                   migrations,
                   bottom=[o + r for o, r in zip(offloads, reloads)],
                   color="#c44e52",
                   label="KV Migration")
    axes[0, 0].set_title("Event Counts")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend()

    avg_offload_ms = [summary["avg_offload_ms"] for summary in summaries]
    avg_reload_ms = [summary["avg_reload_ms"] for summary in summaries]
    avg_migration_ms = [summary["avg_migration_ms"] for summary in summaries]
    axes[0, 1].bar(xs, avg_offload_ms, color="#4c72b0", label="Offload")
    axes[0, 1].bar(xs,
                   avg_reload_ms,
                   bottom=avg_offload_ms,
                   color="#55a868",
                   label="Reload")
    axes[0, 1].bar(xs,
                   avg_migration_ms,
                   bottom=[o + r for o, r in zip(avg_offload_ms, avg_reload_ms)],
                   color="#c44e52",
                   label="KV Migration")
    axes[0, 1].set_title("Average Event Duration")
    axes[0, 1].set_ylabel("Time (ms)")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].set_title("Trigger Timeline")
    axes[1, 0].set_xlabel("Relative Time (ms)")
    axes[1, 0].set_ylabel("Run")
    axes[1, 0].grid(alpha=0.25)

    for index, run in enumerate(runs):
        timestamps = [
            float(event["timestamp"]) for event in run.all_events
            if "timestamp" in event
        ]
        if not timestamps:
            continue
        origin = min(timestamps)
        offload_points = relative_trigger_times(run.expand_events, origin)
        reload_points = relative_trigger_times(run.contract_events, origin)
        migration_points = relative_trigger_times(run.migration_events, origin)
        if offload_points:
            axes[1, 0].scatter(offload_points, [index] * len(offload_points),
                               color="#4c72b0", marker="v", label="Offload" if index == 0 else "")
        if reload_points:
            axes[1, 0].scatter(reload_points, [index] * len(reload_points),
                               color="#55a868", marker="^", label="Reload" if index == 0 else "")
        if migration_points:
            axes[1, 0].scatter(migration_points, [index] * len(migration_points),
                               color="#c44e52", marker="o", label="KV Migration" if index == 0 else "")
    axes[1, 0].set_yticks(xs)
    axes[1, 0].set_yticklabels(labels)
    axes[1, 0].legend(loc="upper right")

    share_pipeline = [
        summary["migration_share_of_pipeline_pct"] for summary in summaries
    ]
    share_wallclock = [
        summary["migration_share_of_wallclock_pct"] for summary in summaries
    ]
    width = 0.36
    axes[1, 1].bar([x - width / 2 for x in xs],
                   share_pipeline,
                   width=width,
                   color="#8172b3",
                   label="Share of pipeline")
    axes[1, 1].bar([x + width / 2 for x in xs],
                   share_wallclock,
                   width=width,
                   color="#937860",
                   label="Share of wallclock")
    axes[1, 1].set_title("Cumulative Migration Time Share")
    axes[1, 1].set_ylabel("Share (%)")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend()
    axes[1, 1].set_xticks(xs)
    axes[1, 1].set_xticklabels(labels, rotation=20, ha="right")

    for axis in (axes[0, 0], axes[0, 1]):
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_specs = build_run_specs(args)
    runs = [load_run_data(spec) for spec in run_specs]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.only_gamma_trace:
        plot_gamma_traces(runs,
                          args.output_dir / f"{args.prefix}_gamma_traces.pdf",
                          args.max_trace_points,
                          args.zero_gamma_scale)
        return

    summaries = save_summary(runs, args.output_dir, args.prefix)

    plot_acceptance_distribution(
        runs, args.output_dir / f"{args.prefix}_acceptance_distribution.pdf")
    plot_acceptance_gamma_traces(
        runs,
        args.output_dir / f"{args.prefix}_acceptance_gamma_traces.pdf",
        args.smooth_window,
        args.max_trace_points,
        args.zero_gamma_scale,
    )
    plot_throughput_gamma_traces(
        runs,
        args.output_dir / f"{args.prefix}_throughput_gamma_traces.pdf",
        args.smooth_window,
        args.max_trace_points,
        args.zero_gamma_scale,
    )
    plot_time_breakdown_bars(
        summaries, args.output_dir / f"{args.prefix}_time_breakdown_bar.pdf")
    plot_time_breakdown_traces(
        runs,
        args.output_dir / f"{args.prefix}_time_breakdown_trace.pdf",
        args.smooth_window,
        args.max_trace_points,
    )
    plot_migration_overhead(
        runs, summaries,
        args.output_dir / f"{args.prefix}_migration_overhead.pdf")


if __name__ == "__main__":
    main()
