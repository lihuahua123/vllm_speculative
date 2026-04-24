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
DISABLED_DECODE_EVENT = "disabled_decode_step"


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
    disabled_decode_events: list[dict[str, Any]]
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


def step_axis(values: list[Any]) -> list[int]:
    return list(range(len(values)))


def zero_value_indices(values: list[float]) -> list[int]:
    return [index for index, value in enumerate(values) if value <= 0]


def relative_time_axis(timestamps: list[float], origin: float) -> list[float]:
    return [timestamp - origin for timestamp in timestamps]


def event_time_origin(run: RunData) -> float:
    timestamps = [
        float(event["timestamp"]) for event in run.all_events
        if "timestamp" in event
    ]
    return min(timestamps) if timestamps else 0.0


def disabled_time_intervals(run: RunData, origin: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    active_start: float | None = None
    for event in sorted(run.all_events, key=lambda item: float(item.get("timestamp", 0.0))):
        if event.get("event_type") != "memory_policy_decision":
            continue
        action = event.get("action")
        timestamp = float(event.get("timestamp", 0.0))
        if action == "expand_kv_cache":
            active_start = timestamp
        elif action == "restore_draft_model" and active_start is not None:
            intervals.append((active_start - origin, timestamp - origin))
            active_start = None
    if active_start is not None:
        timestamps = [
            float(event["timestamp"]) for event in run.all_events
            if "timestamp" in event
        ]
        end = max(timestamps) if timestamps else active_start
        intervals.append((active_start - origin, end - origin))
    return intervals


def shade_disabled_intervals(axis: Any, intervals: list[tuple[float, float]]) -> None:
    for start, end in intervals:
        if end < start:
            continue
        axis.axvspan(start, end,
                     color="#d8d8d8",
                     alpha=0.32,
                     linewidth=0)


def force_zero_in_intervals(xs: list[float], ys: list[float],
                            intervals: list[tuple[float, float]]
                            ) -> tuple[list[float], list[float]]:
    if not xs or not ys or not intervals:
        return xs, ys
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    output: list[tuple[float, float]] = []
    cursor = 0
    for start, end in sorted(intervals):
        if end < start:
            continue
        while cursor < len(pairs) and pairs[cursor][0] < start:
            output.append(pairs[cursor])
            cursor += 1
        previous_y = output[-1][1] if output else pairs[0][1]
        output.append((start, previous_y))
        output.append((start, 0.0))
        while cursor < len(pairs) and pairs[cursor][0] <= end:
            output.append((pairs[cursor][0], 0.0))
            cursor += 1
        output.append((end, 0.0))
        if cursor < len(pairs):
            output.append((end, pairs[cursor][1]))
    output.extend(pairs[cursor:])
    return [item[0] for item in output], [item[1] for item in output]


def mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pct(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole > 0 else 0.0


def speculation_enabled(step: dict[str, Any]) -> bool:
    value = step.get("speculation_enabled")
    if isinstance(value, bool):
        return value
    return int(step.get("proposal_length_gamma", 0)) > 0


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
    disabled_decode_events: list[dict[str, Any]] = []
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
            elif event_type == DISABLED_DECODE_EVENT:
                disabled_decode_events.append(record)
    return RunData(spec=spec,
                   speculative_steps=speculative_steps,
                   expand_events=expand_events,
                   contract_events=contract_events,
                   migration_events=migration_events,
                   disabled_decode_events=disabled_decode_events,
                   all_events=all_events)


def compute_run_summary(run: RunData) -> dict[str, Any]:
    steps = run.speculative_steps
    acceptances = [float(step.get("acceptance_rate", 0.0)) for step in steps]
    gammas = [int(step.get("proposal_length_gamma", 0)) for step in steps]
    enabled_steps = [step for step in steps if speculation_enabled(step)]
    speculative_gammas = [
        int(step.get("proposal_length_gamma", 0)) for step in enabled_steps
    ]
    ar_steps = [step for step in steps if not speculation_enabled(step)]

    draft_ms = sum(float(step.get("draft_time_ms", 0.0))
                   for step in enabled_steps)
    verify_ms = sum(
        float(step.get("scoring_time_ms", 0.0)) +
        float(step.get("verify_time_ms", 0.0)) for step in enabled_steps)
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
        "num_speculative_steps": len(enabled_steps),
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


def is_medium_load(run: RunData) -> bool:
    load = run.spec.load.strip().lower()
    label = run.spec.label.strip().lower()
    return load.startswith("med") or label.endswith("_med")


def medium_load_runs(runs: list[RunData]) -> list[RunData]:
    dataset_order = {"sharegpt": 0, "alpaca": 1, "specbench": 2}
    relevant = [
        run for run in runs if run.speculative_steps and is_medium_load(run)
    ]
    relevant.sort(
        key=lambda run: (
            dataset_order.get(run.spec.dataset.strip().lower(), 99),
            run.spec.label,
        ))
    return relevant


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
            if speculation_enabled(step)
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
    relevant = medium_load_runs(runs)
    if not relevant:
        return
    fig, axes = plt.subplots(len(relevant),
                             1,
                             figsize=(12, max(3.4, 3.0 * len(relevant))),
                             squeeze=False)
    for axis, run in zip(axes[:, 0], relevant):
        acceptance_steps = [
            step for step in run.speculative_steps if speculation_enabled(step)
        ]
        acceptance = [
            float(step.get("acceptance_rate", 0.0)) for step in acceptance_steps
        ]
        gamma = [
            float(step.get("proposal_length_gamma", 0))
            for step in run.speculative_steps
        ]
        timestamps = [
            float(step.get("timestamp", 0.0)) for step in run.speculative_steps
        ]
        acceptance_timestamps = [
            float(step.get("timestamp", 0.0)) for step in acceptance_steps
        ]
        origin = event_time_origin(run)
        steps = relative_time_axis(timestamps, origin)
        acceptance_steps_x = relative_time_axis(acceptance_timestamps, origin)
        intervals = disabled_time_intervals(run, origin)
        acceptance_smooth = moving_average(acceptance, smooth_window)
        x_acc, y_acc = downsample_pairs(acceptance_steps_x, acceptance_smooth,
                                        max_trace_points)
        x_acc, y_acc = force_zero_in_intervals(x_acc, y_acc, intervals)
        x_gamma, y_gamma = force_zero_in_intervals(steps, gamma, intervals)
        zero_indices = zero_value_indices(gamma)

        shade_disabled_intervals(axis, intervals)
        axis.plot(x_acc, y_acc, color="#4c72b0", linewidth=2, label="Acceptance")
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Acceptance", color="#4c72b0")
        axis.tick_params(axis="y", labelcolor="#4c72b0")
        axis.grid(alpha=0.25)
        axis.set_title(format_run_title(run), fontsize=10)

        gamma_axis = axis.twinx()
        shade_disabled_intervals(gamma_axis, intervals)
        gamma_axis.step(x_gamma,
                        y_gamma,
                        where="post",
                        color="#dd8452",
                        alpha=0.85,
                        label="Gamma")
        if zero_indices:
            gamma_axis.scatter([steps[index] for index in zero_indices],
                               [0.0] * len(zero_indices),
                               color="#c44e52",
                               edgecolors="white",
                               linewidths=0.8,
                               s=36,
                               zorder=5,
                               label="Gamma = 0")
        gamma_axis.set_ylim(-0.25, max(gamma + [1.0]) + 0.5)
        gamma_axis.set_ylabel("Gamma", color="#dd8452")
        gamma_axis.tick_params(axis="y", labelcolor="#dd8452")
    axes[-1, 0].set_xlabel("Time since first event (s)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_traces(runs: list[RunData], output_path: Path,
                      smooth_window: int,
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
        batch_size = [
            float(step.get("batch_size", 0.0)) for step in run.speculative_steps
        ]
        timestamps = [
            float(step.get("timestamp", 0.0)) for step in run.speculative_steps
        ]
        origin = event_time_origin(run)
        steps = relative_time_axis(timestamps, origin)
        intervals = disabled_time_intervals(run, origin)
        x_gamma, y_gamma = downsample_pairs(steps, gamma, max_trace_points)
        batch_window = max(5, smooth_window // 2)
        batch_smooth = moving_average(batch_size, batch_window)
        x_batch, y_batch = downsample_pairs(steps, batch_smooth, max_trace_points)
        shade_disabled_intervals(axis, intervals)
        axis.step(x_gamma,
                  y_gamma,
                  where="post",
                  color="#dd8452",
                  linewidth=2.2,
                  label="Selected gamma")
        axis.set_ylabel("Selected gamma", color="#dd8452")
        axis.tick_params(axis="y", labelcolor="#dd8452")
        axis.set_ylim(-0.25, max(gamma + [1.0]) + 0.5)
        axis.grid(alpha=0.25)
        axis.set_title(format_run_title(run), fontsize=10)
        load_axis = axis.twinx()
        shade_disabled_intervals(load_axis, intervals)
        load_axis.plot(x_batch,
                       y_batch,
                       color="#7f7f7f",
                       alpha=0.4,
                       linewidth=2.0,
                       label="Batch size")
        load_axis.fill_between(x_batch,
                               y_batch,
                               color="#bdbdbd",
                               alpha=0.15)
        load_axis.set_ylabel("Batch size", color="#7f7f7f")
        load_axis.tick_params(axis="y", labelcolor="#7f7f7f")
        handles, labels = axis.get_legend_handles_labels()
        load_handles, load_labels = load_axis.get_legend_handles_labels()
        axis.legend(handles + load_handles,
                    labels + load_labels,
                    loc="upper right",
                    frameon=False,
                    fontsize=9)
    axes[-1, 0].set_xlabel("Time since first event (s)")
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
        if not run.speculative_steps:
            continue
        gamma = [
            float(step.get("proposal_length_gamma", 0))
            for step in run.speculative_steps
        ]
        timestamps = [
            float(step.get("timestamp", 0.0)) for step in run.speculative_steps
        ]
        origin = event_time_origin(run)
        steps = relative_time_axis(timestamps, origin)
        intervals = disabled_time_intervals(run, origin)
        throughput = []
        for step in run.speculative_steps:
            batch_size = float(step.get("batch_size", 0.0))
            num_accepted_tokens = float(step.get("num_accepted_tokens", 0.0))
            step_total_time_us = float(step.get("step_total_time_ms", 0.0))
            step_total_time_s = step_total_time_us / 1_000_000.0
            if step_total_time_s > 0:
                throughput.append((batch_size + num_accepted_tokens) /
                                  step_total_time_s)
            else:
                throughput.append(0.0)
        throughput_smooth = moving_average(throughput, smooth_window)
        x_tp, y_tp = downsample_pairs(steps, throughput_smooth,
                                      max_trace_points)
        x_gamma, y_gamma = force_zero_in_intervals(steps, gamma, intervals)
        zero_indices = zero_value_indices(gamma)
        disabled_timestamps = [
            float(step.get("timestamp", 0.0))
            for step in run.disabled_decode_events
        ]
        disabled_steps = relative_time_axis(disabled_timestamps, origin)
        disabled_throughput = [
            float(step.get("throughput_tokens_per_s", 0.0))
            for step in run.disabled_decode_events
        ]

        shade_disabled_intervals(axis, intervals)
        axis.plot(x_tp,
                  y_tp,
                  color="#4c72b0",
                  linewidth=2,
                  label="Spec throughput")
        if disabled_steps:
            axis.plot(disabled_steps,
                      disabled_throughput,
                      color="#c44e52",
                      linewidth=1.8,
                      marker="o",
                      markersize=3.5,
                      label="Disabled decode throughput")
        axis.set_ylabel("Throughput (tokens/s)", color="#4c72b0")
        axis.tick_params(axis="y", labelcolor="#4c72b0")
        axis.grid(alpha=0.25)
        axis.set_title(format_run_title(run), fontsize=10)

        gamma_axis = axis.twinx()
        shade_disabled_intervals(gamma_axis, intervals)
        gamma_axis.step(x_gamma,
                        y_gamma,
                        where="post",
                        color="#dd8452",
                        alpha=0.85,
                        label="Gamma")
        if zero_indices:
            gamma_axis.scatter([steps[index] for index in zero_indices],
                               [0.0] * len(zero_indices),
                               color="#c44e52",
                               edgecolors="white",
                               linewidths=0.8,
                               s=36,
                               zorder=5,
                               label="Gamma = 0")
        if disabled_steps:
            gamma_axis.scatter(disabled_steps,
                               [0.0] * len(disabled_steps),
                               color="#c44e52",
                               edgecolors="white",
                               linewidths=0.5,
                               s=20,
                               alpha=0.8,
                               zorder=4,
                               label="Disabled decode")
        gamma_axis.set_ylim(-0.25, max(gamma + [1.0]) + 0.5)
        gamma_axis.set_ylabel("Gamma", color="#dd8452")
        gamma_axis.tick_params(axis="y", labelcolor="#dd8452")
    axes[-1, 0].set_xlabel("Time since first event (s)")
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
            enabled = speculation_enabled(step)
            draft_ms = float(step.get("draft_time_ms", 0.0)) if enabled else 0.0
            verify_ms = (
                float(step.get("scoring_time_ms", 0.0)) +
                float(step.get("verify_time_ms", 0.0))) if enabled else 0.0
            ar_ms = float(step.get("step_total_time_ms", 0.0)) if not enabled else 0.0
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
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
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
                          args.smooth_window,
                          args.max_trace_points,
                          args.zero_gamma_scale)
        return

    summaries = save_summary(runs, args.output_dir, args.prefix)

    plot_gamma_traces(
        runs,
        args.output_dir / f"{args.prefix}_gamma_traces.pdf",
        args.smooth_window,
        args.max_trace_points,
        args.zero_gamma_scale,
    )

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
