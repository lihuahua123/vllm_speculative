#!/usr/bin/env python3
"""Summarize and visualize speculation-disable events from Nightjar stress tests."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SPEC_STEP_EVENT = "speculative_step"
DISABLED_DECODE_EVENT = "disabled_decode_step"
POLICY_EVENT = "memory_policy_decision"
EXPAND_EVENT = "memory_expand"
CONTRACT_EVENT = "memory_contract"
MIGRATION_EVENT = "kv_block_migration"


@dataclass
class StressCase:
    pattern: str
    variant: str
    event_log: Path
    trace_path: Path | None


PATTERN_ORDER = {
    "burst_spike": 0,
    "high_low_oscillation": 1,
    "sync_migration_worst_case": 2,
}

VARIANT_ORDER = {
    "Nightjar": 0,
    "Nightjar_static_memory": 1,
    "BanditSpec": 2,
    "DSD": 3,
    "SD": 4,
    "TETRIS": 5,
    "wo_sd": 6,
}

VARIANT_LABELS = {
    "Nightjar": "Nightjar",
    # "Nightjar_static_memory": "Nightjar (w/o offload)",
    "BanditSpec": "BanditSpec",
    "DSD": "DSD",
    "SD": "SD",
    "TETRIS": "TETRIS",
    "wo_sd": "w/o SD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze when Nightjar disables speculation in stress tests.")
    parser.add_argument("--summary-csv",
                        type=Path,
                        required=True,
                        help="stress_tests_summary.csv path.")
    parser.add_argument("--trace-dir",
                        type=Path,
                        help="Directory containing per-pattern trace json files.")
    parser.add_argument("--output-dir",
                        type=Path,
                        required=True,
                        help="Directory for summary tables and figures.")
    parser.add_argument("--prefix",
                        default="stress_disable",
                        help="Filename prefix for generated artifacts.")
    return parser.parse_args()


def parse_stress_cases(summary_csv: Path,
                       trace_dir: Path | None) -> list[StressCase]:
    cases: list[StressCase] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern = (row.get("pattern") or "").strip()
            variant = (row.get("variant") or "").strip()
            event_log = Path(row["event_log"]).expanduser()
            if not event_log.exists():
                continue
            trace_path = None
            if trace_dir:
                candidate = trace_dir / f"{pattern}.json"
                if candidate.exists():
                    trace_path = candidate
            cases.append(
                StressCase(pattern=pattern,
                           variant=variant,
                           event_log=event_log,
                           trace_path=trace_path))
    cases.sort(key=case_sort_key)
    return cases


def split_workload_pattern(case_pattern: str, variant: str) -> str:
    suffix = f"_{variant}"
    if variant and case_pattern.endswith(suffix):
        return case_pattern[: -len(suffix)]
    return case_pattern


def pretty_workload_pattern(workload_pattern: str) -> str:
    return workload_pattern.replace("_", " ")


def pretty_variant(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant.replace("_", " "))


def case_sort_key(case: StressCase) -> tuple[int, int, str]:
    workload_pattern = split_workload_pattern(case.pattern, case.variant)
    return (
        PATTERN_ORDER.get(workload_pattern, 99),
        VARIANT_ORDER.get(case.variant, 99),
        case.pattern,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_trace_segments(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    labels = [str(segment.get("label", "")) for segment in segments if segment]
    return " | ".join(label for label in labels if label)


def find_first(events: list[dict[str, Any]],
               event_type: str,
               *,
               action: str | None = None) -> dict[str, Any] | None:
    for event in events:
        if event.get("event_type") != event_type:
            continue
        if action is not None and event.get("action") != action:
            continue
        return event
    return None


def first_disable_step(
        spec_steps: list[dict[str, Any]]) -> tuple[int | None, dict[str, Any] | None]:
    for index, step in enumerate(spec_steps):
        if int(step.get("proposal_length_gamma", 0)) == 0:
            return index, step
    return None, None


def timeline_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = [
        event for event in events
        if event.get("event_type") in (SPEC_STEP_EVENT, DISABLED_DECODE_EVENT)
    ]
    return steps


def rel_ms(anchor_ts: float | None, event_ts: float | None) -> float:
    if anchor_ts is None or event_ts is None:
        return 0.0
    return (event_ts - anchor_ts) * 1000.0


def timeline_rel_ms(steps: list[dict[str, Any]]) -> list[float]:
    anchor_ts = None
    for step in steps:
        ts = step.get("timestamp")
        if isinstance(ts, (int, float)):
            anchor_ts = float(ts)
            break
    return [
        rel_ms(anchor_ts, float(step["timestamp"]))
        if isinstance(step.get("timestamp"), (int, float)) else 0.0
        for step in steps
    ]


def summarize_case(case: StressCase) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = load_jsonl(case.event_log)
    spec_steps = [e for e in events if e.get("event_type") == SPEC_STEP_EVENT]
    steps = timeline_steps(events)
    disable_index, disable_event = first_disable_step(steps)
    expand_policy = find_first(events, POLICY_EVENT, action="expand_kv_cache")
    restore_policy = find_first(events, POLICY_EVENT, action="restore_draft_model")
    expand_event = find_first(events, EXPAND_EVENT)
    migration_event = find_first(events, MIGRATION_EVENT)
    contract_event = find_first(events, CONTRACT_EVENT)

    anchor_ts = None
    for event in events:
        ts = event.get("timestamp")
        if isinstance(ts, (int, float)):
            anchor_ts = float(ts)
            break

    summary = {
        "pattern": case.pattern,
        "trace_segments": load_trace_segments(case.trace_path),
        "event_log": str(case.event_log),
        "num_speculative_steps": len(spec_steps),
        "num_timeline_steps": len(steps),
        "num_disabled_decode_steps": sum(
            1 for event in events
            if event.get("event_type") == DISABLED_DECODE_EVENT),
        "disable_step_count": sum(
            1 for step in steps
            if int(step.get("proposal_length_gamma", 0)) == 0),
        "first_disable_step_index": disable_index if disable_index is not None else -1,
        "first_disable_ts": float(disable_event.get("timestamp", 0.0))
        if disable_event else 0.0,
        "first_disable_rel_ms": rel_ms(anchor_ts, disable_event.get("timestamp"))
        if disable_event else 0.0,
        "disable_batch_size": int(disable_event.get("batch_size", 0))
        if disable_event else 0,
        "disable_queue_len": int(disable_event.get("queue_len", 0))
        if disable_event else 0,
        "disable_num_running": int(disable_event.get("num_running", 0))
        if disable_event else 0,
        "disable_free_gpu_blocks": int(disable_event.get("free_gpu_blocks", 0))
        if disable_event else 0,
        "disable_usable_gpu_blocks": int(disable_event.get("usable_gpu_blocks", 0))
        if disable_event else 0,
        "expand_trigger_ts": float(expand_policy.get("timestamp", 0.0))
        if expand_policy else 0.0,
        "expand_trigger_rel_ms": rel_ms(anchor_ts, expand_policy.get("timestamp"))
        if expand_policy else 0.0,
        "expand_trigger_free_gpu_blocks": int(
            expand_policy.get("free_gpu_blocks", 0)) if expand_policy else 0,
        "expand_trigger_running_len": int(expand_policy.get("running_len", 0))
        if expand_policy else 0,
        "expand_trigger_waiting_len": int(expand_policy.get("waiting_len", 0))
        if expand_policy else 0,
        "persist_steps": int(expand_policy.get("persist_steps", 0))
        if expand_policy else 0,
        "increase_block_threshold": int(
            expand_policy.get("increase_block_threshold", 0))
        if expand_policy else 0,
        "decrease_block_threshold": int(
            expand_policy.get("decrease_block_threshold", 0))
        if expand_policy else 0,
        "restore_trigger_ts": float(restore_policy.get("timestamp", 0.0))
        if restore_policy else 0.0,
        "restore_trigger_rel_ms": rel_ms(anchor_ts, restore_policy.get("timestamp"))
        if restore_policy else 0.0,
        "restore_trigger_free_gpu_blocks": int(
            restore_policy.get("free_gpu_blocks", 0)) if restore_policy else 0,
        "offload_count": sum(1 for event in events
                              if event.get("event_type") == EXPAND_EVENT),
        "reload_count": sum(1 for event in events
                             if event.get("event_type") == CONTRACT_EVENT),
        "migration_count": sum(1 for event in events
                                if event.get("event_type") == MIGRATION_EVENT),
        "migrated_block_count": int(
            migration_event.get("migrated_block_count", 0))
        if migration_event else 0,
        "affected_seq_count": int(
            contract_event.get("affected_seq_count",
                               migration_event.get("affected_seq_count", 0)))
        if contract_event or migration_event else 0,
        "max_blocks_migrated_per_seq": int(
            contract_event.get(
                "max_blocks_migrated_per_seq",
                migration_event.get("max_blocks_migrated_per_seq", 0)))
        if contract_event or migration_event else 0,
        "mean_blocks_migrated_per_seq": float(
            contract_event.get(
                "mean_blocks_migrated_per_seq",
                migration_event.get("mean_blocks_migrated_per_seq", 0.0)))
        if contract_event or migration_event else 0.0,
        "running_len_at_migration": int(
            contract_event.get("running_len_at_migration", 0))
        if contract_event else 0,
        "waiting_len_at_migration": int(
            contract_event.get("waiting_len_at_migration", 0))
        if contract_event else 0,
        "affected_running_ratio": float(
            contract_event.get("affected_running_ratio", 0.0))
        if contract_event else 0.0,
        "expand_duration_ms": float(expand_event.get("duration_ms", 0.0))
        if expand_event else 0.0,
        "migration_duration_ms": float(migration_event.get("duration_ms", 0.0))
        if migration_event else 0.0,
        "contract_duration_ms": float(contract_event.get("duration_ms", 0.0))
        if contract_event else 0.0,
        "time_expand_trigger_to_disable_ms":
        rel_ms(expand_policy.get("timestamp") if expand_policy else None,
               disable_event.get("timestamp") if disable_event else None),
        "time_expand_complete_to_disable_ms":
        rel_ms(expand_event.get("timestamp") if expand_event else None,
               disable_event.get("timestamp") if disable_event else None),
        "time_contract_complete_to_disable_ms":
        rel_ms(contract_event.get("timestamp") if contract_event else None,
               disable_event.get("timestamp") if disable_event else None),
    }
    return summary, steps


def save_summaries(summaries: list[dict[str, Any]], output_dir: Path,
                   prefix: str) -> None:
    json_path = output_dir / f"{prefix}_summary.json"
    csv_path = output_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def plot_cases(cases: list[StressCase],
               timeline_steps_by_pattern: dict[str, list[dict[str, Any]]],
               summaries: list[dict[str, Any]],
               output_path: Path) -> None:
    if not summaries:
        return
    fig, axes = plt.subplots(len(cases),
                             3,
                             figsize=(16, max(3.6, 2.9 * len(cases))),
                             squeeze=False)

    for row_index, case in enumerate(cases):
        steps = timeline_steps_by_pattern[case.pattern]
        summary = next(item for item in summaries if item["pattern"] == case.pattern)
        workload_pattern = split_workload_pattern(case.pattern, case.variant)
        xs = timeline_rel_ms(steps)
        gamma = [int(step.get("proposal_length_gamma", 0)) for step in steps]
        acceptance = [float(step.get("acceptance_rate", 0.0)) for step in steps]
        free_blocks = [int(step.get("free_gpu_blocks", 0)) for step in steps]
        running = [int(step.get("num_running", 0)) for step in steps]
        queue = [int(step.get("queue_len", 0)) for step in steps]

        acceptance_axis = axes[row_index, 0]
        free_axis = axes[row_index, 1]
        load_axis = axes[row_index, 2]

        acceptance_axis.plot(xs,
                             acceptance,
                             color="#4c72b0",
                             linewidth=2,
                             label="Acceptance")
        acceptance_axis.set_ylim(0.0, 1.05)
        acceptance_axis.set_ylabel("Acceptance")
        acceptance_axis.set_title(
            f"{pretty_workload_pattern(workload_pattern)} | {pretty_variant(case.variant)}"
        )
        acceptance_axis.grid(alpha=0.25)

        free_axis.plot(xs, free_blocks, color="#4c72b0", linewidth=2)
        threshold = summary["increase_block_threshold"]
        if threshold > 0:
            free_axis.axhline(threshold,
                              linestyle="--",
                              color="#c44e52",
                              linewidth=1.2,
                              label="Expand threshold")
        free_axis.set_ylabel("Free GPU Blocks")
        free_axis.grid(alpha=0.25)

        load_axis.plot(xs, running, color="#55a868", linewidth=2, label="Running")
        load_axis.plot(xs, queue, color="#8172b3", linewidth=2, label="Queue")
        load_axis.set_ylabel("Requests")
        load_axis.grid(alpha=0.25)

        disable_ts = summary["first_disable_rel_ms"]
        if summary["first_disable_step_index"] >= 0:
            for axis in (acceptance_axis, free_axis, load_axis):
                axis.axvline(disable_ts,
                             color="#000000",
                             linestyle=":",
                             linewidth=1.2)

        if row_index == 0:
            acceptance_axis.legend(loc="upper right")
            free_axis.legend(loc="upper right")
            load_axis.legend(loc="upper right")

    for axis in axes[-1, :]:
        axis.set_xlabel("Relative Time (ms)")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cases_by_workload(
        cases: list[StressCase],
        timeline_steps_by_pattern: dict[str, list[dict[str, Any]]],
        summaries: list[dict[str, Any]],
        output_dir: Path,
        prefix: str) -> None:
    grouped_cases: dict[str, list[StressCase]] = {}
    for case in cases:
        workload_pattern = split_workload_pattern(case.pattern, case.variant)
        grouped_cases.setdefault(workload_pattern, []).append(case)

    for workload_pattern, workload_cases in grouped_cases.items():
        fig, axes = plt.subplots(len(workload_cases),
                                 3,
                                 figsize=(16, max(3.6, 2.9 * len(workload_cases))),
                                 squeeze=False)

        for row_index, case in enumerate(workload_cases):
            steps = timeline_steps_by_pattern[case.pattern]
            summary = next(item for item in summaries
                           if item["pattern"] == case.pattern)
            xs = timeline_rel_ms(steps)
            acceptance = [
                float(step.get("acceptance_rate", 0.0)) for step in steps
            ]
            free_blocks = [int(step.get("free_gpu_blocks", 0)) for step in steps]
            running = [int(step.get("num_running", 0)) for step in steps]
            queue = [int(step.get("queue_len", 0)) for step in steps]

            acceptance_axis = axes[row_index, 0]
            free_axis = axes[row_index, 1]
            load_axis = axes[row_index, 2]

            acceptance_axis.plot(xs,
                                 acceptance,
                                 color="#4c72b0",
                                 linewidth=2,
                                 label="Acceptance")
            acceptance_axis.set_ylim(0.0, 1.05)
            acceptance_axis.set_ylabel("Acceptance")
            acceptance_axis.set_title(
                f"{pretty_workload_pattern(workload_pattern)} | {pretty_variant(case.variant)}"
            )
            acceptance_axis.grid(alpha=0.25)

            free_axis.plot(xs, free_blocks, color="#4c72b0", linewidth=2)
            threshold = summary["increase_block_threshold"]
            if threshold > 0:
                free_axis.axhline(threshold,
                                  linestyle="--",
                                  color="#c44e52",
                                  linewidth=1.2,
                                  label="Expand threshold")
            free_axis.set_ylabel("Free GPU Blocks")
            free_axis.grid(alpha=0.25)

            load_axis.plot(xs,
                           running,
                           color="#55a868",
                           linewidth=2,
                           label="Running")
            load_axis.plot(xs,
                           queue,
                           color="#8172b3",
                           linewidth=2,
                           label="Queue")
            load_axis.set_ylabel("Requests")
            load_axis.grid(alpha=0.25)

            disable_ts = summary["first_disable_rel_ms"]
            if summary["first_disable_step_index"] >= 0:
                for axis in (acceptance_axis, free_axis, load_axis):
                    axis.axvline(disable_ts,
                                 color="#000000",
                                 linestyle=":",
                                 linewidth=1.2)

            if row_index == 0:
                acceptance_axis.legend(loc="upper right")
                free_axis.legend(loc="upper right")
                load_axis.legend(loc="upper right")

        for axis in axes[-1, :]:
            axis.set_xlabel("Relative Time (ms)")

        fig.tight_layout()
        output_path = output_dir / f"{prefix}_{workload_pattern}_timeline.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = parse_stress_cases(args.summary_csv, args.trace_dir)
    summaries: list[dict[str, Any]] = []
    timeline_steps_by_pattern: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        summary, steps = summarize_case(case)
        summaries.append(summary)
        timeline_steps_by_pattern[case.pattern] = steps

    if summaries:
        save_summaries(summaries, args.output_dir, args.prefix)
        plot_cases(cases, timeline_steps_by_pattern, summaries,
                   args.output_dir / f"{args.prefix}_timeline.pdf")
        plot_cases_by_workload(cases, timeline_steps_by_pattern, summaries,
                               args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
