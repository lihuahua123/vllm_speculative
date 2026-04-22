#!/usr/bin/env python3
"""Regenerate stress-test dynamic-behavior response artifacts.

This script reads the latest Nightjar stress-test event logs, regenerates:
1. revision_dynamic_behavior_acceptance_gamma_traces.pdf
2. revision_dynamic_behavior_migration_overhead.pdf
3. stress_gamma_throughput_traces.pdf
4. response_dynamic_migration_overhead_table.tex

Example:
python exps/update_stress_dynamic_response.py \
  --stress-dir benchmark_results/stress_tests \
  --response-dir ../nightjar_paper/Response_Letter_template
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_dynamic_behavior_report import (  # noqa: E402
    RunSpec,
    compute_run_summary,
    load_run_data,
    plot_acceptance_gamma_traces,
    plot_migration_overhead,
    plot_throughput_gamma_traces,
)


CASES = [
    (
        "burst_spike_elastic",
        "Burst spike",
        "Burst spike",
    ),
    (
        "high_low_oscillation_elastic",
        "High/low oscillation",
        "High/low oscillation",
    ),
    (
        "sync_migration_worst_case_elastic",
        "Sync worst case",
        "Sync worst case",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stress-test figures and LaTeX table for the response letter."
    )
    parser.add_argument(
        "--stress-dir",
        type=Path,
        default=Path("benchmark_results/stress_tests"),
        help="Directory containing stress-test case subdirectories.",
    )
    parser.add_argument(
        "--response-dir",
        type=Path,
        default=Path("../nightjar_paper/Response_Letter_template"),
        help="Response letter directory containing figs/.",
    )
    parser.add_argument(
        "--prefix",
        default="revision_dynamic_behavior",
        help="Output filename prefix.",
    )
    parser.add_argument("--smooth-window", type=int, default=25)
    parser.add_argument("--max-trace-points", type=int, default=300)
    parser.add_argument("--zero-gamma-scale", type=float, default=0.08)
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy generated PDFs into response-dir/figs.",
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.2f}"


def fmt_migration(events: list[dict]) -> str:
    if not events:
        return "0"
    return " / ".join(fmt(float(event.get("duration_ms", 0.0))) for event in events)


def moved_blocks(run) -> int:
    total = 0
    for event in run.migration_events:
        total += int(event.get("num_blocks_migrated", 0) or 0)
    return total


def expand_contract_ms(run) -> str:
    expand = [float(event.get("duration_ms", 0.0)) for event in run.expand_events]
    contract = [float(event.get("duration_ms", 0.0)) for event in run.contract_events]
    pairs = []
    for index in range(max(len(expand), len(contract))):
        left = fmt(expand[index]) if index < len(expand) else "0.00"
        right = fmt(contract[index]) if index < len(contract) else "0.00"
        pairs.append(f"{left}--{right}")
    return ", ".join(pairs) if pairs else "0.00--0.00"


def render_table(rows: list[tuple[str, object, dict]]) -> str:
    lines = [
        r"\begin{center}",
        r"\centering",
        r"\captionof{table}{Added summary of speculative-length dynamics and CPU--GPU memory-adaptation overhead.}",
        r"\label{tab:response_dynamic_migration_overhead}",
        r"\resizebox{0.98\linewidth}{!}{",
        r"\begin{tabular}{c|c|c|c|c|c|c}",
        r"\hline",
        r" \textbf{Representative case} & \textbf{Mean acc. (\%)} & \textbf{Median acc. (\%)} & \textbf{Mean $\gamma$ all/enabled} & \textbf{Adapt. cycles} & \textbf{KV migration (ms)} & \textbf{Moved blocks / expand--contract (ms)} \\",
        r" \hline",
    ]
    for label, run, summary in rows:
        gamma = (
            f"{fmt(summary['mean_gamma_all_steps'])} / "
            f"{fmt(summary['mean_gamma_when_enabled'])}"
        )
        lines.append(
            f" {label} & {fmt(summary['mean_acceptance_rate'] * 100.0)} "
            f"& {fmt(summary['median_acceptance_rate'] * 100.0)} "
            f"& {gamma} & {summary['offload_count']} "
            f"& {fmt_migration(run.migration_events)} "
            f"& {moved_blocks(run)} / {expand_contract_ms(run)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}}",
            r"\vspace{0.05cm}",
            r"{\small Acceptance statistics are computed over all recorded decode steps, including steps with $\gamma=0$ after speculation is disabled. The $\gamma$ column reports the mean over all steps and, after the slash, over enabled speculative steps only. Memory-adaptation numbers come from elastic stress-test traces. ``Adapt. cycles'' counts expand--restore cycles; ``moved blocks'' reports actual nonzero KV-block movement, which can be zero when contraction occurs after active KV blocks have drained.}",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    stress_dir = args.stress_dir.resolve()
    response_dir = args.response_dir.resolve()
    output_dir = stress_dir / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    runs = []
    for case_dir, label, load in CASES:
        event_log = stress_dir / case_dir / "nightjar_events.jsonl"
        if not event_log.is_file():
            raise FileNotFoundError(f"Missing event log: {event_log}")
        spec = RunSpec(
            path=event_log,
            label=label,
            dataset="ShareGPT",
            load=load,
        )
        run = load_run_data(spec)
        summary = compute_run_summary(run)
        runs.append(run)
        rows.append((label, run, summary))

    plot_acceptance_gamma_traces(
        runs,
        output_dir / f"{args.prefix}_acceptance_gamma_traces.pdf",
        args.smooth_window,
        args.max_trace_points,
        args.zero_gamma_scale,
    )
    plot_migration_overhead(
        runs,
        [row[2] for row in rows],
        output_dir / f"{args.prefix}_migration_overhead.pdf",
    )
    plot_throughput_gamma_traces(
        runs,
        output_dir / "stress_gamma_throughput_traces.pdf",
        args.smooth_window,
        args.max_trace_points,
        args.zero_gamma_scale,
    )

    table_path = output_dir / "response_dynamic_migration_overhead_table.tex"
    table_path.write_text(render_table(rows), encoding="utf-8")

    if not args.no_copy:
        figs_dir = response_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        pdfs = [
            output_dir / f"{args.prefix}_acceptance_gamma_traces.pdf",
            output_dir / f"{args.prefix}_migration_overhead.pdf",
            output_dir / "stress_gamma_throughput_traces.pdf",
        ]
        for src in pdfs:
            shutil.copy2(src, figs_dir / src.name)

    print(f"Wrote figures to: {output_dir}")
    print(f"Wrote LaTeX table to: {table_path}")
    if not args.no_copy:
        print(f"Copied PDFs to: {response_dir / 'figs'}")


if __name__ == "__main__":
    main()
