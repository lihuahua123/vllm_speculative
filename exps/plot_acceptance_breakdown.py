#!/usr/bin/env python3
"""Plot Nightjar acceptance/gamma/breakdown figures from JSONL event logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_speculative_steps(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event_type") == "speculative_step":
                rows.append(record)
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    averaged = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        averaged.append(running_sum / min(idx + 1, window))
    return averaged


def plot_acceptance_and_gamma(rows: list[dict], output_path: Path,
                              smooth_window: int) -> None:
    steps = list(range(len(rows)))
    acceptance = [float(row.get("acceptance_rate", 0.0)) for row in rows]
    gamma = [int(row.get("proposal_length_gamma", 0)) for row in rows]
    batch = [int(row.get("batch_size", 0)) for row in rows]
    acceptance_smooth = moving_average(acceptance, smooth_window)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(steps, acceptance, alpha=0.25, color="#4c72b0")
    axes[0].plot(steps, acceptance_smooth, color="#4c72b0", linewidth=2)
    axes[0].set_ylabel("Acceptance")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)

    axes[1].step(steps, gamma, where="post", color="#dd8452")
    axes[1].set_ylabel("Gamma")
    axes[1].grid(alpha=0.25)

    axes[2].step(steps, batch, where="post", color="#55a868")
    axes[2].set_ylabel("Batch")
    axes[2].set_xlabel("Decode Step")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_time_breakdown(rows: list[dict], output_path: Path,
                        smooth_window: int) -> None:
    steps = list(range(len(rows)))
    draft = [float(row.get("draft_time_ms", 0.0)) for row in rows]
    scoring = [float(row.get("scoring_time_ms", 0.0)) for row in rows]
    verify = [float(row.get("verify_time_ms", 0.0)) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(steps, moving_average(draft, smooth_window), label="Draft")
    ax.plot(steps, moving_average(scoring, smooth_window), label="Scoring")
    ax.plot(steps, moving_average(verify, smooth_window), label="Verify")
    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Time (ms)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Nightjar acceptance, gamma, and timing breakdown.")
    parser.add_argument("event_log", type=Path, help="Nightjar JSONL event log")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("figs"),
                        help="Directory to store generated figures")
    parser.add_argument("--prefix",
                        type=str,
                        default="nightjar",
                        help="Filename prefix for generated figures")
    parser.add_argument("--smooth-window",
                        type=int,
                        default=20,
                        help="Moving average window size")
    args = parser.parse_args()

    rows = load_speculative_steps(args.event_log)
    if not rows:
        raise SystemExit("No speculative_step events found in the provided log.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_acceptance_and_gamma(
        rows, args.output_dir / f"{args.prefix}_acceptance_gamma.pdf",
        args.smooth_window)
    plot_time_breakdown(
        rows, args.output_dir / f"{args.prefix}_time_breakdown.pdf",
        args.smooth_window)


if __name__ == "__main__":
    main()
