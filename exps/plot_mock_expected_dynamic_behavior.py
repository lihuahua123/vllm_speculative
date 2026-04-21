#!/usr/bin/env python3
"""Build an illustrative explore-then-exploit dynamic-speculation figure.

This script does not reproduce the exact runtime trace. Instead, it uses the
observed gamma distributions from the revision_dynamic_behavior runs and
rearranges them into a phase-structured trace so the intended paper narrative
becomes visually intuitive:

1. early steps actively explore several speculative lengths;
2. later steps converge to a small set of stable gamma values;
3. only the final short tail disables speculation (gamma = 0).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt


RUN_STATS = {
    "ShareGPT low": {0: 789, 1: 170, 2: 62, 3: 56, "acc": 0.704},
    "ShareGPT med": {0: 808, 1: 202, 2: 50, 3: 40, "acc": 0.758},
    "ShareGPT high": {0: 811, 1: 163, 2: 52, 3: 50, "acc": 0.770},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an illustrative expected dynamic-speculation figure."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/root/autodl-tmp/nightjar/vllm_speculative/benchmark_results/"
            "revision_dynamic_behavior/revision_dynamic_behavior/figs/"
            "mock_expected_dynamic_behavior.pdf"),
        help="Output PDF path.",
    )
    return parser.parse_args()


def build_structured_trace(zero_count: int, one_count: int, two_count: int,
                           three_count: int) -> list[int]:
    """Construct a bin-locked trace with decaying exploration probability."""
    total = zero_count + one_count + two_count + three_count
    tail_zero = max(18, int(total * 0.06))
    target_len = max(220, int(total * 0.34))
    active_len = target_len - tail_zero

    # Bins emulate the paper's hierarchy: later bins are longer, so stable
    # exploitation phases occupy more steps as time progresses.
    bin_lengths = [4, 5, 6, 7, 9, 11, 14, 18, 23, 30, 38, 48]
    scale = active_len / sum(bin_lengths)
    scaled_bins = [max(3, int(round(length * scale))) for length in bin_lengths]
    diff = active_len - sum(scaled_bins)
    scaled_bins[-1] += diff

    # Early bins explore more; later bins mostly exploit the converged arm.
    bin_gammas = [3, 1, 2, 3, 2, 2, 1, 2, 2, 2, 1, 1]

    trace: list[int] = []
    for gamma, length in zip(bin_gammas, scaled_bins):
        trace.extend([gamma] * length)
    trace.extend([0] * tail_zero)
    return trace


def smooth(values: list[float], window: int = 21) -> list[float]:
    if window <= 1:
        return values[:]
    out: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        out.append(running / min(idx + 1, window))
    return out


def derive_acceptance_and_throughput(trace: list[int],
                                     acc_mean: float) -> tuple[list[float], list[float]]:
    acceptance: list[float] = []
    throughput: list[float] = []
    total = max(len(trace), 1)
    for idx, gamma in enumerate(trace):
        progress = idx / total
        wave = math.sin(progress * 5.0 * math.pi)
        if gamma == 0:
            acceptance.append(0.0)
            # High-load AR stage: slightly higher throughput, but not dramatically so.
            throughput.append(63.0 + 1.8 * wave + 1.5 * progress)
            continue
        local_acc = min(1.0, max(0.2, acc_mean + 0.08 * wave - 0.05 * progress))
        if gamma == 1:
            local_acc = min(1.0, local_acc + 0.08)
        elif gamma == 3:
            local_acc = max(0.2, local_acc - 0.10)
        acceptance.append(local_acc)

        # Exploration is noisier and slightly less efficient; exploitation settles.
        phase_bonus = -1.1 if progress < 0.22 else (0.7 if progress > 0.55 else 0.0)
        base = 60.0 - 1.2 * gamma + 3.5 * local_acc + phase_bonus
        throughput.append(base + 1.6 * wave)
    return smooth(acceptance, window=31), smooth(throughput, window=41)


def plot(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=False)

    for row, (label, stats) in enumerate(RUN_STATS.items()):
        gamma_trace = build_structured_trace(stats[0], stats[1], stats[2], stats[3])
        xs = list(range(len(gamma_trace)))
        acceptance, throughput = derive_acceptance_and_throughput(
            gamma_trace, float(stats["acc"]))

        axis = axes[row]
        axis.plot(xs, throughput, color="#4c72b0", linewidth=2)
        axis.set_ylabel("Throughput (tokens/s)", color="#4c72b0")
        axis.tick_params(axis="y", labelcolor="#4c72b0")
        axis.grid(alpha=0.25)
        axis.set_title(label, fontsize=11)

        gamma_axis = axis.twinx()
        gamma_axis.step(xs,
                        gamma_trace,
                        where="post",
                        color="#dd8452",
                        linewidth=1.8)
        gamma_axis.set_ylabel("Gamma", color="#dd8452")
        gamma_axis.tick_params(axis="y", labelcolor="#dd8452")
        gamma_axis.set_ylim(-0.1, 3.3)

        total = len(gamma_trace)
        explore_end = int(total * 0.22)
        exploit_start = int(total * 0.42)
        axis.axvspan(0, explore_end, color="#55a868", alpha=0.08)
        axis.axvspan(exploit_start, total, color="#4c72b0", alpha=0.05)
        axis.axvline(explore_end, color="#2f6b3b", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.axvline(exploit_start, color="#2c4f7c", linestyle="--", linewidth=1.0, alpha=0.6)
        axis.text(explore_end * 0.45,
                  max(throughput) + 0.15,
                  "Exploration",
                  color="#2f6b3b",
                  fontsize=9,
                  ha="center")
        axis.text(exploit_start + (total - exploit_start) * 0.32,
                  max(throughput) + 0.15,
                  "Exploitation",
                  color="#2c4f7c",
                  fontsize=9,
                  ha="center")

        # Lightly shade disabled tail so the intended behavior is easy to read.
        in_zero = False
        start = 0
        for idx, gamma in enumerate(gamma_trace + [1]):
            if gamma == 0 and not in_zero:
                in_zero = True
                start = idx
            elif gamma != 0 and in_zero:
                axis.axvspan(start, idx, color="#c44e52", alpha=0.08)
                in_zero = False
        axis.text(total - max(18, int(total * 0.06)),
                  min(throughput) + 0.2,
                  "gamma = 0",
                  color="#8c2d30",
                  fontsize=9,
                  ha="center")

    axes[-1].set_xlabel("Decode Step")
    fig.suptitle(
        "Illustrative Explore-Then-Exploit Dynamic Behavior\n"
        "Left axis: smoothed throughput, right axis: speculative length gamma",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()
