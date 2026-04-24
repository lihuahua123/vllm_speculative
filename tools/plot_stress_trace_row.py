#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


DEFAULT_INPUT = Path(
    "/root/autodl-tmp/nightjar/vllm_speculative/benchmark_results/stress_tests/"
    "burst_spike_Nightjar/nightjar_events.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/root/autodl-tmp/nightjar/nightjar_paper/Response_Letter_template/figs"
)
CONTRACTION_MARKER_ITEM_INDEX = 629


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {jsonl_path}"
                ) from exc
            if "timestamp" not in obj:
                continue
            records.append(obj)
    if not records:
        raise ValueError(f"No valid timestamped records found in {jsonl_path}")
    records.sort(key=lambda x: x["timestamp"])
    return records


def fill_missing_usable_gpu_blocks(records: list[dict]) -> list[dict]:
    last_usable = None
    filled = []
    for record in records:
        current = dict(record)
        if "usable_gpu_blocks" in current:
            last_usable = current["usable_gpu_blocks"]
        elif last_usable is not None:
            current["usable_gpu_blocks"] = last_usable
        filled.append(current)
    return filled


def get_contraction_marker_x(records: list[dict], xs: list[float]) -> float:
    if len(records) < CONTRACTION_MARKER_ITEM_INDEX:
        raise ValueError(
            "Not enough timestamped records to place Memory Contraction at "
            f"item {CONTRACTION_MARKER_ITEM_INDEX}"
        )
    return xs[CONTRACTION_MARKER_ITEM_INDEX - 1]


def plot_traces(records: list[dict], output_path: Path) -> None:
    base_ts = records[0]["timestamp"]
    xs = [r["timestamp"] - base_ts for r in records]
    contraction_marker_x = get_contraction_marker_x(records, xs)
    free_gpu_blocks = [r.get("free_gpu_blocks") for r in records]
    used_usable_blocks = [
        r.get("usable_gpu_blocks", 0) - r.get("free_gpu_blocks", 0)
        for r in records
    ]
    num_running = [r.get("num_running", 0) for r in records]
    queue_len = [r.get("queue_len", 0) for r in records]
    marker_event_types = {
        "memory_expand": "#e377c2",
    }
    event_markers = []
    for r in records:
        event_type = r.get("event_type")
        if event_type in marker_event_types:
            event_markers.append(
                (r["timestamp"] - base_ts, event_type,
                 marker_event_types[event_type])
            )
    event_markers.append(
        (contraction_marker_x, "Memory Contraction", "#000000")
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2), constrained_layout=True)

    for ax in axes:
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for x, _, color in event_markers:
            ax.axvline(x=x, color=color, linestyle="--", linewidth=1.2, alpha=0.9)

    axes[0].plot(xs, free_gpu_blocks, color="#1f77b4", linewidth=2.0)
    axes[0].set_title("(a)")
    axes[0].set_xlabel("Relative Time (s)")
    axes[0].set_ylabel("Free GPU Blocks")

    axes[1].plot(xs, used_usable_blocks, color="#ff7f0e", linewidth=2.0)
    axes[1].set_title("(b)")
    axes[1].set_xlabel("Relative Time (s)")
    axes[1].set_ylabel("Occupied Usable Blocks")

    axes[2].plot(xs, num_running, label="Running Requests", color="#2ca02c",
                 linewidth=2.0)
    axes[2].plot(xs, queue_len, label="Queue Length", color="#d62728",
                 linewidth=2.0)
    axes[2].set_title("(c)")
    axes[2].set_xlabel("Relative Time (s)")
    axes[2].set_ylabel("Number of Requests")
    axes[2].legend(frameon=False)

    event_handles = [
        Line2D([0], [0], color=color, linestyle="--", linewidth=1.2,
               label="Memory Expansion")
        for event_type, color in marker_event_types.items()
    ]
    event_handles.append(
        Line2D([0], [0], color="#000000", linestyle="--", linewidth=1.2,
               label="Memory Contraction")
    )
    axes[0].legend(handles=event_handles, frameon=False, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot three stress-test traces in a single row."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to nightjar_events.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Defaults to "
            "nightjar_paper/Response_Letter_template/figs/stress_trace_row.pdf"
        ),
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / "stress_trace_row.pdf"

    records = fill_missing_usable_gpu_blocks(load_records(input_path))
    plot_traces(records, output_path)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
