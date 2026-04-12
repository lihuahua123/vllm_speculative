#!/usr/bin/env python3
"""Summarize Nightjar migration and memory events from a JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_events(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    migrations, expands, contracts = [], [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            event_type = record.get("event_type")
            if event_type == "kv_block_migration":
                migrations.append(record)
            elif event_type == "memory_expand":
                expands.append(record)
            elif event_type == "memory_contract":
                contracts.append(record)
    return migrations, expands, contracts


def plot_migration_events(migrations: list[dict], output_path: Path) -> None:
    indices = list(range(len(migrations)))
    counts = [int(event.get("migrated_block_count", 0)) for event in migrations]
    durations = [float(event.get("duration_ms", 0.0)) for event in migrations]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(indices, counts, color="#4c72b0", alpha=0.7)
    ax1.set_xlabel("Migration Event")
    ax1.set_ylabel("Migrated Blocks", color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(indices, durations, color="#dd8452", marker="o")
    ax2.set_ylabel("Duration (ms)", color="#dd8452")
    ax2.tick_params(axis="y", labelcolor="#dd8452")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(migrations: list[dict], expands: list[dict],
                  contracts: list[dict], output_path: Path) -> None:
    total_migrated = sum(int(event.get("migrated_block_count", 0))
                         for event in migrations)
    total_migration_ms = sum(float(event.get("duration_ms", 0.0))
                             for event in migrations)
    summary = {
        "num_migration_events": len(migrations),
        "num_expand_events": len(expands),
        "num_contract_events": len(contracts),
        "total_migrated_blocks": total_migrated,
        "total_migration_time_ms": total_migration_ms,
        "avg_migrated_blocks":
        (total_migrated / len(migrations) if migrations else 0.0),
        "avg_migration_time_ms":
        (total_migration_ms / len(migrations) if migrations else 0.0),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Nightjar migration overhead statistics.")
    parser.add_argument("event_log", type=Path, help="Nightjar JSONL event log")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("figs"),
                        help="Directory to store generated outputs")
    parser.add_argument("--prefix",
                        type=str,
                        default="nightjar",
                        help="Filename prefix for generated outputs")
    args = parser.parse_args()

    migrations, expands, contracts = load_events(args.event_log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(migrations, expands, contracts,
                  args.output_dir / f"{args.prefix}_migration_summary.json")
    if migrations:
        plot_migration_events(
            migrations,
            args.output_dir / f"{args.prefix}_migration_overhead.pdf")


if __name__ == "__main__":
    main()
