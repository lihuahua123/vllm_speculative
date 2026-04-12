#!/usr/bin/env python3
"""Generate reusable trace plans for Nightjar stress tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def burst_spike() -> list[dict[str, float]]:
    return [
        {"request_rate": 2, "num_requests": 30},
        {"request_rate": 8, "num_requests": 40},
        {"request_rate": 24, "num_requests": 80},
        {"request_rate": 40, "num_requests": 120},
        {"request_rate": 6, "num_requests": 40},
    ]


def high_low_oscillation() -> list[dict[str, float]]:
    return [
        {"request_rate": 4, "num_requests": 30},
        {"request_rate": 28, "num_requests": 30},
        {"request_rate": 4, "num_requests": 30},
        {"request_rate": 28, "num_requests": 30},
        {"request_rate": 4, "num_requests": 30},
        {"request_rate": 28, "num_requests": 30},
    ]


def sync_migration_worst_case() -> list[dict[str, float]]:
    return [
        {"request_rate": 6, "num_requests": 40},
        {"request_rate": 36, "num_requests": 140},
        {"request_rate": 36, "num_requests": 140},
        {"request_rate": 3, "num_requests": 20},
        {"request_rate": 30, "num_requests": 120},
    ]


PATTERNS = {
    "burst_spike": burst_spike,
    "high_low_oscillation": high_low_oscillation,
    "sync_migration_worst_case": sync_migration_worst_case,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stress-test trace plans for Nightjar.")
    parser.add_argument("--pattern",
                        choices=sorted(PATTERNS),
                        required=True,
                        help="Trace pattern to generate.")
    parser.add_argument("--output",
                        type=Path,
                        required=True,
                        help="Output JSON file path.")
    args = parser.parse_args()

    segments = PATTERNS[args.pattern]()
    payload = {
        "pattern": args.pattern,
        "segments": segments,
        "total_requests": sum(int(seg["num_requests"]) for seg in segments),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
