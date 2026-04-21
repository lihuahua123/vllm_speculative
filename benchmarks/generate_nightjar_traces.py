#!/usr/bin/env python3
"""Generate reusable trace plans for Nightjar stress tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def burst_spike() -> list[dict[str, float]]:
    return [
        {
            "label": "pre_spike",
            "request_rate": 2,
            "num_requests": 30,
            "burstiness": 1.0
        },
        {
            "label": "ramp_up",
            "request_rate": 8,
            "num_requests": 40,
            "burstiness": 0.9
        },
        {
            "label": "burst_peak",
            "request_rate": 24,
            "num_requests": 80,
            "burstiness": 0.5
        },
        {
            "label": "migration_pressure_peak",
            "request_rate": 40,
            "num_requests": 120,
            "burstiness": 0.35
        },
        {
            "label": "recovery",
            "request_rate": 6,
            "num_requests": 40,
            "burstiness": 1.2
        },
    ]


def high_low_oscillation() -> list[dict[str, float]]:
    return [
        {
            "label": "low_0",
            "request_rate": 4,
            "num_requests": 30,
            "burstiness": 1.1
        },
        {
            "label": "high_0",
            "request_rate": 28,
            "num_requests": 30,
            "burstiness": 0.7
        },
        {
            "label": "low_1",
            "request_rate": 4,
            "num_requests": 30,
            "burstiness": 1.1
        },
        {
            "label": "high_1",
            "request_rate": 28,
            "num_requests": 30,
            "burstiness": 0.7
        },
        {
            "label": "low_2",
            "request_rate": 4,
            "num_requests": 30,
            "burstiness": 1.1
        },
        {
            "label": "high_2",
            "request_rate": 28,
            "num_requests": 30,
            "burstiness": 0.7
        },
    ]


def sync_migration_worst_case() -> list[dict[str, float]]:
    return [
        {
            "label": "warmup_low_0",
            "request_rate": 2,
            "num_requests": 50,
            "burstiness": 1.3
        },
        {
            "label": "sync_peak_0",
            "request_rate": 44,
            "num_requests": 110,
            "burstiness": 0.25
        },
        {
            "label": "drain_reload_0",
            "request_rate": 1,
            "num_requests": 70,
            "burstiness": 1.5
        },
        {
            "label": "sync_peak_1",
            "request_rate": 48,
            "num_requests": 120,
            "burstiness": 0.2
        },
        {
            "label": "drain_reload_1",
            "request_rate": 1,
            "num_requests": 70,
            "burstiness": 1.5
        },
        {
            "label": "sync_peak_2",
            "request_rate": 46,
            "num_requests": 120,
            "burstiness": 0.22
        },
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
