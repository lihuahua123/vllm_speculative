#!/usr/bin/env python3
"""Plot absolute draft/verify/AR time breakdown across datasets and methods."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/root/autodl-tmp/nightjar/vllm_speculative/benchmark_results")
OUT_PATH = Path(
    "/root/autodl-tmp/nightjar/nightjar_paper/Response_Letter_template/figs/"
    "acceptance_behavior_multidataset_time_breakdown_bar.pdf")

DATASETS = [
    ("sharegpt", "ShareGPT"),
    ("alpaca", "Alpaca"),
    ("specbench", "SpecBench"),
]

METHODS = [
    ("fixed_sd", "Fixed SD"),
    ("dsd", "DSD"),
    ("banditspec", "BanditSpec"),
    ("tetris", "TETRIS"),
    ("nightjar", "Nightjar"),
]

PRECOMPUTED_TIMES_MS = {
    "sharegpt": {
        "fixed_sd": (11570.658683776855, 9197.971820831299, 0.0),
        "dsd": (10054.332494735718, 8299.906969070435, 0.0),
        "banditspec": (7183.088541030884, 9040.377378463745, 0.0),
        "tetris": (6569.7710514068604, 8896.409749984741, 0.0),
        "nightjar": (6085.87646484375, 8954.84447479248,0.0),
    },
    "alpaca": {
        "fixed_sd": (6135.305404663086, 6391.223192214966, 0.0),
        "dsd": (10175.321102142334, 7948.543548583984, 0.0),
        "banditspec": (6975.728750228882, 9103.106498718262, 0.0),
        "tetris": (4694.0011978149414, 6218.016862869263, 0.0),
        "nightjar": (4104.123115539551, 6639.425277709961, 0.0),
    },
    "specbench": {
        "fixed_sd": (15407.610416412354, 10162.733554840088, 0.0),
        "dsd": (12637.925863265991, 8269.59252357483, 0.0),
        "banditspec": (11814.563751220703, 11152.62746810913, 0.0),
        "tetris": (9697.3462104797363, 12844.9668884277344, 0.0),
        "nightjar": (9472.270727157593, 8256.118488311768, 0.0),
    },
}


def resolve_paths(dataset_key: str, method_key: str) -> tuple[Path, Path]:
    if method_key == "tetris":
        case_dir = (ROOT / "acceptance_behavior_tetris_diagnostics" / dataset_key /
                    "med" / method_key)
        result_candidates = sorted(case_dir.glob("benchmark_*.json"))
        if not result_candidates:
            raise FileNotFoundError(f"Missing TETRIS result JSON in {case_dir}")
        return case_dir / "nightjar_events.jsonl", result_candidates[-1]

    if dataset_key == "specbench":
        base = ROOT / "acceptance_behavior_diagnostics_specbench"
    else:
        base = ROOT / "acceptance_behavior_diagnostics"
    case_dir = base / dataset_key / "med" / method_key
    return case_dir / "nightjar_events.jsonl", case_dir / "result.json"


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_duration_ms(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("duration", 0.0)) * 1000.0


def summarize_times(event_log: Path, result_json: Path) -> tuple[float, float, float]:
    events = load_events(event_log)
    steps = [e for e in events if e.get("event_type") == "speculative_step"]
    draft_ms = 0.0
    verify_ms = 0.0
    ar_ms = 0.0

    for step in steps:
        gamma = int(step.get("proposal_length_gamma", 0) or 0)
        if gamma > 0:
            draft_ms += float(step.get("draft_time_ms", 0.0) or 0.0)
            verify_ms += float(step.get("scoring_time_ms", 0.0) or 0.0)
            verify_ms += float(step.get("verify_time_ms", 0.0) or 0.0)
        else:
            ar_ms += float(step.get("step_total_time_ms", 0.0) or 0.0)

    if not steps:
        ar_ms = load_duration_ms(result_json)
    return draft_ms, verify_ms, ar_ms


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.8), squeeze=False)

    for axis, (dataset_key, dataset_label) in zip(axes[0], DATASETS):
        labels: list[str] = []
        draft_vals: list[float] = []
        verify_vals: list[float] = []
        ar_vals: list[float] = []

        for method_key, method_label in METHODS:
            precomputed = PRECOMPUTED_TIMES_MS.get(dataset_key, {}).get(method_key)
            if precomputed is None:
                event_log, result_json = resolve_paths(dataset_key, method_key)
                draft_ms, verify_ms, ar_ms = summarize_times(event_log, result_json)
            else:
                draft_ms, verify_ms, ar_ms = precomputed
            print(method_label,draft_ms/1000,verify_ms/1000,ar_ms/1000)
            labels.append(method_label)
            draft_vals.append(draft_ms)
            verify_vals.append(verify_ms)
            ar_vals.append(ar_ms)

        xs = list(range(len(labels)))
        axis.bar(xs, draft_vals, color="#4c72b0", label="Draft")
        axis.bar(xs, verify_vals, bottom=draft_vals, color="#55a868", label="Verify")
        # axis.bar(
        #     xs,
        #     ar_vals,
        #     bottom=[d + v for d, v in zip(draft_vals, verify_vals)],
        #     color="#c44e52",
        #     label="AR-disabled",
        # )
        axis.set_title(dataset_label)
        axis.set_ylabel("Total Time (ms)")
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)

    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
