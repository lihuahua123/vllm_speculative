#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
import statistics


def mean(values):
    return statistics.fmean(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_events(path):
    records = []
    if not path or not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize_events(events):
    spec_steps = [e for e in events if e.get("event_type") == "speculative_step"]
    enabled = [e for e in spec_steps if int(e.get("proposal_length_gamma", 0) or 0) > 0]
    disabled = [e for e in spec_steps if int(e.get("proposal_length_gamma", 0) or 0) == 0]

    gammas = [int(e.get("proposal_length_gamma", 0) or 0) for e in spec_steps]
    accept = [float(e.get("acceptance_rate", 0.0) or 0.0) for e in enabled]
    accepted = [int(e.get("num_accepted_tokens", 0) or 0) for e in enabled]
    proposed = [
        int(e.get("proposal_length_gamma", 0) or 0) * int(e.get("batch_size", 0) or 0)
        for e in enabled
    ]

    draft_ms = sum(float(e.get("draft_time_ms", 0.0) or 0.0) for e in enabled)
    scoring_ms = sum(float(e.get("scoring_time_ms", 0.0) or 0.0) for e in enabled)
    verify_ms = sum(float(e.get("verify_time_ms", 0.0) or 0.0) for e in enabled)
    disabled_ms = sum(float(e.get("step_total_time_ms", 0.0) or 0.0) for e in disabled)
    total_ms = draft_ms + scoring_ms + verify_ms + disabled_ms

    batch = [int(e.get("batch_size", 0) or 0) for e in spec_steps]
    queue = [int(e.get("queue_len", 0) or 0) for e in spec_steps]
    context = [int(e.get("context_length", 0) or 0) for e in spec_steps]
    free_blocks = [int(e.get("free_gpu_blocks", 0) or 0) for e in spec_steps]

    return {
        "num_steps": len(spec_steps),
        "num_enabled_steps": len(enabled),
        "num_disabled_steps": len(disabled),
        "enabled_ratio": len(enabled) / len(spec_steps) if spec_steps else 0.0,
        "mean_gamma_all": mean(gammas),
        "mean_gamma_enabled": mean([g for g in gammas if g > 0]),
        "max_gamma": max(gammas) if gammas else 0,
        "mean_acceptance_rate": mean(accept),
        "median_acceptance_rate": median(accept),
        "accepted_tokens": sum(accepted),
        "proposed_tokens": sum(proposed),
        "accepted_per_step": mean(accepted),
        "draft_ms": draft_ms,
        "scoring_ms": scoring_ms,
        "verify_ms": verify_ms,
        "disabled_ms": disabled_ms,
        "draft_share_pct": draft_ms / total_ms * 100.0 if total_ms else 0.0,
        "scoring_share_pct": scoring_ms / total_ms * 100.0 if total_ms else 0.0,
        "verify_share_pct": verify_ms / total_ms * 100.0 if total_ms else 0.0,
        "disabled_share_pct": disabled_ms / total_ms * 100.0 if total_ms else 0.0,
        "mean_batch_size": mean(batch),
        "max_batch_size": max(batch) if batch else 0,
        "mean_queue_len": mean(queue),
        "max_queue_len": max(queue) if queue else 0,
        "mean_context_length": mean(context),
        "max_context_length": max(context) if context else 0,
        "min_free_gpu_blocks": min(free_blocks) if free_blocks else 0,
    }


def summarize_trace(trace):
    windows = trace.get("windows", []) if trace else []
    active = [
        w for w in windows
        if w.get("arrivals", 0) or w.get("completed", 0)
        or w.get("mean_e2el_ms", 0) or w.get("p95_e2el_ms", 0)
    ]
    high = [w for w in active if float(w.get("mean_e2el_ms", 0.0) or 0.0) > 10000.0]
    return {
        "active_windows": len(active),
        "zero_completion_windows": sum(1 for w in active if int(w.get("completed", 0) or 0) == 0),
        "mean_completion_rps": mean([float(w.get("completion_rps", 0.0) or 0.0) for w in active]),
        "high_latency_window_ratio": len(high) / len(active) if active else 0.0,
    }


def result_key(path):
    name = os.path.basename(path)
    match = re.match(
        r"benchmark_(?P<sub_strategy>.+?)_(?P<dataset>alpaca|sharegpt|specbench)_"
        r"(?P<num_prompts>\d+)_(?P<rate>[0-9.]+)_",
        name,
    )
    if not match:
        return None
    sub_strategy = match.group("sub_strategy")
    method = {
        "nospec_3": "nospec",
        "deep_3": "deep_3",
        "ucb_3": "banditspec",
        "smart_spec_3": "dsd",
        "ada_bin_greedy_3": "nightjar",
    }.get(sub_strategy, sub_strategy)
    return match.group("dataset"), match.group("rate"), method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="benchmark_results/vicuna13B")
    parser.add_argument("--diag-dir", default="benchmark_results/vicuna13B_diagnostics")
    parser.add_argument("--output", default="benchmark_results/vicuna13B_diagnostics/summary.csv")
    args = parser.parse_args()

    result_files = glob.glob(os.path.join(args.result_dir, "**", "benchmark_*.json"), recursive=True)
    rows = []
    for result_path in sorted(result_files):
        key = result_key(result_path)
        if key is None:
            continue
        dataset, rate, method = key
        result = load_json(result_path)

        # Prefer explicit diagnostic names emitted by run_table6_vicuna13b.sh.
        candidates = glob.glob(os.path.join(args.diag_dir, f"{dataset}_{rate}_*events.jsonl"))
        event_path = ""
        for c in candidates:
            base = os.path.basename(c)
            if method == "deep_3" and "_sd_deep3_" in base:
                event_path = c
            elif method == "dsd" and "_dsd_smart_spec_" in base:
                event_path = c
            elif method == "banditspec" and "_banditspec_ucb_" in base:
                event_path = c
            elif method == "nospec" and "_nospec_" in base:
                event_path = c
            elif method == "nightjar" and "_nightjar_ada_bin_greedy_" in base:
                event_path = c
        trace_path = event_path.replace("_events.jsonl", "_trace_summary.json") if event_path else ""

        events = summarize_events(load_events(event_path))
        trace = summarize_trace(load_json(trace_path) if trace_path and os.path.exists(trace_path) else result.get("trace_summary", {}))

        row = {
            "dataset": dataset,
            "rate": rate,
            "method": method,
            "result_json": result_path,
            "event_log": event_path,
            "completed": result.get("completed", ""),
            "duration_s": result.get("duration", ""),
            "total_token_throughput": result.get("total_token_throughput", ""),
            "mean_e2el_ms": result.get("mean_e2el_ms", ""),
            "p99_e2el_ms": result.get("p99_e2el_ms", ""),
            "avg_input_len": mean(result.get("input_lens", [])),
            "avg_expected_output_len": (result.get("expected_total_output_tokens", 0) or 0) / max(1, int(result.get("num_prompts", 1) or 1)),
            **events,
            **trace,
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
