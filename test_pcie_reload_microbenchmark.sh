#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/pcie_reload_microbenchmark}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULT_DIR/pcie_transfer_summary.csv}"
RUN_LOG="${RUN_LOG:-$RESULT_DIR/pcie_transfer.log}"

DIRECTION="${DIRECTION:-h2d}"
DEVICE="${DEVICE:-0}"
ITERS="${ITERS:-100}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
PINNED="${PINNED:-True}"

if [ "$#" -gt 0 ]; then
    CONCURRENCY_VALUES=("$@")
else
    CONCURRENCY_VALUES=(1 2 4 8)
fi

SIZE_VALUES=(${SIZE_VALUES:-64 256 512 1024})

mkdir -p "$RESULT_DIR"
rm -f "$SUMMARY_CSV" "$RUN_LOG"
printf "direction,size_mb,concurrency,pinned,dispatch_overhead_us,per_transfer_mean_ms,per_transfer_p95_ms,batch_completion_mean_ms,batch_completion_p95_ms,draft_available_mean_ms,agg_gib_s,result_json\n" > "$SUMMARY_CSV"

for size_mb in "${SIZE_VALUES[@]}"; do
    for concurrency in "${CONCURRENCY_VALUES[@]}"; do
        result_json="$RESULT_DIR/pcie_${DIRECTION}_${size_mb}mb_c${concurrency}.json"
        cmd=(
            python "$ROOT_DIR/benchmarks/pcie_transfer_microbenchmark.py"
            --direction "$DIRECTION"
            --size-mb "$size_mb"
            --concurrency "$concurrency"
            --iters "$ITERS"
            --warmup-iters "$WARMUP_ITERS"
            --device "$DEVICE"
            --output "$result_json"
        )
        if [[ "$PINNED" == "True" ]]; then
            cmd+=(--pinned)
        fi

        {
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start size_mb=$size_mb concurrency=$concurrency"
            "${cmd[@]}"
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done size_mb=$size_mb concurrency=$concurrency"
        } >> "$RUN_LOG" 2>&1

        python - "$result_json" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys

result_json = sys.argv[1]
summary_csv = sys.argv[2]

with open(result_json, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        data["direction"],
        data["size_mb"],
        data["concurrency"],
        data["pinned"],
        data.get("dispatch_overhead_us", 0.0),
        data.get("per_transfer_elapsed_ms", data["latency_ms"])["mean"],
        data.get("per_transfer_elapsed_ms", data["latency_ms"])["p95"],
        data.get("batch_completion_ms", data["wall_ms"])["mean"],
        data.get("batch_completion_ms", data["wall_ms"])["p95"],
        data.get("draft_available_ms", data["wall_ms"])["mean"],
        data["aggregate_bandwidth_gib_s"],
        result_json,
    ])
PY
    done
done

echo "Summary written to $SUMMARY_CSV"
echo "Run log written to $RUN_LOG"
