#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/pcie_reload_microbenchmark}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULT_DIR/pcie_transfer_summary.csv}"
MARKDOWN_TABLE="${MARKDOWN_TABLE:-$RESULT_DIR/pcie_transfer_summary.md}"
LATEX_TABLE="${LATEX_TABLE:-$RESULT_DIR/pcie_transfer_summary.tex}"
RUN_LOG="${RUN_LOG:-$RESULT_DIR/pcie_transfer.log}"

DIRECTION="${DIRECTION:-h2d}"
DEVICE="${DEVICE:-0}"
ITERS="${ITERS:-100}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
PINNED="${PINNED:-True}"
SIZE_VALUES=(${SIZE_VALUES:-64 256 512 1024})
LATEX_CONCURRENCY_VALUES=(${LATEX_CONCURRENCY_VALUES:-1 2 4 8})
COMPACT_TABLE="${COMPACT_TABLE:-False}"

if [ "$#" -gt 0 ]; then
    CONCURRENCY_VALUES=("$@")
else
    CONCURRENCY_VALUES=(1 2 4 8)
fi

mkdir -p "$RESULT_DIR"
rm -f "$SUMMARY_CSV" "$MARKDOWN_TABLE" "$LATEX_TABLE" "$RUN_LOG"
printf "direction,size_mb,concurrency,pinned,dispatch_overhead_us,per_transfer_mean_ms,per_transfer_p95_ms,batch_completion_mean_ms,batch_completion_p95_ms,draft_available_mean_ms,draft_available_p95_ms,agg_gib_s,result_json\n" > "$SUMMARY_CSV"

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
            printf 'CMD:'
            printf ' %q' "${cmd[@]}"
            printf '\n'
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
        data.get("draft_available_ms", data["wall_ms"])["p95"],
        data["aggregate_bandwidth_gib_s"],
        result_json,
    ])
PY
    done
done

python - "$SUMMARY_CSV" "$MARKDOWN_TABLE" "$LATEX_TABLE" "$COMPACT_TABLE" "${LATEX_CONCURRENCY_VALUES[@]}" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

summary_csv = Path(sys.argv[1])
markdown_table = Path(sys.argv[2])
latex_table = Path(sys.argv[3])
compact_table = sys.argv[4].lower() == "true"
latex_concurrency_values = {int(x) for x in sys.argv[5:]}

rows = []
with summary_csv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["size_mb"] = int(row["size_mb"])
        row["concurrency"] = int(row["concurrency"])
        for key in [
            "dispatch_overhead_us",
            "per_transfer_mean_ms",
            "per_transfer_p95_ms",
            "batch_completion_mean_ms",
            "batch_completion_p95_ms",
            "draft_available_mean_ms",
            "draft_available_p95_ms",
            "agg_gib_s",
        ]:
            row[key] = float(row[key])
        rows.append(row)

rows.sort(key=lambda r: (r["size_mb"], r["concurrency"]))

with markdown_table.open("w", encoding="utf-8") as f:
    f.write("# PCIe Reload Microbenchmark Summary\n\n")
    f.write(
        "Measured with pinned host memory and asynchronous H2D copies. "
        "`draft_available_mean_ms` is the batch wall-clock completion time, i.e. the time until all concurrent reloads finish.\n\n"
    )
    f.write("| Payload (MB) | Concurrency | Mean copy-event (ms) | P95 copy-event (ms) | Batch completion / draft-available (ms) | P95 draft-available (ms) | Aggregate bandwidth (GiB/s) |\n")
    f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        f.write(
            f"| {row['size_mb']} | {row['concurrency']} | {row['per_transfer_mean_ms']:.2f} | "
            f"{row['per_transfer_p95_ms']:.2f} | {row['draft_available_mean_ms']:.2f} | "
            f"{row['draft_available_p95_ms']:.2f} | {row['agg_gib_s']:.2f} |\n"
        )

latex_rows = [
    row for row in rows
    if row["concurrency"] in latex_concurrency_values
]
if compact_table:
    grouped = defaultdict(list)
    for row in latex_rows:
        grouped[row["size_mb"]].append(row)
    latex_rows = []
    for size_mb in sorted(grouped):
        candidates = sorted(grouped[size_mb], key=lambda r: r["concurrency"])
        if candidates:
            latex_rows.append(candidates[0])
        if len(candidates) > 1:
            latex_rows.append(candidates[-1])

with latex_table.open("w", encoding="utf-8") as f:
    f.write("% Auto-generated by test_pcie_reload_microbenchmark.sh\n")
    f.write("\\begin{table}[t]\n")
    f.write("\\centering\n")
    f.write("\\caption{PCIe reload microbenchmark with pinned host memory and concurrent H2D transfers. Mean copy-event duration reports CUDA-event elapsed time for an individual async copy, while draft-available time reports the wall-clock time until the full concurrent reload group completes.}\n")
    f.write("\\label{tab:response_pcie_reload}\n")
    f.write("\\resizebox{0.98\\linewidth}{!}{%\n")
    f.write("\\begin{tabular}{c|c|c|c|c}\n")
    f.write("\\hline\n")
    f.write("Payload & Concurrency & Mean copy-event (ms) & Draft-available time (ms) & Aggregate bandwidth (GiB/s) \\\\\n")
    f.write("\\hline\n")
    current_size = None
    for row in latex_rows:
        if current_size is not None and row["size_mb"] != current_size:
            f.write("\\hline\n")
        current_size = row["size_mb"]
        f.write(
            f"{row['size_mb']} MB & {row['concurrency']} & {row['per_transfer_mean_ms']:.2f} & "
            f"{row['draft_available_mean_ms']:.2f} & {row['agg_gib_s']:.2f} \\\\\n"
        )
    f.write("\\hline\n")
    f.write("\\end{tabular}%\n")
    f.write("}\n")
    f.write("\\end{table}\n")
PY

echo "Summary written to $SUMMARY_CSV"
echo "Markdown table written to $MARKDOWN_TABLE"
echo "LaTeX table written to $LATEX_TABLE"
echo "Run log written to $RUN_LOG"
