#!/usr/bin/env bash

set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/revision_dynamic_behavior}"
MANIFEST_PATH="${MANIFEST_PATH:-$RESULT_DIR/dynamic_behavior_manifest.json}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULT_DIR/dynamic_behavior_runs.csv}"
RUN_LOG="${RUN_LOG:-$RESULT_DIR/dynamic_behavior.log}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/deep05b}"
STRATEGY="${STRATEGY:-ilp}"
SUB_STRATEGY="${SUB_STRATEGY:-ada_bin_greedy}"
SELECT_STRATEGY="${SELECT_STRATEGY:-capacity}"

SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-/root/autodl-tmp/sharegpt.json}"
ALPACA_DATASET_PATH="${ALPACA_DATASET_PATH:-tatsu-lab/alpaca}"
SPECBENCH_DATASET_PATH="${SPECBENCH_DATASET_PATH:-/root/autodl-tmp/nightjar/vllm_speculative/question_shuffled.jsonl}"

NUM_PROMPTS="${NUM_PROMPTS:-410}"
START_INDEX="${START_INDEX:-0}"
SPECULATIVE_LEN="${SPECULATIVE_LEN:-3}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-4112}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
INCREASE_BLOCK_THRESHOLD="${INCREASE_BLOCK_THRESHOLD:-2000}"
DECREASE_BLOCK_THRESHOLD="${DECREASE_BLOCK_THRESHOLD:-2000}"
PERSIST_STEPS="${PERSIST_STEPS:-3}"
BURSTINESS="${BURSTINESS:-1.0}"
ENABLE_TRACE="${ENABLE_TRACE:-True}"
SAVE_TRACE="${SAVE_TRACE:-False}"
SEED="${SEED:-42}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SPECULATIVE_DRAFT_TP_SIZE="${SPECULATIVE_DRAFT_TP_SIZE:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-}"

LOW_RATE="${LOW_RATE:-5}"
MED_RATE="${MED_RATE:-10}"
HIGH_RATE="${HIGH_RATE:-20}"

ANALYZE_AT_END="${ANALYZE_AT_END:-True}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$RESULT_DIR/figs}"
ANALYSIS_PREFIX="${ANALYSIS_PREFIX:-revision_dynamic_behavior}"

mkdir -p "$RESULT_DIR"
rm -f "$RUN_LOG" "$SUMMARY_CSV" "$MANIFEST_PATH"

printf "dataset,load,request_rate,result_json,event_log,label\n" > "$SUMMARY_CSV"

require_local_file() {
    local path="$1"
    local name="$2"
    if [[ -z "$path" || ! -f "$path" ]]; then
        echo "Missing $name file: $path" >&2
        exit 1
    fi
}

require_nonempty() {
    local value="$1"
    local name="$2"
    if [[ -z "$value" ]]; then
        echo "Missing required setting: $name" >&2
        exit 1
    fi
}

require_local_file "$SHAREGPT_DATASET_PATH" "SHAREGPT_DATASET_PATH"
require_nonempty "$ALPACA_DATASET_PATH" "ALPACA_DATASET_PATH"
require_local_file "$SPECBENCH_DATASET_PATH" "SPECBENCH_DATASET_PATH"

MANIFEST_ENTRIES_FILE="$(mktemp)"
cleanup() {
    rm -f "$MANIFEST_ENTRIES_FILE"
}
trap cleanup EXIT

run_one() {
    local dataset_name="$1"
    local dataset_path="$2"
    local load_name="$3"
    local request_rate="$4"

    local combo_dir="$RESULT_DIR/${dataset_name}_${load_name}"
    local label="${dataset_name}_${load_name}"
    local event_log="$combo_dir/nightjar_events.jsonl"

    mkdir -p "$combo_dir"
    rm -f "$event_log"
    export NIGHTJAR_EVENT_LOG_PATH="$event_log"

    local before_file after_file result_json
    before_file="$(mktemp)"
    after_file="$(mktemp)"
    find "$combo_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$before_file"

    local cmd=(
        python "$ROOT_DIR/run_benchmark_tests.py"
        --strategy "$STRATEGY"
        --sub-strategy "$SUB_STRATEGY"
        --select-strategy "$SELECT_STRATEGY"
        --model "$MODEL_NAME"
        --draft-model "$DRAFT_MODEL_NAME"
        --dataset-name "$dataset_name"
        --dataset-path "$dataset_path"
        --speculative-len "$SPECULATIVE_LEN"
        --num-prompts "$NUM_PROMPTS"
        --request-rates "$request_rate"
        --start-index "$START_INDEX"
        --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE"
        --enable-trace "$ENABLE_TRACE"
        --save-trace "$SAVE_TRACE"
        --burstiness "$BURSTINESS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --speculative-draft-tensor-parallel-size "$SPECULATIVE_DRAFT_TP_SIZE"
        --increase-block-threshold "$INCREASE_BLOCK_THRESHOLD"
        --decrease-block-threshold "$DECREASE_BLOCK_THRESHOLD"
        --persist-steps "$PERSIST_STEPS"
        --result-dir "$combo_dir"
        --seed "$SEED"
    )
    if [[ -n "$OUTPUT_LEN" ]]; then
        cmd+=(--output-len "$OUTPUT_LEN")
    fi

    {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] dataset=$dataset_name load=$load_name rate=$request_rate start"
        "${cmd[@]}"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] dataset=$dataset_name load=$load_name rate=$request_rate done"
    } >> "$RUN_LOG" 2>&1

    find "$combo_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$after_file"
    result_json="$(comm -13 "$before_file" "$after_file" | tail -n 1)"
    if [[ -z "$result_json" ]]; then
        result_json="$(find "$combo_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort | tail -n 1)"
    fi
    rm -f "$before_file" "$after_file"

    printf "%s,%s,%s,%s,%s,%s\n" \
        "$dataset_name" "$load_name" "$request_rate" "$result_json" "$event_log" "$label" \
        >> "$SUMMARY_CSV"

    python - "$MANIFEST_ENTRIES_FILE" "$event_log" "$label" "$dataset_name" "$load_name" "$request_rate" "$result_json" <<'PY'
import json
import sys

manifest_entries_file, event_log, label, dataset, load_name, rate, result_json = sys.argv[1:8]
entry = {
    "path": event_log,
    "label": label,
    "dataset": dataset,
    "load": f"{load_name} ({rate} req/s)",
    "request_rate": float(rate),
    "result_json": result_json,
}
with open(manifest_entries_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
}

run_one "sharegpt" "$SHAREGPT_DATASET_PATH" "low" "$LOW_RATE"
run_one "sharegpt" "$SHAREGPT_DATASET_PATH" "med" "$MED_RATE"
run_one "sharegpt" "$SHAREGPT_DATASET_PATH" "high" "$HIGH_RATE"

run_one "alpaca" "$ALPACA_DATASET_PATH" "low" "$LOW_RATE"
run_one "alpaca" "$ALPACA_DATASET_PATH" "med" "$MED_RATE"
run_one "alpaca" "$ALPACA_DATASET_PATH" "high" "$HIGH_RATE"

run_one "specbench" "$SPECBENCH_DATASET_PATH" "low" "$LOW_RATE"
run_one "specbench" "$SPECBENCH_DATASET_PATH" "med" "$MED_RATE"
run_one "specbench" "$SPECBENCH_DATASET_PATH" "high" "$HIGH_RATE"

python - "$MANIFEST_ENTRIES_FILE" "$MANIFEST_PATH" <<'PY'
import json
import sys

entries_path, manifest_path = sys.argv[1:3]
entries = []
with open(entries_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)
PY

if [[ "$ANALYZE_AT_END" == "True" ]]; then
    python "$ROOT_DIR/exps/plot_dynamic_behavior_report.py" \
        --manifest "$MANIFEST_PATH" \
        --output-dir "$ANALYSIS_OUTPUT_DIR" \
        --prefix "$ANALYSIS_PREFIX" \
        >> "$RUN_LOG" 2>&1
fi

echo "Run log: $RUN_LOG"
echo "Summary CSV: $SUMMARY_CSV"
echo "Manifest: $MANIFEST_PATH"
if [[ "$ANALYZE_AT_END" == "True" ]]; then
    echo "Figures: $ANALYSIS_OUTPUT_DIR"
fi
