#!/usr/bin/env bash

set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/t_persist_sensitivity}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULT_DIR/t_persist_sensitivity_summary.csv}"
RUN_LOG="${RUN_LOG:-$RESULT_DIR/t_persist_sensitivity.log}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/deep05b}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/sharegpt.json}"
SUB_STRATEGY="${SUB_STRATEGY:-epsilon_greedy_with_offload}"
SELECT_STRATEGY="${SELECT_STRATEGY:-capacity}"

NUM_PROMPTS="${NUM_PROMPTS:-410}"
REQUEST_RATE="${REQUEST_RATE:-20}"
START_INDEX="${START_INDEX:-0}"
SPECULATIVE_LEN="${SPECULATIVE_LEN:-4}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-4112}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
INCREASE_BLOCK_THRESHOLD="${INCREASE_BLOCK_THRESHOLD:-2000}"
DECREASE_BLOCK_THRESHOLD="${DECREASE_BLOCK_THRESHOLD:-2000}"
BURSTINESS="${BURSTINESS:-1.0}"
ENABLE_TRACE="${ENABLE_TRACE:-True}"
SAVE_TRACE="${SAVE_TRACE:-False}"
SEED="${SEED:-42}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SPECULATIVE_DRAFT_TP_SIZE="${SPECULATIVE_DRAFT_TP_SIZE:-1}"

if [ "$#" -gt 0 ]; then
    PERSIST_VALUES=("$@")
else
    PERSIST_VALUES=(1 2 3 5 8)
fi

mkdir -p "$RESULT_DIR"
rm -f "$RUN_LOG" "$SUMMARY_CSV"
printf "persist_steps,total_token_throughput,mean_e2el_ms,offload_count,result_json,event_log\n" > "$SUMMARY_CSV"

for persist in "${PERSIST_VALUES[@]}"; do
    EVENT_LOG="$RESULT_DIR/nightjar_events_persist_${persist}.jsonl"
    rm -f "$EVENT_LOG"
    export NIGHTJAR_EVENT_LOG_PATH="$EVENT_LOG"

    before_file="$(mktemp)"
    after_file="$(mktemp)"
    find "$RESULT_DIR" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$before_file"

    python "$ROOT_DIR/run_benchmark_tests.py" \
        --strategy ilp \
        --sub-strategy "$SUB_STRATEGY" \
        --select-strategy "$SELECT_STRATEGY" \
        --model "$MODEL_NAME" \
        --draft-model "$DRAFT_MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$DATASET_PATH" \
        --speculative-len "$SPECULATIVE_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rates "$REQUEST_RATE" \
        --start-index "$START_INDEX" \
        --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE" \
        --enable-trace "$ENABLE_TRACE" \
        --save-trace "$SAVE_TRACE" \
        --burstiness "$BURSTINESS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --speculative-draft-tensor-parallel-size "$SPECULATIVE_DRAFT_TP_SIZE" \
        --increase-block-threshold "$INCREASE_BLOCK_THRESHOLD" \
        --decrease-block-threshold "$DECREASE_BLOCK_THRESHOLD" \
        --persist-steps "$persist" \
        --result-dir "$RESULT_DIR" \
        --seed "$SEED" \
        >> "$RUN_LOG" 2>&1

    find "$RESULT_DIR" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$after_file"
    result_json="$(comm -13 "$before_file" "$after_file" | tail -n 1)"
    rm -f "$before_file" "$after_file"

    python - "$persist" "$result_json" "$EVENT_LOG" "$SUMMARY_CSV" <<'PY'
import csv
import json
import os
import sys

persist = int(sys.argv[1])
result_json = sys.argv[2]
event_log = sys.argv[3]
summary_csv = sys.argv[4]

throughput = ""
latency = ""
offload_count = 0

if result_json and os.path.exists(result_json):
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    throughput = data.get("total_token_throughput", "")
    latency = data.get("mean_e2el_ms", "")

if event_log and os.path.exists(event_log):
    with open(event_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (record.get("event_type") == "memory_policy_decision"
                    and record.get("action") == "expand_kv_cache"):
                offload_count += 1

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        persist,
        throughput,
        latency,
        offload_count,
        result_json,
        event_log,
    ])

print(
    f"persist_steps={persist}, total_token_throughput={throughput}, "
    f"mean_e2el_ms={latency}, offload_count={offload_count}"
)
PY
done

echo "Summary written to $SUMMARY_CSV"
