#!/usr/bin/env bash

set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/long_context_r3_1}"
RUN_LOG="${RUN_LOG:-$RESULT_DIR/long_context_r3_1.log}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULT_DIR/long_context_r3_1_summary.csv}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/Llama-3.1-8B-Instruct}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/Llama-3.2-1B}"
STRATEGIES="${STRATEGIES:-epsilon_greedy epsilon_greedy_with_offload}"
SPECULATIVE_LEN="${SPECULATIVE_LEN:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SELECT_STRATEGY="${SELECT_STRATEGY:-capacity}"
DATASET_NAME="${DATASET_NAME:-random}"
DATASET_PATH="${DATASET_PATH:-unused_for_random_dataset}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
TRACE_WINDOW_SEC="${TRACE_WINDOW_SEC:-1.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SPECULATIVE_DRAFT_TP_SIZE="${SPECULATIVE_DRAFT_TP_SIZE:-1}"
ENABLE_TRACE="${ENABLE_TRACE:-True}"
SAVE_TRACE="${SAVE_TRACE:-False}"
SEED="${SEED:-42}"

# Keep some headroom for output tokens and special tokens. The bucket names stay
# 8K/32K/64K/128K, but the actual prompt length is slightly smaller on purpose.
# 8k
DEFAULT_CONTEXT_BUCKETS=(64K 128K)
if [ "$#" -gt 0 ]; then
    CONTEXT_BUCKETS=("$@")
else
    CONTEXT_BUCKETS=("${DEFAULT_CONTEXT_BUCKETS[@]}")
fi

mkdir -p "$RESULT_DIR"
rm -f "$RUN_LOG"

if [ "${OUTPUT_LEN}" -lt 1 ]; then
    echo "OUTPUT_LEN must be >= 1 for vLLM /generate; got ${OUTPUT_LEN}" >&2
    exit 1
fi

if [ ! -f "$SUMMARY_CSV" ]; then
    printf "%s\n" \
      "context_bucket,max_model_len,input_len,output_len,strategy,request_rate,num_prompts,num_gpu_blocks_override,increase_block_threshold,decrease_block_threshold,persist_steps,completed,total_token_throughput,mean_ttft_ms,p99_ttft_ms,mean_e2el_ms,p99_e2el_ms,avg_queue_len,p95_queue_len,max_queue_len,avg_num_swapped,max_num_swapped,min_free_gpu_blocks,draft_transfer_done,kv_expand_events,kv_migration_events,speculative_steps,result_json,trace_summary_json,event_log" \
      > "$SUMMARY_CSV"
fi

set_profile() {
    local bucket="$1"
    case "$bucket" in
        8k|8K)
            CONTEXT_BUCKET="8k"
            MAX_MODEL_LEN=8192
            RANDOM_INPUT_LEN=7680
            NUM_PROMPTS=24
            REQUEST_RATES=(1 2 4)
            NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE_8K:-4608}"
            INCREASE_BLOCK_THRESHOLD=256
            DECREASE_BLOCK_THRESHOLD=192
            PERSIST_STEPS=2
            ;;
        32k|32K)
            CONTEXT_BUCKET="32k"
            MAX_MODEL_LEN=32768
            RANDOM_INPUT_LEN=32256
            NUM_PROMPTS=16
            REQUEST_RATES=(0.5 1 2)
            NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE_32K:-4480}"
            INCREASE_BLOCK_THRESHOLD=384
            DECREASE_BLOCK_THRESHOLD=256
            PERSIST_STEPS=3
            ;;
        64k|64K)
            CONTEXT_BUCKET="64k"
            MAX_MODEL_LEN=65536
            RANDOM_INPUT_LEN=65024
            NUM_PROMPTS=12
            REQUEST_RATES=(0.25 0.5 1)
            NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE_64K:-4608}"
            INCREASE_BLOCK_THRESHOLD=512
            DECREASE_BLOCK_THRESHOLD=384
            PERSIST_STEPS=4
            ;;
        128k|128K)
            CONTEXT_BUCKET="128k"
            MAX_MODEL_LEN=131072
            RANDOM_INPUT_LEN=130048
            NUM_PROMPTS=8
            REQUEST_RATES=(0.125 0.25 0.5)
            NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE_128K:-3968}"
            INCREASE_BLOCK_THRESHOLD=640
            DECREASE_BLOCK_THRESHOLD=512
            PERSIST_STEPS=5
            ;;
        *)
            echo "Unsupported context bucket: $bucket" >&2
            exit 1
            ;;
    esac
}

summarize_run() {
    local result_json="$1"
    local trace_summary_json="$2"
    local event_log="$3"
    local strategy="$4"
    local request_rate="$5"

    python - "$SUMMARY_CSV" "$CONTEXT_BUCKET" "$MAX_MODEL_LEN" "$RANDOM_INPUT_LEN" \
        "$OUTPUT_LEN" "$strategy" "$request_rate" "$NUM_PROMPTS" "$NUM_GPU_BLOCKS_OVERRIDE" \
        "$INCREASE_BLOCK_THRESHOLD" "$DECREASE_BLOCK_THRESHOLD" "$PERSIST_STEPS" \
        "$result_json" "$trace_summary_json" "$event_log" <<'PY'
import csv
import json
import math
import os
import statistics
import sys

summary_csv = sys.argv[1]
context_bucket = sys.argv[2]
max_model_len = int(sys.argv[3])
input_len = int(sys.argv[4])
output_len = int(sys.argv[5])
strategy = sys.argv[6]
request_rate = sys.argv[7]
num_prompts = int(sys.argv[8])
num_gpu_blocks_override = int(sys.argv[9])
increase_block_threshold = int(sys.argv[10])
decrease_block_threshold = int(sys.argv[11])
persist_steps = int(sys.argv[12])
result_json = sys.argv[13]
trace_summary_json = sys.argv[14]
event_log = sys.argv[15]

def p99(values):
    if not values:
        return ""
    values = sorted(values)
    idx = min(len(values) - 1, math.ceil(len(values) * 0.99) - 1)
    return values[idx]

completed = throughput = mean_ttft_ms = p99_ttft_ms = ""
mean_e2el_ms = p99_e2el_ms = ""

if os.path.exists(result_json):
    with open(result_json, "r", encoding="utf-8") as f:
        result = json.load(f)
    completed = result.get("completed", "")
    throughput = result.get("total_token_throughput", "")
    mean_ttft_ms = result.get("mean_ttft_ms", "")
    p99_ttft_ms = result.get("p99_ttft_ms", "")
    mean_e2el_ms = result.get("mean_e2el_ms", "")
    p99_e2el_ms = result.get("p99_e2el_ms", "")

queue_lens = []
num_swapped = []
free_gpu_blocks = []
draft_transfer_done = 0
kv_expand_events = 0
kv_migration_events = 0
speculative_steps = 0

if os.path.exists(event_log):
    with open(event_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            event_type = record.get("event_type")
            if event_type == "speculative_step":
                speculative_steps += 1
                if "queue_len" in record:
                    queue_lens.append(record["queue_len"])
                if "num_swapped" in record:
                    num_swapped.append(record["num_swapped"])
                if "free_gpu_blocks" in record:
                    free_gpu_blocks.append(record["free_gpu_blocks"])
            elif event_type == "draft_transfer_done":
                draft_transfer_done += 1
            elif event_type == "kv_block_migration":
                kv_migration_events += 1
            elif event_type == "memory_policy_decision" and record.get("action") == "expand_kv_cache":
                kv_expand_events += 1

avg_queue_len = statistics.mean(queue_lens) if queue_lens else ""
p95_queue_len = ""
if queue_lens:
    q = sorted(queue_lens)
    idx = min(len(q) - 1, math.ceil(len(q) * 0.95) - 1)
    p95_queue_len = q[idx]
max_queue_len = max(queue_lens) if queue_lens else ""
avg_num_swapped = statistics.mean(num_swapped) if num_swapped else ""
max_num_swapped = max(num_swapped) if num_swapped else ""
min_free_gpu_blocks = min(free_gpu_blocks) if free_gpu_blocks else ""

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        context_bucket,
        max_model_len,
        input_len,
        output_len,
        strategy,
        request_rate,
        num_prompts,
        num_gpu_blocks_override,
        increase_block_threshold,
        decrease_block_threshold,
        persist_steps,
        completed,
        throughput,
        mean_ttft_ms,
        p99_ttft_ms,
        mean_e2el_ms,
        p99_e2el_ms,
        avg_queue_len,
        p95_queue_len,
        max_queue_len,
        avg_num_swapped,
        max_num_swapped,
        min_free_gpu_blocks,
        draft_transfer_done,
        kv_expand_events,
        kv_migration_events,
        speculative_steps,
        result_json,
        trace_summary_json,
        event_log,
    ])

print(
    f"context={context_bucket} strategy={strategy} qps={request_rate} "
    f"throughput={throughput} mean_ttft_ms={mean_ttft_ms} "
    f"mean_e2el_ms={mean_e2el_ms} max_queue_len={max_queue_len} "
    f"draft_transfer_done={draft_transfer_done} kv_migration_events={kv_migration_events}"
)
PY
}

run_one() {
    local strategy="$1"
    local request_rate="$2"

    local run_prefix="${CONTEXT_BUCKET}_${strategy}_qps_${request_rate}"
    local event_log="$RESULT_DIR/${run_prefix}_events.jsonl"
    local trace_summary_json="$RESULT_DIR/${run_prefix}_trace_summary.json"
    local result_json="$RESULT_DIR/benchmark_${run_prefix}.json"

    rm -f "$event_log" "$trace_summary_json" "$result_json"
    local before_file
    local after_file
    before_file="$(mktemp)"
    after_file="$(mktemp)"
    find "$RESULT_DIR" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$before_file"

    (
        cd "$ROOT_DIR"
        python run_benchmark_tests.py \
            --strategy ilp \
            --sub-strategy "$strategy" \
            --select-strategy "$SELECT_STRATEGY" \
            --model "$MODEL_NAME" \
            --draft-model "$DRAFT_MODEL_NAME" \
            --dataset-name "$DATASET_NAME" \
            --dataset-path "$DATASET_PATH" \
            --speculative-len "$SPECULATIVE_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --request-rates "$request_rate" \
            --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE" \
            --enable-trace "$ENABLE_TRACE" \
            --save-trace "$SAVE_TRACE" \
            --burstiness 1.0 \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --speculative-draft-tensor-parallel-size "$SPECULATIVE_DRAFT_TP_SIZE" \
            --increase-block-threshold "$INCREASE_BLOCK_THRESHOLD" \
            --decrease-block-threshold "$DECREASE_BLOCK_THRESHOLD" \
            --persist-steps "$PERSIST_STEPS" \
            --result-dir "$RESULT_DIR" \
            --seed "$SEED" \
            --trace-window-sec "$TRACE_WINDOW_SEC" \
            --export-step-log "$event_log" \
            --export-trace-summary "$trace_summary_json" \
            --random-input-len "$RANDOM_INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --random-range-ratio 1.0 \
            --random-prefix-len 0 \
            >>"$RUN_LOG" 2>&1
    )

    find "$RESULT_DIR" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$after_file"
    result_json="$(comm -13 "$before_file" "$after_file" | tail -n 1)"
    rm -f "$before_file" "$after_file"

    if [ -z "$result_json" ]; then
        echo "Failed to locate benchmark result json for $run_prefix" | tee -a "$RUN_LOG"
        exit 1
    fi

    summarize_run "$result_json" "$trace_summary_json" "$event_log" "$strategy" "$request_rate"
}

for bucket in "${CONTEXT_BUCKETS[@]}"; do
    set_profile "$bucket"
    for strategy in $STRATEGIES; do
        for request_rate in "${REQUEST_RATES[@]}"; do
            printf "[run] bucket=%s strategy=%s qps=%s prompts=%s input_len=%s output_len=%s\n" \
                "$CONTEXT_BUCKET" "$strategy" "$request_rate" "$NUM_PROMPTS" "$RANDOM_INPUT_LEN" "$OUTPUT_LEN" \
                | tee -a "$RUN_LOG"
            run_one "$strategy" "$request_rate"
        done
        sleep 3
    done
done

echo "Summary written to $SUMMARY_CSV"
