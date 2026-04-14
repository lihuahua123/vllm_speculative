#!/usr/bin/env bash

set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/stress_tests}"
RUN_LOG="${RUN_LOG:-$RESULT_DIR/stress_tests.log}"
SUMMARY_CSV="${SUMMARY_CSV:-$RESULT_DIR/stress_tests_summary.csv}"
TRACE_DIR="${TRACE_DIR:-$RESULT_DIR/traces}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/deep05b}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/sharegpt.json}"
SUB_STRATEGY="${SUB_STRATEGY:-epsilon_greedy_with_offload}"
SELECT_STRATEGY="${SELECT_STRATEGY:-capacity}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
NUM_PROMPTS="${NUM_PROMPTS:-430}"
START_INDEX="${START_INDEX:-0}"
SPECULATIVE_LEN="${SPECULATIVE_LEN:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-4112}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
INCREASE_BLOCK_THRESHOLD="${INCREASE_BLOCK_THRESHOLD:-2000}"
DECREASE_BLOCK_THRESHOLD="${DECREASE_BLOCK_THRESHOLD:-2000}"
PERSIST_STEPS="${PERSIST_STEPS:-3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SPECULATIVE_DRAFT_TP_SIZE="${SPECULATIVE_DRAFT_TP_SIZE:-1}"
BURSTINESS="${BURSTINESS:-1.0}"
ENABLE_TRACE="${ENABLE_TRACE:-True}"
SAVE_TRACE="${SAVE_TRACE:-False}"
SEED="${SEED:-42}"
TRACE_WINDOW_SEC="${TRACE_WINDOW_SEC:-1.0}"

mkdir -p "$RESULT_DIR" "$TRACE_DIR"
rm -f "$RUN_LOG" "$SUMMARY_CSV"
printf "pattern,result_json,event_log,trace_summary\n" > "$SUMMARY_CSV"

generate_trace() {
    local pattern="$1"
    local trace_path="$TRACE_DIR/${pattern}.json"
    python "$ROOT_DIR/benchmarks/generate_nightjar_traces.py" \
        --pattern "$pattern" \
        --output "$trace_path"
    printf "%s" "$trace_path"
}

run_one() {
    local pattern="$1"
    local case_dir="$RESULT_DIR/$pattern"
    local event_log="$case_dir/nightjar_events.jsonl"
    local trace_summary="$case_dir/trace_summary.json"
    local trace_plan

    mkdir -p "$case_dir"
    rm -f "$event_log" "$trace_summary"
    export NIGHTJAR_EVENT_LOG_PATH="$event_log"

    trace_plan="$(generate_trace "$pattern")"

    local before_file after_file result_json
    before_file="$(mktemp)"
    after_file="$(mktemp)"
    find "$case_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$before_file"

    local cmd=(
        python "$ROOT_DIR/run_benchmark_tests.py"
        --strategy ilp
        --sub-strategy "$SUB_STRATEGY"
        --select-strategy "$SELECT_STRATEGY"
        --model "$MODEL_NAME"
        --draft-model "$DRAFT_MODEL_NAME"
        --host "$HOST"
        --port "$PORT"
        --dataset-name "$DATASET_NAME"
        --dataset-path "$DATASET_PATH"
        --speculative-len "$SPECULATIVE_LEN"
        --num-prompts "$NUM_PROMPTS"
        --request-rates 1
        --start-index "$START_INDEX"
        --output-len "$OUTPUT_LEN"
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
        --trace-plan "$trace_plan"
        --trace-window-sec "$TRACE_WINDOW_SEC"
        --export-trace-summary "$trace_summary"
        --result-dir "$case_dir"
        --seed "$SEED"
    )

    {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pattern=$pattern start"
        "${cmd[@]}"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pattern=$pattern done"
    } >> "$RUN_LOG" 2>&1

    find "$case_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$after_file"
    result_json="$(comm -13 "$before_file" "$after_file" | tail -n 1)"
    if [[ -z "$result_json" ]]; then
        result_json="$(find "$case_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort | tail -n 1)"
    fi
    rm -f "$before_file" "$after_file"

    printf "%s,%s,%s,%s\n" \
        "$pattern" "$result_json" "$event_log" "$trace_summary" \
        >> "$SUMMARY_CSV"
}

run_one burst_spike
run_one high_low_oscillation
run_one sync_migration_worst_case

echo "Run log: $RUN_LOG"
echo "Summary CSV: $SUMMARY_CSV"
