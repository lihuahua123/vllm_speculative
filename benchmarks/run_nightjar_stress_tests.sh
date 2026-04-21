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
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$RESULT_DIR/figs}"
ANALYSIS_PREFIX="${ANALYSIS_PREFIX:-stress_disable}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/deep05b}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/sharegpt.json}"
SUB_STRATEGY="${SUB_STRATEGY:-epsilon_greedy_with_offload}"
SELECT_STRATEGY="${SELECT_STRATEGY:-capacity}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
NUM_PROMPTS="${NUM_PROMPTS:-600}"
START_INDEX="${START_INDEX:-0}"
SPECULATIVE_LEN="${SPECULATIVE_LEN:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-4112}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
INCREASE_BLOCK_THRESHOLD="${INCREASE_BLOCK_THRESHOLD:-2000}"
DECREASE_BLOCK_THRESHOLD="${DECREASE_BLOCK_THRESHOLD:-2000}"
PERSIST_STEPS="${PERSIST_STEPS:-3}"
STATIC_INCREASE_BLOCK_THRESHOLD="${STATIC_INCREASE_BLOCK_THRESHOLD:-999999}"
STATIC_DECREASE_BLOCK_THRESHOLD="${STATIC_DECREASE_BLOCK_THRESHOLD:-999999}"
STATIC_PERSIST_STEPS="${STATIC_PERSIST_STEPS:-999999}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SPECULATIVE_DRAFT_TP_SIZE="${SPECULATIVE_DRAFT_TP_SIZE:-1}"
BURSTINESS="${BURSTINESS:-1.0}"
ENABLE_TRACE="${ENABLE_TRACE:-True}"
SAVE_TRACE="${SAVE_TRACE:-False}"
SEED="${SEED:-42}"
TRACE_WINDOW_SEC="${TRACE_WINDOW_SEC:-1.0}"

mkdir -p "$RESULT_DIR" "$TRACE_DIR"
rm -f "$RUN_LOG" "$SUMMARY_CSV"
printf "pattern,variant,result_json,event_log,trace_summary\n" > "$SUMMARY_CSV"

generate_trace() {
    local trace_key="$1"
    local workload_pattern="$2"
    local trace_path="$TRACE_DIR/${trace_key}.json"
    python "$ROOT_DIR/benchmarks/generate_nightjar_traces.py" \
        --pattern "$workload_pattern" \
        --output "$trace_path"
    printf "%s" "$trace_path"
}

run_one() {
    local workload_pattern="$1"
    local variant="$2"
    local increase_threshold="$3"
    local decrease_threshold="$4"
    local persist_steps="$5"
    local case_id="${workload_pattern}_${variant}"
    local case_dir="$RESULT_DIR/$case_id"
    local event_log="$case_dir/nightjar_events.jsonl"
    local trace_summary="$case_dir/trace_summary.json"
    local trace_plan

    mkdir -p "$case_dir"
    rm -f "$event_log" "$trace_summary"
    export NIGHTJAR_EVENT_LOG_PATH="$event_log"

    trace_plan="$(generate_trace "$case_id" "$workload_pattern")"

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
        --increase-block-threshold "$increase_threshold"
        --decrease-block-threshold "$decrease_threshold"
        --persist-steps "$persist_steps"
        --trace-plan "$trace_plan"
        --trace-window-sec "$TRACE_WINDOW_SEC"
        --export-trace-summary "$trace_summary"
        --result-dir "$case_dir"
        --seed "$SEED"
    )

    {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pattern=$workload_pattern variant=$variant start"
        "${cmd[@]}"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pattern=$workload_pattern variant=$variant done"
    } >> "$RUN_LOG" 2>&1

    find "$case_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort > "$after_file"
    result_json="$(comm -13 "$before_file" "$after_file" | tail -n 1)"
    if [[ -z "$result_json" ]]; then
        result_json="$(find "$case_dir" -maxdepth 1 -type f -name 'benchmark_*.json' | sort | tail -n 1)"
    fi
    rm -f "$before_file" "$after_file"

    printf "%s,%s,%s,%s,%s\n" \
        "$case_id" "$variant" "$result_json" "$event_log" "$trace_summary" \
        >> "$SUMMARY_CSV"
}

run_pair() {
    local workload_pattern="$1"
    run_one "$workload_pattern" "elastic" \
        "$INCREASE_BLOCK_THRESHOLD" \
        "$DECREASE_BLOCK_THRESHOLD" \
        "$PERSIST_STEPS"
    run_one "$workload_pattern" "static_memory" \
        "$STATIC_INCREASE_BLOCK_THRESHOLD" \
        "$STATIC_DECREASE_BLOCK_THRESHOLD" \
        "$STATIC_PERSIST_STEPS"
}

run_pair burst_spike
run_pair high_low_oscillation
run_pair sync_migration_worst_case

python "$ROOT_DIR/exps/analyze_stress_disable_cases.py" \
    --summary-csv "$SUMMARY_CSV" \
    --trace-dir "$TRACE_DIR" \
    --output-dir "$ANALYSIS_OUTPUT_DIR" \
    --prefix "$ANALYSIS_PREFIX" \
    >> "$RUN_LOG" 2>&1

echo "Run log: $RUN_LOG"
echo "Summary CSV: $SUMMARY_CSV"
echo "Analysis dir: $ANALYSIS_OUTPUT_DIR"
