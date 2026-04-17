#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/deep05b}"
HOST="${HOST:-127.0.0.1}"
PORT_BASE="${PORT_BASE:-8110}"

NUM_PROMPTS="${NUM_PROMPTS:-350}"
REQUEST_RATES="${REQUEST_RATES:-5 10 15 20 25}"
START_INDEX="${START_INDEX:-0}"
OUTPUT_LEN="${OUTPUT_LEN:-150}"
MAX_SPECULATIVE_LEN="${MAX_SPECULATIVE_LEN:-3}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-4112}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SPECULATIVE_DRAFT_TP_SIZE="${SPECULATIVE_DRAFT_TP_SIZE:-1}"
BURSTINESS="${BURSTINESS:-1.0}"
SEED="${SEED:-42}"
ENABLE_TRACE="${ENABLE_TRACE:-False}"
SAVE_TRACE="${SAVE_TRACE:-False}"
EXPLORE="${EXPLORE:-False}"

# Static-memory cases use unreachable thresholds to disable elastic KV cache.
STATIC_INCREASE_BLOCK_THRESHOLD="${STATIC_INCREASE_BLOCK_THRESHOLD:-999999}"
STATIC_DECREASE_BLOCK_THRESHOLD="${STATIC_DECREASE_BLOCK_THRESHOLD:-999999}"
STATIC_PERSIST_STEPS="${STATIC_PERSIST_STEPS:-999999}"

# Nightjar full keeps the original elastic-memory thresholds.
ELASTIC_INCREASE_BLOCK_THRESHOLD="${ELASTIC_INCREASE_BLOCK_THRESHOLD:-150}"
ELASTIC_DECREASE_BLOCK_THRESHOLD="${ELASTIC_DECREASE_BLOCK_THRESHOLD:-100}"
ELASTIC_PERSIST_STEPS="${ELASTIC_PERSIST_STEPS:-3}"

DATASETS="${DATASETS:-sharegpt:/root/autodl-tmp/sharegpt.json alpaca:tatsu-lab/alpaca specbench:/root/autodl-tmp/nightjar/vllm_speculative/question_shuffled.jsonl}"

# Format: method_name:sub_strategy:select_strategy
BASELINE_CASES="${BASELINE_CASES:-smart_spec:smart_spec: banditspec:ucb: tetris:deep:capacity}"

RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/benchmark_results/fairness_ablation}"
MANIFEST_CSV="$RESULT_ROOT/run_manifest.csv"
GLOBAL_CSV="$ROOT_DIR/benchmark_results.csv"

mkdir -p "$RESULT_ROOT"
rm -f "$GLOBAL_CSV" "$MANIFEST_CSV"
printf "dataset,group_name,method_name,sub_strategy,select_strategy,elastic_memory,speculative_len,result_dir\n" > "$MANIFEST_CSV"

run_case() {
    local dataset_name="$1"
    local dataset_path="$2"
    local group_name="$3"
    local method_name="$4"
    local sub_strategy="$5"
    local select_strategy="$6"
    local elastic_memory="$7"
    local port="$8"

    local result_dir="$RESULT_ROOT/${dataset_name}/${group_name}/${method_name}"
    mkdir -p "$result_dir"

    local increase_threshold="$STATIC_INCREASE_BLOCK_THRESHOLD"
    local decrease_threshold="$STATIC_DECREASE_BLOCK_THRESHOLD"
    local persist_steps="$STATIC_PERSIST_STEPS"

    if [[ "$elastic_memory" == "true" ]]; then
        increase_threshold="$ELASTIC_INCREASE_BLOCK_THRESHOLD"
        decrease_threshold="$ELASTIC_DECREASE_BLOCK_THRESHOLD"
        persist_steps="$ELASTIC_PERSIST_STEPS"
    fi

    local cmd=(
        python "$ROOT_DIR/run_benchmark_tests.py"
        --strategy ilp
        --sub-strategy "$sub_strategy"
        --explore "$EXPLORE"
        --save-trace "$SAVE_TRACE"
        --model "$MODEL_NAME"
        --draft-model "$DRAFT_MODEL_NAME"
        --host "$HOST"
        --port "$port"
        --dataset-name "$dataset_name"
        --dataset-path "$dataset_path"
        --speculative-len "$MAX_SPECULATIVE_LEN"
        --num-prompts "$NUM_PROMPTS"
        --request-rates ${REQUEST_RATES}
        --start-index "$START_INDEX"
        --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE"
        --enable-trace "$ENABLE_TRACE"
        --burstiness "$BURSTINESS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$MAX_MODEL_LEN"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --speculative-draft-tensor-parallel-size "$SPECULATIVE_DRAFT_TP_SIZE"
        --increase-block-threshold "$increase_threshold"
        --decrease-block-threshold "$decrease_threshold"
        --persist-steps "$persist_steps"
        --result-dir "$result_dir"
        --seed "$SEED"
    )

    if [[ -n "$OUTPUT_LEN" ]]; then
        cmd+=(--output-len "$OUTPUT_LEN")
    fi

    if [[ -n "$select_strategy" ]]; then
        cmd+=(--select-strategy "$select_strategy")
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$dataset_name" "$group_name" "$method_name" "$sub_strategy" \
        "$select_strategy" "$elastic_memory" "$MAX_SPECULATIVE_LEN" \
        "$result_dir" >> "$MANIFEST_CSV"

    echo "[fairness-ablation] dataset=$dataset_name group=$group_name method=$method_name elastic=$elastic_memory port=$port"
    "${cmd[@]}"
}

cat <<EOF
Running fairness ablation with:
  MODEL_NAME=$MODEL_NAME
  DRAFT_MODEL_NAME=$DRAFT_MODEL_NAME
  NUM_PROMPTS=$NUM_PROMPTS
  REQUEST_RATES=$REQUEST_RATES
  MAX_SPECULATIVE_LEN=$MAX_SPECULATIVE_LEN
  NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE

Outputs:
  1. Nightjar w/o offload
  2. Nightjar full
  3. Baseline static-memory
EOF

port_offset=0
for dataset in $DATASETS; do
    dataset_name="${dataset%%:*}"
    dataset_path="${dataset#*:}"

    run_case "$dataset_name" "$dataset_path" "nightjar_wo_offload" "nightjar_wo_offload" "epsilon_greedy" "" "false" "$((PORT_BASE + port_offset))"
    port_offset=$((port_offset + 1))

    run_case "$dataset_name" "$dataset_path" "nightjar_full" "nightjar_full" "epsilon_greedy_with_offload" "" "true" "$((PORT_BASE + port_offset))"
    port_offset=$((port_offset + 1))

    for baseline_case in $BASELINE_CASES; do
        method_name="${baseline_case%%:*}"
        remainder="${baseline_case#*:}"
        sub_strategy="${remainder%%:*}"
        select_strategy="${remainder#*:}"
        if [[ "$select_strategy" == "$remainder" ]]; then
            select_strategy=""
        fi

        run_case "$dataset_name" "$dataset_path" "baseline_static_memory" "$method_name" "$sub_strategy" "$select_strategy" "false" "$((PORT_BASE + port_offset))"
        port_offset=$((port_offset + 1))
    done
done

if [[ -f "$GLOBAL_CSV" ]]; then
    mv "$GLOBAL_CSV" "$RESULT_ROOT/benchmark_results_fairness_ablation.csv"
fi

echo "Fairness ablation finished."
echo "Results root: $RESULT_ROOT"
echo "Manifest: $MANIFEST_CSV"
