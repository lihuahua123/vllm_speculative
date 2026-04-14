#!/usr/bin/env bash

set -euo pipefail

# Reviewer #1 - Comment 13:
# Lightweight model-free draft control experiment on DeepSeek-R1-Distill-Qwen-7B.
# Goal: show adaptive length (Nightjar) is still effective when the draft is N-gram.

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/root/autodl-tmp/nightjar/vllm_speculative}"

NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATES_STR="${REQUEST_RATES:-1 2 4 8 12 16}"
read -r -a REQUEST_RATES <<< "$REQUEST_RATES_STR"
FILE_NAME="${FILE_NAME:-$SCRIPT_DIR/r1_13_model_free_ngram_7b.log}"
MAX_SPECULATIVE_LEN="${MAX_SPECULATIVE_LEN:-4}"
START_INDEX="${START_INDEX:-0}"
ENABLE_TRACE="${ENABLE_TRACE:-False}"
SAVE_TRACE="${SAVE_TRACE:-False}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-4938}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
INCREASE_BLOCK_THRESHOLD="${INCREASE_BLOCK_THRESHOLD:-150}"
DECREASE_BLOCK_THRESHOLD="${DECREASE_BLOCK_THRESHOLD:-100}"
BURSTINESS="${BURSTINESS:-1.0}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B}"
NGRAM_DRAFT_NAME="${NGRAM_DRAFT_NAME:-[ngram]}"

DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/sharegpt.json}"
# 可切到其他数据集做补充：
# DATASET_NAME=alpaca
# DATASET_PATH=tatsu-lab/alpaca
# DATASET_NAME=specbench
# DATASET_PATH=/root/autodl-tmp/nightjar/vllm_speculative/question_shuffled.jsonl

# 不设 output_len 时，沿用数据集样本自身的 expected_output_len。
OUTPUT_LEN="${OUTPUT_LEN:-}"

kill_server() {
    local pids
    pids="$(pgrep -f "adaptive_engine_example" || true)"
    if [[ -n "$pids" ]]; then
        kill $pids || true
        sleep 5
    fi
}

run_case() {
    local sub_strategy="$1"
    local speculative_len="$2"
    local prompt_rate="$3"
    local explore="${4:-False}"

    local cmd=(
        python run_benchmark_tests.py
        --strategy ilp
        --sub-strategy "$sub_strategy"
        --explore "$explore"
        --save-trace "$SAVE_TRACE"
        --model "$MODEL_NAME"
        --draft-model "$NGRAM_DRAFT_NAME"
        --dataset-name "$DATASET_NAME"
        --dataset-path "$DATASET_PATH"
        --speculative-len "$speculative_len"
        --num-prompts "$NUM_PROMPTS"
        --request-rate "$prompt_rate"
        --start-index "$START_INDEX"
        --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE"
        --enable-trace "$ENABLE_TRACE"
        --burstiness "$BURSTINESS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --increase-block-threshold "$INCREASE_BLOCK_THRESHOLD"
        --decrease-block-threshold "$DECREASE_BLOCK_THRESHOLD"
    )

    if [[ -n "$OUTPUT_LEN" ]]; then
        cmd+=(--output-len "$OUTPUT_LEN")
    fi

    {
        echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
        echo "sub_strategy=${sub_strategy} speculative_len=${speculative_len} request_rate=${prompt_rate}"
        "${cmd[@]}"
        echo
    } &>> "$FILE_NAME"

    kill_server
}

cd "$REPO_DIR"
rm -f "$FILE_NAME"

{
    echo "# Reviewer #1 comment 13"
    echo "# 7B model-free draft (ngram) lightweight control experiment"
    echo "# model=${MODEL_NAME}"
    echo "# draft=${NGRAM_DRAFT_NAME}"
    echo "# dataset=${DATASET_NAME}"
    echo "# dataset_path=${DATASET_PATH}"
    echo "# request_rates=${REQUEST_RATES[*]}"
    echo "# num_prompts=${NUM_PROMPTS}"
    echo "# max_speculative_len=${MAX_SPECULATIVE_LEN}"
    echo
} >> "$FILE_NAME"

for PROMPT_RATE in "${REQUEST_RATES[@]}"; do
    explore="False"

    # Nightjar adaptive length with model-free draft.
    run_case "ada_bin_greedy" "$MAX_SPECULATIVE_LEN" "$PROMPT_RATE" "$explore"

    # Fixed ngram baselines.
    run_case "ngram" "1" "$PROMPT_RATE"
    run_case "deep" "1" "$PROMPT_RATE"
    run_case "deep" "2" "$PROMPT_RATE"
    run_case "deep" "3" "$PROMPT_RATE"
    run_case "deep" "4" "$PROMPT_RATE"

    # No speculation baseline.
    run_case "nospec" "$MAX_SPECULATIVE_LEN" "$PROMPT_RATE"
done

echo "Finished. Log saved to: $FILE_NAME"
