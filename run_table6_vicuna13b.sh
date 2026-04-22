#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Vicuna-13B side of paper Table 6 as closely as the local
# benchmark scripts expose it: Alpaca/ShareGPT/SpecBench, 200 requests, 5 QPS.

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/vicuna-13b-v1.3}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/vicuna-68m}"

NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATES="${REQUEST_RATES:-5}"
START_INDEX="${START_INDEX:-0}"
BURSTINESS="${BURSTINESS:-1.0}"
SEED="${SEED:-42}"

MAX_SPECULATIVE_LEN="${MAX_SPECULATIVE_LEN:-3}"
GPU_BLOCKS="${GPU_BLOCKS:-1285}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
INCREASE_BLOCK_THRESHOLD="${INCREASE_BLOCK_THRESHOLD:-150}"
DECREASE_BLOCK_THRESHOLD="${DECREASE_BLOCK_THRESHOLD:-100}"
PERSIST_STEPS="${PERSIST_STEPS:-3}"

ENABLE_TRACE="${ENABLE_TRACE:-False}"
SAVE_TRACE="${SAVE_TRACE:-False}"
RESULT_DIR="${RESULT_DIR:-benchmark_results_table6_vicuna13b}"
LOG_FILE="${LOG_FILE:-table6_vicuna13b.log}"

# Table 6 baselines:
# nospec             -> w/o SD
# deep:3             -> SD, gamma=3
# ucb                -> BanditSpec-style MAB baseline
# smart_spec         -> DSD-style baseline in this repo
# deep:3:capacity    -> TETRIS/capacity-style baseline used by existing scripts
# ada_bin_greedy     -> Nightjar
METHODS="${METHODS:-nospec deep:3 ucb smart_spec deep:3:capacity ada_bin_greedy}"
DATASETS="${DATASETS:-alpaca sharegpt specbench}"

cd "$(dirname "$0")"
rm -f "$LOG_FILE"

for required_path in "$MODEL_NAME" "$DRAFT_MODEL_NAME"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required model path: $required_path" | tee -a "$LOG_FILE"
        echo "Override MODEL_NAME or DRAFT_MODEL_NAME if the model is stored elsewhere." | tee -a "$LOG_FILE"
        exit 1
    fi
done

run_one() {
    local dataset_name="$1"
    local dataset_path="$2"
    local sub_strategy="$3"
    local speculative_len="$4"
    local select_strategy="$5"
    local rate="$6"

    local cmd=(
        python run_benchmark_tests.py
        --strategy ilp
        --sub-strategy "$sub_strategy"
        --model "$MODEL_NAME"
        --draft-model "$DRAFT_MODEL_NAME"
        --dataset-name "$dataset_name"
        --dataset-path "$dataset_path"
        --speculative-len "$speculative_len"
        --num-prompts "$NUM_PROMPTS"
        --request-rates "$rate"
        --start-index "$START_INDEX"
        --num-gpu-blocks-override "$GPU_BLOCKS"
        --enable-trace "$ENABLE_TRACE"
        --save-trace "$SAVE_TRACE"
        --burstiness "$BURSTINESS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$MAX_MODEL_LEN"
        --increase-block-threshold "$INCREASE_BLOCK_THRESHOLD"
        --decrease-block-threshold "$DECREASE_BLOCK_THRESHOLD"
        --persist-steps "$PERSIST_STEPS"
        --seed "$SEED"
        --result-dir "$RESULT_DIR"
    )

    if [[ -n "$select_strategy" ]]; then
        cmd+=(--select-strategy "$select_strategy")
    fi

    {
        printf '\n[%s] dataset=%s method=%s len=%s qps=%s\n' \
            "$(date '+%F %T')" "$dataset_name" "$sub_strategy" "$speculative_len" "$rate"
        printf 'Command:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    } | tee -a "$LOG_FILE"

    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
}

for dataset in $DATASETS; do
    case "$dataset" in
        alpaca)
            dataset_name="alpaca"
            dataset_path="tatsu-lab/alpaca"
            ;;
        sharegpt)
            dataset_name="sharegpt"
            dataset_path="/root/autodl-tmp/sharegpt.json"
            ;;
        specbench)
            dataset_name="specbench"
            dataset_path="./question_shuffled.jsonl"
            ;;
        *)
            echo "Unknown dataset: $dataset" | tee -a "$LOG_FILE"
            exit 1
            ;;
    esac

    for rate in $REQUEST_RATES; do
        for method in $METHODS; do
            sub_strategy="$method"
            speculative_len="$MAX_SPECULATIVE_LEN"
            select_strategy=""

            IFS=':' read -r sub_strategy maybe_len maybe_select <<< "$method"
            if [[ -n "${maybe_len:-}" ]]; then
                speculative_len="$maybe_len"
            fi
            if [[ -n "${maybe_select:-}" ]]; then
                select_strategy="$maybe_select"
            fi

            run_one "$dataset_name" "$dataset_path" "$sub_strategy" \
                "$speculative_len" "$select_strategy" "$rate"
        done
    done
done

echo "Done. Log: $LOG_FILE, result dir: $RESULT_DIR"
