#!/usr/bin/env bash

set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/Llama-3.1-8B-Instruct}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-/root/autodl-tmp/Llama-3.2-1B}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/sharegpt.json}"
FILE_NAME="${FILE_NAME:-llama31_llama32_smart_spec.log}"
RESULT_DIR="${RESULT_DIR:-benchmark_results/llama31_llama32_smart_spec}"

HOST="${HOST:-127.0.0.1}"
BENCHMARK_PORT="${BENCHMARK_PORT:-8010}"
PROBE_PORT="${PROBE_PORT:-8020}"

NUM_PROMPTS="${NUM_PROMPTS:-150}"
PROMPT_RATE="${PROMPT_RATE:-20}"
START_INDEX="${START_INDEX:-0}"
SPECULATIVE_LEN="${SPECULATIVE_LEN:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SPECULATIVE_DRAFT_TP_SIZE="${SPECULATIVE_DRAFT_TP_SIZE:-1}"
ENABLE_TRACE="${ENABLE_TRACE:-False}"
BURSTINESS="${BURSTINESS:-1.0}"
INCREASE_BLOCK_THRESHOLD="${INCREASE_BLOCK_THRESHOLD:-150}"
DECREASE_BLOCK_THRESHOLD="${DECREASE_BLOCK_THRESHOLD:-100}"
SUB_STRATEGY="${SUB_STRATEGY:-smart_spec}"

PROBE_LOG="${PROBE_LOG:-$ROOT_DIR/llama31_probe_gpu_blocks.log}"
probe_pid=""

cleanup() {
    if [[ -n "${probe_pid}" ]] && kill -0 "${probe_pid}" 2>/dev/null; then
        kill "${probe_pid}" 2>/dev/null || true
        wait "${probe_pid}" 2>/dev/null || true
    fi
}

trap cleanup EXIT

require_gpu() {
    local gpu_list
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found. This script requires a CUDA-visible GPU." >&2
        return 1
    fi
    gpu_list="$(nvidia-smi -L 2>/dev/null || true)"
    if [[ -z "${gpu_list}" ]] || [[ "${gpu_list}" == "No devices found." ]]; then
        echo "No CUDA-visible GPU detected. This script requires a CUDA-visible GPU." >&2
        return 1
    fi
}

extract_num_gpu_blocks() {
    local log_file="$1"
    local blocks

    blocks="$(grep -oE '# GPU blocks: [0-9]+' "$log_file" 2>/dev/null | tail -n 1 | awk '{print $4}')"
    if [[ -n "${blocks}" ]]; then
        printf '%s\n' "$blocks"
        return 0
    fi

    blocks="$(grep -oE '# cuda blocks: [0-9]+' "$log_file" 2>/dev/null | tail -n 1 | awk '{print $4}')"
    if [[ -n "${blocks}" ]]; then
        printf '%s\n' "$blocks"
        return 0
    fi

    blocks="$(grep -oE 'num_gpu_blocks [0-9]+' "$log_file" 2>/dev/null | tail -n 1 | awk '{print $2}')"
    if [[ -n "${blocks}" ]]; then
        printf '%s\n' "$blocks"
        return 0
    fi

    blocks="$(grep -oE 'num_gpu_blocks=[0-9]+' "$log_file" 2>/dev/null | tail -n 1 | cut -d '=' -f 2)"
    if [[ -n "${blocks}" ]]; then
        printf '%s\n' "$blocks"
        return 0
    fi

    return 1
}

get_num_gpu_blocks_override() {
    rm -f "$PROBE_LOG"

    local speculative_model="$DRAFT_MODEL_NAME"
    local -a probe_cmd=(
        python -m vllm.entrypoints.openai.api_server
        --model "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PROBE_PORT" \
        --device "$VLLM_TARGET_DEVICE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --enforce-eager \
        --no-enable-prefix-caching \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    )

    if [[ -n "${speculative_model}" ]]; then
        probe_cmd+=(
            --speculative-model "$speculative_model"
            --num-speculative-tokens "$SPECULATIVE_LEN"
        )
        if [[ "${speculative_model}" == "[ngram]" ]]; then
            probe_cmd+=(
                --ngram-prompt-lookup-min 1
                --ngram_prompt_lookup_max "$SPECULATIVE_LEN"
            )
        else
            probe_cmd+=(
                --speculative-draft-tensor-parallel-size "$SPECULATIVE_DRAFT_TP_SIZE"
            )
        fi
    fi

    "${probe_cmd[@]}" >"$PROBE_LOG" 2>&1 &
    probe_pid=$!

    for _ in $(seq 1 180); do
        local detected_blocks
        detected_blocks="$(extract_num_gpu_blocks "$PROBE_LOG" || true)"
        if [[ -n "${detected_blocks}" ]]; then
            printf '%s\n' "${detected_blocks}"
            return 0
        fi

        if ! kill -0 "$probe_pid" 2>/dev/null; then
            echo "Failed to probe num_gpu_blocks_override. Probe server exited early." >&2
            tail -n 80 "$PROBE_LOG" >&2 || true
            return 1
        fi
        sleep 2
    done

    echo "Timed out while probing num_gpu_blocks_override." >&2
    tail -n 80 "$PROBE_LOG" >&2 || true
    return 1
}

require_gpu

echo "Probing num_gpu_blocks_override for model: $MODEL_NAME"
NUM_GPU_BLOCKS_OVERRIDE=4000 #"$(get_num_gpu_blocks_override)"
echo "Detected num_gpu_blocks_override=$NUM_GPU_BLOCKS_OVERRIDE"

cleanup
probe_pid=""

mkdir -p "$RESULT_DIR"

python run_benchmark_tests.py \
    --strategy ilp \
    --sub-strategy "$SUB_STRATEGY" \
    --model "$MODEL_NAME" \
    --draft-model "$DRAFT_MODEL_NAME" \
    --host "$HOST" \
    --port "$BENCHMARK_PORT" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --speculative-len "$SPECULATIVE_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rates "$PROMPT_RATE" \
    --start-index "$START_INDEX" \
    --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE" \
    --enable-trace "$ENABLE_TRACE" \
    --burstiness "$BURSTINESS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --speculative-draft-tensor-parallel-size "$SPECULATIVE_DRAFT_TP_SIZE" \
    --increase-block-threshold "$INCREASE_BLOCK_THRESHOLD" \
    --decrease-block-threshold "$DECREASE_BLOCK_THRESHOLD" \
    --result-dir "$RESULT_DIR" \
    &>> "$FILE_NAME"

echo "Finished. Log: $FILE_NAME"
