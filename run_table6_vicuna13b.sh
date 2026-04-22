#!/usr/bin/env bash
set -u

# Table 6 13B setting:
# target/draft: Vicuna-13B + vicuna-68m
# datasets: Alpaca, ShareGPT, SpecBench
# local Table-6-like CSV rows use 200 requests, 5 QPS, burstiness 1.0.

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

NUM_PROMPTS=${NUM_PROMPTS:-200}
PROMPT_RATES="${PROMPT_RATES:-5}"
FILE_NAME=${FILE_NAME:-table6_vicuna13b.log}
RESULT_DIR=${RESULT_DIR:-benchmark_results/vicuna13B}
MAX_SPECULATIVE_LEN=${MAX_SPECULATIVE_LEN:-3}
START_INDEX=${START_INDEX:-0}
ENABLE_TRACE=${ENABLE_TRACE:-False}
SAVE_TRACE=${SAVE_TRACE:-False}

# Follow test3_diff_length_13B.sh. Override if your GPU needs a different value.
num_gpu_blocks_override=${num_gpu_blocks_override:-4938}
gpu_memory_utilization=${gpu_memory_utilization:-0.85}
increase_block_threshold=${increase_block_threshold:-150}
decrease_block_threshold=${decrease_block_threshold:-100}
burstiness=${burstiness:-1.0}

model_name=${model_name:-/root/autodl-tmp/vicuna-13b-v1.3}
draft_model_name=${draft_model_name:-/root/autodl-tmp/vicuna-68m}

# Table 6 methods in this repo's implementation names:
# nospec: w/o SD
# deep gamma=3: SD
# ucb: BanditSpec
# smart_spec: DSD
# deep --select-strategy capacity: TETRIS/capacity baseline
# ada_bin_greedy: Nightjar
RUN_NOSPEC=${RUN_NOSPEC:-1}
RUN_SD=${RUN_SD:-1}
RUN_BANDITSPEC=${RUN_BANDITSPEC:-1}
RUN_DSD=${RUN_DSD:-1}
RUN_TETRIS=${RUN_TETRIS:-1}
RUN_NIGHTJAR=${RUN_NIGHTJAR:-1}

cd "$(dirname "$0")"
mkdir -p "$RESULT_DIR"
rm -rf "$FILE_NAME"

run_cmd() {
    echo "" | tee -a "$FILE_NAME"
    echo "[$(date '+%F %T')] $*" | tee -a "$FILE_NAME"
    "$@" &>> "$FILE_NAME"

    pid=$(pgrep -f "adaptive_engine_example" || true)
    if [ -n "$pid" ]; then
        kill $pid || true
    fi
    sleep 3
}

run_dataset() {
    data_set_name=$1
    data_set_path=$2

    for PROMPT_RATE in $PROMPT_RATES
    do
        explore="False"

        if [ "$RUN_NOSPEC" = "1" ]; then
            run_cmd python run_benchmark_tests.py --strategy ilp --sub-strategy nospec \
                --model "$model_name" --save-trace "$SAVE_TRACE" --draft-model "$draft_model_name" \
                --dataset-name "$data_set_name" --dataset-path "$data_set_path" \
                --result-dir "$RESULT_DIR" \
                --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts "$NUM_PROMPTS" \
                --request-rate "$PROMPT_RATE" --start-index "$START_INDEX" \
                --num-gpu-blocks-override "$num_gpu_blocks_override" \
                --enable-trace "$ENABLE_TRACE" --burstiness "$burstiness" \
                --gpu-memory-utilization "$gpu_memory_utilization" \
                --increase-block-threshold "$increase_block_threshold" \
                --decrease-block-threshold "$decrease_block_threshold"
        fi

        if [ "$RUN_SD" = "1" ]; then
            run_cmd python run_benchmark_tests.py --strategy ilp --sub-strategy deep \
                --model "$model_name" --save-trace "$SAVE_TRACE" --draft-model "$draft_model_name" \
                --dataset-name "$data_set_name" --dataset-path "$data_set_path" \
                --result-dir "$RESULT_DIR" \
                --speculative-len 3 --num-prompts "$NUM_PROMPTS" \
                --request-rate "$PROMPT_RATE" --start-index "$START_INDEX" \
                --num-gpu-blocks-override "$num_gpu_blocks_override" \
                --enable-trace "$ENABLE_TRACE" --burstiness "$burstiness" \
                --gpu-memory-utilization "$gpu_memory_utilization" \
                --increase-block-threshold "$increase_block_threshold" \
                --decrease-block-threshold "$decrease_block_threshold"
        fi

        if [ "$RUN_BANDITSPEC" = "1" ]; then
            run_cmd python run_benchmark_tests.py --strategy ilp --sub-strategy ucb \
                --explore "$explore" --save-trace "$SAVE_TRACE" \
                --model "$model_name" --draft-model "$draft_model_name" \
                --dataset-name "$data_set_name" --dataset-path "$data_set_path" \
                --result-dir "$RESULT_DIR" \
                --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts "$NUM_PROMPTS" \
                --request-rate "$PROMPT_RATE" --start-index "$START_INDEX" \
                --num-gpu-blocks-override "$num_gpu_blocks_override" \
                --enable-trace "$ENABLE_TRACE" --burstiness "$burstiness" \
                --gpu-memory-utilization "$gpu_memory_utilization" \
                --increase-block-threshold "$increase_block_threshold" \
                --decrease-block-threshold "$decrease_block_threshold"
        fi

        if [ "$RUN_DSD" = "1" ]; then
            run_cmd python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec \
                --model "$model_name" --save-trace "$SAVE_TRACE" --draft-model "$draft_model_name" \
                --dataset-name "$data_set_name" --dataset-path "$data_set_path" \
                --result-dir "$RESULT_DIR" \
                --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts "$NUM_PROMPTS" \
                --request-rate "$PROMPT_RATE" --start-index "$START_INDEX" \
                --num-gpu-blocks-override "$num_gpu_blocks_override" \
                --enable-trace "$ENABLE_TRACE" --burstiness "$burstiness" \
                --gpu-memory-utilization "$gpu_memory_utilization" \
                --increase-block-threshold "$increase_block_threshold" \
                --decrease-block-threshold "$decrease_block_threshold"
        fi

        if [ "$RUN_TETRIS" = "1" ]; then
            run_cmd python run_benchmark_tests.py --strategy ilp --sub-strategy deep \
                --select-strategy capacity --save-trace "$SAVE_TRACE" \
                --model "$model_name" --draft-model "$draft_model_name" \
                --dataset-name "$data_set_name" --dataset-path "$data_set_path" \
                --result-dir "$RESULT_DIR" \
                --speculative-len 3 --num-prompts "$NUM_PROMPTS" \
                --request-rate "$PROMPT_RATE" --start-index "$START_INDEX" \
                --num-gpu-blocks-override "$num_gpu_blocks_override" \
                --enable-trace "$ENABLE_TRACE" --burstiness "$burstiness" \
                --gpu-memory-utilization "$gpu_memory_utilization" \
                --increase-block-threshold "$increase_block_threshold" \
                --decrease-block-threshold "$decrease_block_threshold"
        fi

        if [ "$RUN_NIGHTJAR" = "1" ]; then
            run_cmd python run_benchmark_tests.py --strategy ilp --sub-strategy ada_bin_greedy \
                --explore "$explore" --save-trace "$SAVE_TRACE" \
                --model "$model_name" --draft-model "$draft_model_name" \
                --dataset-name "$data_set_name" --dataset-path "$data_set_path" \
                --result-dir "$RESULT_DIR" \
                --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts "$NUM_PROMPTS" \
                --request-rate "$PROMPT_RATE" --start-index "$START_INDEX" \
                --num-gpu-blocks-override "$num_gpu_blocks_override" \
                --enable-trace "$ENABLE_TRACE" --burstiness "$burstiness" \
                --gpu-memory-utilization "$gpu_memory_utilization" \
                --increase-block-threshold "$increase_block_threshold" \
                --decrease-block-threshold "$decrease_block_threshold"
        fi
    done
}

for i in 1
do
    run_dataset alpaca tatsu-lab/alpaca
    run_dataset sharegpt /root/autodl-tmp/sharegpt.json
    run_dataset specbench ./question_shuffled.jsonl
done
