#!/bin/bash
# Script to train and evaluate the multi-armed bandit optimizer

# Model parameters
MODEL_NAME="deepseek-aiDeepSeek-R1-Distill-Qwen-7B"
MODEL_PATH="/data/model/${MODEL_NAME}"
SPEC_MODEL="alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B"
DATASET_PATH="/data/sharegpt.json"

# Output directory for results
RESULTS_DIR="bandit_results"
mkdir -p $RESULTS_DIR

# Print execution info
echo "========================================================"
echo "Starting bandit training and evaluation for ${MODEL_NAME}"
echo "Using speculative model: ${SPEC_MODEL}"
echo "========================================================"

# Run the training and evaluation
python tools/train_bandit_optimizer.py \
    --model $MODEL_PATH \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 9432 \
    --speculative-model $SPEC_MODEL \
    --num-speculative-tokens 4 \
    --dataset-path $DATASET_PATH \
    --num-training-requests 50 \
    --num-test-requests 20 \
    --max-output-tokens 150 \
    --training-iterations 3 \
    --request-rates 1.0 2.0 4.0 8.0 \
    --test-duration 120.0 \
    --monitoring-interval 3.0 \
    --cooldown-period 15.0 \
    --num_gpu_blocks_override 6270 \
    --save-results \
    --results-dir $RESULTS_DIR

echo "========================================================"
echo "Bandit training and evaluation completed"
echo "Results saved to ${RESULTS_DIR}"
echo "========================================================"

# # Now run the benchmark tests for comparison
# echo "Running benchmark tests for comparison..."

# # Standard benchmark without bandit optimization for comparison
# echo "[" >> $RESULTS_DIR/standard_benchmark.json
# python benchmarks/benchmark_serving.py --backend vllm --model $MODEL_PATH --dataset-name sharegpt --dataset-path $DATASET_PATH --num-prompts 20 --request-rate 1 --goodput tpot:30 --save-result --result-dir $RESULTS_DIR --percentile-metrics ttft,tpot,itl,e2el --result-filename standard_benchmark.json --metadata "type=standard"
# echo "," >> $RESULTS_DIR/standard_benchmark.json
# sleep 2
# python benchmarks/benchmark_serving.py --backend vllm --model $MODEL_PATH --dataset-name sharegpt --dataset-path $DATASET_PATH --num-prompts 20 --request-rate 2 --goodput tpot:30 --save-result --result-dir $RESULTS_DIR --percentile-metrics ttft,tpot,itl,e2el --result-filename standard_benchmark.json --metadata "type=standard" 
# echo "," >> $RESULTS_DIR/standard_benchmark.json
# sleep 2
# python benchmarks/benchmark_serving.py --backend vllm --model $MODEL_PATH --dataset-name sharegpt --dataset-path $DATASET_PATH --num-prompts 20 --request-rate 4 --goodput tpot:30 --save-result --result-dir $RESULTS_DIR --percentile-metrics ttft,tpot,itl,e2el --result-filename standard_benchmark.json --metadata "type=standard"
# echo "," >> $RESULTS_DIR/standard_benchmark.json
# sleep 2
# python benchmarks/benchmark_serving.py --backend vllm --model $MODEL_PATH --dataset-name sharegpt --dataset-path $DATASET_PATH --num-prompts 20 --request-rate 8 --goodput tpot:30 --save-result --result-dir $RESULTS_DIR --percentile-metrics ttft,tpot,itl,e2el --result-filename standard_benchmark.json --metadata "type=standard"
# echo "]" >> $RESULTS_DIR/standard_benchmark.json

# echo "All tests completed successfully!" 