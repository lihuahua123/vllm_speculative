#!/bin/bash

# Prefill 性能测试脚本
# 支持不同的 input length 和 batch size
# 注意：此脚本只启动一次 vLLM 服务器，然后执行所有测试配置，最后关闭服务器
# 这样可以避免重复启动服务器的开销，提高测试效率

# 模型配置
model_name=/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B  # 主模型路径
draft_model_name=/root/autodl-tmp/deep05b  # Draft 模型路径（如果使用 speculative 策略）

# 基础配置
FILE_NAME=prefill_benchmark.log
SPECULATIVE_LEN=3
ENABLE_TRACE="False"
num_gpu_blocks_override=4938  # 根据你的GPU调整，参考 test3.sh 中的注释
# 717 #26064 #9369 #A600 50% #4800 4090 # deep A6000 26064 llama8b A6000 xxx
# 通过下面命令行得到: 788 for 33B vicuna and eagle 1285 for 13B vicuna and eagle
gpu_memory_utilization=0.85
burstiness=1.0

# 服务器配置
HOST="127.0.0.1"
PORT=8010

# 策略配置
STRATEGY="ilp"  # 可选: baseline, ilp, no-spec
SUB_STRATEGY=("deep" "nospec")  # 所有可用的子策略 "nospec"
data_set_name=sharegpt
data_set_path=/root/autodl-tmp/sharegpt.json # $data_set_path

# 测试参数配置
# 不同的 input length 列表（token 数）
INPUT_LENGTHS=(100 200 300 400 500 600 700 800 900 1000)
# for ((i=1; i<=128; i++)); do
#     INPUT_LENGTHS+=($i)
# done


# 不同的 batch size（通过 request-rate 控制并发请求数）
# 注意：实际 batch size 由 vLLM 调度器决定，request-rate 影响并发请求数
BATCH_SIZES=(1 2 4 8 16 32 64)

# 每个配置的测试次数
NUM_PROMPTS=5

# 输出长度（prefill 测试通常设为 1）
OUTPUT_LEN=1



export HF_ENDPOINT='https://hf-mirror.com'

# 清理旧的日志文件
rm -rf $FILE_NAME

echo "========================================="
echo "开始 Prefill 性能测试"
echo "模型: $model_name"
if [ "$STRATEGY" != "no-spec" ]; then
    echo "Draft 模型: $draft_model_name"
fi
echo "策略: $STRATEGY"
echo "将测试的子策略: ${SUB_STRATEGY[@]}"
echo "服务器地址: $HOST:$PORT"
echo "测试的 Input Lengths: ${INPUT_LENGTHS[@]}"
echo "测试的 Batch Sizes (并发请求数): ${BATCH_SIZES[@]}"
echo "每个配置的 Prompts 数: $NUM_PROMPTS"
echo "输出长度: $OUTPUT_LEN"
echo "========================================="
echo ""

# 遍历所有子策略
for sub_strategy in "${SUB_STRATEGY[@]}"; do
    echo ""
    echo "========================================="
    echo "开始测试子策略: $sub_strategy"
    echo "========================================="
    
    # 为每个策略设置独立的结果目录
    RESULT_DIR="prefill_benchmark_results_${sub_strategy}"
    
    # 使用专门的 benchmark_prefill.py 脚本
    # 这个脚本支持 random 数据集和 input length 参数
    # 使用与 run_benchmark_tests.py 相同的服务器启动方式
    # 注意：benchmark_prefill.py 会：
    #   1. 启动一次 vLLM 服务器
    #   2. 循环执行所有 (input_len, batch_size) 组合的测试
    #   3. 所有测试完成后关闭服务器
    # 这样可以避免重复启动服务器的开销
    python benchmark_prefill.py \
        --model $model_name \
        --draft-model $draft_model_name \
        --host $HOST \
        --port $PORT \
        --input-lengths ${INPUT_LENGTHS[@]} \
        --batch-sizes ${BATCH_SIZES[@]} \
        --num-prompts $NUM_PROMPTS \
        --output-len $OUTPUT_LEN \
        --strategy $STRATEGY \
        --sub-strategy $sub_strategy \
        --speculative-len $SPECULATIVE_LEN \
        --num-gpu-blocks-override $num_gpu_blocks_override \
        --gpu-memory-utilization $gpu_memory_utilization \
        --enable-trace "$ENABLE_TRACE" \
        --burstiness $burstiness \
        --result-dir $RESULT_DIR \
        --log-file $FILE_NAME \
        &>> $FILE_NAME
    
    echo ""
    echo "========================================="
    echo "子策略 $sub_strategy 测试完成！"
    echo "结果保存在: $RESULT_DIR/"
    echo "========================================="
done

echo ""
echo "========================================="
echo "所有子策略的 Prefill 性能测试完成！"
echo "日志文件: $FILE_NAME"
echo "========================================="

