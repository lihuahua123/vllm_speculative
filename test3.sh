# CUDA 多进程必须用 spawn，不能用 fork，否则会报 "Cannot re-initialize CUDA in forked subprocess"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#python run_benchmark_tests.py --strategy ilp --sub-strategy ngram --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 442 --request-rate 10 --output-len $output_len 1>> a120new.log 2>> a120new_error.log 
NUM_PROMPTS=150
PROMPT_RATE=5
FILE_NAME=medusa4.log # 
MAX_SPECULATIVE_LEN=3
START_INDEX=0
ENABLE_TRACE="False"
SAVE_TRACE="False"
num_gpu_blocks_override=4112 #717 #26064 #9369 #A600 50% #4800 4090 # deep A6000 26064 llama8b A6000 xxx  通过下面命令行得到 788 for 33B vicuna and eagle 1285 for 13B vicuna and eagle
gpu_memory_utilization=0.85
increase_block_threshold=150
decrease_block_threshold=100
# python -m vllm.entrypoints.openai.api_server  --model /root/autodl-tmp/Qwen2.5-32B-Instruct --gpu-memory-utilization 0.85 --speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 2048 --tensor_parallel_size 2
# python -m vllm.entrypoints.openai.api_server  --model /root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B --gpu-memory-utilization 0.85 --speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 2048 --tensor_parallel_size 1
rm -rf $FILE_NAME
data_set_name=sharegpt
data_set_path=/root/autodl-tmp/sharegpt.json # $data_set_path
# data_set_name=specbench    #--dataset-path tatsu-lab/alpaca
# data_set_path=./question_shuffled.jsonl # $data_set_path
# Hugging Face 镜像与缓存：使用固定缓存目录，避免每次重新下载
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
# 首次运行会下载到上述目录，之后自动用缓存。若需强制仅用本地缓存（不联网）可取消下一行注释：
# export HF_DATASETS_OFFLINE=1

# data_set_name=alpaca
# --dataset-path tatsu-lab/alpaca
# data_set_path=tatsu-lab/alpaca
burstiness=1.0
# 不设 output_len 时用数据集每条样本的 expected_output_len；同一批数据(seed 已固定)+ignore_eos 下，total_output_tokens 应一致
# output_len=256 # 若需强制所有请求统一长度可取消注释并传 --output-len $output_len
#model_name=/root/autodl-tmp/deepseek 
#  python -m vllm.entrypoints.openai.api_server  --model /model/vicuna-7b-v1.3 --gpu-memory-utilization 0.85 --speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 2048
model_name=/root/autodl-tmp/Qwen2.5-32B-Instruct #/root/autodl-tmp/vicuna-13b-v1.3  #/model/vicuna-7b-v1.3
draft_model_name=/root/autodl-tmp/Qwen2.5-0.5B-Instruct-152064 #/root/autodl-tmp/vllm-medusa-vicuna-7b-v1.3 #/root/autodl-tmp/vicuna-68m #/root/autodl-tmp/vllm-medusa-vicuna-7b-v1.3
# model_name=/data/model/llama3instruct
# draft_model_name=/data/model/yuhuiliEAGLE-LLaMA3-Instruct-8B-vllm
#/data/model/DeepSeek-R1-DRAFT-Qwen2.5-0.5B
export HF_ENDPOINT='https://hf-mirror.com'
for i in 1  # 2 3 4 5
do
    for PROMPT_RATE in 20 #5 10 15 20 25 #25 # 24 # 6 8 10 12 14 # 10 15 # 20 25 #0.5 2 5 10
    do
        # if [ $PROMPT_RATE -eq 5 ]; then
        #     explore="True"
        # else
        explore="False"
        # fi
        # sleep 3

        

        # # ADABinGreedySimple
        # python run_benchmark_tests.py --strategy ilp --sub-strategy ada_bin_greedy_simple --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --increase-block-threshold $increase_block_threshold --decrease-block-threshold $decrease_block_threshold --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid

        # # # EpsilonGreedySimple
        # python run_benchmark_tests.py --strategy ilp --sub-strategy epsilon_greedy_simple --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --increase-block-threshold $increase_block_threshold --decrease-block-threshold $decrease_block_threshold --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid

       
        # # LinUCBSpec
        # python run_benchmark_tests.py --strategy ilp --sub-strategy lin_ucb --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --increase-block-threshold $increase_block_threshold --decrease-block-threshold $decrease_block_threshold --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid
        ## c prefill
        # python run_benchmark_tests.py --strategy ilp --sub-strategy epsilon_greedy_with_c_prefill --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --increase-block-threshold $increase_block_threshold --decrease-block-threshold $decrease_block_threshold --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid
        # # offload
        # python run_benchmark_tests.py --strategy ilp --sub-strategy epsilon_greedy_with_offload --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --increase-block-threshold $increase_block_threshold --decrease-block-threshold $decrease_block_threshold &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid
        # ucb
    #     python run_benchmark_tests.py --strategy ilp --sub-strategy ucb --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
    #     pid=$(pgrep -f "adaptive_engine_example")
    #     kill $pid
    #     # threshold
    # #    python run_benchmark_tests.py --strategy ilp --sub-strategy threshold --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
    # #     pid=$(pgrep -f "adaptive_engine_example")
    # #     kill $pid
    #     # # capacity
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep  --select-strategy capacity --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid
    #     # # # smart_spec
        python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec  --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 4 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
        pid=$(pgrep -f "adaptive_engine_example")
        kill $pid
        # deep-1
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid
        # # deep-2
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid
        # # deep-3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid
        # deep-4
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 4 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid
        # # deep-5
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
        # pid=$(pgrep -f "adaptive_engine_example")
        # kill $pid

        # # Nightjar
        python run_benchmark_tests.py --strategy ilp --sub-strategy ada_bin_greedy --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --increase-block-threshold $increase_block_threshold --decrease-block-threshold $decrease_block_threshold --output-len $output_len &>> $FILE_NAME
        pid=$(pgrep -f "adaptive_engine_example")
        kill $pid
    # # #     # nospec
    #     python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len ${MAX_SPECULATIVE_LEN} --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    #     pid=$(pgrep -f "adaptive_engine_example")
    #     kill $pid

        
    #     sleep 3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
        # sleep 3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
        # sleep 3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy threshold --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
    #     sleep 3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
    #     sleep 3
    #     python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
    #     sleep 3
    #     python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization --output-len $output_len &>> $FILE_NAME
    done
done
# data_set_name=alpaca
# --dataset-path tatsu-lab/alpaca
# data_set_path=tatsu-lab/alpaca
# FILE_NAME=a154new.log
# rm -rf $FILE_NAME
# for PROMPT_RATE in 1 5 10 15
# do
#     python run_benchmark_tests.py --strategy ilp --sub-strategy daspec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --output-len $output_len &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --output-len $output_len&>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --output-len $output_len &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --output-len $output_len &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --output-len $output_len &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --output-len $output_len &>> $FILE_NAME
# done


# NUM_PROMPTS=480
# PROMPT_RATE=10   
# FILE_NAME=a162new.log #  150 是 1 qps 151 是10qps 152是在调教
# SPECULATIVE_LEN=5
# START_INDEX=0
# rm -rf $FILE_NAME
# data_set_name=specbench
# data_set_path=./question.jsonl # $data_set_path
# output_len=256
# num_gpu_blocks_override=26064
# #model_name=/root/autodl-tmp/deepseek 
# model_name=/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B
# draft_model_name=/data/model/DeepSeek-R1-DRAFT-Qwen2.5-0.5B
# #draft_model_name=/root/autodl-tmp/DeepSeek-R1-DRAFT-Qwen2.5-0.5B #/data/model/DeepSeek-R1-DRAFT-Qwen2.5-0.5B
# for PROMPT_RATE in 10 15
# do
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy daspec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --output-len $output_len --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --output-len $output_len --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --output-len $output_len --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --output-len $output_len --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --output-len $output_len --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --output-len $output_len --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
# done

# profile
# model_name=/data/model/Llama-3.1-8B
# draft_model_name=/data/model/eagle-llama3

# NUM_PROMPTS=480
# PROMPT_RATE=40   
# #  134 是100 sharegpt 136 是 $NUM_PROMPTS sharegpt
# SPECULATIVE_LEN=5
# ENABLE_TRACE="False"
# FILE_NAME=test.log
# num_gpu_blocks_override=12373 #26064 # deep A6000 26064 llama8b A6000 11466  通过下面命令行得到
# # python -m vllm.entrypoints.openai.api_server  --model /data/model/Llama-3.1-8B --gpu-memory-utilization 0.85 --speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 28432
# rm -rf $FILE_NAME
# data_set_name=specbench #alpaca    #--dataset-path tatsu-lab/alpaca
# data_set_path=./question_shuffled.jsonl # $data_set_path
# #model_name=/root/autodl-tmp/deepseek 
# model_name=/data/model/llama3instruct
# draft_model_name=/data/model/eagle-llama3

# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1> a122new.log 2> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1> a122new.log 2> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1>> a122new.log 2>> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name        --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_namet         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override 1>> a122new.log 2>> a122new_error.log 

