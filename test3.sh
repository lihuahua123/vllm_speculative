#python run_benchmark_tests.py --strategy ilp --sub-strategy ngram --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 442 --request-rate 10 1>> a120new.log 2>> a120new_error.log 
NUM_PROMPTS=300
PROMPT_RATE=5
FILE_NAME=dynamic_alpaca_llama3_tiaojiao49.log #  134 是100 sharegpt 136 是 $NUM_PROMPTS sharegpt
SPECULATIVE_LEN=2
START_INDEX=0
ENABLE_TRACE="True"
SAVE_TRACE="True"
num_gpu_blocks_override=4800 #26064 #9369 #A600 50% #4800 4090 # deep A6000 26064 llama8b A6000 xxx  通过下面命令行得到
gpu_memory_utilization=0.85
# python -m vllm.entrypoints.openai.api_server  --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B --gpu-memory-utilization 0.85 --speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 3000
rm -rf $FILE_NAME
# data_set_name=sharegpt
# data_set_path=/data/sharegpt.json # $data_set_path
# data_set_name=specbench    #--dataset-path tatsu-lab/alpaca
# data_set_path=./question_shuffled.jsonl # $data_set_path
data_set_name=alpaca
data_set_path=tatsu-lab/alpaca
burstiness=1.0
# output_len=256 # 
#model_name=/root/autodl-tmp/deepseek 
model_name=/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B
draft_model_name=/data/model/DeepSeek-R1-DRAFT-Qwen2.5-0.5B
# model_name=/data/model/llama3instruct
# draft_model_name=/data/model/yuhuiliEAGLE-LLaMA3-Instruct-8B-vllm
#/data/model/DeepSeek-R1-DRAFT-Qwen2.5-0.5B
for i in 1 # 2 3 4 5
do
    for PROMPT_RATE in  5 # 24 # 6 8 10 12 14 # 10 15 # 20 25 #0.5 2 5 10
    do
        # if [ $PROMPT_RATE -eq 5 ]; then
        #     explore="True"
        # else
        explore="False"
        # fi
        # sleep 3
        python run_benchmark_tests.py --strategy ilp --sub-strategy epsilon_greedy  --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        
        python run_benchmark_tests.py --strategy ilp --sub-strategy ucb --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        python run_benchmark_tests.py --strategy ilp --sub-strategy threshold --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        
        python run_benchmark_tests.py --strategy ilp --sub-strategy deep --select-strategy capacity --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        
        python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
       
        python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        
        python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        
        # sleep 3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        # sleep 3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
        # sleep 3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy threshold --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    #     sleep 3
        # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    #     sleep 3
    #     python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    #     sleep 3
    #     python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    done
done
# data_set_name=alpaca
# data_set_path=tatsu-lab/alpaca
# FILE_NAME=a154new.log
# rm -rf $FILE_NAME
# for PROMPT_RATE in 1 5 10 15
# do
#     python run_benchmark_tests.py --strategy ilp --sub-strategy daspec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override&>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
#     # python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override &>> $FILE_NAME
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

