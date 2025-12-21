

# profile
model_name=/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B
draft_model_name=/root/autodl-tmp/deep05b

NUM_PROMPTS=480
NUM_PROMPTS_FOR_4_5=100
PROMPT_RATE=40   
#  134 是100 sharegpt 136 是 $NUM_PROMPTS sharegpt
SPECULATIVE_LEN=5
ENABLE_TRACE="False"
FILE_NAME=test.log
num_gpu_blocks_override=4681 #26064 # deep A6000 26064 llama8b A6000 11466  通过下面命令行得到
gpu_memory_utilization=0.85  # GPU 内存利用率，可以调整为 0.65, 0.75, 0.85 等
# python -m vllm.entrypoints.openai.api_server  --model /data/model/Llama-3.1-8B --gpu-memory-utilization 0.85 --speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 28432
rm -rf $FILE_NAME
data_set_name=specbench #alpaca    #--dataset-path tatsu-lab/alpaca
data_set_path=./question_shuffled.jsonl # $data_set_path

# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1> a122new.log 2> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1> a122new.log 2> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts ${NUM_PROMPTS_FOR_4_5} --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts ${NUM_PROMPTS_FOR_4_5} --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1>> a122new.log 2>> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name        --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1>> a122new.log 2>> a122new_error.log 
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1>> a122new.log 2>> a122new_error.log 
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1>> a122new.log 2>> a122new_error.log 
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_namet         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts ${NUM_PROMPTS_FOR_4_5} --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1>> a122new.log 2>> a122new_error.log 
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts ${NUM_PROMPTS_FOR_4_5} --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization 1>> a122new.log 2>> a122new_error.log 

