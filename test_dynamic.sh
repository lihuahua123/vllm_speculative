#python run_benchmark_tests.py --strategy ilp --sub-strategy ngram --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 442 --request-rate 10 1>> a120new.log 2>> a120new_error.log 
NUM_PROMPTS=200
PROMPT_RATE=5
FILE_NAME=dynamic.log 
SPECULATIVE_LEN=3
START_INDEX=0
ENABLE_TRACE="True" # Dynamic
SAVE_TRACE="True"
num_gpu_blocks_override=717 
gpu_memory_utilization=0.85
# python -m vllm.entrypoints.openai.api_server  --model /root/autodl-tmp/vicuna-13b-v1.3 --gpu-memory-utilization 0.85 --speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 2048
rm -rf $FILE_NAME
data_set_name=sharegpt
data_set_path=/root/autodl-tmp/sharegpt.json # $data_set_path
# data_set_name=specbench    #--dataset-path tatsu-lab/alpaca
# data_set_path=./question_shuffled.jsonl # $data_set_path
# data_set_name=alpaca
# data_set_path=tatsu-lab/alpaca
burstiness=1.0
model_name=/root/autodl-tmp/vicuna-13b-v1.3
draft_model_name=/root/autodl-tmp/vicuna-68m
export HF_ENDPOINT='https://hf-mirror.com'
explore="False"
# Nightjar
for i in 1  2 3 4 5
do
    python run_benchmark_tests.py --strategy ilp --sub-strategy Nightjar --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    # BanditSpec
    python run_benchmark_tests.py --strategy ilp --sub-strategy ucb --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    # DSD
    python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --explore $explore --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    # TETRIS
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep  --select-strategy capacity --save-trace $SAVE_TRACE --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    # SD 1
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    # SD 2
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    # SD 3
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 4 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    # SD 4
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
    # SD 5
    python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --save-trace $SAVE_TRACE --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX --num-gpu-blocks-override $num_gpu_blocks_override --enable-trace "$ENABLE_TRACE" --burstiness $burstiness --gpu-memory-utilization $gpu_memory_utilization &>> $FILE_NAME
done