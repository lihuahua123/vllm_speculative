NUM_PROMPTS=200
PROMPT_RATE=10
num_gpu_blocks_override=717
gpu_memory_utilization=0.85
data_set_name=sharegpt
data_set_path=/root/autodl-tmp/sharegpt.json # $data_set_path
model_name=/root/autodl-tmp/vicuna-13b-v1.3
draft_model_name=/root/autodl-tmp/vicuna-68m


python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override  --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override   --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override  --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override   --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name        --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override  --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override  --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override  --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override  --gpu-memory-utilization $gpu_memory_utilization
python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --num-gpu-blocks-override $num_gpu_blocks_override  --gpu-memory-utilization $gpu_memory_utilization

# python ./exps/profile22.py
