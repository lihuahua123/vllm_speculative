
#python run_benchmark_tests.py --strategy ilp --sub-strategy ngram --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 442 --request-rate 10 1>> a120new.log 2>> a120new_error.log 
NUM_PROMPTS=400
PROMPT_RATE=10   
FILE_NAME=a148new.log #  134 是100 sharegpt 136 是 300 sharegpt
SPECULATIVE_LEN=5
START_INDEX=300
rm -rf $FILE_NAME
data_set_name=sharegpt
data_set_path=/data/sharegpt.json # $data_set_path
#model_name=/root/autodl-tmp/deepseek 
model_name=/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B
draft_model_name=/data/model/DeepSeek-R1-DRAFT-Qwen2.5-0.5B
#draft_model_name=/root/autodl-tmp/DeepSeek-R1-DRAFT-Qwen2.5-0.5B #/data/model/DeepSeek-R1-DRAFT-Qwen2.5-0.5B
for PROMPT_RATE in 1 5 10 15 #0.5 2 5 10
do
    python run_benchmark_tests.py --strategy ilp --sub-strategy daspec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
done

data_set_name=alpaca
data_set_path=tatsu-lab/alpaca
FILE_NAME=a147new.log
rm -rf $FILE_NAME
for PROMPT_RATE in 1 5 10 15
do
    python run_benchmark_tests.py --strategy ilp --sub-strategy daspec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy smart_spec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 1 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 3 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len 5 --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
    python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model $model_name --draft-model $draft_model_name       --dataset-name $data_set_name         --dataset-path $data_set_path   --speculative-len $SPECULATIVE_LEN --num-prompts $NUM_PROMPTS --request-rate $PROMPT_RATE --start-index $START_INDEX &>> $FILE_NAME
done

# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts 300 --request-rate 10 1> a122new.log 2> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts 300 --request-rate 5 1>> a122new.log 2>> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts 300 --request-rate 5 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts 300 --request-rate 5 1>> a122new.log 2>> a122new_error.log  
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts 300 --request-rate 5 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy deep --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts 300 --request-rate 5 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 1 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 2 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 3 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 4 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy  nospec --model $model_name --draft-model $draft_model_name       --dataset-name sharegpt         --dataset-path $data_set_path   --profile --speculative-len 5 --num-prompts 300 --request-rate 10 1>> a122new.log 2>> a122new_error.log 


# python run_benchmark_tests.py --strategy ilp --sub-strategy daspec --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 100 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
#python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 100 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy daspec --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 100 --request-rate 5 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 100 --request-rate 5 1>> a122new.log 2>> a122new_error.log
# python run_benchmark_tests.py --strategy ilp --sub-strategy daspec --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 100 --request-rate 10 1>> a122new.log 2>> a122new_error.log 
# python run_benchmark_tests.py --strategy ilp --sub-strategy nospec --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json   --num-prompts 100 --request-rate 10 1>> a122new.log 2>> a122new_error.log