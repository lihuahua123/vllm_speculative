
model_name=llama3instruct
file_name=llama3instruct_ngram_small_qps
rm logs2/$file_name.log
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 0.2 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 0.6 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 1 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 1.4 &>> logs2/$file_name.log
