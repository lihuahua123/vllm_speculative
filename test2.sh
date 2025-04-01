
model_name=Llama-2-7b-hf
file_name=llama7b_68m
rm logs2/$file_name.log
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 0.5 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 1 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 2 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 3 &>> logs2/$file_name.log