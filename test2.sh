
model_name=deepseek-aiDeepSeek-R1-Distill-Qwen-7B
file_name=deepseek_SLO_big_reqrate_deep05b
rm logs2/$file_name.log
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 300 --request-rate 5 --goodput tpot:30 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 300 --request-rate 10 --goodput tpot:30 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 300 --request-rate 15 --goodput tpot:30 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 300 --request-rate 20 --goodput tpot:30 &>> logs2/$file_name.log
