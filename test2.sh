
model_name=deepseek-aiDeepSeek-R1-Distill-Qwen-7B
file_name=deepseek_qps_deepseek_my
rm logs2/$file_name.log
# python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 1 --goodput tpot:30 &>> logs2/$file_name.log
# sleep 2
# python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 2 --goodput tpot:30 &>> logs2/$file_name.log
# sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 4 --goodput tpot:30 &>> logs2/$file_name.log
sleep 2
python benchmarks/benchmark_serving.py         --backend vllm        --model /data/model/$model_name  --dataset-name sharegpt    --dataset-path /data/sharegpt.json      --num-prompts 20 --request-rate 8 --goodput tpot:30 &>> logs2/$file_name.log
