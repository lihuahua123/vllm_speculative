model_name=deepseek-aiDeepSeek-R1-Distill-Qwen-7B
dirs=logs3
num_prompts=30
file_name=deepseek_qps_ngram_${num_prompts}_new.log
# rm logs2/$file_name.log
    
# running_cmd='VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server  --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B --gpu-memory-utilization 0.9 --enforce-eager  --no-enable-prefix-caching --max-model-len 9432  --disable_switch_draft_model  &> a.log'
running_cmd='VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server  --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B --gpu-memory-utilization 0.9 speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 9432  --disable_switch_draft_model'
# running_cmd='VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server  --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B --gpu-memory-utilizspeculative-model alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 9432  --disable_switch_draft_model'
echo "[" >> $dirs/$file_name
# Escape the running_cmd by wrapping it in single quotes to avoid interpretation of spaces as argument separators
python benchmarks/benchmark_serving.py --backend vllm --model /data/model/$model_name --dataset-name sharegpt --dataset-path /data/sharegpt.json --num-prompts $num_prompts --request-rate 5 --goodput tpot:30 --save-result --result-dir $dirs/ --percentile-metrics ttft,tpot,itl,e2el --result-filename $file_name  --metadata "running_cmd='$running_cmd'"
echo "," >> $dirs/$file_name
sleep 2
python benchmarks/benchmark_serving.py --backend vllm --model /data/model/$model_name --dataset-name sharegpt --dataset-path /data/sharegpt.json --num-prompts $num_prompts --request-rate 10 --goodput tpot:30 --save-result --result-dir $dirs/ --percentile-metrics ttft,tpot,itl,e2el --result-filename $file_name  --metadata "running_cmd='$running_cmd'"
echo "," >> $dirs/$file_name
sleep 2
python benchmarks/benchmark_serving.py --backend vllm --model /data/model/$model_name --dataset-name sharegpt --dataset-path /data/sharegpt.json --num-prompts $num_prompts --request-rate 15 --goodput tpot:30 --save-result --result-dir $dirs/ --percentile-metrics ttft,tpot,itl,e2el --result-filename $file_name  --metadata "running_cmd='$running_cmd'"
echo "," >> $dirs/$file_name
sleep 2
python benchmarks/benchmark_serving.py --backend vllm --model /data/model/$model_name --dataset-name sharegpt --dataset-path /data/sharegpt.json --num-prompts $num_prompts --request-rate 20 --goodput tpot:30 --save-result --result-dir $dirs/ --percentile-metrics ttft,tpot,itl,e2el --result-filename $file_name  --metadata "running_cmd='$running_cmd'"
echo "]" >> $dirs/$file_name
