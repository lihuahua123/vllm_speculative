python -m vllm.entrypoints.openai.api_server  --model /root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B --gpu-memory-utilization 0.85 --speculative-model [ngram] --ngram_prompt_lookup_max 4 --num-speculative-tokens 4 --enforce-eager  --no-enable-prefix-caching --max-model-len 2048 --tensor_parallel_size 1

利用这个命令行得到7B在4090 最多可以num_usable_gpu_blocks 4938 个block，这是没有草稿模型的时候

NUM_GPU_BLOCKS_OVERRIDE=4038 \
  INCREASE_BLOCK_THRESHOLD=1000 \
  REQUEST_RATE=20 \
  NUM_PROMPTS=350 \
  ./test_t_persist_sensitivity.sh 1