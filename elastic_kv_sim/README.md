# Elastic KV Simulator

This directory contains a standalone simulator for Nightjar/vLLM dynamic memory
behavior:

- KV cache expansion, matching `CacheEngine.increase_gpu_blocks()` with
  `torch.nn.functional.pad`.
- KV cache shrink migration, matching the block-migration path in
  `CacheEngine.decrease_gpu_blocks()`. In benchmark mode the default
  `--shrink-backend triton` directly launches the
  `kv_cache_block_migration_kernel` Triton kernel for live KV-block movement;
  `--shrink-backend torch` keeps the PyTorch fallback.
- Draft model offload/reload, modeled as GPU-to-CPU and CPU-to-GPU tensor
  movement based on draft weight size.

Example:

```bash
python elastic_kv_sim/simulate_elastic_kv.py \
  --current-blocks 4096 \
  --change-blocks 1024 \
  --migrated-blocks 200 \
  --num-layers 32 \
  --block-size 16 \
  --num-kv-heads 32 \
  --head-size 128 \
  --dtype float16 \
  --draft-model-size 1GiB
```

To execute the Triton live-KV migration path explicitly:

```bash
python elastic_kv_sim/simulate_elastic_kv.py \
  --model-config Qwen_32B.conf.json \
  --current-blocks 4096 \
  --change-blocks 1024 \
  --migrated-blocks 512 \
  --draft-model-size 1GiB \
  --shrink-backend triton
```

Using a HuggingFace-style model config, for example the Qwen 32B config in this
repo:

```bash
python elastic_kv_sim/simulate_elastic_kv.py \
  --model-config Qwen_32B.conf.json \
  --current-blocks 4096 \
  --change-blocks 1024 \
  --migrated-blocks 200 \
  --draft-model-size 1GiB \
  --transfer-concurrency 8 \
  --estimate-only
```

For a fast estimate without CUDA allocation:

```bash
python elastic_kv_sim/simulate_elastic_kv.py \
  --current-blocks 4096 \
  --change-blocks 1024 \
  --draft-model-size 1GiB \
  --estimate-only \
  --gpu-mem-bw-gib-s 700 \
  --h2d-bw-gib-s 23 \
  --d2h-bw-gib-s 23
```

Notes:

- `--change-blocks` is the number of blocks added during expansion or removed
  during contraction.
- `--migrated-blocks` is the number of live blocks in the removed tail region
  that must be moved to lower block IDs during contraction. If omitted, it
  defaults to `--change-blocks`.
- `--transfer-concurrency` estimates the wall-clock completion time when
  multiple draft models concurrently offload/reload through the same host--GPU
  path. This is the relevant quantity for PCIe saturation.
- `--model-config` reads `num_hidden_layers`, `num_key_value_heads`,
  `hidden_size`, `num_attention_heads`, and `torch_dtype` from a config JSON.
- The default benchmark mode runs small bounded samples and scales by bytes, so
  it can estimate very large block counts without allocating the full requested
  KV cache.
- `shrink_migration` estimates the active block movement path. `shrink_full_trim_copy`
  is a conservative full-copy reference if the implementation copies all kept
  blocks into a new tensor.
