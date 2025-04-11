# Multi-Armed Bandit Optimization for vLLM

This feature introduces dynamic model switching and optimization based on system metrics and request load to maximize throughput and system efficiency.

## Overview

The multi-armed bandit implementation provides an automated way to dynamically choose between different decoding strategies:

1. **N-gram draft model** - Use n-gram draft models when system load is high to reduce memory consumption
2. **Neural draft model** - Use neural draft models when system load is low for higher quality speculative decoding
3. **Disable speculative decoding** - Fallback option when memory is constrained but we want to maintain quality

The optimizer monitors system metrics including:
- Throughput (tokens/second)
- Request load (number of active requests)
- Memory usage (KV cache utilization)

It then makes intelligent decisions about which strategy to use, learning from observed rewards over time.

## Components

- **`MultiArmedBanditOptimizer`**: Core UCB (Upper Confidence Bound) algorithm for decision making
- **`ThresholdSwitcher`**: Simpler, threshold-based approach for quick responses to changing load
- **`BanditOptimizationManager`**: Integration layer between the optimizer and LLMEngine
- **`LLMEngineWithBandit`**: Extension of LLMEngine with built-in bandit optimization

## How to Use

### 1. Direct Integration with LLMEngineWithBandit

The simplest way to use the bandit optimizer is to use `LLMEngineWithBandit` as a drop-in replacement for `LLMEngine`:

```python
from vllm.engine.llm_engine_with_bandit import LLMEngineWithBandit
from vllm.engine.arg_utils import AsyncEngineArgs

# Create engine args
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    
    # Enable speculative decoding with both models available
    speculative_config={
        "draft_model": "ngram",  # Default starting with n-gram
        "ngram_path": "cache/ngram_model",
        "ngram_config": {
            "ngram_level": 3
        },
        "num_speculative_tokens": 3
    },
)

# Create engine with bandit optimization
engine = LLMEngineWithBandit.from_engine_args(
    engine_args,
    bandit_monitoring_interval=5.0,  # Check every 5 seconds
    bandit_cooldown_period=30.0      # Minimum time between switches
)

# Use the engine normally - optimization happens automatically
```

### 2. Manual Integration with Existing LLMEngine

If you want to add bandit optimization to an existing LLMEngine:

```python
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.bandit_integration import BanditOptimizationManager

# Create your LLMEngine as usual
engine = LLMEngine.from_engine_args(engine_args)

# Create the bandit optimization manager
optimizer = BanditOptimizationManager(
    llm_engine=engine,
    monitoring_interval=5.0,
    cooldown_period=30.0
)

# In your inference loop:
while engine.has_unfinished_requests():
    outputs = engine.step()
    
    # Record token generation for throughput calculation
    tokens_generated = 0
    for output in outputs:
        for o in output.outputs:
            tokens_generated += len(o.token_ids)
    
    # Update the optimizer
    optimizer.record_tokens(tokens_generated)
    
    # Run optimization step
    action = optimizer.step()
    if action:
        print(f"Optimizer selected action: {action.name}")
```

## Configuration Options

### BanditOptimizationManager

- `monitoring_interval`: How often to check metrics and potentially switch models (seconds)
- `cooldown_period`: Minimum time between model switches (seconds)
- `throughput_window`: Number of measurements to use for calculating throughput
- `ngram_high_request_threshold`: Request count threshold to switch to n-gram
- `neural_low_request_threshold`: Request count threshold to switch to neural
- `memory_high_threshold`: Memory usage threshold to disable speculative decoding
- `memory_low_threshold`: Memory usage threshold to re-enable speculative decoding

### MultiArmedBanditOptimizer

- `exploration_weight`: Controls exploration vs. exploitation (higher = more exploration)
- `reward_window_size`: Number of rewards to keep in history
- `min_samples_per_arm`: Minimum samples before using UCB algorithm
- `throughput_weight`: Weight of throughput in reward calculation
- `memory_weight`: Weight of memory efficiency in reward calculation

## Example

See `examples/bandit_optimization_example.py` for a complete example of using the bandit optimization with different request loads.

## Requirements

The bandit optimization depends on the following vLLM features:

1. Speculative decoding with support for both n-gram and neural draft models
2. Dynamic model switching in the LLMEngine
3. Ability to enable/disable speculative decoding at runtime
4. Access to memory usage metrics through the block manager

Make sure your vLLM version supports these features.

## Testing

Unit tests for the bandit optimizer are available in `tests/engine/test_bandit_optimizer.py`. 