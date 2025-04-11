# Q-Learning for Dynamic Speculative Decoding in vLLM

This extension to vLLM adds a Q-learning based approach for dynamically managing speculative decoding strategies based on system load, memory usage, and throughput.

## Overview

Speculative decoding in vLLM can use different draft models:
1. Neural draft model - more accurate, but consumes more memory
2. N-gram draft model - less accurate, but lightweight
3. No speculative decoding - fallback when resources are constrained

This implementation uses Reinforcement Learning (Q-learning specifically) to dynamically switch between these options based on system state.

## Implementation

The implementation consists of four main components:

1. `q_learning_optimizer.py` - Core Q-learning implementation
2. `q_learning_integration.py` - Integration with the LLMEngine
3. `llm_engine_with_q_learning.py` - Extension of LLMEngine that uses Q-learning
4. `adaptive_engine_factory.py` - Factory to create engines with different optimization strategies

## Markov Decision Process (MDP) Formulation

The Q-learning approach models the speculative decoding decision as a Markov Decision Process:

- **State Space**: 
  - Request load (discretized into buckets)
  - Memory usage (discretized into buckets)
  - Current model state (ngram, neural, or disabled)

- **Action Space**:
  - SWITCH_TO_NGRAM - Switch to n-gram draft model
  - SWITCH_TO_NEURAL - Switch to neural draft model
  - DISABLE_SPEC_DECODING - Disable speculative decoding

- **Reward Function**:
  - Primary Component: Throughput (tokens/second)
  - Penalty Component: Memory usage penalty (to discourage excessive memory consumption)
  
  Reward = Throughput - λ * Memory_Usage^2

- **State Transition**: Deterministic based on actions taken

## Q-learning Algorithm

The Q-learning algorithm uses an epsilon-greedy policy with the standard Q-update rule:

```
Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
```

Where:
- Q(s,a) is the current Q-value
- α is the learning rate
- r is the reward
- γ is the discount factor for future rewards
- max(Q(s',a')) is the maximum Q-value for the next state
- The exploration rate (epsilon) decays over time

## Using the Implementation

### Option 1: Using the Factory

The simplest way to use this implementation is through the `adaptive_engine_factory.py`:

```python
from vllm.engine.adaptive_engine_factory import create_adaptive_engine

# Create an engine with Q-learning
engine = create_adaptive_engine(
    engine_args=engine_args,
    optimization_strategy="q_learning",
    monitoring_interval=5.0,
    cooldown_period=30.0,
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=0.2
)

# Run inference as usual
output = engine.generate("Your prompt here", SamplingParams())
```

### Option 2: Direct Instantiation

You can also directly instantiate the `LLMEngineWithQLearning` class:

```python
from vllm.engine.llm_engine_with_q_learning import LLMEngineWithQLearning

engine = LLMEngineWithQLearning.from_engine_args(
    engine_args=engine_args,
    q_learning_monitoring_interval=5.0,
    q_learning_cooldown_period=30.0,
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=0.2
)
```

### Training and Evaluating

For best results, you should train the Q-learning model with representative workloads:

```python
await engine.train_and_evaluate_q_learning(
    training_requests=training_requests,
    test_requests=test_requests,
    sampling_params=sampling_params,
    training_iterations=100,
    request_rates=[1.0, 2.0, 4.0, 8.0],
    save_q_table=True
)
```

After training, you can save the Q-table for future use:

```python
engine.save_q_table("q_table.json")
```

And load it in a future session:

```python
engine = create_adaptive_engine(
    engine_args=engine_args,
    optimization_strategy="q_learning",
    load_model_path="q_table.json"
)
```

## Comparison with Bandit Optimization

This implementation provides an alternative to the Multi-Armed Bandit approach. Key differences:

1. **State Awareness**: Q-learning maintains a state representation, allowing it to learn different policies for different system states.

2. **Temporal Credit Assignment**: Q-learning can propagate rewards back through time, learning from sequences of decisions.

3. **Exploration Strategy**: Uses epsilon-greedy policy with decaying exploration rate.

4. **Memory Requirements**: Requires more memory to store the Q-table for all state-action pairs.

## Example

See `examples/adaptive_engine_example.py` for a complete example that:
1. Trains both bandit and Q-learning optimizers
2. Tests them with different request rates
3. Compares their performance
4. Generates visualizations

Run it with:
```
python examples/adaptive_engine_example.py --model facebook/opt-125m --strategy both
```

## Configuration Parameters

Key parameters for the Q-learning approach:

- `learning_rate`: How quickly new information overrides old (alpha in the Q-learning equation)
- `discount_factor`: Weight given to future rewards (gamma in the Q-learning equation)
- `exploration_rate`: Initial probability of choosing random actions
- `exploration_decay`: Rate at which exploration decreases
- `min_exploration_rate`: Minimum exploration probability
- `memory_penalty_coefficient`: Weight for the memory usage penalty
- `request_load_buckets`: Thresholds for discretizing request load
- `memory_usage_buckets`: Thresholds for discretizing memory usage 