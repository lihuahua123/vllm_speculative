import numpy as np
import math
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import random

logger = logging.getLogger(__name__)

class QLearningAction(Enum):
    """Actions that the Q-learning optimizer can take"""
    SWITCH_TO_NGRAM = 0
    SWITCH_TO_NEURAL = 1
    DISABLE_SPEC_DECODING = 2

class QLearningOptimizer:
    """Q-learning optimizer for dynamic model selection in speculative decoding.
    
    This implements a tabular Q-learning approach with discretized state space to learn
    the optimal policy for switching between neural and n-gram draft models, or
    disabling speculative decoding altogether based on system state.
    
    The state space consists of:
    - Request load (discretized into buckets)
    - Memory usage (discretized into buckets)
    - Current draft model type (neural, ngram, or disabled)
    
    The reward function prioritizes throughput while penalizing high memory usage.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 0.2,
                 exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01,
                 memory_penalty_coefficient: float = 0.3,
                 request_load_buckets: List[int] = [5, 10, 15, 20, 25, 30],
                 memory_usage_buckets: List[float] = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
                 cooldown_period: float = 30.0):
        """Initialize the Q-learning optimizer.
        
        Args:
            learning_rate: Alpha parameter controlling how much new information overrides old
            discount_factor: Gamma parameter for valuing future rewards
            exploration_rate: Initial epsilon value for exploration vs exploitation
            exploration_decay: Factor to multiply exploration_rate by after each learning step
            min_exploration_rate: Minimum exploration rate
            memory_penalty_coefficient: Weight for memory usage penalty in reward function
            request_load_buckets: Thresholds for discretizing request load
            memory_usage_buckets: Thresholds for discretizing memory usage
            cooldown_period: Minimum time between actions in seconds
        """
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.memory_penalty_coefficient = memory_penalty_coefficient
        
        # State discretization
        self.request_load_buckets = request_load_buckets
        self.memory_usage_buckets = memory_usage_buckets
        
        # Tracking current state
        self.using_ngram_model = False
        self.using_neural_model = True  # Default is usually neural
        self.spec_decoding_disabled = False
        
        # Timing and cooldown
        self.last_action_time = 0.0
        self.cooldown_period = cooldown_period
        
        # Initialize Q-table
        self.q_table = {}
        self._initialize_q_table()
        
        # Metrics tracking
        self.last_action = None
        self.last_state = None
        self.last_reward = None
        self.throughput_history = []
        self.request_load_history = []
        self.memory_usage_history = []
        self.current_state = None
        self.baseline_throughput = None
        
        # Training metrics
        self.episode_count = 0
        self.total_rewards = 0
        self.step_count = 0
    
    def _initialize_q_table(self):
        """Initialize the Q-table with zeros for all state-action pairs."""
        # Generate all possible states
        # State format: (request_load_bucket, memory_usage_bucket, using_ngram, using_neural, spec_disabled)
        for req_idx in range(len(self.request_load_buckets) + 1):
            for mem_idx in range(len(self.memory_usage_buckets) + 1):
                # Three possible model states: ngram, neural, or disabled speculative decoding
                self.q_table[self._state_key(req_idx, mem_idx, True, False, False)] = {
                    action: 0.0 for action in QLearningAction
                }
                self.q_table[self._state_key(req_idx, mem_idx, False, True, False)] = {
                    action: 0.0 for action in QLearningAction
                }
                self.q_table[self._state_key(req_idx, mem_idx, False, False, True)] = {
                    action: 0.0 for action in QLearningAction
                }
        
        logger.info(f"Initialized Q-table with {len(self.q_table)} states")
    
    def _state_key(self, 
                  request_load_bucket: int, 
                  memory_usage_bucket: int, 
                  using_ngram: bool, 
                  using_neural: bool, 
                  spec_disabled: bool) -> str:
        """Convert state components to a hashable key for the Q-table."""
        return f"{request_load_bucket}_{memory_usage_bucket}_{int(using_ngram)}_{int(using_neural)}_{int(spec_disabled)}"
    
    def _discretize_request_load(self, request_load: int) -> int:
        """Convert continuous request load to discrete bucket index."""
        for i, threshold in enumerate(self.request_load_buckets):
            if request_load < threshold:
                return i
        return len(self.request_load_buckets)
    
    def _discretize_memory_usage(self, memory_usage: float) -> int:
        """Convert continuous memory usage to discrete bucket index."""
        for i, threshold in enumerate(self.memory_usage_buckets):
            if memory_usage < threshold:
                return i
        return len(self.memory_usage_buckets)
    
    def _get_current_state_key(self) -> str:
        """Get the key for the current state."""
        if self.current_state is None:
            # Default to middle buckets if no state information available
            request_bucket = len(self.request_load_buckets) // 2
            memory_bucket = len(self.memory_usage_buckets) // 2
        else:
            request_bucket = self._discretize_request_load(self.current_state["request_load"])
            memory_bucket = self._discretize_memory_usage(self.current_state["memory_usage"])
        
        return self._state_key(
            request_bucket, 
            memory_bucket,
            self.using_ngram_model,
            self.using_neural_model,
            self.spec_decoding_disabled
        )
    
    def _calculate_reward(self, current_throughput: float, memory_usage: float) -> float:
        """Calculate reward based on throughput and memory usage.
        
        Args:
            current_throughput: Current tokens per second throughput
            memory_usage: Current memory usage as a fraction (0.0-1.0)
            
        Returns:
            Reward value combining throughput and memory penalty
        """
        # Set baseline throughput if not set
        if self.baseline_throughput is None and len(self.throughput_history) >= 3:
            self.baseline_throughput = np.mean(self.throughput_history)
            
        throughput_component = current_throughput
        
        # Apply memory penalty that increases exponentially as usage approaches 1.0
        memory_penalty = self.memory_penalty_coefficient * (memory_usage ** 2)
        
        # Combine the components
        reward = throughput_component - memory_penalty
        
        return reward
    
    def record_metrics(self, throughput: float, request_load: int, memory_usage: float) -> None:
        """Record current system metrics for state tracking.
        
        Args:
            throughput: Current tokens per second
            request_load: Number of active requests
            memory_usage: Memory usage ratio (0.0-1.0)
        """
        self.throughput_history.append(throughput)
        self.request_load_history.append(request_load)
        self.memory_usage_history.append(memory_usage)
        
        # Keep history bounded
        max_history = 20
        if len(self.throughput_history) > max_history:
            self.throughput_history.pop(0)
            self.request_load_history.pop(0)
            self.memory_usage_history.pop(0)
        
        # Update current state
        self.current_state = {
            "throughput": throughput,
            "request_load": request_load,
            "memory_usage": memory_usage
        }
    
    def update_reward(self, new_throughput: float, new_memory_usage: float) -> None:
        """Process reward for the last action and update Q-table.
        
        Args:
            new_throughput: Observed throughput after last action
            new_memory_usage: Observed memory usage after last action
        """
        if self.last_action is None or self.last_state is None:
            return
            
        # Calculate reward
        reward = self._calculate_reward(new_throughput, new_memory_usage)
        self.last_reward = reward
        self.total_rewards += reward
        
        # Get current state key after the action
        current_state_key = self._get_current_state_key()
        
        # Q-learning update
        old_q_value = self.q_table[self.last_state][self.last_action]
        
        # Get max future Q-value
        max_future_q = max(self.q_table[current_state_key].values())
        
        # Q-learning formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        new_q_value = old_q_value + self.learning_rate * (
            reward + self.discount_factor * max_future_q - old_q_value
        )
        
        # Update Q-value
        self.q_table[self.last_state][self.last_action] = new_q_value
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
        # Log the update
        logger.debug(f"Q-learning update: state={self.last_state}, action={self.last_action.name}, "
                    f"reward={reward:.2f}, new_q={new_q_value:.2f}, "
                    f"exploration_rate={self.exploration_rate:.3f}")
        
        self.step_count += 1
    
    def select_action(self, throughput: float, request_load: int, memory_usage: float) -> Optional[QLearningAction]:
        """Select the best action using epsilon-greedy policy.
        
        Args:
            throughput: Current tokens per second
            request_load: Number of active requests
            memory_usage: Memory usage ratio (0.0-1.0)
            
        Returns:
            Selected action or None if in cooldown period
        """
        # Record metrics
        self.record_metrics(throughput, request_load, memory_usage)
        
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown_period:
            return None
        
        # Get current state key
        current_state_key = self._get_current_state_key()
        
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            # Exploration: choose random action
            action = random.choice(list(QLearningAction))
            logger.debug(f"Exploring with random action: {action.name}")
        else:
            # Exploitation: choose best action according to Q-table
            q_values = self.q_table[current_state_key]
            max_q = max(q_values.values())
            
            # Get all actions with the maximum Q-value (could be multiple)
            best_actions = [action for action, q_val in q_values.items() 
                           if q_val == max_q]
            action = random.choice(best_actions)
            logger.debug(f"Exploiting with best action: {action.name}, Q={max_q:.2f}")
        
        # Validate the action based on current state
        if action == QLearningAction.SWITCH_TO_NGRAM and self.using_ngram_model:
            logger.debug("Already using ngram model, skipping action")
            return None
        
        if action == QLearningAction.SWITCH_TO_NEURAL and self.using_neural_model:
            logger.debug("Already using neural model, skipping action")
            return None
        
        if action == QLearningAction.DISABLE_SPEC_DECODING and self.spec_decoding_disabled:
            logger.debug("Speculative decoding already disabled, skipping action")
            return None
        
        # Store current state for Q-learning update
        self.last_state = current_state_key
        self.last_action = action
        self.last_action_time = current_time
        
        # Update model state flags
        if action == QLearningAction.SWITCH_TO_NGRAM:
            self.using_ngram_model = True
            self.using_neural_model = False
            self.spec_decoding_disabled = False
        elif action == QLearningAction.SWITCH_TO_NEURAL:
            self.using_ngram_model = False
            self.using_neural_model = True
            self.spec_decoding_disabled = False
        elif action == QLearningAction.DISABLE_SPEC_DECODING:
            self.using_ngram_model = False
            self.using_neural_model = False
            self.spec_decoding_disabled = True
            
        logger.info(f"Selected action: {action.name}, state: {current_state_key}")
        return action
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and metrics of the Q-learning optimizer."""
        state_key = self._get_current_state_key() if self.current_state else None
        
        # Get top actions for current state
        top_actions = []
        if state_key and state_key in self.q_table:
            q_values = self.q_table[state_key]
            sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
            top_actions = [(action.name, round(q_val, 3)) for action, q_val in sorted_actions]
        
        return {
            "current_metrics": self.current_state,
            "last_action": self.last_action.name if self.last_action else None,
            "last_reward": self.last_reward,
            "exploration_rate": self.exploration_rate,
            "using_ngram_model": self.using_ngram_model,
            "using_neural_model": self.using_neural_model,
            "spec_decoding_disabled": self.spec_decoding_disabled,
            "current_state": state_key,
            "top_q_values": top_actions,
            "episode_count": self.episode_count,
            "total_rewards": self.total_rewards,
            "step_count": self.step_count,
            "q_table_size": len(self.q_table)
        }
        
    def save_q_table(self, file_path: str) -> None:
        """Save the Q-table to a file.
        
        Args:
            file_path: Path to save the Q-table
        """
        import json
        
        # Convert Q-table to serializable format
        serializable_q_table = {}
        for state_key, actions in self.q_table.items():
            serializable_q_table[state_key] = {action.name: q_val for action, q_val in actions.items()}
        
        data = {
            "q_table": serializable_q_table,
            "params": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate,
                "request_load_buckets": self.request_load_buckets,
                "memory_usage_buckets": self.memory_usage_buckets,
                "memory_penalty_coefficient": self.memory_penalty_coefficient
            },
            "metrics": {
                "episode_count": self.episode_count,
                "total_rewards": self.total_rewards,
                "step_count": self.step_count
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Q-table saved to {file_path}")
        
    def load_q_table(self, file_path: str) -> None:
        """Load a Q-table from a file.
        
        Args:
            file_path: Path to load the Q-table from
        """
        import json
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Restore parameters if available
        params = data.get("params", {})
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.discount_factor = params.get("discount_factor", self.discount_factor)
        self.exploration_rate = params.get("exploration_rate", self.exploration_rate)
        self.request_load_buckets = params.get("request_load_buckets", self.request_load_buckets)
        self.memory_usage_buckets = params.get("memory_usage_buckets", self.memory_usage_buckets)
        self.memory_penalty_coefficient = params.get("memory_penalty_coefficient", 
                                                    self.memory_penalty_coefficient)
        
        # Restore metrics if available
        metrics = data.get("metrics", {})
        self.episode_count = metrics.get("episode_count", 0)
        self.total_rewards = metrics.get("total_rewards", 0)
        self.step_count = metrics.get("step_count", 0)
        
        # Restore Q-table
        serialized_q_table = data.get("q_table", {})
        self.q_table = {}
        
        for state_key, actions in serialized_q_table.items():
            self.q_table[state_key] = {
                QLearningAction[action_name]: q_val 
                for action_name, q_val in actions.items()
            }
        
        logger.info(f"Loaded Q-table from {file_path} with {len(self.q_table)} states") 