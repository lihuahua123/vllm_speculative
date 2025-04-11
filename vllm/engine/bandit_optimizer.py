import numpy as np
import math
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class BanditAction(Enum):
    """Actions that the bandit optimizer can take"""
    SWITCH_TO_NGRAM = 0
    SWITCH_TO_NEURAL = 1
    DISABLE_SPEC_DECODING = 2

class ThresholdSwitcher:
    """A simple threshold-based switcher that tracks token generation rate"""
    def __init__(self, 
                 window_size: int = 5,
                 high_threshold: float = 40.0,  # tokens/s
                 low_threshold: float = 20.0,   # tokens/s
                 cooldown_period: float = 10.0):  # seconds
        self.token_history: List[int] = []
        self.timestamp_history: List[float] = []
        self.window_size = window_size
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.cooldown_period = cooldown_period
        self.last_switch_time = 0.0
    
    def record_tokens(self, tokens_generated: int) -> None:
        """Record tokens generated in the latest step"""
        self.token_history.append(tokens_generated)
        self.timestamp_history.append(time.time())
        
        # Keep only the last window_size entries
        if len(self.token_history) > self.window_size:
            self.token_history.pop(0)
            self.timestamp_history.pop(0)
    
    def get_current_throughput(self) -> float:
        """Calculate current throughput in tokens per second"""
        if len(self.token_history) < 2:
            return 0.0
        
        total_tokens = sum(self.token_history)
        time_span = self.timestamp_history[-1] - self.timestamp_history[0]
        
        if time_span <= 0:
            return 0.0
            
        return total_tokens / time_span
    
    def check_and_update(self) -> Optional[BanditAction]:
        """Check throughput and return a suggested action if needed"""
        throughput = self.get_current_throughput()
        current_time = time.time()
        
        # Don't switch if in cooldown period
        if current_time - self.last_switch_time < self.cooldown_period:
            return None
            
        if throughput > self.high_threshold:
            self.last_switch_time = current_time
            return BanditAction.SWITCH_TO_NGRAM
        elif throughput < self.low_threshold:
            self.last_switch_time = current_time
            return BanditAction.SWITCH_TO_NEURAL
        
        return None

class MultiArmedBanditOptimizer:
    """A UCB-based multi-armed bandit optimizer for dynamic model switching."""
    
    def __init__(self, 
                 exploration_weight: float = 2.0,
                 reward_window_size: int = 10,
                 min_samples_per_arm: int = 3,
                 cooldown_period: float = 30.0,  # seconds
                 throughput_weight: float = 0.7,
                 memory_weight: float = 0.3):
        # Initialize counts for each arm
        self.counts = {
            BanditAction.SWITCH_TO_NGRAM: 0,
            BanditAction.SWITCH_TO_NEURAL: 0,
            BanditAction.DISABLE_SPEC_DECODING: 0
        }
        
        # Initialize reward history for each arm
        self.reward_history = {
            BanditAction.SWITCH_TO_NGRAM: [],
            BanditAction.SWITCH_TO_NEURAL: [],
            BanditAction.DISABLE_SPEC_DECODING: []
        }
        
        # Running metrics
        self.throughput_history = []
        self.request_load_history = []
        self.memory_usage_history = []
        
        # State tracking
        self.last_action: Optional[BanditAction] = None
        self.last_action_time = 0.0
        self.baseline_throughput = None
        self.current_state = None
        
        # Configuration
        self.exploration_weight = exploration_weight
        self.reward_window_size = reward_window_size
        self.min_samples_per_arm = min_samples_per_arm
        self.cooldown_period = cooldown_period
        self.throughput_weight = throughput_weight
        self.memory_weight = memory_weight
        
        # Current model state
        self.using_ngram_model = False
        self.using_neural_model = False
        self.spec_decoding_disabled = False
    
    def record_metrics(self, throughput: float, request_load: int, memory_usage: float) -> None:
        """Record current system metrics"""
        self.throughput_history.append(throughput)
        self.request_load_history.append(request_load)
        self.memory_usage_history.append(memory_usage)
        
        # Keep history bounded
        if len(self.throughput_history) > self.reward_window_size:
            self.throughput_history.pop(0)
            self.request_load_history.pop(0)
            self.memory_usage_history.pop(0)
        
        # Set baseline throughput if not set
        if self.baseline_throughput is None and len(self.throughput_history) >= 3:
            self.baseline_throughput = np.mean(self.throughput_history)
        
        # Store current state for reward calculation
        self.current_state = {
            "throughput": throughput,
            "request_load": request_load,
            "memory_usage": memory_usage
        }
    
    def _calculate_reward(self, current_throughput: float, memory_usage: float) -> float:
        """Calculate reward based on throughput improvement and memory usage"""
        # If no baseline, can't calculate reward
        if self.baseline_throughput is None:
            return 0.0
        
        # Calculate throughput improvement
        throughput_improvement = (current_throughput - self.baseline_throughput) / self.baseline_throughput
        
        # Penalize high memory usage (1.0 = full usage, 0.0 = no usage)
        memory_efficiency = 1.0 - memory_usage
        
        # Combined reward
        reward = (self.throughput_weight * throughput_improvement + 
                  self.memory_weight * memory_efficiency)
        
        return reward
    
    def update_reward(self, new_throughput: float, new_memory_usage: float) -> None:
        """Update reward for the last selected action"""
        if self.last_action is None or self.current_state is None:
            return
        
        reward = self._calculate_reward(new_throughput, new_memory_usage)
        
        # Update reward history
        self.reward_history[self.last_action].append(reward)
        
        # Keep reward history bounded
        if len(self.reward_history[self.last_action]) > self.reward_window_size:
            self.reward_history[self.last_action].pop(0)
        
        # Log the reward
        logger.info(f"Action {self.last_action.name} received reward: {reward:.4f}")
        
        # Update baseline as a moving average
        if self.baseline_throughput is not None:
            alpha = 0.3  # Weight for new observation
            self.baseline_throughput = (1 - alpha) * self.baseline_throughput + alpha * new_throughput
    
    def _get_ucb_value(self, action: BanditAction, total_count: int) -> float:
        """Calculate the UCB value for an action"""
        if self.counts[action] == 0:
            return float('inf')
        
        # Calculate the average reward
        reward_mean = np.mean(self.reward_history[action]) if self.reward_history[action] else 0.0
        
        # Calculate the exploration bonus
        exploration_bonus = self.exploration_weight * math.sqrt(2 * math.log(total_count) / self.counts[action])
        
        return reward_mean + exploration_bonus
    
    def _has_min_samples(self) -> bool:
        """Check if all arms have been tried the minimum number of times"""
        return all(self.counts[action] >= self.min_samples_per_arm for action in BanditAction)
    
    def _select_action_by_state(self, request_load: int, memory_usage: float) -> BanditAction:
        """Select action based on current system state when not enough samples are available"""
        # High request load -> switch to ngram model
        if request_load > 15:
            return BanditAction.SWITCH_TO_NGRAM
        
        # High memory usage -> disable speculative decoding
        if memory_usage > 0.9:
            return BanditAction.DISABLE_SPEC_DECODING
        
        # Low request load and memory not full -> switch to neural model
        if request_load < 10 and memory_usage < 0.8:
            return BanditAction.SWITCH_TO_NEURAL
        
        # Default to disable speculative decoding as a middle ground
        return BanditAction.DISABLE_SPEC_DECODING
    
    def select_action(self, throughput: float, request_load: int, memory_usage: float) -> Optional[BanditAction]:
        """Select the best action based on current metrics"""
        # Record current metrics
        self.record_metrics(throughput, request_load, memory_usage)
        
        # Check if we're in cooldown period
        current_time = time.time()
        if self.last_action_time > 0 and current_time - self.last_action_time < self.cooldown_period:
            logger.debug("In cooldown period, no action selected")
            return None
        
        # If we don't have enough samples yet, use heuristic selection
        if not self._has_min_samples():
            action = self._select_action_by_state(request_load, memory_usage)
        else:
            # Calculate UCB values for each arm
            total_count = sum(self.counts.values())
            ucb_values = {action: self._get_ucb_value(action, total_count) for action in BanditAction}
            
            # Select the arm with the highest UCB value
            action = max(ucb_values, key=ucb_values.get)
            
            # Log UCB values
            logger.debug(f"UCB values: {ucb_values}")
        
        # Check if the selected action is valid based on current state
        if action == BanditAction.SWITCH_TO_NGRAM and self.using_ngram_model:
            logger.debug("Already using ngram model, skipping action")
            return None
        
        if action == BanditAction.SWITCH_TO_NEURAL and self.using_neural_model:
            logger.debug("Already using neural model, skipping action")
            return None
        
        if action == BanditAction.DISABLE_SPEC_DECODING and self.spec_decoding_disabled:
            logger.debug("Speculative decoding already disabled, skipping action")
            return None
        
        # Update state
        self.last_action = action
        self.last_action_time = current_time
        self.counts[action] += 1
        
        if action == BanditAction.SWITCH_TO_NGRAM:
            self.using_ngram_model = True
            self.using_neural_model = False
            self.spec_decoding_disabled = False
        elif action == BanditAction.SWITCH_TO_NEURAL:
            self.using_ngram_model = False
            self.using_neural_model = True
            self.spec_decoding_disabled = False
        elif action == BanditAction.DISABLE_SPEC_DECODING:
            self.spec_decoding_disabled = True
        
        logger.info(f"Selected action: {action}, counts: {self.counts[action]}")
        return action
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the optimizer"""
        return {
            "counts": self.counts,
            "reward_means": {
                action.name: np.mean(rewards) if rewards else 0.0 
                for action, rewards in self.reward_history.items()
            },
            "current_throughput": self.throughput_history[-1] if self.throughput_history else 0.0,
            "baseline_throughput": self.baseline_throughput,
            "last_action": self.last_action.name if self.last_action else None,
            "using_ngram_model": self.using_ngram_model,
            "using_neural_model": self.using_neural_model,
            "spec_decoding_disabled": self.spec_decoding_disabled
        } 