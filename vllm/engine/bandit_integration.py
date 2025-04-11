import time
import logging
from typing import Optional, Dict, Any

from vllm.engine.bandit_optimizer import MultiArmedBanditOptimizer, BanditAction, ThresholdSwitcher

logger = logging.getLogger(__name__)

class BanditOptimizationManager:
    """Manages the integration between LLMEngine and the bandit optimizer"""
    
    def __init__(self, 
                 llm_engine,
                 monitoring_interval: float = 5.0,  # seconds
                 cooldown_period: float = 30.0,     # seconds
                 throughput_window: int = 5,
                 ngram_high_request_threshold: int = 15,
                 neural_low_request_threshold: int = 10,
                 memory_high_threshold: float = 0.9,
                 memory_low_threshold: float = 0.8):
        
        self.llm_engine = llm_engine
        self.monitoring_interval = monitoring_interval
        
        # Create optimizer
        self.optimizer = MultiArmedBanditOptimizer(
            cooldown_period=cooldown_period,
            reward_window_size=throughput_window
        )
        
        # Create simpler threshold-based switcher as a fallback
        self.threshold_switcher = ThresholdSwitcher(
            window_size=throughput_window
        )
        
        # Configure thresholds
        self.ngram_high_request_threshold = ngram_high_request_threshold
        self.neural_low_request_threshold = neural_low_request_threshold
        self.memory_high_threshold = memory_high_threshold
        self.memory_low_threshold = memory_low_threshold
        
        # Metrics tracking
        self.last_check_time = 0.0
        self.tokens_generated_since_last_check = 0
        self.last_throughput = 0.0
        
        # State tracking 
        self.last_action: Optional[BanditAction] = None
        self.action_in_progress = False
        
        # Initialize state from engine
        self.update_state_from_engine()
        self.has_increased_block_number = False
    
    def update_state_from_engine(self) -> None:
        """Update internal state to match current engine state"""
        # Check if we're using ngram or neural model
        if hasattr(self.llm_engine, 'using_ngram_draft_model'):
            self.optimizer.using_ngram_model = self.llm_engine.using_ngram_draft_model
            self.optimizer.using_neural_model = not self.llm_engine.using_ngram_draft_model
        
        # Check if speculative decoding is disabled
        if hasattr(self.llm_engine, 'model_executor') and hasattr(self.llm_engine.model_executor, 'get_speculative_decoding_disabled'):
            self.optimizer.spec_decoding_disabled = self.llm_engine.model_executor.get_speculative_decoding_disabled()
    
    def record_tokens(self, tokens_generated: int) -> None:
        """Record tokens generated in the latest step"""
        self.tokens_generated_since_last_check += tokens_generated
        self.threshold_switcher.record_tokens(tokens_generated)
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics from the engine"""
        # Calculate throughput
        current_time = time.time()
        time_elapsed = current_time - self.last_check_time
        
        throughput = 0.0
        if time_elapsed > 0 and self.tokens_generated_since_last_check > 0:
            throughput = self.tokens_generated_since_last_check / time_elapsed
            self.last_throughput = throughput
        else:
            throughput = self.last_throughput
        
        # Reset counters
        self.tokens_generated_since_last_check = 0
        self.last_check_time = current_time
        
        # Get request load
        request_load = self.llm_engine.get_num_running_requests()
        
        # Get memory usage (GPU cache usage percentage)
        memory_usage = 0.0
        if hasattr(self.llm_engine, 'cache_config') and self.llm_engine.cache_config.num_gpu_blocks:
            total_gpu_blocks = self.llm_engine.cache_config.num_gpu_blocks
            free_gpu_blocks = sum(
                scheduler.block_manager.get_num_free_gpu_blocks()
                for scheduler in self.llm_engine.scheduler
            )
            memory_usage = 1.0 - (free_gpu_blocks / total_gpu_blocks)
        
        return {
            "throughput": throughput,
            "request_load": request_load,
            "memory_usage": memory_usage,
            "time": current_time
        }
    
    def _execute_action(self, action: BanditAction) -> bool:
        """Execute the selected action on the engine"""
        self.action_in_progress = True
        
        try:
            if action == BanditAction.SWITCH_TO_NGRAM:
                logger.info("Bandit optimizer: Switching to n-gram draft model")
                if not self.llm_engine.using_ngram_draft_model:
                    if not self.has_increased_block_number:
                        self.llm_engine.increase_block_number()
                        self.has_increased_block_number = True
                    self.llm_engine.switch_to_ngram_draft_model()
                    self.last_action = action
                    return True
                
            elif action == BanditAction.SWITCH_TO_NEURAL:
                logger.info("Bandit optimizer: Switching to neural draft model")
                if (hasattr(self.llm_engine, 'has_loaded_neural_model') and 
                    not self.llm_engine.has_loaded_neural_model):
                    # First load the neural model
                    print("Decreasing block number and loading neural model")
                    if self.has_increased_block_number:
                        self.llm_engine.decrease_block_number()
                        self.has_increased_block_number = False
                    self.llm_engine.load_neural_model_async()
                    self.llm_engine.has_loaded_neural_model = True
                
                if self.llm_engine.using_ngram_draft_model:
                    print("Switching to neural draft model!!!")
                    self.llm_engine.switch_to_neural_draft_model()
                    self.last_action = action
                    return True
                
            elif action == BanditAction.DISABLE_SPEC_DECODING:
                logger.info("Bandit optimizer: Disabling speculative decoding")
                if hasattr(self.llm_engine.model_executor, 'set_disable_speculative_decoding'):
                    current_state = self.llm_engine.model_executor.get_speculative_decoding_disabled()
                    if not current_state:
                        self.llm_engine.model_executor.set_disable_speculative_decoding(True)
                        self.last_action = action
                        return True
                    
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            
        finally:
            self.action_in_progress = False
            
        return False
    
    def step(self) -> Optional[BanditAction]:
        """Run one optimization step, returning the executed action if any"""
        # Skip if we're already processing an action
        if self.action_in_progress:
            return None
            
        # Check if it's time to monitor again
        current_time = time.time()
        if current_time - self.last_check_time < self.monitoring_interval:
            return None
            
        # Get current metrics
        metrics = self._get_current_metrics()
        
        # If we previously executed an action, update its reward
        if self.last_action:
            self.optimizer.update_reward(
                metrics["throughput"], 
                metrics["memory_usage"]
            )
            self.last_action = None
            
        # Get optimizer's suggestion
        optimizer_action = self.optimizer.select_action(
            metrics["throughput"],
            metrics["request_load"],
            metrics["memory_usage"]
        )
        
        # Prioritize threshold_action over optimizer_action for more responsiveness
        action =  optimizer_action
        
        # Execute the action if any
        if action:
            if self._execute_action(action):
                logger.info(f"Executed action: {action.name}")
                return action
                
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of optimization"""
        metrics = self._get_current_metrics()
        
        return {
            "current_metrics": metrics,
            "optimizer_status": self.optimizer.get_status(),
            "threshold_throughput": self.threshold_switcher.get_current_throughput(),
            "last_action": self.last_action.name if self.last_action else None,
            "action_in_progress": self.action_in_progress
        } 