import time
import logging
from typing import Optional, Dict, Any, List

from vllm.engine.ilp_optimizer import ILPOptimizer, ILPAction

logger = logging.getLogger(__name__)

class ILPOptimizationManager:
    """Manages the integration between LLMEngine and the ILP optimizer
    
    This class implements the interface between the LLM engine and the ILP
    optimizer, collecting metrics, running optimization steps, and applying
    the selected actions to the engine.
    """
    
    def __init__(self, 
                 llm_engine,
                 monitoring_interval: float = 5.0,  # seconds
                 cooldown_period: float = 30.0,     # seconds
                 throughput_window: int = 5,
                 memory_threshold: float = 0.9,
                 min_acceptable_throughput: float = 10.0,  # tokens/s
                 num_small_models: int = 3):
        
        self.llm_engine = llm_engine
        self.monitoring_interval = monitoring_interval
        
        # Create optimizer
        self.optimizer = ILPOptimizer(
            llm_engine=self.llm_engine,
            num_small_models=num_small_models,
            reward_window_size=throughput_window,
            cooldown_period=cooldown_period,
            min_acceptable_throughput=min_acceptable_throughput
        )
        
        # Configure thresholds
        self.memory_threshold = memory_threshold
        
        # Metrics tracking
        self.last_check_time = 0.0
        self.tokens_generated_since_last_check = 0
        self.last_throughput = 0.0
        
        # State tracking 
        self.last_action: Optional[ILPAction] = None
        self.action_in_progress = False
    
   
    def record_tokens(self, tokens_generated: int) -> None:
        """Record tokens generated in the latest step"""
        self.tokens_generated_since_last_check += tokens_generated
    
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
        
        # Get speculative decoding metrics
        acceptance_rate = self.llm_engine.model_executor.get_speculative_metrics()[0]
        spec_length = self.llm_engine.scheduler_config.num_lookahead_slots
        
        return {
            "throughput": throughput,
            "acceptance_rate": acceptance_rate,
            "spec_length": spec_length,
            "time": current_time
        }
    
    def _execute_action(self, action: ILPAction) -> bool:
        """Execute the selected action on the engine"""
        self.action_in_progress = True
        
        try:
            # Handle small model actions
            if action in [ILPAction.USE_SMALL_MODEL_1, ILPAction.USE_SMALL_MODEL_2, ILPAction.USE_SMALL_MODEL_3]:
                model_index = action.value
                model_name = f"small_model_{model_index+1}"
                
                logger.info(f"ILP optimizer: Switching to {model_name}")
                
                if action == ILPAction.USE_SMALL_MODEL_1:
                    self.llm_engine.decrease_block_number()
                    self.llm_engine.load_neural_model_async()
                    self.llm_engine.switch_to_neural_draft_model()
                elif action == ILPAction.USE_SMALL_MODEL_2:
                    self.llm_engine.model_executor.offload_proposer_worker()
                    self.llm_engine.increase_block_number()
                    self.llm_engine.switch_to_ngram_draft_model()
               
                self.optimizer.current_model_index = model_index
                self.last_action = action
                return True
                
            elif action == ILPAction.DISABLE_SPEC_DECODING:
                logger.info("ILP optimizer: Disabling speculative decoding")
                self.llm_engine.model_executor.set_disable_speculative_decoding(True)
                self.optimizer.current_model_index = -1  # No speculative decoding
                self.last_action = action
                return True
                    
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            
        finally:
            self.action_in_progress = False
            
        return False
    
    def step(self) -> Optional[ILPAction]:
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
        
        
        # Get optimizer's suggestion
        action = self.optimizer.select_action(
            metrics["throughput"],
            metrics["acceptance_rate"],
            metrics["spec_length"]
        )
        
        # Execute the action if any
        if action:
            if self._execute_action(action):
                logger.info(f"Executed action: {action.name}")
                self.optimizer.current_model_index = action.value
                return action
                
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of optimization"""
        metrics = self._get_current_metrics()
        
        return {
            "current_metrics": metrics,
            "optimizer_status": self.optimizer.get_status(),
            "last_action": self.last_action.name if self.last_action else None,
            "action_in_progress": self.action_in_progress,
        } 