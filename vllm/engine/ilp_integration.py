import time
import logging
import threading
from typing import Optional, Dict, Any, List, Set
from concurrent.futures import ThreadPoolExecutor
from vllm.engine.ilp_optimizer import ILPOptimizer, ILPAction

logger = logging.getLogger(__name__)

class ILPOptimizationManager:
    """Manages the integration between LLMEngine and the ILP optimizer
    
    This class implements the interface between the LLM engine and the ILP
    optimizer, collecting metrics, running optimization steps, and applying
    the selected actions to the engine.
    """
    
    def __init__(self, 
                 engine,
                 monitoring_interval: float = 30.0,  # seconds
                 throughput_window: int = 50,
                 memory_threshold: float = 0.9,
                 min_monitoring_interval: float = 5.0,
                 max_monitoring_interval: float = 60.0,
                 stability_threshold: float = 0.2):
        
        self.engine = engine
        self.initial_monitoring_interval = monitoring_interval
        self.monitoring_interval = monitoring_interval
        self.min_monitoring_interval = min_monitoring_interval
        self.max_monitoring_interval = max_monitoring_interval
        self.stability_threshold = stability_threshold
        
        # 用于自适应监控间隔的历史数据
        self.throughput_history = []
        self.history_window_size = 5
        
        # Create optimizer
        self.optimizer = ILPOptimizer(
            llm_engine=self.engine,
            reward_window_size=throughput_window
        )
        # self.optimizer.load_action_time_history()
        # Configure thresholds
        self.memory_threshold = memory_threshold
        
        # Metrics tracking
        self.last_check_time = 0.0
        self.tokens_generated_since_last_check = 0
        self.last_throughput = 0.0
        
        self.optimizer.last_action = ILPAction.USE_SMALL_MODEL_1
        self.last_predict_throughput = 0.0
        self.metrics = None
        self.last_action_time = 0.0
        self.action_interval = 30
        self.static_action = None
        self.profile = False
        self.strategy = "daspec_spec"
        self.last_batch_size = 0
        
    
    def _collect_metrics(self,stage_data):
        """实际执行指标收集工作"""
        # 这里是原来set_record_metrics的核心逻辑
        metrics = self._get_current_metrics(stage_data)
        self.metrics = metrics
        self.optimizer.record_metrics(metrics)
        
        tokens_generated = 0
        for output in self.outputs:
            for o in output.outputs:
                tokens_generated += len(o.token_ids)
        tokens_throughput_per_step = tokens_generated / self.duration_time_per_step
        
        # 更新令牌计数器
        self.record_tokens(tokens_generated, tokens_throughput_per_step)
    
    
    def record_tokens(self, tokens_generated: int, tokens_throughput_per_step: float) -> None:
        """Record tokens generated in the latest step"""
        self.tokens_generated_since_last_check += tokens_generated
        self.tokens_throughput_per_step = tokens_throughput_per_step
    
    def _get_current_metrics(self,stage_data) -> Dict[str, Any]:
        """Get current system metrics from the engine"""
      
        
        # Get speculative decoding metrics
        # 0: draft, 1: scoring, 2: verification 3: batch size 4: num_accepted_tokens 5: context_length 6: stage
        
        acceptance_rate = stage_data[4]/stage_data[3]
        batch_size = stage_data[3]
        accepted_tokens_length = stage_data[4]
        total_latency = stage_data[0]+stage_data[1]+stage_data[2]
        spec_length = self.engine.scheduler_config.num_lookahead_slots

        return {
            #"throughput": throughput,
            "acceptance_rate": acceptance_rate,
            "spec_length": spec_length,
            #"time": current_time,
            "proposal_time": stage_data[0],
            "scoring_time": stage_data[1],
            "verification_time": stage_data[2],
            "total_latency": total_latency,
            "accepted_tokens_length": accepted_tokens_length,
            "batch_size": batch_size,
            "context_length": stage_data[5],
            "stage": stage_data[6]
        }
        
    
    def _execute_action(self, action: ILPAction) -> bool:
        """Execute the selected action on the engine"""
        # Handle small model actions
        virtual_engine = 0

        if action in [ILPAction.USE_SMALL_MODEL_1, ILPAction.USE_SMALL_MODEL_2]:
            model_index = action.value
            model_name = f"small_model_{model_index+1}"
            print("action",action)
            logger.info(f"ILP optimizer: Switching to {model_name}")
            # FIXME：load_neural_model_async 和 switch_to_neural_draft_model 需要隔一个step
            if action == ILPAction.USE_SMALL_MODEL_1:
                self.engine.decrease_block_number()
                self.engine.load_neural_model_async()
                self.engine.scheduler[virtual_engine].smart_spec = None
                self.engine.switch_to_neural_draft_model()
                self.engine.set_disable_speculative_decoding(False)
            elif action == ILPAction.USE_SMALL_MODEL_2:
                self.engine.set_disable_speculative_decoding(False)
                self.engine.switch_to_ngram_draft_model()
                # 先转移再offload TODO: proposer KV cache offload
                self.engine.scheduler[virtual_engine].smart_spec = None
                if self.engine.speculative_config is not None: self.engine.offload_proposer_worker()
                self.engine.increase_block_number()
                
            
            self.optimizer.current_model_index = model_index
            self.optimizer.last_action = action
            return True
            
        elif action == ILPAction.DISABLE_SPEC_DECODING:
            logger.info("ILP optimizer: Disabling speculative decoding")
            self.engine.set_disable_speculative_decoding(True)
            self.engine.scheduler[virtual_engine].smart_spec = None
            self.engine.scheduler[virtual_engine].daspec = None
            self.engine.scheduler[virtual_engine].scheduler_config.num_lookahead_slots = 0
            if self.engine.speculative_config is not None: self.engine.offload_proposer_worker()
            self.engine.increase_block_number()
            self.optimizer.current_model_index = -1  # No speculative decoding
            self.optimizer.last_action = action
            return True

        return True
    
   


    
    def change_speculative_action(self, action:int, save_action_time_history=False, profile=False,file_name=None, offload=False):
        self.offload = offload
        self.profile = profile
        if action >= 0:
            self.static_action = self.optimizer.actions[action]
            logger.info(f"Change speculative action to {self.static_action.name}")
            action = self.static_action
            if self._execute_action(action):
                logger.info(f"For static action,Executed action: {action.name}")
                return action
            else:
                logger.info(f"For static action,Failed to execute action: {action.name}")
                return None
        else:
            if save_action_time_history:
                self.optimizer.save_action_time_history(file_name)
            self.static_action = None # 自由选择
            logger.info(f"Change speculative action to None")