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
        self.static = False
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
    
    def _update_monitoring_interval(self, current_throughput: float) -> None:
        """根据吞吐量变化动态调整监控间隔"""
        if len(self.tried_actions) < len(self.optimizer.actions):
            # 还没探索完所有动作
            return
        # 添加当前吞吐量到历史记录
        self.throughput_history.append(current_throughput)
        
        # 保持历史记录在窗口大小范围内
        if len(self.throughput_history) > self.history_window_size:
            self.throughput_history.pop(0)
            
        # 至少需要2个数据点才能计算变化率
        if len(self.throughput_history) < 2:
            return
            
        # 计算最近吞吐量的变化率
        throughput_changes = []
        for i in range(1, len(self.throughput_history)):
            prev = self.throughput_history[i-1]
            curr = self.throughput_history[i]
            # 避免除以零
            if prev > 0:
                change_rate = abs(curr - prev) / prev
                throughput_changes.append(change_rate)
        
        # 计算平均变化率
        if throughput_changes:
            avg_change_rate = sum(throughput_changes) / len(throughput_changes)
            
            # 根据变化率调整监控间隔
            if avg_change_rate > self.stability_threshold:
                # 较大变化，缩短监控间隔
                self.monitoring_interval = max(
                    self.monitoring_interval * 0.7,  # 减少30%
                    self.min_monitoring_interval
                )
                # logger.info(f"系统负载变化明显 ({avg_change_rate:.2f})，缩短监控间隔至 {self.monitoring_interval:.1f}秒")
            else:
                # 较小变化，延长监控间隔
                self.monitoring_interval = min(
                    self.monitoring_interval * 1.2,  # 增加20%
                    self.max_monitoring_interval
                )
                # logger.info(f"系统负载稳定 ({avg_change_rate:.2f})，延长监控间隔至 {self.monitoring_interval:.1f}秒")
   
    def record_tokens(self, tokens_generated: int, tokens_throughput_per_step: float) -> None:
        """Record tokens generated in the latest step"""
        self.tokens_generated_since_last_check += tokens_generated
        self.tokens_throughput_per_step = tokens_throughput_per_step
    
    def _get_current_metrics(self,stage_data) -> Dict[str, Any]:
        """Get current system metrics from the engine"""
        # Calculate throughput
        # current_time = time.time()
        # time_elapsed = current_time - self.last_check_time
        
        # throughput = 0.0
        # if time_elapsed > 0 and self.tokens_generated_since_last_check > 0:
        #     throughput = self.tokens_generated_since_last_check / time_elapsed
        #     self.last_throughput = throughput
        # else:
        #     throughput = self.last_throughput
            
        # 更新自适应监控间隔
        # self._update_monitoring_interval(self.tokens_throughput_per_step)
        
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
        
    
    def _execute_action(self, action: ILPAction, static=False) -> bool:
        """Execute the selected action on the engine"""
        # Handle small model actions
        self.static = static
        virtual_engine = 0
        if action in [ILPAction.USE_SMALL_MODEL_1, ILPAction.USE_SMALL_MODEL_2]:
            model_index = action.value
            model_name = f"small_model_{model_index+1}"
            print("action",action)
            logger.info(f"ILP optimizer: Switching to {model_name}")
            # FIXME：load_neural_model_async 和 switch_to_neural_draft_model 需要隔一个step
            if action == ILPAction.USE_SMALL_MODEL_1:
                if static:
                    self.engine.decrease_block_number()
                    self.engine.load_neural_model_async()
                    self.engine.scheduler[virtual_engine].smart_spec = None
                self.engine.switch_to_neural_draft_model()
                self.engine.set_disable_speculative_decoding(False)
            elif action == ILPAction.USE_SMALL_MODEL_2:
                self.engine.set_disable_speculative_decoding(False)
                self.engine.switch_to_ngram_draft_model()
                # 先转移再offload TODO: proposer KV cache offload
                if static:
                    self.engine.scheduler[virtual_engine].smart_spec = None
                    self.engine.offload_proposer_worker()
                    self.engine.increase_block_number()
                
            
            self.optimizer.current_model_index = model_index
            self.optimizer.last_action = action
            return True
            
        elif action == ILPAction.DISABLE_SPEC_DECODING:
            logger.info("ILP optimizer: Disabling speculative decoding")
            self.engine.set_disable_speculative_decoding(True)
            if static:
                self.engine.scheduler[virtual_engine].smart_spec = None
                self.engine.offload_proposer_worker()
                self.engine.increase_block_number()
            self.optimizer.current_model_index = -1  # No speculative decoding
            self.optimizer.last_action = action
            return True

        return True
    
    def set_record_metrics(self):
        """现在这个方法的功能由后台线程实现"""
        # 直接触发后台收集
        self.trigger_metrics_collection()


    def step(self,scheduler_outputs,stage_data) -> Optional[ILPAction]:
        if self.profile:
            self._collect_metrics(stage_data)
            return None
        if self.static_action is not None and self.static_action == self.optimizer.last_action:
            return None
        # 限制调整频率
        request_load = len(scheduler_outputs.scheduled_seq_groups)
        if request_load > 10 and scheduler_outputs.num_lookahead_slots >0 and abs(request_load - self.last_batch_size) > 2:
            self.last_batch_size = request_load
            time_start = time.time()
            action = self.optimizer.select_action(request_load)
            time_end = time.time()
            logger.info(f"select_action time: {time_end - time_start}")
            """
            memory adjustment
            """
            # time_start = time.time()
            # virtual_engine = 0
            # can_increase_space, can_decrease_space = False, False
            # if self.engine.scheduler[virtual_engine].block_manager.num_usable_gpu_blocks < self.engine.scheduler[virtual_engine].block_manager.num_total_gpu_blocks and \
            #     (len(scheduler_outputs.scheduled_seq_groups) < len(self.engine.scheduler[virtual_engine].running) or \
            #     len(scheduler_outputs.scheduled_seq_groups) <  len(self.engine.scheduler[virtual_engine].waiting)):
            #     can_increase_space = True
            # else:
            #     if self.engine.scheduler[virtual_engine].block_manager.num_usable_gpu_blocks == self.engine.scheduler[virtual_engine].block_manager.num_total_gpu_blocks and \
            #         self.engine.cache_config.num_virtual_blocks <  self.engine.scheduler[virtual_engine].block_manager.get_num_free_gpu_blocks():
            #         can_decrease_space = True
            # if can_increase_space:
            #     self.engine.offload_proposer_worker()
            #     self.engine.increase_block_number()
            # if can_decrease_space:
            #     self.engine.decrease_block_number()
            #     self.engine.load_neural_model_async()
            # time_end = time.time()
            # logger.info(f"memory adjustment time: {time_end - time_start}")
            if action and self._execute_action(action):
                logger.info(f"Executed action: {action.name}")
                self.optimizer.current_model_index = action.value
                return action
            return None


    
    def change_speculative_action(self, action:int, save_action_time_history=False, profile=False,file_name=None):
        self.profile = profile
        if action >= 0:
            self.static_action = self.optimizer.actions[action]
            logger.info(f"Change speculative action to {self.static_action.name}")
            action = self.static_action
            if self._execute_action(action,static=True):
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