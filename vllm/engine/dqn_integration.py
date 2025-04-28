import time
import logging
import torch
from typing import Optional, Dict, Any, Set
import numpy as np

from vllm.engine.dqn_optimizer import DQNOptimizer, DQNAction

logger = logging.getLogger(__name__)

class DQNOptimizationManager:
    """管理LLMEngine和DQN优化器之间的集成
    
    此类实现LLM引擎和DQN优化器之间的接口，收集指标，
    运行优化步骤，并将选定的操作应用于引擎。
    """
    
    def __init__(self, 
                 llm_engine,
                 monitoring_interval: float = 30.0,  # 秒
                 cooldown_period: float = 10.0,     # 动作之间的冷却期（秒）
                 throughput_window: int = 5,
                 memory_threshold: float = 0.9,
                 min_acceptable_throughput: float = 10.0,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 hidden_dim: int = 64,
                 batch_size: int = 64,
                 target_update: int = 10,
                 model_path: Optional[str] = None):
        
        self.llm_engine = llm_engine
        self.monitoring_interval = monitoring_interval
        self.cooldown_period = cooldown_period
        
        # 创建优化器
        self.optimizer = DQNOptimizer(
            llm_engine=self.llm_engine,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            target_update=target_update,
            reward_window_size=throughput_window,
            min_acceptable_throughput=min_acceptable_throughput
        )
        
        # 如果提供了模型路径，加载预训练模型
        if model_path:
            self.optimizer.load_model(model_path)
        
        # 配置阈值
        self.memory_threshold = memory_threshold
        
        # 指标跟踪
        self.last_check_time = 0.0
        self.last_action_time = 0.0
        self.tokens_generated_since_last_check = 0
        self.last_throughput = 0.0
        
        # 添加用于跟踪已尝试过的动作
        self.tried_actions: Set[DQNAction] = set()
        
        self.action_in_progress = False
    
    def record_tokens(self, tokens_generated: int) -> None:
        """记录最新步骤中生成的标记数"""
        self.tokens_generated_since_last_check += tokens_generated
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """从引擎获取当前系统指标"""
        # 计算吞吐量
        current_time = time.time()
        time_elapsed = current_time - self.last_check_time
        
        throughput = 0.0
        if time_elapsed > 0 and self.tokens_generated_since_last_check > 0:
            throughput = self.tokens_generated_since_last_check / time_elapsed
            self.last_throughput = throughput
        else:
            throughput = self.last_throughput
            
        # 获取请求负载
        scheduler_outputs = self.llm_engine.try_scheduler()
        request_load = len(scheduler_outputs.scheduled_seq_groups) - scheduler_outputs.num_prefill_groups
        
        # 获取推测解码指标
        acceptance_rate = self.llm_engine.model_executor.get_speculative_metrics()[0]
        spec_length = self.llm_engine.scheduler_config.num_lookahead_slots
        
        return {
            "throughput": throughput,
            "request_load": request_load,
            "acceptance_rate": acceptance_rate,
            "spec_length": spec_length,
            "time": current_time
        }
    
    def _execute_action(self, action: DQNAction) -> bool:
        """在引擎上执行选定的动作"""
        self.action_in_progress = True
        
        try:
            # 处理模型切换动作
            if action == DQNAction.SWITCH_TO_NEURAL:
                logger.info(f"DQN优化器: 切换到神经模型")
                # 需要确保我们正确加载和切换模型
                self.llm_engine.model_executor.set_disable_speculative_decoding(False)
                self.llm_engine.decrease_block_number()
                self.llm_engine.load_neural_model_async()
                self.llm_engine.switch_to_neural_draft_model()
                
                self.optimizer.using_neural_model = True
                self.optimizer.using_ngram_model = False
                self.optimizer.spec_decoding_disabled = False
                self.optimizer.last_action = action
                self.last_action_time = time.time()
                return True
                
            elif action == DQNAction.SWITCH_TO_NGRAM:
                logger.info("DQN优化器: 切换到n-gram模型")
                self.llm_engine.model_executor.set_disable_speculative_decoding(False)
                self.llm_engine.model_executor.offload_proposer_worker()
                self.llm_engine.increase_block_number()
                self.llm_engine.switch_to_ngram_draft_model()
                
                self.optimizer.using_ngram_model = True
                self.optimizer.using_neural_model = False
                self.optimizer.spec_decoding_disabled = False
                self.optimizer.last_action = action
                self.last_action_time = time.time()
                return True
                
            elif action == DQNAction.DISABLE_SPEC_DECODING:
                logger.info("DQN优化器: 禁用推测解码")
                self.llm_engine.model_executor.set_disable_speculative_decoding(True)
                self.optimizer.using_ngram_model = False
                self.optimizer.using_neural_model = False
                self.optimizer.spec_decoding_disabled = True
                self.optimizer.last_action = action
                self.last_action_time = time.time()
                return True
                    
        except Exception as e:
            logger.error(f"执行动作 {action} 时出错: {e}")
            
        finally:
            self.action_in_progress = False
            
        return False
    
    def step(self) -> Optional[DQNAction]:
        """运行一个优化步骤，如果有执行的动作则返回"""
        # 如果我们已经在处理动作，则跳过
        if self.action_in_progress:
            return None
            
        # 获取当前指标
        metrics = self._get_current_metrics()
        
        # 记录指标给优化器
        self.optimizer.record_metrics(
            throughput=metrics["throughput"],
            acceptance_rate=metrics["acceptance_rate"],
            spec_length=metrics["spec_length"],
            request_load=metrics["request_load"]
        )
        
        # 检查是否该再次监控
        current_time = time.time()
        if current_time - self.last_check_time < self.monitoring_interval:
            return None
            
        # 检查是否在冷却期内
        if current_time - self.last_action_time < self.cooldown_period:
            return None
            
        # 重置计数器
        self.tokens_generated_since_last_check = 0
        self.last_check_time = current_time
        
        # 检查是否有未尝试过的动作
        untried_actions = [action for action in DQNAction if action not in self.tried_actions]
        
        if untried_actions and len(self.optimizer.replay_buffer) < self.optimizer.batch_size:
            # 优先选择未尝试过的动作，直到收集足够的样本
            action = np.random.choice(untried_actions)
            logger.info(f"尝试未尝试过的动作: {action.name}")
        else:
            # 使用DQN的选择逻辑
            action = self.optimizer.select_action()
        
        # 如果有，执行动作
        if action:
            if self._execute_action(action):
                logger.info(f"已执行动作: {action.name}")
                # 记录此动作已被尝试
                self.tried_actions.add(action)
                return action
                
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取优化的当前状态"""
        metrics = self._get_current_metrics()
        
        return {
            "current_metrics": metrics,
            "optimizer_status": self.optimizer.get_status(),
            "last_action_time": self.last_action_time,
            "action_in_progress": self.action_in_progress,
            "tried_actions": [a.name for a in self.tried_actions],
            "time_since_last_action": time.time() - self.last_action_time
        }
    
    def save_model(self, path: str) -> bool:
        """保存DQN模型到指定路径"""
        return self.optimizer.save_model(path) 