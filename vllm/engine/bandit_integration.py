import time
import logging
from typing import Optional, Dict, Any, Set
import torch
from vllm.engine.bandit_optimizer import BanditOptimizer, BanditAction

logger = logging.getLogger(__name__)

class BanditOptimizationManager:
    """管理LLMEngine和Bandit优化器之间的集成
    
    此类实现LLM引擎和Bandit优化器之间的接口，收集指标，
    运行优化步骤，并将选定的操作应用于引擎。
    """
    
    def __init__(self, 
                 llm_engine,
                 monitoring_interval: float = 30.0,  # 秒
                 throughput_window: int = 5,
                 memory_threshold: float = 0.9,
                 min_acceptable_throughput: float = 10.0,
                 baseline_window_size: int = 10):  # 新增参数：基准吞吐率窗口大小
        
        self.llm_engine = llm_engine
        self.monitoring_interval = monitoring_interval
        
        # 创建优化器
        self.optimizer = BanditOptimizer(
            llm_engine=self.llm_engine,
            reward_window_size=throughput_window,
            min_acceptable_throughput=min_acceptable_throughput
        )
        
        # 配置阈值
        self.memory_threshold = memory_threshold
        
        # 指标跟踪
        self.last_check_time = 0.0
        self.tokens_generated_since_last_check = 0
        self.last_throughput = 0.0
        
        # 添加用于跟踪已尝试过的动作
        self.tried_actions: Set[BanditAction] = set()
        
        self.action_in_progress = False
        self.last_action = None
        
        # 新增：用于记录禁用投机推理时的吞吐率
        self.baseline_throughput_samples = []
        self.baseline_window_size = baseline_window_size  # 添加窗口大小参数
        self.baseline_throughput = min_acceptable_throughput  # 初始值设为参数传入的默认值
        self.has_measured_baseline = False
    
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
            
        # 获取推测解码指标
        acceptance_rate = self.llm_engine.model_executor.get_speculative_metrics()[0]
        spec_length = self.llm_engine.scheduler_config.num_lookahead_slots
        
        return {
            "throughput": throughput,
            "acceptance_rate": acceptance_rate,
            "spec_length": spec_length,
            "time": current_time
        }
    
    def _execute_action(self, action: BanditAction) -> bool:
        """在引擎上执行选定的动作"""
        self.action_in_progress = True
        
        try:
            # 处理模型切换动作
            if action == BanditAction.SWITCH_TO_NEURAL:
                logger.info(f"Bandit优化器: 切换到神经模型")
                # 需要确保我们正确加载和切换模型
                self.llm_engine.model_executor.set_disable_speculative_decoding(False)
                self.llm_engine.decrease_block_number()
                self.llm_engine.load_neural_model_async()
                self.llm_engine.switch_to_neural_draft_model()
                
                self.optimizer.using_neural_model = True
                self.optimizer.using_ngram_model = False
                self.optimizer.spec_decoding_disabled = False
                self.last_action = action
                return True
                
            elif action == BanditAction.SWITCH_TO_NGRAM:
                logger.info("Bandit优化器: 切换到n-gram模型")
                self.llm_engine.model_executor.set_disable_speculative_decoding(False)
                self.llm_engine.offload_proposer_worker()
                self.llm_engine.increase_block_number()
                self.llm_engine.switch_to_ngram_draft_model()
                
                self.optimizer.using_ngram_model = True
                self.optimizer.using_neural_model = False
                self.optimizer.spec_decoding_disabled = False
                self.last_action = action
                return True
                
            elif action == BanditAction.DISABLE_SPEC_DECODING:
                logger.info("Bandit优化器: 禁用推测解码")
                self.llm_engine.model_executor.set_disable_speculative_decoding(True)
                self.optimizer.using_ngram_model = False
                self.optimizer.using_neural_model = False
                self.optimizer.spec_decoding_disabled = True
                self.last_action = action
                
                # 新增：标记开始收集禁用投机推理的吞吐率基准
                self.baseline_throughput_samples = []
                return True
                    
        except Exception as e:
            logger.error(f"执行动作 {action} 时出错: {e}")
            
        finally:
            self.action_in_progress = False
            
        return False
    
    def step(self) -> Optional[BanditAction]:
        """运行一个优化步骤，如果有执行的动作则返回"""
        # 如果我们已经在处理动作，则跳过
        if self.action_in_progress:
            return None
            
        # 记录当前指标
        metrics = self._get_current_metrics()
        self.optimizer.record_metrics(metrics["throughput"], 
                           metrics["acceptance_rate"], 
                           metrics["spec_length"])
        
        # 新增：如果当前处于禁用推测解码状态，收集吞吐率样本
        if self.optimizer.spec_decoding_disabled:
            if metrics["throughput"] > 0:  # 确保吞吐率有效
                # 添加样本并维持窗口大小
                self.baseline_throughput_samples.append(metrics["throughput"])
                # 如果样本数超过窗口大小，移除最旧的样本
                if len(self.baseline_throughput_samples) > self.baseline_window_size:
                    self.baseline_throughput_samples.pop(0)
                    
                # 当收集到足够样本后，计算平均值作为基线吞吐率
                if len(self.baseline_throughput_samples) >= min(5, self.baseline_window_size):  # 至少收集5个样本或窗口大小
                    new_baseline = sum(self.baseline_throughput_samples) / len(self.baseline_throughput_samples)
                    # 只有在有显著变化时才更新
                    if not self.has_measured_baseline or abs(new_baseline - self.baseline_throughput) / self.baseline_throughput > 0.2:
                        logger.info(f"更新基准吞吐率: {self.baseline_throughput:.2f} -> {new_baseline:.2f}")
                        self.baseline_throughput = new_baseline
                        # 更新优化器中的最低可接受吞吐率
                        self.optimizer.min_acceptable_throughput = self.baseline_throughput
                        self.has_measured_baseline = True
        # 为每个动作分配特定的接受率、推测长度和吞吐率
        action_metrics = {}
        for action in self.optimizer.actions:
            if action == BanditAction.DISABLE_SPEC_DECODING:
                action_metrics[action] = {
                    "acceptance_rate": 0.0,
                    "spec_length": 1,
                    "throughput": sum(self.optimizer.action_metrics_history[action]["throughputs"]) / len(self.optimizer.action_metrics_history[action]["throughputs"]) if self.optimizer.action_metrics_history[action]["throughputs"] else (self.optimizer.current_state["throughput"] if self.optimizer.current_state else 10.0)
                }
            else:
                rates = self.optimizer.action_metrics_history[action]["acceptance_rates"]
                lengths = self.optimizer.action_metrics_history[action]["spec_lengths"]
                throughputs = self.optimizer.action_metrics_history[action]["throughputs"]
                
                action_metrics[action] = {
                    "acceptance_rate": sum(rates) / len(rates) if rates else (self.optimizer.current_state["acceptance_rate"] if self.optimizer.current_state else 0.5),
                    "spec_length": sum(lengths) / len(lengths) if lengths else (self.optimizer.current_state["spec_length"] if self.optimizer.current_state else 3),
                    "throughput": sum(throughputs) / len(throughputs) if throughputs else (self.optimizer.current_state["throughput"] if self.optimizer.current_state else 10.0)
                }
        
        # 计算每个动作的预期奖励
        expected_rewards = {}
        for action in self.optimizer.actions:
            reward = self.optimizer._calculate_reward(
                action,
                action_metrics[action]["acceptance_rate"],
                action_metrics[action]["spec_length"],
                action_metrics[action]["throughput"]
            )
            expected_rewards[action] = reward
            # 确保奖励不是 CUDA 张量
            if isinstance(reward, torch.Tensor):
                reward = reward.cpu().detach().numpy()
            self.optimizer.reward_history[action].append(reward)
            # 保持奖励历史在限定范围内
            if len(self.optimizer.reward_history[action]) > self.optimizer.reward_window_size:
                self.optimizer.reward_history[action].pop(0)
        self.optimizer.expected_rewards = expected_rewards
        # 检查是否该再次监控
        current_time = time.time()
        if current_time - self.last_check_time < self.monitoring_interval:
            return None
            
        # 重置计数器
        self.tokens_generated_since_last_check = 0
        self.last_check_time = current_time
        # 修改：确保至少尝试一次禁用投机推理以获取基准吞吐率
        if not self.has_measured_baseline and BanditAction.DISABLE_SPEC_DECODING not in self.tried_actions:
            action = BanditAction.DISABLE_SPEC_DECODING
            logger.info(f"为测量基准吞吐率而禁用推测解码")
        else:
            # 检查是否有未尝试过的动作
            untried_actions = set()
            for action in BanditAction:
                if action not in self.tried_actions:
                    untried_actions.add(action)
                    break
            
            if untried_actions:
                # 优先选择未尝试过的动作
                action = next(iter(untried_actions))
                logger.info(f"尝试未尝试过的动作: {action.name}")
            else:
                # 如果所有动作都已尝试过，则使用正常的选择逻辑
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
            "last_action": self.last_action.name if self.last_action else None,
            "action_in_progress": self.action_in_progress,
            "tried_actions": [a.name for a in self.tried_actions],
        } 