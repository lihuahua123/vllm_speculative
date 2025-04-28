import numpy as np
import math
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import joblib
import torch

logger = logging.getLogger(__name__)

class BanditAction(Enum):
    """Bandit优化器可以采取的行动"""
    SWITCH_TO_NEURAL = 0 
    SWITCH_TO_NGRAM = 1
    DISABLE_SPEC_DECODING = 2

class ThresholdSwitcher:
    """A simple threshold-based switcher that tracks token generation rate"""
    def __init__(self, 
                 window_size: int = 5,
                 high_threshold: float = 40.0,  # tokens/s
                 low_threshold: float = 20.0):   # tokens/s
        self.token_history: List[int] = []
        self.timestamp_history: List[float] = []
        self.window_size = window_size
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
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
        
        if throughput > self.high_threshold:
            self.last_switch_time = current_time
            return BanditAction.SWITCH_TO_NGRAM
        elif throughput < self.low_threshold:
            self.last_switch_time = current_time
            return BanditAction.SWITCH_TO_NEURAL
        
        return None

class BanditOptimizer:
    """基于Multi-armed Bandit的动态模型切换优化器。
    
    此优化器使用UCB（Upper Confidence Bound）算法来动态选择最佳的推测采样方法：
    1. 在神经模型和n-gram模型之间切换
    2. 必要时禁用推测解码
    
    优化器会考虑吞吐量和内存使用情况，以及接受率等指标来做出决策。
    """
    
    def __init__(self, 
                 llm_engine,
                 reward_window_size: int = 10,
                 min_samples_per_arm: int = 3,
                 throughput_weight: float = 0.7,
                 memory_weight: float = 0.3,
                 min_acceptable_throughput: float = 10.0):
        
        self.llm_engine = llm_engine
        # 创建动作列表
        self.actions = [
            BanditAction.SWITCH_TO_NEURAL,
            BanditAction.SWITCH_TO_NGRAM,
            BanditAction.DISABLE_SPEC_DECODING
        ]
        
        # 初始化每个动作的计数
        self.counts = {action: 0 for action in self.actions}
        
        # 初始化每个动作的奖励历史
        self.reward_history = {action: [] for action in self.actions}
        
        # 运行指标
        self.request_load_history = []
        self.memory_usage_history = []
        self.acceptance_rate_history = []
        self.spec_length_history = []
        
        # 状态跟踪
        self.last_action = None
        self.last_action_time = 0.0
        self.current_state = None
        
        # 配置
        self.reward_window_size = reward_window_size
        self.min_samples_per_arm = min_samples_per_arm
        self.throughput_weight = throughput_weight
        self.memory_weight = memory_weight
        self.min_acceptable_throughput = min_acceptable_throughput
        
        # 当前模型状态
        self.using_ngram_model = False
        self.using_neural_model = False
        self.spec_decoding_disabled = False
        
        # 探索权重 (UCB算法)
        self.exploration_weight = 2.0
        
        # 新增用于记录指标的动作历史
        self.action_metrics_history = {action: {
            "acceptance_rates": [],
            "spec_lengths": [],
            "throughputs": []  # 新增吞吐率历史记录
        } for action in self.actions}

        self.expected_rewards = {}
        
    def record_metrics(self, 
                      throughput: float, 
                      acceptance_rate: Optional[float] = None,
                      spec_length: Optional[int] = None) -> None:
        """记录当前系统指标"""
        # 记录推测解码指标（如果可用）
        if acceptance_rate is not None:
            self.acceptance_rate_history.append(acceptance_rate)
        if spec_length is not None:
            self.spec_length_history.append(spec_length)
        
        avg_acceptance_rate = sum(self.acceptance_rate_history)/len(self.acceptance_rate_history) if self.acceptance_rate_history else 0.0
        
        # 保持历史记录在有限范围内
        if len(self.request_load_history) > self.reward_window_size:
            if self.acceptance_rate_history:
                self.acceptance_rate_history.pop(0)
            if self.spec_length_history:
                self.spec_length_history.pop(0)
        
        # 记录当前动作的指标
        if self.last_action is not None:
            # 为每个动作记录吞吐率
            self.action_metrics_history[self.last_action]["throughputs"].append(throughput)
            
            if acceptance_rate is not None and spec_length is not None and self.last_action != BanditAction.DISABLE_SPEC_DECODING:
                self.action_metrics_history[self.last_action]["acceptance_rates"].append(acceptance_rate)
                self.action_metrics_history[self.last_action]["spec_lengths"].append(spec_length)
            
            # 保持历史记录在有限范围内
            if len(self.action_metrics_history[self.last_action]["throughputs"]) > self.reward_window_size:
                self.action_metrics_history[self.last_action]["throughputs"].pop(0)
                
            if self.last_action != BanditAction.DISABLE_SPEC_DECODING:
                if len(self.action_metrics_history[self.last_action]["acceptance_rates"]) > self.reward_window_size:
                    self.action_metrics_history[self.last_action]["acceptance_rates"].pop(0)
                    self.action_metrics_history[self.last_action]["spec_lengths"].pop(0)

        # 存储当前状态用于计算
        self.current_state = {
            "throughput": throughput,
            "acceptance_rate": avg_acceptance_rate,
            "spec_length": spec_length if spec_length is not None else 0
        }
    
    def _calculate_reward(self, 
                          action: BanditAction, 
                          acceptance_rate: float,
                          spec_length: int,
                          throughput: float) -> float:
        """计算动作的奖励值
        
        参数:
            action: 要评估的动作
            acceptance_rate: 推测被接受的比率 [0,1]
            spec_length: 平均推测长度
            throughput: 当前吞吐量(tokens/s)
            
        返回:
            计算的奖励值
        """
        # 如果吞吐量低于最低可接受值，施加惩罚
        throughput_penalty = 0.0
        if throughput < self.min_acceptable_throughput:
            throughput_penalty = (self.min_acceptable_throughput - throughput) * 0.5
        
        # 根据不同动作计算奖励
        if action == BanditAction.DISABLE_SPEC_DECODING:
            # 禁用推测解码时只考虑吞吐量
            reward = self.throughput_weight * throughput - throughput_penalty
        else:
            # 对于神经模型和n-gram模型，考虑吞吐量、接受率和推测长度
            # 吞吐量是主要优化目标
            throughput_reward = self.throughput_weight * throughput
            
            # 推测效率 = 接受率 * 推测长度
            # 高接受率和长推测长度的组合能带来最大收益
            spec_efficiency = acceptance_rate * spec_length
            spec_reward = (1 - self.throughput_weight) * spec_efficiency
            
            # 神经模型通常有更高的接受率但可能更慢
            if action == BanditAction.SWITCH_TO_NEURAL:
                # 稍微提高神经模型的奖励，以平衡其更高的计算成本
                model_bonus = 0.05 * throughput if acceptance_rate > 0.7 else 0.0
                reward = throughput_reward + spec_reward + model_bonus - throughput_penalty
            else:  # SWITCH_TO_NGRAM
                # n-gram模型更快但接受率可能较低
                # 仅当接受率合理时才给予奖励
                model_bonus = 0.1 * throughput if acceptance_rate > 0.4 else 0.0
                reward = throughput_reward + spec_reward + model_bonus - throughput_penalty
        
        return reward
    
    def _get_ucb_value(self, action: BanditAction, total_count: int) -> float:
        """计算一个动作的UCB值"""
        if self.counts[action] == 0:
            return float('inf')  # 未尝试过的动作有无限大的价值
        
        # 计算平均奖励
        if self.reward_history[action]:
            # 处理奖励可能是张量列表的情况
            rewards = []
            for r in self.reward_history[action]:
                if isinstance(r, torch.Tensor):
                    rewards.append(r.cpu().numpy())
                else:
                    rewards.append(r)
            reward_mean = np.mean(rewards)
        else:
            reward_mean = 0.0
        
        # 计算探索奖励
        exploration_bonus = self.exploration_weight * math.sqrt(2 * math.log(total_count) / self.counts[action])
        return reward_mean + exploration_bonus
        
    def _has_min_samples(self) -> bool:
        """检查是否所有动作都已被尝试了最小次数"""
        return all(self.counts[action] >= self.min_samples_per_arm for action in self.actions)
    
    def select_action(self) -> Optional[BanditAction]:
        """基于当前指标选择最佳动作"""
        # 检查是否在冷却期
        current_time = time.time()
        
       
            
        # 如果我们没有足够的样本，使用探索-利用平衡(UCB)
        if not self._has_min_samples():
            logger.info("没有足够的样本,探索！")
            # 计算每个动作的UCB值
            total_count = sum(self.counts.values()) + 1  # 避免除零
            ucb_values = {action: self._get_ucb_value(action, total_count) for action in self.actions}
            print(ucb_values)
            # 选择UCB值最高的动作
            action = max(ucb_values, key=ucb_values.get)
        else:
            # 否则选择预期奖励最高的动作
            logger.info("有足够的样本,选择预期奖励最高的动作")
            action = max(self.expected_rewards, key=self.expected_rewards.get)
        
        # 检查选择的动作是否有效（不是当前已选择的状态）
        if action == BanditAction.SWITCH_TO_NEURAL and self.using_neural_model:
            logger.info("已经使用神经模型，跳过动作")
            return None
        
        if action == BanditAction.SWITCH_TO_NGRAM and self.using_ngram_model:
            logger.info("已经使用n-gram模型，跳过动作")
            return None
        
        if action == BanditAction.DISABLE_SPEC_DECODING and self.spec_decoding_disabled:
            logger.info("推测解码已禁用，跳过动作")
            return None
        
        # 更新状态
        self.last_action = action
        self.last_action_time = current_time
        self.counts[action] += 1
        
        
        logger.info(f"选择动作: {action.name}, 计数: {self.counts[action]}")
        return action
    
    def get_status(self) -> Dict[str, Any]:
        """获取优化器的当前状态"""
        
        # 处理奖励历史中可能包含的张量
        reward_means = {}
        for action, rewards_list in self.reward_history.items():
            if rewards_list:
                processed_rewards = []
                for r in rewards_list:
                    if isinstance(r, torch.Tensor):
                        processed_rewards.append(r.cpu().numpy())
                    else:
                        processed_rewards.append(r)
                reward_means[action.name] = np.mean(processed_rewards)
            else:
                reward_means[action.name] = 0.0
        
        return {
            "action_counts": {
                action.name: self.counts[action]
                for action in self.actions
            },
            "reward_means": reward_means,
            "last_action": self.last_action.name if self.last_action else None,
            "exploration_strategy": self.exploration_weight,
            "current_throughput": self.current_state["throughput"] if self.current_state else 0.0,
            "using_ngram_model": self.using_ngram_model,
            "using_neural_model": self.using_neural_model,
            "spec_decoding_disabled": self.spec_decoding_disabled
        } 