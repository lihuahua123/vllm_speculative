import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
from collections import deque
import os
import math

logger = logging.getLogger(__name__)

# 与其他优化器保持一致的动作枚举
class DQNAction(Enum):
    """DQN优化器可以采取的动作"""
    SWITCH_TO_NEURAL = 0  # 使用神经网络草稿模型
    SWITCH_TO_NGRAM = 1   # 使用n-gram草稿模型
    DISABLE_SPEC_DECODING = 2  # 禁用推测解码

# 定义DQN网络结构
class DQNNetwork(nn.Module):
    """DQN神经网络模型，用于预测不同动作的Q值"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """初始化DQN网络
        
        参数:
            state_dim: 状态向量的维度
            action_dim: 可能的动作数量
            hidden_dim: 隐藏层的维度
        """
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """前向传播，计算给定状态的Q值"""
        return self.network(state)

# 定义经验回放缓冲区
class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样转换"""
    
    def __init__(self, capacity: int):
        """初始化缓冲区
        
        参数:
            capacity: 缓冲区的最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """将转换添加到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """从缓冲区随机采样一批转换
        
        参数:
            batch_size: 要采样的批大小
            
        返回:
            包含状态、动作、奖励、下一个状态和完成标志的批量数据
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
    
    def __len__(self):
        """返回缓冲区中的转换数量"""
        return len(self.buffer)

class DQNOptimizer:
    """基于DQN的动态模型切换优化器
    
    此优化器使用深度Q网络（DQN）和马尔科夫决策过程（MDP）来动态选择最佳的推测采样方法:
    1. 在神经模型和n-gram模型之间切换
    2. 必要时禁用推测解码
    
    优化器会考虑当前的系统状态（吞吐量、请求量、接受率等）来做出决策。
    """
    
    def __init__(self, 
                 llm_engine,
                 state_dim: int = 5,  # 状态维度：[吞吐量, 请求量, 接受率, 推测长度, 当前模型]
                 action_dim: int = 3,  # 动作维度：[神经模型, n-gram模型, 禁用推测解码]
                 hidden_dim: int = 64,  # 隐藏层维度
                 learning_rate: float = 0.001,  # 学习率
                 gamma: float = 0.99,  # 折扣因子
                 epsilon_start: float = 1.0,  # 初始探索率
                 epsilon_end: float = 0.1,  # 最终探索率
                 epsilon_decay: float = 0.995,  # 探索率衰减
                 buffer_capacity: int = 10000,  # 回放缓冲区容量
                 batch_size: int = 64,  # 批大小
                 target_update: int = 10,  # 目标网络更新频率
                 min_acceptable_throughput: float = 10.0,  # 最低可接受吞吐量
                 reward_window_size: int = 5):  # 奖励窗口大小
        
        self.llm_engine = llm_engine
        
        # 初始化状态和动作空间
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建动作列表
        self.actions = [
            DQNAction.SWITCH_TO_NEURAL,
            DQNAction.SWITCH_TO_NGRAM,
            DQNAction.DISABLE_SPEC_DECODING
        ]
        
        # 初始化DQN网络
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不进行梯度更新
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练参数
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.train_count = 0
        
        # 初始化状态
        self.current_state = None
        self.last_state = None
        self.last_action = None
        self.last_action_time = 0.0
        
        # 动作状态
        self.using_ngram_model = False
        self.using_neural_model = False
        self.spec_decoding_disabled = False
        
        # 指标历史
        self.action_metrics_history = {action: {
            "acceptance_rates": [],
            "spec_lengths": [],
            "throughputs": []
        } for action in self.actions}
        
        self.reward_history = {action: [] for action in self.actions}
        self.counts = {action: 0 for action in self.actions}
        
        # 配置
        self.min_acceptable_throughput = min_acceptable_throughput
        self.reward_window_size = reward_window_size

        # 指标跟踪
        self.acceptance_rate_history = []
        self.spec_length_history = []
        self.throughput_history = []
        self.request_load_history = []
    
    def _normalize_state(self, state_dict: Dict[str, float]) -> np.ndarray:
        """标准化状态向量
        
        参数:
            state_dict: 包含状态变量的字典
            
        返回:
            标准化的状态向量
        """
        # 提取相关状态变量
        throughput = state_dict.get("throughput", 0.0)
        request_load = state_dict.get("request_load", 0.0)
        acceptance_rate = state_dict.get("acceptance_rate", 0.0)
        spec_length = state_dict.get("spec_length", 0.0)
        
        # 编码当前模型状态
        model_state = np.zeros(3)  # [神经模型, n-gram模型, 禁用推测解码]
        if self.using_neural_model:
            model_state[0] = 1.0
        elif self.using_ngram_model:
            model_state[1] = 1.0
        elif self.spec_decoding_disabled:
            model_state[2] = 1.0
        
        # 标准化值
        normalized_throughput = min(throughput / 100.0, 1.0)  # 假设最大吞吐量为100 tokens/s
        normalized_request_load = min(request_load / 50.0, 1.0)  # 假设最大请求负载为50
        
        # 组合成状态向量
        state = np.array([
            normalized_throughput,
            normalized_request_load,
            acceptance_rate,  # 接受率已经在[0,1]范围内
            spec_length / 10.0,  # 假设最大推测长度为10
            model_state[0],
            model_state[1],
            model_state[2],
        ])
        
        return state
    
    def record_metrics(self, 
                      throughput: float, 
                      acceptance_rate: Optional[float] = None,
                      spec_length: Optional[int] = None,
                      request_load: Optional[int] = None) -> None:
        """记录当前系统指标
        
        参数:
            throughput: 当前吞吐量（tokens/s）
            acceptance_rate: 推测被接受的比率 [0,1]
            spec_length: 平均推测长度
            request_load: 当前请求负载
        """
        # 记录指标历史
        self.throughput_history.append(throughput)
        if request_load is not None:
            self.request_load_history.append(request_load)
        
        # 记录推测解码指标（如果可用）
        if acceptance_rate is not None:
            self.acceptance_rate_history.append(acceptance_rate)
        if spec_length is not None:
            self.spec_length_history.append(spec_length)
        
        # 计算平均接受率
        avg_acceptance_rate = sum(self.acceptance_rate_history)/len(self.acceptance_rate_history) if self.acceptance_rate_history else 0.0
        
        # 保持历史记录在有限范围内
        if len(self.throughput_history) > self.reward_window_size:
            self.throughput_history.pop(0)
            if self.request_load_history:
                self.request_load_history.pop(0)
            if self.acceptance_rate_history:
                self.acceptance_rate_history.pop(0)
            if self.spec_length_history:
                self.spec_length_history.pop(0)
        
        # 获取当前请求负载
        if request_load is None:
            scheduler_outputs = self.llm_engine.try_scheduler()
            request_load = len(scheduler_outputs.scheduled_seq_groups) - scheduler_outputs.num_prefill_groups
        
        # 记录当前动作的指标
        if self.last_action is not None:
            # 为每个动作记录吞吐率
            self.action_metrics_history[self.last_action]["throughputs"].append(throughput)
            
            if acceptance_rate is not None and spec_length is not None and self.last_action != DQNAction.DISABLE_SPEC_DECODING:
                self.action_metrics_history[self.last_action]["acceptance_rates"].append(acceptance_rate)
                self.action_metrics_history[self.last_action]["spec_lengths"].append(spec_length)
            
            # 保持历史记录在有限范围内
            if len(self.action_metrics_history[self.last_action]["throughputs"]) > self.reward_window_size:
                self.action_metrics_history[self.last_action]["throughputs"].pop(0)
                
            if self.last_action != DQNAction.DISABLE_SPEC_DECODING:
                if len(self.action_metrics_history[self.last_action]["acceptance_rates"]) > self.reward_window_size:
                    self.action_metrics_history[self.last_action]["acceptance_rates"].pop(0)
                    self.action_metrics_history[self.last_action]["spec_lengths"].pop(0)
        
        # 构建当前状态
        current_state_dict = {
            "throughput": throughput,
            "request_load": request_load,
            "acceptance_rate": avg_acceptance_rate,
            "spec_length": spec_length if spec_length is not None else 0
        }
        
        # 标准化并保存当前状态
        self.current_state = current_state_dict
        
        # 如果有上一个状态和动作，计算奖励并存储转换
        if self.last_state is not None and self.last_action is not None:
            reward = self._calculate_reward(self.last_action, throughput, avg_acceptance_rate, spec_length)
            
            # 存储奖励历史
            self.reward_history[self.last_action].append(reward)
            if len(self.reward_history[self.last_action]) > self.reward_window_size:
                self.reward_history[self.last_action].pop(0)
            
            # 构建转换并添加到回放缓冲区
            last_state_vector = self._normalize_state(self.last_state)
            current_state_vector = self._normalize_state(current_state_dict)
            
            # 转换动作枚举为整数
            action_idx = self.last_action.value
            
            # 添加转换到回放缓冲区
            self.replay_buffer.push(
                last_state_vector,
                action_idx,
                reward,
                current_state_vector,
                False  # 在这种情况下，我们不会有真正的终止状态
            )
        
        # 更新上一个状态
        self.last_state = current_state_dict
    
    def _calculate_reward(self, 
                          action: DQNAction, 
                          throughput: float,
                          acceptance_rate: float,
                          spec_length: int) -> float:
        """计算动作的奖励值
        
        参数:
            action: 要评估的动作
            throughput: 当前吞吐量(tokens/s)
            acceptance_rate: 推测被接受的比率 [0,1]
            spec_length: 平均推测长度
            
        返回:
            计算的奖励值
        """
        # 如果吞吐量低于最低可接受值，施加惩罚
        throughput_penalty = 0.0
        if throughput < self.min_acceptable_throughput:
            throughput_penalty = (self.min_acceptable_throughput - throughput) * 0.5
        
        # 标准化吞吐量(假设最大值为100 tokens/s)
        normalized_throughput = min(throughput / 100.0, 1.0)
        
        # 根据不同动作计算奖励
        if action == DQNAction.DISABLE_SPEC_DECODING:
            # 禁用推测解码时只考虑吞吐量
            reward = normalized_throughput - throughput_penalty
        else:
            # 对于神经模型和n-gram模型
            # 推测效率 = 接受率 * 推测长度
            spec_efficiency = acceptance_rate * min(spec_length / 10.0, 1.0)
            
            if action == DQNAction.SWITCH_TO_NEURAL:
                # 神经模型通常有更高的接受率但可能更慢
                model_bonus = 0.1 if acceptance_rate > 0.7 else 0.0
                reward = normalized_throughput * 0.7 + spec_efficiency * 0.3 + model_bonus - throughput_penalty
            else:  # SWITCH_TO_NGRAM
                # n-gram模型更快但接受率可能较低
                model_bonus = 0.1 if acceptance_rate > 0.4 else 0.0
                reward = normalized_throughput * 0.8 + spec_efficiency * 0.2 + model_bonus - throughput_penalty
        
        return float(reward)
    
    def train(self):
        """训练DQN网络"""
        # 如果缓冲区中没有足够的样本，跳过训练
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从回放缓冲区采样一批数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 计算当前Q值
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_state_values = next_q_values.max(1)[0]
            target_values = rewards + (1 - dones) * self.gamma * next_state_values
        
        # 计算损失
        loss = self.criterion(state_action_values, target_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.train_count += 1
        if self.train_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def select_action(self) -> Optional[DQNAction]:
        """基于当前状态选择动作"""
        # 如果没有当前状态，无法选择动作
        if self.current_state is None:
            return None
        
        # 探索-利用权衡
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            action = random.choice(self.actions)
            logger.info(f"DQN探索: 随机选择动作 {action.name}")
        else:
            # 利用：选择Q值最高的动作
            state = torch.FloatTensor(self._normalize_state(self.current_state)).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            
            # 获取Q值最高的动作索引
            action_idx = q_values.argmax().item()
            action = self.actions[action_idx]
            logger.info(f"DQN利用: 选择Q值最高的动作 {action.name}")
        
        # 检查选择的动作是否有效（不是当前已选择的状态）
        if action == DQNAction.SWITCH_TO_NEURAL and self.using_neural_model:
            logger.info("已经使用神经模型，跳过动作")
            return None
        
        if action == DQNAction.SWITCH_TO_NGRAM and self.using_ngram_model:
            logger.info("已经使用n-gram模型，跳过动作")
            return None
        
        if action == DQNAction.DISABLE_SPEC_DECODING and self.spec_decoding_disabled:
            logger.info("推测解码已禁用，跳过动作")
            return None
        
        # 更新状态
        self.last_action = action
        self.last_action_time = time.time()
        self.counts[action] += 1
        
        logger.info(f"选择动作: {action.name}, 计数: {self.counts[action]}")
        
        # 训练网络
        self.train()
        
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
                        processed_rewards.append(r.cpu().item())
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
            "epsilon": self.epsilon,
            "train_count": self.train_count,
            "buffer_size": len(self.replay_buffer),
            "current_throughput": self.current_state["throughput"] if self.current_state else 0.0,
            "using_ngram_model": self.using_ngram_model,
            "using_neural_model": self.using_neural_model,
            "spec_decoding_disabled": self.spec_decoding_disabled
        }
    
    def save_model(self, path: str) -> bool:
        """保存DQN网络模型到文件
        
        参数:
            path: 保存模型的路径
            
        返回:
            bool: 模型是否成功保存
        """
        try:
            # 创建包含所有相关组件的字典
            save_dict = {
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'train_count': self.train_count,
                'reward_history': self.reward_history,
                'counts': self.counts
            }
            
            # 保存模型
            torch.save(save_dict, path)
            logger.info(f"DQN模型已保存到 {path}")
            return True
        except Exception as e:
            logger.error(f"保存DQN模型时出错: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """从文件加载DQN网络模型
        
        参数:
            path: 加载模型的路径
            
        返回:
            bool: 模型是否成功加载
        """
        try:
            # 加载保存的字典
            checkpoint = torch.load(path)
            
            # 加载网络参数
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            
            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # 加载其他状态
            self.epsilon = checkpoint['epsilon']
            self.train_count = checkpoint['train_count']
            self.reward_history = checkpoint['reward_history']
            self.counts = checkpoint['counts']
            
            logger.info(f"DQN模型已从 {path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载DQN模型时出错: {e}")
            return False 