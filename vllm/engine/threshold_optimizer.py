# SPDX-License-Identifier: Apache-2.0

import numpy as np
import time
import logging
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ThresholdConfig:
    """存储当前阈值配置的数据类"""
    high_load_threshold: int
    high_load_threshold2: int
    speculative_batch_size: int
    
    def __repr__(self) -> str:
        return (f"ThresholdConfig(high_load={self.high_load_threshold}, "
                f"high_load2={self.high_load_threshold2}, "
                f"spec_batch={self.speculative_batch_size})")


class BanditArm:
    """表示多臂老虎机中的一个臂（即一个阈值配置）"""
    def __init__(self, config: ThresholdConfig):
        self.config = config
        # 贝塔分布参数 - Thompson Sampling
        self.alpha = 1.0  # 成功次数 + 1
        self.beta = 1.0   # 失败次数 + 1
        self.total_reward = 0.0
        self.pulls = 0
        self.avg_reward = 0.0
        self.last_reward = None

    def update(self, reward: float):
        """更新根据获得的奖励更新臂的状态"""
        # 规范化奖励，确保在0到1之间
        normalized_reward = min(max(reward / 100.0, 0.0), 1.0)
        
        # 更新贝塔分布参数
        self.alpha += normalized_reward
        self.beta += (1.0 - normalized_reward)
        
        # 更新统计信息
        self.pulls += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.pulls
        self.last_reward = reward
        
    def sample_value(self) -> float:
        """从贝塔分布采样一个值，用于Thompson Sampling"""
        return np.random.beta(self.alpha, self.beta)
    
    def to_dict(self):
        """将臂的状态转换为字典，用于序列化"""
        return {
            'config': asdict(self.config),
            'alpha': self.alpha,
            'beta': self.beta,
            'total_reward': self.total_reward,
            'pulls': self.pulls,
            'avg_reward': self.avg_reward,
            'last_reward': self.last_reward
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典恢复臂的状态"""
        arm = cls(ThresholdConfig(**data['config']))
        arm.alpha = data['alpha']
        arm.beta = data['beta']
        arm.total_reward = data['total_reward']
        arm.pulls = data['pulls']
        arm.avg_reward = data['avg_reward']
        arm.last_reward = data['last_reward']
        return arm


class ThresholdOptimizer:
    """使用多臂老虎机算法优化阈值配置"""
    
    def __init__(self, state_file_path=None):
        # 可能的配置空间
        self.high_load_values = [1, 2, 3, 5, 8]
        self.high_load2_values = [0, 1, 2]
        self.batch_size_values = [8, 16, 24, 32, 48]
        
        # 创建所有可能的配置组合
        self.arms: Dict[str, BanditArm] = {}
        
        # 当前选择的配置
        self.current_arm: Optional[BanditArm] = None
        self.current_arm_key: Optional[str] = None
        self.arm_switch_time = time.time()
        
        # 吞吐率监控
        self.token_count = 0
        self.start_time = time.time()
        self.last_optimization_time = time.time()
        self.optimization_interval = 30  # 30秒
        self.min_tokens_before_switch = 5000  # 至少处理这么多token后才考虑切换
        
        # 流量趋势监测
        self.throughput_history = []  # 存储历史吞吐率
        self.throughput_check_interval = 60  # 每分钟检查一次吞吐率
        self.running_time_threshold = 60  # 至少运行1分钟
        self.last_throughput_check = time.time()
        # self.min_trend_samples = 3  # 至少需要这么多样本才能检测趋势
        # self.trend_threshold = 0.15  # 吞吐率变化超过15%才认为有趋势
        # self.trend_consistency_required = 3  # 连续几次同向变化才认为是趋势
        
        # 尝试从文件加载状态，如果失败则初始化新的arms
        if state_file_path and os.path.exists(state_file_path):
            self.load_state(state_file_path)
            logger.info(f"已从 {state_file_path} 加载多臂老虎机状态")
        else:
            self._initialize_arms()
            logger.info("已初始化新的多臂老虎机状态")
        
        # 初始选择一个配置
        self.select_next_arm()
        
        # 保存状态路径
        self.state_file_path = state_file_path
        
    def _initialize_arms(self):
        """初始化所有可能的配置组合"""
        for h1 in self.high_load_values:
            for h2 in self.high_load2_values:
                if h2 >= h1:  # 跳过无效配置
                    continue
                for bs in self.batch_size_values:
                    config = ThresholdConfig(h1, h2, bs)
                    key = self._config_to_key(config)
                    self.arms[key] = BanditArm(config)
    
    def _config_to_key(self, config: ThresholdConfig) -> str:
        """将配置转换为唯一键"""
        return f"{config.high_load_threshold}_{config.high_load_threshold2}_{config.speculative_batch_size}"
    
    def select_next_arm(self) -> ThresholdConfig:
        """使用Thompson Sampling选择下一个配置"""
        # 计算每个臂的采样值
        samples = {key: arm.sample_value() for key, arm in self.arms.items()}
        
        # 选择采样值最高的臂
        best_key = max(samples, key=samples.get)
        
        # 如果选择了新的臂，记录时间
        if best_key != self.current_arm_key:
            prev_config = self.current_arm.config if self.current_arm else None
            self.current_arm_key = best_key
            self.current_arm = self.arms[best_key]
            self.arm_switch_time = time.time()
            
            logger.info(f"切换阈值配置: {prev_config} -> {self.current_arm.config}")
            logger.info(f"当前最佳臂平均奖励: {self.current_arm.avg_reward:.2f} tokens/s")
        
        return self.current_arm.config
    
    def record_tokens(self, token_count: int):
        """记录生成的token数量，用于计算吞吐率"""
        self.token_count += token_count
    
    def should_optimize(self) -> bool:
        """检查是否应该优化阈值配置"""
        now = time.time()
        # 确保当前配置已经运行足够长的时间
        if now - self.arm_switch_time < self.running_time_threshold:  # 至少运行1分钟
            return False
            
        # 确保处理了足够多的token
        if self.token_count < self.min_tokens_before_switch:
            return False
            
        # 检查吞吐率趋势
        # if now - self.last_throughput_check >= self.throughput_check_interval:
        #     self.last_throughput_check = now
            
        #     # 计算当前吞吐率
        #     elapsed = now - self.start_time
        #     if elapsed > 0 and self.token_count > 0:
        #         current_throughput = self.token_count / elapsed
        #         self.throughput_history.append(current_throughput)
                
        #         # 只保留最近的样本
        #         if len(self.throughput_history) > 10:
        #             self.throughput_history = self.throughput_history[-10:]
                
        #         # 检测趋势
        #         if len(self.throughput_history) >= self.min_trend_samples:
        #             # 计算连续变化的方向
        #             directions = []
        #             for i in range(1, len(self.throughput_history)):
        #                 prev = self.throughput_history[i-1]
        #                 curr = self.throughput_history[i]
        #                 if curr > prev * (1 + self.trend_threshold):
        #                     directions.append(1)  # 上升
        #                 elif curr < prev * (1 - self.trend_threshold):
        #                     directions.append(-1)  # 下降
        #                 else:
        #                     directions.append(0)  # 稳定
                    
        #             # 检查是否有连续的上升或下降趋势
        #             if len(directions) >= self.trend_consistency_required:
        #                 recent_directions = directions[-self.trend_consistency_required:]
        #                 if all(d > 0 for d in recent_directions) or all(d < 0 for d in recent_directions):
        #                     logger.info(f"检测到持续{'上升' if recent_directions[0] > 0 else '下降'}趋势，触发优化")
        #                     return True
            
        # 检查是否到了优化间隔
        return now - self.last_optimization_time >= self.optimization_interval
    
    def optimize(self):
        """计算当前配置的奖励并选择下一个配置"""
        now = time.time()
        elapsed = now - self.start_time
        
        if elapsed < 1.0 or self.token_count < 100:
            return  # 数据不足，跳过优化
        
        # 计算吞吐率作为奖励
        throughput = self.token_count / elapsed
        
        # 更新当前臂的奖励
        if self.current_arm:
            logger.info(f"当前配置 {self.current_arm.config} 的吞吐率: {throughput:.2f} tokens/s")
            self.current_arm.update(throughput)
        
        # 重置统计
        self.token_count = 0
        self.start_time = now
        self.last_optimization_time = now
        
        # 重置流量趋势监测
        self.throughput_history = []
        self.last_throughput_check = now
        
        # 选择下一个配置
        self.select_next_arm()
        
        # 优化后保存状态
        if self.state_file_path:
            self.save_state(self.state_file_path)
    
    def get_current_thresholds(self) -> Tuple[int, int, int]:
        """获取当前的阈值配置"""
        if not self.current_arm:
            # 默认值
            return 3, 1, 32
            
        config = self.current_arm.config
        return config.high_load_threshold, config.high_load_threshold2, config.speculative_batch_size
    
    def log_stats(self):
        """记录当前所有臂的统计信息"""
        # 按平均奖励排序
        sorted_arms = sorted(self.arms.values(), key=lambda arm: arm.avg_reward, reverse=True)
        
        logger.info("当前阈值优化统计:")
        for i, arm in enumerate(sorted_arms[:5]):  # 只显示前5个
            logger.info(f"{i+1}. {arm.config}: 拉动次数={arm.pulls}, 平均奖励={arm.avg_reward:.2f} tokens/s")
    
    def save_state(self, file_path):
        """保存多臂老虎机状态到文件"""
        try:
            # 收集需要保存的状态
            state = {
                'arms': {key: arm.to_dict() for key, arm in self.arms.items()},
                'current_arm_key': self.current_arm_key,
                'timestamp': time.time(),
                'throughput_history': self.throughput_history,
                'last_throughput_check': self.last_throughput_check
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 保存到文件 - 使用pickle格式保存
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"已保存多臂老虎机状态到 {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存多臂老虎机状态失败: {e}")
            return False
    
    def load_state(self, file_path):
        """从文件加载多臂老虎机状态"""
        try:
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
            
            # 重建arms对象
            self.arms = {}
            for key, arm_data in state['arms'].items():
                self.arms[key] = BanditArm.from_dict(arm_data)
            
            # 恢复当前选中的臂
            self.current_arm_key = state['current_arm_key']
            if self.current_arm_key and self.current_arm_key in self.arms:
                self.current_arm = self.arms[self.current_arm_key]
                
            # 恢复流量趋势监测相关变量
            if 'throughput_history' in state:
                self.throughput_history = state['throughput_history']
            if 'last_throughput_check' in state:
                self.last_throughput_check = state['last_throughput_check']
            
            logger.info(f"已加载多臂老虎机状态，包含 {len(self.arms)} 个臂")
            return True
        except Exception as e:
            logger.error(f"加载多臂老虎机状态失败: {e}")
            self._initialize_arms()  # 加载失败时初始化新的arms
            return False


class DraftModelSwitcher:
    """控制草稿模型切换的类，整合阈值优化器"""
    
    def __init__(self, llm_engine, state_file_path=None):
        self.llm_engine = llm_engine
        # 使用指定的状态文件路径初始化优化器
        if state_file_path is None:
            # 默认状态保存路径
            state_file_path = os.path.join(os.path.expanduser("~"), ".vllm", "threshold_optimizer_state.pkl")
        
        self.optimizer = ThresholdOptimizer(state_file_path)
        self.last_check_time = time.time()
        self.check_interval = 5  # 5秒检查一次
        
        # 定期保存状态的计时
        self.last_save_time = time.time()
        self.save_interval = 1800  # 30分钟保存一次
        
        # 更新初始阈值
        self._update_engine_thresholds()
        
    def _update_engine_thresholds(self):
        """将优化器的阈值应用到引擎"""
        h1, h2, bs = self.optimizer.get_current_thresholds()
        
        # 更新LLM引擎的阈值
        self.llm_engine.request_load_tracker['high_load_threshold'] = h1
        self.llm_engine.request_load_tracker['high_load_threshold2'] = h2
        self.llm_engine.model_executor.set_disable_by_batch_size(bs)
        print(f"update thresholds: h1: {h1}, h2: {h2}, bs: {bs}")
    
    def record_tokens(self, token_count: int):
        """记录生成的token数量"""
        self.optimizer.record_tokens(token_count)
    
    def check_and_update(self):
        """检查是否需要优化阈值并更新"""
        now = time.time()
        if now - self.last_check_time < self.check_interval:
            return
            
        self.last_check_time = now
        
        # 检查是否需要执行优化
        if self.optimizer.should_optimize():
            self.optimizer.optimize()
            self._update_engine_thresholds()
            # 每5次优化记录一次详细统计
            if np.random.random() < 0.2:
                self.optimizer.log_stats()
        
        # 定期保存状态，即使没有优化也保存
        if now - self.last_save_time >= self.save_interval:
            if self.optimizer.state_file_path:
                self.optimizer.save_state(self.optimizer.state_file_path)
                self.last_save_time = now
                logger.info("已定期保存多臂老虎机状态")
    
    def save_state(self):
        """手动保存当前状态"""
        if self.optimizer.state_file_path:
            return self.optimizer.save_state(self.optimizer.state_file_path)
        return False 