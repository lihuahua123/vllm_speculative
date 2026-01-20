import numpy as np
import pickle
import time
from joblib import load
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, List, Optional
import json
from pathlib import Path

import numpy as np

class OnlineDASpec:
    def __init__(self, verify_model_path, draft_model_path, max_proposed_length=5):
        """
        初始化在线DASpec
        """
        self.max_proposed_length = max_proposed_length
        self.prev_alphas = []
        self.correction_factors = {k:{b:1 for b in range(1,300)} for k in range(self.max_proposed_length + 1)}


        # 加载在线模型
        try:
            with open(verify_model_path, 'rb') as f:
                self.verify_model = pickle.load(f)
            with open(draft_model_path, 'rb') as f:
                self.draft_model = pickle.load(f)
            print(f"成功加载在线模型: {verify_model_path}, {draft_model_path}")
        except Exception as e:
            print(f"加载在线模型失败: {e}")
            self.verify_model = None
            self.draft_model = None

        # 加载训练表
        try:
            with open('./train_table_avg_llama.pkl', 'rb') as f:
                self.train_table_avg = pickle.load(f)
        except:
            print("警告: 无法加载train_table_avg_llama.pkl")
            self.train_table_avg = {}

    def moving_average(self, window_size=10):
        if len(self.prev_alphas) == 0:
            return 0.7
        window = self.prev_alphas[-window_size:]
        return sum(window) / len(window)

    def moving_average_history(self, speculative_metrics_history, window_size=10):
        if len(speculative_metrics_history) == 0:
            return 0
        window_history = speculative_metrics_history[-window_size:]
        #  0: draft, 1: scoring, 2: verification 3: batch size 4: num_accepted_tokens 5: context_length 6: stage 7: proposal_length
        scoring_time = 0
        verification_time = 0
        for i in range(len(window_history)):
            scoring_time += window_history[i][1]
            #verification_time += window_history[i][2]
        scoring_time = scoring_time / len(window_history)
        #verification_time = verification_time / len(window_history)
        return scoring_time
    def online_correction_factor(self,k,batch_size,actual_accepted_tokens):
        # After each batch, update:
        observed = actual_accepted_tokens
        predicted = self.train_table_avg[k][batch_size]
        correction = observed / (predicted + 1e-6)
        # Use a moving average for correction
        self.correction_factors[k][batch_size] = 0.5 * self.correction_factors[k][batch_size] + 0.5 * correction
        # print("correction_factors",self.correction_factors[k][batch_size],k,batch_size)

    def estimate_generated_length(self, alpha, k, batch_size):
        """
        估计生成的token长度。带bonus的
        :param alpha: token接受率。
        :param k: 推测长度。
        :return: 生成的token长度。
        """
        if alpha == 1:
            return batch_size * (k + 1) # 如果接受率为1，生成k+1个token
        Analytical_length = batch_size * (1 - alpha ** (k + 1)) / (1 - alpha)
        if self.train_table_avg[k][batch_size] < 0:
            return Analytical_length
        p10 = self.train_table_avg[k][batch_size] * self.correction_factors[k][batch_size]
        return max(p10,Analytical_length)


    def estimate_batch_execution_time(self, context_length, batch_size, proposed_length, moving_average_scoring_time):
        if self.draft_model is None or self.verify_model is None:
            return 0.1

        draft_features = {
            'context_length': context_length,
            'batch_size': batch_size,
            'gamma': proposed_length
        }
        draft_time = self.draft_model.predict_one(draft_features)
        if draft_time is None:
            draft_time = 0.01

        verify_features = {
            'context_length': context_length,
            'batch_size': batch_size * (proposed_length + 1),
            'gamma': 1
        }
        verify_time = self.verify_model.predict_one(verify_features)
        verify_time = max(moving_average_scoring_time, verify_time)
        if verify_time is None:
            verify_time = 0.01

        return draft_time + verify_time

    def goodput_estimation(self, context_length, batch_size, proposed_length, alpha, speculative_metrics_history):
        if proposed_length == 0:
            verify_features = {
                'context_length': context_length,
                'batch_size': batch_size,
                'gamma': 1
            }
            time_predict =  self.verify_model.predict_one(verify_features)
            if time_predict is None or time_predict <= 0:
                time_predict = 0.01
            return batch_size / time_predict

        generated_length = self.estimate_generated_length(alpha, proposed_length, batch_size)
        execution_time = self.estimate_batch_execution_time(context_length, batch_size, proposed_length, self.moving_average_history(speculative_metrics_history))
        print("proposed_length", proposed_length, "generated_length", generated_length, "execution_time", execution_time)
        if execution_time <= 0:
            execution_time = 0.01

        return generated_length / execution_time

    def optimize_proposed_length(self, context_length, batch_size, speculative_metrics_history=None):
        best_goodput = -1
        best_length = 0
        alpha = self.moving_average()
        for k in range(0, self.max_proposed_length + 1):
            goodput = self.goodput_estimation(context_length, batch_size, k, alpha, speculative_metrics_history)
            if goodput > best_goodput:
                best_goodput = goodput
                best_length = k

        return best_length

    def update_with_feedback(self, context_length, batch_size, proposed_length,
                           actual_draft_time, actual_verify_time, num_accepted_tokens):
        if self.draft_model is None or self.verify_model is None:
            return

        try:
            if proposed_length > 0:
                draft_features = {
                    'context_length': context_length,
                    'batch_size': batch_size,
                    'gamma': proposed_length
                }
                #self.draft_model.learn_one(draft_features, actual_draft_time)

            verify_features = {
                'context_length': context_length,
                'batch_size': batch_size * (proposed_length + 1) if proposed_length > 0 else batch_size,
                'gamma': 1
            }
            #self.verify_model.learn_one(verify_features, actual_verify_time)

            # if proposed_length > 0:
            #     acceptance_rate = num_accepted_tokens / (proposed_length * batch_size)
            #     self.prev_alphas.append(acceptance_rate)
            #     if len(self.prev_alphas) > 100:
            #         self.prev_alphas = self.prev_alphas[-50:]

        except Exception as e:
            print(f"在线学习更新失败: {e}")
class SmartSpec:
    def __init__(self, model, draft_model, max_proposed_length=5):
        """
        初始化SmartSpec。
        :param model: 模型，用于计算执行时间。
        :param max_proposed_length: 最大推测长度。
        """
        self.model = model
        self.draft_model = draft_model
        self.max_proposed_length = max_proposed_length
        self.prev_alphas = []  # 用于存储历史token接受率
        self.smoothed = -1
        self.explore_for_low_alpha_cnt = 0

    def moving_average(self, window_size=10):
        """
        计算历史token接受率的移动平均值。
        :param window_size: 移动平均的窗口大小。
        :return: 移动平均值。
        """
        if len(self.prev_alphas) == 0:
            return 0.7  # 默认值，如果没有历史数据
        window = self.prev_alphas[-window_size:]
        return sum(window) / len(window)

    def estimate_generated_length(self, alpha, k):
        """
        估计生成的token长度。带bonus的
        :param alpha: token接受率。
        :param k: 推测长度。
        :return: 生成的token长度。
        """
        if alpha == 1:
            return k + 1  # 如果接受率为1，生成k+1个token
        return (1 - alpha ** (k + 1)) / (1 - alpha)

    def estimate_batch_execution_time(self, context_length, batch_size, proposed_length, speculative_metrics=None):
        """
        估计批处理的执行时间。
        :param batch_size: 批处理大小。
        :param context_length: 上下文长度。
        :return: 执行时间。
        """

        draft = self.draft_model.predict([[context_length,batch_size]])[0]
        target = self.model.predict([[context_length, batch_size*proposed_length]])[0]

        return draft *  proposed_length + target
    def goodput_estimation(self, context_length, batch_size, proposed_length, alpha, speculative_metrics=None):
        """
        计算goodput。
        :param batch_size: 批处理大小。
        :param proposed_length: 推测长度。
        :return: goodput值。
        """
        if proposed_length == 0:
            # 0: draft, 1: scoring, 2: verification 3: batch size 4: num_accepted_tokens 5: context_length 6: stage 7: proposed_length
            time_predict = self.model.predict([[context_length, 0]])[0]
            return batch_size/time_predict
        generated_length = batch_size * self.estimate_generated_length(alpha, proposed_length)
        execution_time = self.estimate_batch_execution_time(context_length,batch_size,proposed_length, speculative_metrics)
        return generated_length / execution_time

    def optimize_proposed_length(self, context_length, batch_size, speculative_metrics=None,context_lengths=None):
        """
        优化推测长度，选择最大化goodput的长度。
        :param batch_size: 批处理大小。
        :return: 最优的推测长度。
        """
        best_goodput = -1
        best_length = 0
        min_k = 0
        alpha = self.moving_average() #self.exponential_smoothing()

        for k in range(min_k, self.max_proposed_length + 1):
            goodput = self.goodput_estimation(context_length,batch_size, k, alpha, speculative_metrics)
            #print("proposed_length",k,"goodput",goodput)
            if goodput > best_goodput:
                best_goodput = goodput
                best_length = k
        return best_length, best_goodput
def goodput_estimation_unpack(args):
    return DASpec.goodput_estimation(*args)
class DASpec:
    def __init__(self, model, draft_model,max_proposed_length=5):
        """
        初始化DASpec
        :param model: 模型，用于计算执行时间。
        :param max_proposed_length: 最大推测长度。
        """
        self.model = load(model)
        self.draft_model = load(draft_model)
        self.max_proposed_length = max_proposed_length
        self.prev_alphas = []  # 用于存储历史token接受率
        self.smoothed = -1
        self.continue_low_alpha = 0
        #self.generated_token_num_predict_model = generated_token_num_predict_model
        # train_table_avg
        with open('./train_table_avg_specbench.pkl', 'rb') as f:
            #with open('./train_table_avg_llama.pkl', 'rb') as f:
            #with open('./train_table_avg_alpaca.pkl', 'rb') as f:
            self.train_table_avg = pickle.load(f)
        self.correction_factors = {k:{b:1 for b in range(1,300)} for k in range(self.max_proposed_length + 1)}
        # goodput_estimation 必须是静态方法或全局函数
        from functools import partial
        self.goodput_estimation_func = partial(self.goodput_estimation)  # 如果是静态方法可直接用

    def exponential_smoothing(self, alpha=0.1):
        """
        计算指数平滑的alpha值。
        :param alpha: 平滑系数。
        :return: 平滑系数。
        """
        if len(self.prev_alphas) == 0:
            return 0.7
        if self.smoothed == -1:
            self.smoothed = self.prev_alphas[-1]
            return self.smoothed
        actual = self.prev_alphas[-1]
        self.smoothed = alpha * actual + (1 - alpha) * self.smoothed
        return self.smoothed

    def moving_average(self, window_size=10):
        """
        计算历史token接受率的移动平均值。
        :param window_size: 移动平均的窗口大小。
        :return: 移动平均值。
        """
        if len(self.prev_alphas) == 0:
            return 0.7  # 默认值，如果没有历史数据
        window = self.prev_alphas[-window_size:]
        #print("window",window)
        return sum(window) / len(window)
    def moving_average_history(self, speculative_metrics_history, window_size=10):
        if len(speculative_metrics_history) == 0:
            return 0
        window_history = speculative_metrics_history[-window_size:]
        #  0: draft, 1: scoring, 2: verification 3: batch size 4: num_accepted_tokens 5: context_length 6: stage 7: proposal_length
        scoring_time = 0
        verification_time = 0
        for i in range(len(window_history)):
            scoring_time += window_history[i][1]
            #verification_time += window_history[i][2]
        scoring_time = scoring_time / len(window_history)
        #verification_time = verification_time / len(window_history)
        return scoring_time

    def online_correction_factor(self,k,batch_size,actual_accepted_tokens):
        # After each batch, update:
        observed = actual_accepted_tokens
        predicted = self.train_table_avg[k][batch_size]
        correction = observed / (predicted + 1e-6)
        # Use a moving average for correction
        self.correction_factors[k][batch_size] = 0.5 * self.correction_factors[k][batch_size] + 0.5 * correction
        # print("correction_factors",self.correction_factors[k][batch_size],k,batch_size)

    def estimate_generated_length(self, alpha, k, batch_size):
        """
        估计生成的token长度。带bonus的
        :param alpha: token接受率。
        :param k: 推测长度。
        :return: 生成的token长度。
        """
        if alpha == 1:
            return batch_size * (k + 1) # 如果接受率为1，生成k+1个token
        Analytical_length = batch_size * (1 - alpha ** (k + 1)) / (1 - alpha)
        if self.train_table_avg[k][batch_size] < 0:
            return Analytical_length
        p10 = self.train_table_avg[k][batch_size] * self.correction_factors[k][batch_size]
        return p10 #max(p10,Analytical_length)


    def estimate_batch_execution_time(self, context_length, batch_size, proposed_length, speculative_metrics=None,draft_predict=None,average_scoring_time=0):
        """
        估计批处理的执行时间。
        :param batch_size: 批处理大小。
        :param context_length: 上下文长度。
        :return: 执行时间。
        """
        # 0: draft, 1: scoring, 2: verification 3: batch size 4: num_accepted_tokens 5: context_length 6: stage 7: proposed_length
        # 假设执行时间是线性的，基于模型系数 FIXME 万一前面的和后面的batch size 不一样，得到的时间也不一样


        draft = self.draft_model.predict([[context_length, batch_size, proposed_length]])[0]
        target = self.model.predict([[context_length, batch_size*(proposed_length+1),1]])[0]
        return draft + target # verification
    def chage_table(self,k,batch_size,new):
        self.train_table_avg[k][batch_size] = new

    @staticmethod
    def goodput_estimation_parallel(context_length, batch_size, proposed_length, alpha, speculative_metrics, draft_predict, model, draft_model, train_table_avg, correction_factors):
        """
        计算goodput。
        :param batch_size: 批处理大小。
        :param proposed_length: 推测长度。
        :return: goodput值。
        """
        def estimate_generated_length(alpha, k, batch_size, train_table_avg, correction_factors):
            if alpha == 1:
                return batch_size * (k + 1)
            Analytical_length = batch_size * (1 - alpha ** (k + 1)) / (1 - alpha)
            if train_table_avg[k][batch_size] < 0:
                return Analytical_length
            p10 = train_table_avg[k][batch_size] * correction_factors[k][batch_size]
            return max(p10, Analytical_length)

        def estimate_batch_execution_time(context_length, batch_size, proposed_length, speculative_metrics, draft_predict, model, draft_model):
            draft = draft_model.predict([[context_length, batch_size, proposed_length]])[0]
            target = model.predict([[context_length, batch_size * (proposed_length + 1), 1]])[0]
            return draft + target

        if proposed_length == 0:
            time_predict = model.predict([[context_length, batch_size, 1]])[0]
            return batch_size / time_predict
        generated_length = estimate_generated_length(alpha, proposed_length, batch_size, train_table_avg, correction_factors)
        execution_time = estimate_batch_execution_time(context_length, batch_size, proposed_length, speculative_metrics, draft_predict, model, draft_model)
        return generated_length / execution_time

    def goodput_estimation(self,context_length, batch_size, proposed_length, alpha, speculative_metrics, draft_predict):
        """
        计算goodput。
        :param batch_size: 批处理大小。
        :param proposed_length: 推测长度。
        :return: goodput值。
        """
        draft_predict = draft_predict
        model = self.model
        draft_model = self.draft_model
        train_table_avg = self.train_table_avg
        correction_factors = self.correction_factors
        def estimate_generated_length(alpha, k, batch_size, train_table_avg, correction_factors):
            if alpha == 1:
                return batch_size * (k + 1)
            Analytical_length = batch_size * (1 - alpha ** (k + 1)) / (1 - alpha)
            if train_table_avg[k][batch_size] < 0:
                return Analytical_length
            p10 = train_table_avg[k][batch_size] * correction_factors[k][batch_size]
            return max(p10, Analytical_length)

        def estimate_batch_execution_time(context_length, batch_size, proposed_length, speculative_metrics, draft_predict, model, draft_model):
            draft = draft_model.predict([[context_length, batch_size, proposed_length]])[0]
            target = model.predict([[context_length, batch_size * (proposed_length + 1), 1]])[0]
            return draft + target

        if proposed_length == 0:
            time_predict = model.predict([[context_length, batch_size, 1]])[0]
            return batch_size / time_predict
        generated_length = estimate_generated_length(alpha, proposed_length, batch_size, train_table_avg, correction_factors)
        execution_time = estimate_batch_execution_time(context_length, batch_size, proposed_length, speculative_metrics, draft_predict, model, draft_model)
        return generated_length / execution_time

    def optimize_proposed_length(self, context_length, batch_size, speculative_metrics=None):
        best_goodput = -1
        best_length = 0
        alpha = self.moving_average()
        next_alpha = alpha
        draft_predict = None
        results = []
        args_list = [
            (context_length, batch_size, k, next_alpha, speculative_metrics, draft_predict, self.model, self.draft_model, self.train_table_avg, self.correction_factors)
            for k in range(0, self.max_proposed_length + 1)
        ]

        for k in [0,self.max_proposed_length]:#range(0, self.max_proposed_length + 1):
            results.append([k,self.goodput_estimation(context_length, batch_size, k, next_alpha, speculative_metrics, draft_predict)])


        for k, goodput in results:
            if goodput > best_goodput:
                best_goodput = goodput
                best_length = k
        return best_length

    def change_prev_alphas(self,prev_alphas):
        self.prev_alphas = prev_alphas

class DASpecWithExploration:
    def __init__(self, model, draft_model, max_proposed_length=5,
                 exploration_rate=0.1, exploration_decay=0.99, min_exploration_rate=0.01,
                 context_bins=200, context_min=1, context_max=200,
                 consecutive_threshold=5, forced_exploration_arms=[2]):
        """
        初始化带探索机制的DASpec包装器
        
        Args:
            model: 模型路径，用于计算执行时间
            draft_model: 草稿模型路径
            max_proposed_length: 最大推测长度
            exploration_rate: 初始探索概率
            exploration_decay: 探索概率衰减率
            min_exploration_rate: 最小探索概率
            context_bins: 上下文分箱数量
            context_min: 最小上下文值
            context_max: 最大上下文值
            consecutive_threshold: 连续选择同一arm的阈值
            forced_exploration_arms: 强制探索的候选arm列表
        """
        # 初始化原始的DASpec
        self.daspec = DASpec(model, draft_model, max_proposed_length)

        # 探索参数
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # 上下文分箱
        self.context_bins = context_bins
        self.context_min = context_min
        self.context_max = context_max
        self.context_bounds = np.linspace(context_min, context_max, context_bins + 1)

        # 每个arm-context组合的统计信息
        self.arm_stats = {
            'counts': np.zeros((max_proposed_length + 1, context_bins)),  # 选择次数
            'rewards': np.zeros((max_proposed_length + 1, context_bins)),  # 累积奖励
            'ucb_values': np.zeros((max_proposed_length + 1, context_bins)),  # UCB值
        }

        # 连续选择检测
        self.consecutive_threshold = consecutive_threshold
        self.forced_exploration_arms = forced_exploration_arms
        self.consecutive_selections = {}  # {context_bin: {'last_arm': arm, 'count': count}}
        self.performance_history = {
            'avg_rewards': np.zeros((max_proposed_length + 1, context_bins)),  # 平均奖励
            'forced_exploration_results': {}  # 强制探索结果记录
        }

        # 时间步计数
        self.t = 0

        print(f"DASpecWithExploration initialized: max_proposed_length={max_proposed_length}, "
              f"exploration_rate={exploration_rate}, context_bins={context_bins}, "
              f"consecutive_threshold={consecutive_threshold}, forced_exploration_arms={forced_exploration_arms}")

    def change_prev_alphas(self,prev_alphas):
        self.daspec.change_prev_alphas(prev_alphas)
    def online_correction_factor(self,k,batch_size,actual_accepted_tokens):
        return self.daspec.online_correction_factor(k,batch_size,actual_accepted_tokens)
    def chage_table(self,k,batch_size,new):
        self.daspec.chage_table(k,batch_size,new)
    def _get_context_bin(self, context: int) -> int:
        """
        将连续上下文映射到离散的bin索引
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            bin索引
        """
        # 确保context在有效范围内
        context = max(self.context_min, min(context, self.context_max))

        # 找到对应的bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _update_ucb_values(self, context_bin: int):
        """
        更新指定上下文bin的UCB值
        
        Args:
            context_bin: 上下文bin索引
        """
        for arm in range(self.daspec.max_proposed_length + 1):
            count = self.arm_stats['counts'][arm, context_bin]
            if count > 0:
                # 计算经验均值
                mean_reward = self.arm_stats['rewards'][arm, context_bin] / count

                # 计算置信区间（使用UCB1公式）
                confidence_radius = np.sqrt(2 * np.log(self.t + 1) / count)

                # 更新UCB值
                self.arm_stats['ucb_values'][arm, context_bin] = mean_reward + confidence_radius
            else:
                # 如果arm还没有被选择过，设置UCB为无穷大
                self.arm_stats['ucb_values'][arm, context_bin] = float('inf')

    def optimize_proposed_length(self, context_length, batch_size, speculative_metrics=None):
        """
        优化推测长度，结合DASpec的预测和探索机制
        
        Args:
            context_length: 上下文长度
            batch_size: 批处理大小
            speculative_metrics: 推测指标
            
        Returns:
            最优的推测长度
        """
        self.t += 1
        context_bin = self._get_context_bin(batch_size)

        # 更新UCB值
        self._update_ucb_values(context_bin)

        # 检查是否需要强制探索
        forced_exploration_needed = self._check_forced_exploration(context_bin)

        if forced_exploration_needed:
            # 强制探索：选择强制探索候选arm中的一个
            selected_arm = self._select_forced_exploration_arm(context_bin)
            print(f"DASpecWithExploration (forced_exploration): context={batch_size}, bin={context_bin}, "
                  f"selected_arm={selected_arm}, consecutive_count={self.consecutive_selections[context_bin]['count']}")
        elif np.random.random() < self.exploration_rate:
            # 探索：优先选择未尝试的arm
            unexplored_arms = np.where(self.arm_stats['counts'][:, context_bin] == 0)[0]
            if len(unexplored_arms) > 0:
                # 随机选择一个未尝试的arm
                selected_arm = np.random.choice(unexplored_arms)
                print(f"DASpecWithExploration (explore): context={batch_size}, bin={context_bin}, "
                      f"selected_unexplored_arm={selected_arm}")
            else:
                # 如果所有arm都尝试过了，随机选择一个
                selected_arm = np.random.randint(0, self.daspec.max_proposed_length + 1)
                print(f"DASpecWithExploration (explore): context={batch_size}, bin={context_bin}, "
                      f"selected_random_arm={selected_arm}")
        else:
            # 利用：结合DASpec预测和UCB值
            # 获取DASpec的预测结果
            daspec_prediction = self.daspec.optimize_proposed_length(
                context_length, batch_size, speculative_metrics
            )

            # 获取当前上下文bin的UCB值
            ucb_values = self.arm_stats['ucb_values'][:, context_bin]

            # 检查是否应该使用强制探索的结果
            selected_arm = self._check_forced_exploration_performance(context_bin, daspec_prediction, ucb_values)

            if selected_arm is None:
                # 如果DASpec预测的arm的UCB值不是最低的，选择UCB值最高的arm
                if ucb_values[daspec_prediction] < np.max(ucb_values):
                    # 选择UCB值最高的arm
                    selected_arm = np.argmax(ucb_values)
                    print(f"DASpecWithExploration (exploit): context={batch_size}, bin={context_bin}, "
                          f"daspec_prediction={daspec_prediction}, ucb_best={selected_arm}, "
                          f"ucb_values={ucb_values}")
                else:
                    # 使用DASpec的预测
                    selected_arm = daspec_prediction
                    print(f"DASpecWithExploration (exploit): context={batch_size}, bin={context_bin}, "
                          f"using_daspec_prediction={selected_arm}")
            else:
                print(f"DASpecWithExploration (exploit): context={batch_size}, bin={context_bin}, "
                      f"using_forced_exploration_result={selected_arm}")

        # 更新连续选择计数
        self._update_consecutive_selections(context_bin, selected_arm)

        return selected_arm

    def update_with_feedback(self, context_length, batch_size, proposed_length,
                           actual_draft_time, actual_verify_time, num_accepted_tokens):
        """
        更新反馈信息
        
        Args:
            context_length: 上下文长度
            batch_size: 批处理大小
            proposed_length: 推测长度
            actual_draft_time: 实际草稿时间
            actual_verify_time: 实际验证时间
            num_accepted_tokens: 接受的token数量
        """

        # 计算奖励（吞吐量）
        total_time = actual_draft_time + actual_verify_time
        reward = num_accepted_tokens / (total_time + 1e-6)

        # 更新统计信息
        context_bin = self._get_context_bin(batch_size)
        self.arm_stats['counts'][proposed_length, context_bin] += 1
        self.arm_stats['rewards'][proposed_length, context_bin] += reward

        # 更新平均奖励
        count = self.arm_stats['counts'][proposed_length, context_bin]
        self.performance_history['avg_rewards'][proposed_length, context_bin] = (
            self.arm_stats['rewards'][proposed_length, context_bin] / count
        )

        # 如果是强制探索的arm，记录其性能
        if proposed_length in self.forced_exploration_arms:
            key = f"{context_bin}"
            if key not in self.performance_history['forced_exploration_results']:
                self.performance_history['forced_exploration_results'][key] = {}

            self.performance_history['forced_exploration_results'][key][proposed_length] = (
                self.performance_history['avg_rewards'][proposed_length, context_bin]
            )

            print(f"DASpecWithExploration: Forced exploration arm {proposed_length} performance updated: "
                  f"context={batch_size}, reward={reward:.3f}, avg_reward={self.performance_history['avg_rewards'][proposed_length, context_bin]:.3f}")

        # 衰减探索概率
        # self.exploration_rate = max(self.min_exploration_rate,
        #                            self.exploration_rate * self.exploration_decay)

        # print(f"DASpecWithExploration Update: arm={proposed_length}, context={batch_size}, "
        #       f"reward={reward:.3f}, exploration_rate={self.exploration_rate:.4f}")

    def get_exploration_stats(self, context: int) -> dict:
        """
        获取指定上下文的探索统计信息
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            dict: 包含统计信息的字典
        """
        context_bin = self._get_context_bin(context)
        stats = {}

        for arm in range(self.daspec.max_proposed_length + 1):
            count = self.arm_stats['counts'][arm, context_bin]
            if count > 0:
                mean_reward = self.arm_stats['rewards'][arm, context_bin] / count
                ucb_value = self.arm_stats['ucb_values'][arm, context_bin]
            else:
                mean_reward = 0.0
                ucb_value = float('inf')

            stats[f'arm_{arm}'] = {
                'count': int(count),
                'mean_reward': mean_reward,
                'ucb_value': ucb_value
            }

        return {
            'context': context,
            'context_bin': context_bin,
            'exploration_rate': self.exploration_rate,
            'arm_stats': stats
        }
    def reset(self):
        """重置算法状态"""
        # 重置DASpec
        self.daspec = DASpec(self.daspec.model, self.daspec.draft_model, self.daspec.max_proposed_length)

        # 重置探索统计
        self.arm_stats['counts'].fill(0)
        self.arm_stats['rewards'].fill(0)
        self.arm_stats['ucb_values'].fill(0)

        # 重置时间步和探索概率
        self.t = 0
        self.exploration_rate = 0.1  # 重置为初始值

        print("DASpecWithExploration: Reset completed")

    def save_state(self, filepath: str):
        pass

    def load_state(self, filepath: str):
        pass

    def _check_forced_exploration(self, context_bin):
        """
        检查是否需要强制探索
        
        Args:
            context_bin: 上下文bin索引
            
        Returns:
            bool: 是否需要强制探索
        """
        if context_bin not in self.consecutive_selections:
            return False

        consecutive_info = self.consecutive_selections[context_bin]
        return consecutive_info['count'] >= self.consecutive_threshold

    def _select_forced_exploration_arm(self, context_bin):
        """
        选择强制探索的arm
        
        Args:
            context_bin: 上下文bin索引
            
        Returns:
            int: 选择的arm索引
        """
        # 优先选择强制探索候选arm中未尝试或尝试较少的
        best_arm = None
        min_count = float('inf')

        for arm in self.forced_exploration_arms:
            if arm <= self.daspec.max_proposed_length:
                count = self.arm_stats['counts'][arm, context_bin]
                if count < min_count:
                    min_count = count
                    best_arm = arm

        # 如果没有找到合适的arm，随机选择一个
        if best_arm is None:
            best_arm = np.random.choice(self.forced_exploration_arms)

        return best_arm

    def _check_forced_exploration_performance(self, context_bin, daspec_prediction, ucb_values):
        """
        检查强制探索的性能，如果表现更好则优先选择
        
        Args:
            context_bin: 上下文bin索引
            daspec_prediction: DASpec预测的arm
            ucb_values: UCB值数组
            
        Returns:
            int or None: 如果有更好的强制探索结果返回arm索引，否则返回None
        """
        # 检查强制探索历史结果
        key = f"{context_bin}"
        if key in self.performance_history['forced_exploration_results']:
            forced_results = self.performance_history['forced_exploration_results'][key]

            # 获取当前预测arm的平均奖励
            current_avg_reward = self.performance_history['avg_rewards'][daspec_prediction, context_bin]

            # 检查是否有强制探索的arm表现更好
            best_forced_arm = None
            best_forced_reward = current_avg_reward

            for arm, avg_reward in forced_results.items():
                if avg_reward > best_forced_reward:
                    best_forced_reward = avg_reward
                    best_forced_arm = arm

            if best_forced_arm is not None:
                return best_forced_arm

        return None

    def _update_consecutive_selections(self, context_bin, selected_arm):
        """
        更新连续选择计数
        
        Args:
            context_bin: 上下文bin索引
            selected_arm: 选择的arm
        """
        if context_bin not in self.consecutive_selections:
            self.consecutive_selections[context_bin] = {'last_arm': selected_arm, 'count': 1}
        else:
            if self.consecutive_selections[context_bin]['last_arm'] == selected_arm:
                self.consecutive_selections[context_bin]['count'] += 1
            else:
                self.consecutive_selections[context_bin] = {'last_arm': selected_arm, 'count': 1}


class UCBSpec:
    def __init__(self, num_arms, confidence_param=0.5, max_spec_length=4):
        """
        Initialize the UCBSPEC algorithm.
        
        Args:
            num_arms (int): Number of hyperparameter configurations (arms)
            confidence_param (float): Confidence parameter (δ in the paper)
            max_spec_length (int): Maximum speculation length (L in the paper)
        """
        self.K = num_arms
        self.delta = confidence_param
        self.L = max_spec_length

        # Initialize statistics for each arm
        self.arm_counts = np.zeros(num_arms)  # n_i,t
        self.arm_rewards = np.zeros(num_arms)  # Sum of rewards for each arm
        self.ucb_values = np.zeros(num_arms)   # UCB values for each arm
        self.round_robin = False
        # Track the total number of rounds
        self.t = 0

    def select_arm(self, context, current_qps):
        """
        Select an arm according to the UCBSPEC algorithm.
        
        Returns:
            int: Index of the selected arm
        """


        if self.round_robin:
            # Round-robin for the first K rounds
            return  self.t % self.K
        elif self.t < self.K:
            return self.t
        else:
            # Select arm with highest UCB value
            # print("self.ucb_values",self.ucb_values,self.arm_counts,self.arm_rewards)
            return np.argmax(self.ucb_values)

    def update(self, chosen_arm, context, generated_tokens: int, elasped_time: float):
        """
        Update the statistics after observing a reward.
        
        Args:
            chosen_arm (int): Index of the arm that was pulled
            reward (float): Observed reward (length of accepted tokens)
        """
        self.t += 1
        reward = generated_tokens / (elasped_time + 1e-6)
        # Update counts and rewards for the chosen arm
        self.arm_counts[chosen_arm] += 1
        # self.arm_rewards[chosen_arm] += reward
        self.arm_rewards[chosen_arm] += reward / context

        # Update UCB values for all arms
        for i in range(self.K):
            if self.arm_counts[i] > 0:
                # Calculate empirical mean
                mu_hat = self.arm_rewards[i] / self.arm_counts[i]
                # Calculate confidence radius
                n = self.arm_counts[i]
                log_term = math.log((self.K * (self.t**2) * math.sqrt(1 + n)) / self.delta)
                cr = (self.L / 2) * math.sqrt((1 + n) / (n**2) * (1 + 2 * log_term))
                self.ucb_values[i] = mu_hat + cr
            else:
                # If arm hasn't been pulled yet, set UCB to infinity
                self.ucb_values[i] = float('inf')

    def reset(self):
        """Reset the algorithm's state."""
        self.arm_counts = np.zeros(self.K)
        self.arm_rewards = np.zeros(self.K)
        self.ucb_values = np.zeros(self.K)
        self.t = 0

    def save_state(self, filepath):
        """Save the internal state to a file."""
        state = {
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'ucb_values': self.ucb_values,
            't': self.t,
            'K': self.K,
            'delta': self.delta,
            'L': self.L,
        }
        print("save_state",state,filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath):
        """Load the internal state from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print("load_state",filepath,state)
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.t = state['t']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']
class UCBSpec2:
    def __init__(self,
                 K: int,                 # 候选γ值数量
                 max_spec_length: int,                 # 最大推测长度
                 delta: float = 0.1,    # 置信参数
                ):
        # === 状态空间 ===
        self.arm_stats = {
            'n': np.zeros(K),            # 选择次数
            'throughput_hat': np.zeros(K), # 平均吞吐量估计
            'cr': np.zeros(K)            # 置信半径
        }
        self.gamma_values = np.linspace(1, max_spec_length, K)  # γ的候选值（论文6.2节）
        self.last_update_time = time.time()
        self.round_robin = True
        self.t = 0
        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.delta = delta

    def select_arm(self,context) -> int:
        """ 选择γ值索引（改进UCB公式） """
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K
        elif self.t < self.K:
            return self.t

        # 计算吞吐量UCB（论文6.2节思想）
        ucb = self.arm_stats['throughput_hat'] + self._calc_confidence_radius()
        return np.argmax(ucb)

    def _calc_confidence_radius(self) -> np.ndarray:
        """ 基于吞吐量方差调整置信半径 """
        t = self.arm_stats['n'].sum()
        return np.sqrt(
            2 * np.log(t**2 / self.delta) / (self.arm_stats['n'] + 1e-6)
        )

    def update(self, arm_idx: int, context, generated_tokens: int, elasped_time: float):
        """ 
        更新吞吐量统计量
        :param generated_tokens: 本轮生成的token总数（含拒绝token）
        """
        throughput = generated_tokens / (elasped_time + 1e-6)  # tokens/sec
        self.t += 1
        # 更新统计量（指数加权平均）
        n_prev = self.arm_stats['n'][arm_idx]
        self.arm_stats['throughput_hat'][arm_idx] = (
            n_prev * self.arm_stats['throughput_hat'][arm_idx] + throughput
        ) / (n_prev + 1)
        self.arm_stats['n'][arm_idx] += 1
    def save_state(self, filepath):
        """Save the internal state to a file."""
        state = {
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'ucb_values': self.ucb_values,
            't': self.t,
            'K': self.K,
            'delta': self.delta,
            'L': self.L,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath):
        """Load the internal state from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.t = state['t']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']

    def load_state(self, filepath):
        """Load the internal state from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.t = state['t']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']

class ThompsonSamplingSpec:
    def __init__(self,
                 K: int,                 # 候选γ值数量
                 max_spec_length: int,   # 最大推测长度
                 context_bins: int = 10, # 上下文分箱数量
                 mu_prior: float = 0.0,  # 高斯分布先验均值
                 sigma_prior: float = 1.0, # 高斯分布先验标准差
                 sigma_noise: float = 1.0, # 观测噪声标准差
                ):
        """
        初始化Thompson Sampling算法，基于上下文（请求量）决定投机长度
        
        使用高斯-高斯共轭先验模型：
        - 吞吐量服从高斯分布
        - 每个arm-context组合维护高斯后验分布
        - 通过采样后验分布来选择最优arm
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_bins: 上下文分箱数量，用于离散化请求量
            mu_prior: 高斯分布先验均值
            sigma_prior: 高斯分布先验标准差
            sigma_noise: 观测噪声标准差
        """
        # === 状态空间 ===
        self.arm_stats = {
            'n': np.zeros((K, context_bins)),           # 每个arm-context组合的选择次数
            'sum_rewards': np.zeros((K, context_bins)), # 总奖励（吞吐量之和）
            'sum_squared_rewards': np.zeros((K, context_bins)), # 奖励平方和
        }

        # === 高斯后验分布参数 ===
        # 后验均值: mu_post = (mu_prior/sigma_prior^2 + sum_rewards/sigma_noise^2) / (1/sigma_prior^2 + n/sigma_noise^2)
        # 后验方差: sigma_post^2 = 1 / (1/sigma_prior^2 + n/sigma_noise^2)
        self.posterior_params = {
            'mu': np.full((K, context_bins), mu_prior),     # 后验均值
            'sigma_squared': np.full((K, context_bins), sigma_prior**2)  # 后验方差
        }

        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.context_bins = context_bins
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.sigma_noise = sigma_noise

        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0
        self.context_bounds = None  # 上下文边界，将在第一次使用时初始化

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)

        print(f"ThompsonSamplingSpec initialized: K={K}, L={max_spec_length}, context_bins={context_bins}")

    def _get_context_bin(self, context: int) -> int:
        """
        将连续上下文（请求量）映射到离散的bin索引
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            bin索引
        """
        if self.context_bounds is None:
            # 初始化上下文边界（假设请求量范围是1-100）
            self.context_bounds = np.linspace(1, 100, self.context_bins + 1)

        # 确保context在有效范围内
        context = max(1, min(context, 100))

        # 找到对应的bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _update_posterior(self, arm_idx: int, context_bin: int, reward: float):
        """
        更新高斯后验分布参数
        
        Args:
            arm_idx: arm索引
            context_bin: 上下文bin索引
            reward: 观测到的奖励（吞吐量）
        """
        n = self.arm_stats['n'][arm_idx, context_bin]

        # 更新统计量
        self.arm_stats['n'][arm_idx, context_bin] += 1
        self.arm_stats['sum_rewards'][arm_idx, context_bin] += reward
        self.arm_stats['sum_squared_rewards'][arm_idx, context_bin] += reward**2

        # 更新后验分布参数
        # 高斯-高斯共轭先验的更新公式
        mu_prev = self.posterior_params['mu'][arm_idx, context_bin]
        sigma_squared_prev = self.posterior_params['sigma_squared'][arm_idx, context_bin]

        # 新的后验参数
        precision_prior = 1.0 / (self.sigma_prior**2)
        precision_likelihood = 1.0 / (self.sigma_noise**2)
        precision_post = precision_prior + (n + 1) * precision_likelihood

        self.posterior_params['sigma_squared'][arm_idx, context_bin] = 1.0 / precision_post

        # 后验均值更新
        mu_post = (precision_prior * self.mu_prior +
                   precision_likelihood * self.arm_stats['sum_rewards'][arm_idx, context_bin]) / precision_post
        self.posterior_params['mu'][arm_idx, context_bin] = mu_post

    def select_arm(self, context: int) -> int:
        """
        使用Thompson Sampling选择投机长度
        
        Args:
            context: 当前请求量（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K
        elif self.t < self.K:
            return self.t
        context_bin = self._get_context_bin(context)

        # 为每个arm采样高斯后验分布
        sampled_values = np.zeros(self.K)
        for k in range(self.K):
            mu = self.posterior_params['mu'][k, context_bin]
            sigma = np.sqrt(self.posterior_params['sigma_squared'][k, context_bin])

            # 从高斯分布采样
            if sigma > 0:
                sampled_values[k] = np.random.normal(mu, sigma)
            else:
                sampled_values[k] = mu

        # 选择采样值最大的arm
        selected_arm = np.argmax(sampled_values)

        print(f"Thompson Sampling: context={context}, bin={context_bin}, "
              f"sampled_values={sampled_values}, selected_arm={selected_arm}")

        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新Thompson Sampling统计量
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求量（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        self.t += 1

        # 计算奖励（吞吐量）
        reward = generated_tokens / (elapsed_time + 1e-6)

        context_bin = self._get_context_bin(context)

        # 更新后验分布
        self._update_posterior(arm_idx, context_bin, reward)

        # print(f"Thompson Update: arm={arm_idx}, context={context}, bin={context_bin}, "
        #       f"reward={reward:.2f}, mu={self.posterior_params['mu'][arm_idx, context_bin]:.2f}, "
        #       f"sigma={np.sqrt(self.posterior_params['sigma_squared'][arm_idx, context_bin]):.2f}")

    def get_expected_rewards(self, context: int) -> np.ndarray:
        """
        获取每个arm在当前上下文下的期望奖励
        
        Args:
            context: 请求量
            
        Returns:
            每个arm的期望奖励数组
        """
        context_bin = self._get_context_bin(context)
        expected_rewards = np.zeros(self.K)

        for k in range(self.K):
            expected_rewards[k] = self.posterior_params['mu'][k, context_bin]

        return expected_rewards

    def get_uncertainty(self, context: int) -> np.ndarray:
        """
        获取每个arm在当前上下文下的不确定性（后验标准差）
        
        Args:
            context: 请求量
            
        Returns:
            每个arm的不确定性数组
        """
        context_bin = self._get_context_bin(context)
        uncertainty = np.zeros(self.K)

        for k in range(self.K):
            uncertainty[k] = np.sqrt(self.posterior_params['sigma_squared'][k, context_bin])

        return uncertainty

    def reset(self):
        """重置算法状态"""
        self.arm_stats['n'].fill(0)
        self.arm_stats['sum_rewards'].fill(0)
        self.arm_stats['sum_squared_rewards'].fill(0)
        self.posterior_params['mu'].fill(self.mu_prior)
        self.posterior_params['sigma_squared'].fill(self.sigma_prior**2)
        self.t = 0
        self.round_robin = True

    def save_state(self, filepath: str):
        """保存内部状态到文件"""
        state = {
            'arm_stats': self.arm_stats,
            'posterior_params': self.posterior_params,
            't': self.t,
            'K': self.K,
            'L': self.L,
            'context_bins': self.context_bins,
            'mu_prior': self.mu_prior,
            'sigma_prior': self.sigma_prior,
            'sigma_noise': self.sigma_noise,
            'context_bounds': self.context_bounds,
            'spec_lengths': self.spec_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """从文件加载内部状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.arm_stats = state['arm_stats']
        self.posterior_params = state['posterior_params']
        self.t = state['t']
        self.K = state['K']
        self.L = state['L']
        self.context_bins = state['context_bins']
        self.mu_prior = state['mu_prior']
        self.sigma_prior = state['sigma_prior']
        self.sigma_noise = state['sigma_noise']
        self.context_bounds = state['context_bounds']
        self.spec_lengths = state['spec_lengths']

class LinUCBSpec:
    def __init__(self,
                 K: int,                 # 候选投机长度数量
                 max_spec_length: int,   # 最大推测长度
                 context_dim: int = 1,   # 上下文维度（这里固定为1，表示请求数）
                 alpha: float = 1.0,     # 探索参数
                 lambda_reg: float = 1.0, # 正则化参数
                ):
        """
        初始化LinUCB算法，基于上下文（请求数）决定投机长度
        
        使用线性模型：
        - 每个arm维护一个线性模型参数
        - 上下文是请求数（batch size）
        - 通过UCB公式选择最优arm
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_dim: 上下文维度，固定为1（请求数）
            alpha: 探索参数，控制探索-利用平衡
            lambda_reg: 正则化参数，防止过拟合
        """
        # === 状态空间 ===
        self.arm_params = {
            'A': [np.eye(context_dim) * lambda_reg for _ in range(K)],  # 每个arm的A矩阵
            'b': [np.zeros(context_dim) for _ in range(K)],             # 每个arm的b向量
            'theta': [np.zeros(context_dim) for _ in range(K)],         # 每个arm的线性参数
        }

        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)
        self.has_explore_context = [[0 for _ in range(K)] for _ in range(200)]
        print(f"LinUCBSpec initialized: K={K}, L={max_spec_length}, context_dim={context_dim}, alpha={alpha}")

    def _update_arm_params(self, arm_idx: int, context: np.ndarray, reward: float):
        """
        更新指定arm的线性模型参数
        
        Args:
            arm_idx: arm索引
            context: 上下文向量（请求数）
            reward: 观测到的奖励（吞吐量）
        """
        # 确保context是列向量
        context = context.reshape(-1, 1)
        # 更新A矩阵和b向量
        self.arm_params['A'][arm_idx] += context @ context.T
        self.arm_params['b'][arm_idx] += reward * context.flatten()

        # 更新theta参数（线性回归解）
        try:
            A_inv = np.linalg.inv(self.arm_params['A'][arm_idx])
            self.arm_params['theta'][arm_idx] = A_inv @ self.arm_params['b'][arm_idx]
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            A_inv = np.linalg.pinv(self.arm_params['A'][arm_idx])
            self.arm_params['theta'][arm_idx] = A_inv @ self.arm_params['b'][arm_idx]

    def select_arm(self, context: int) -> int:
        """
        使用LinUCB选择投机长度
        
        Args:
            context: 当前请求数（batch size）
            
        Returns:
            选择的投机长度索引
        """
        # if self.round_robin:  # 初始轮次：Round-Robin
        #     return self.t % self.K
        # elif self.t < self.K:
        #     return self.t
        # for arm_idx in range(self.K):
        #     if self.has_explore_context[context][arm_idx] == 0:
        #         return arm_idx
        # 将context转换为向量
        context_vec = np.array([context])

        # 计算每个arm的UCB值
        ucb_values = np.zeros(self.K)
        for k in range(self.K):
            # 预测奖励
            predicted_reward = self.arm_params['theta'][k] @ context_vec

            # 计算置信区间
            try:
                A_inv = np.linalg.inv(self.arm_params['A'][k])
                context_vec_col = context_vec.reshape(-1, 1)
                confidence = self.alpha * np.sqrt(context_vec_col.T @ A_inv @ context_vec_col)
                ucb_values[k] = predicted_reward + confidence
            except np.linalg.LinAlgError:
                # 如果矩阵不可逆，使用伪逆
                A_inv = np.linalg.pinv(self.arm_params['A'][k])
                context_vec_col = context_vec.reshape(-1, 1)
                confidence = self.alpha * np.sqrt(context_vec_col.T @ A_inv @ context_vec_col)
                ucb_values[k] = predicted_reward + confidence

        # 选择UCB值最大的arm
        selected_arm = np.argmax(ucb_values)

        print(f"LinUCB: context={context}, ucb_values={ucb_values}, selected_arm={selected_arm}")

        return 2 #selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新LinUCB统计量
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求数（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        # self.t += 1

        # # 计算奖励（吞吐量）
        # reward = generated_tokens / (elapsed_time + 1e-6)
        # self.has_explore_context[context][arm_idx] += 1
        # # 将context转换为向量
        # context_vec = np.array([context])

        # # 更新线性模型参数
        # self._update_arm_params(arm_idx, context_vec, reward)

        # print(f"LinUCB Update: arm={arm_idx}, context={context}, "
        #       f"reward={reward:.2f}, theta={self.arm_params['theta'][arm_idx]}")

    def reset(self):
        """重置算法状态"""
        for k in range(self.K):
            self.arm_params['A'][k] = np.eye(self.context_dim) * self.lambda_reg
            self.arm_params['b'][k] = np.zeros(self.context_dim)
            self.arm_params['theta'][k] = np.zeros(self.context_dim)
        self.t = 0
        self.round_robin = True

    def save_state(self, filepath: str):
        """保存内部状态到文件"""
        state = {
            'arm_params': self.arm_params,
            't': self.t,
            'K': self.K,
            'L': self.L,
            'context_dim': self.context_dim,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'spec_lengths': self.spec_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """从文件加载内部状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.arm_params = state['arm_params']
        self.t = state['t']
        self.K = state['K']
        self.L = state['L']
        self.context_dim = state['context_dim']
        self.alpha = state['alpha']
        self.lambda_reg = state['lambda_reg']
        self.spec_lengths = state['spec_lengths']



class ContextualLinUCB:
    def __init__(self,
                 K: int,                 # 候选投机长度数量
                 max_spec_length: int,   # 最大推测长度
                 context_dim: int = 1,   # 上下文维度（这里固定为1，表示请求数）
                 alpha: float = 1.0,     # 探索参数
                 lambda_reg: float = 1.0, # 正则化参数
                ):
        """
        初始化ContextualLinUCB算法，基于上下文（请求数）决定投机长度
        
        使用线性模型：
        - 每个arm维护一个线性模型参数
        - 上下文是请求数（batch size）
        - 通过UCB公式选择最优arm
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_dim: 上下文维度，固定为1（请求数）
            alpha: 探索参数，控制探索-利用平衡
            lambda_reg: 正则化参数，防止过拟合
        """
        # === 状态空间 ===
        self.n_arms = K
        self.d = context_dim
        self.L = max_spec_length  # 最大投机长度
        self.A = [np.eye(self.d) * lambda_reg for _ in range(K)]  # 臂的精度矩阵
        self.b = [np.zeros(self.d) for _ in range(K)]
        self.θ = [np.zeros(self.d) for _ in range(K)]

        # === 超参数 ===
        self.K = K
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)

        print(f"ContextualLinUCB initialized: K={K}, L={max_spec_length}, context_dim={context_dim}, alpha={alpha}")

    def _update_arm_params(self, arm_idx: int, context: np.ndarray, reward: float):
        """
        更新指定arm的线性模型参数
        
        Args:
            arm_idx: arm索引
            context: 上下文向量（请求数）
            reward: 观测到的奖励（吞吐量）
        """
        # BanditSPEC式增量更新（算法4思想）
        self.A[arm_idx] += np.outer(context, context)
        self.b[arm_idx] += reward * context
        self.θ[arm_idx] = np.linalg.solve(self.A[arm_idx], self.b[arm_idx])

    def select_arm(self, context: int) -> int:
        """
        使用ContextualLinUCB选择投机长度
        
        Args:
            context: 当前请求数（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K
        elif self.t < self.K:
            return self.t
        # 将context转换为向量
        context_vec = np.array([context])

        # 带请求量上下文的决策
        ucb_values = []
        for a in range(self.n_arms):
            try:
                A_inv = np.linalg.inv(self.A[a])
                std = np.sqrt(context_vec.T @ A_inv @ context_vec)
                # 论文式(13)的改进置信半径
                cr = self.L/2 * std * np.sqrt(1 + 2*np.log(len(self.A[a])))
                ucb = context_vec @ self.θ[a] + cr * (1 + context)
                ucb_values.append(ucb)
            except np.linalg.LinAlgError:
                # 如果矩阵不可逆，使用伪逆
                A_inv = np.linalg.pinv(self.A[a])
                std = np.sqrt(context_vec.T @ A_inv @ context_vec)
                cr = self.L/2 * std * np.sqrt(1 + 2*np.log(len(self.A[a])))
                ucb = context_vec @ self.θ[a] + cr * (1 + context)
                ucb_values.append(ucb)

        # 选择UCB值最大的arm
        selected_arm = np.argmax(ucb_values)

        print(f"ContextualLinUCB: context={context}, ucb_values={ucb_values}, selected_arm={selected_arm}")

        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新ContextualLinUCB统计量
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求数（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        self.t += 1

        # 计算奖励（吞吐量）
        reward = generated_tokens / (elapsed_time + 1e-6)

        # 将context转换为向量
        context_vec = np.array([context])

        # 更新线性模型参数
        self._update_arm_params(arm_idx, context_vec, reward)

        print(f"ContextualLinUCB Update: arm={arm_idx}, context={context}, "
              f"reward={reward:.2f}, theta={self.θ[arm_idx]}")

    def get_expected_rewards(self, context: int) -> np.ndarray:
        """
        获取每个arm在当前上下文下的期望奖励
        
        Args:
            context: 请求数
            
        Returns:
            每个arm的期望奖励数组
        """
        context_vec = np.array([context])
        expected_rewards = np.zeros(self.K)

        for k in range(self.K):
            expected_rewards[k] = context_vec @ self.θ[k]

        return expected_rewards

    def get_uncertainty(self, context: int) -> np.ndarray:
        """
        获取每个arm在当前上下文下的不确定性（置信区间）
        
        Args:
            context: 请求数
            
        Returns:
            每个arm的不确定性数组
        """
        context_vec = np.array([context])
        uncertainty = np.zeros(self.K)

        for k in range(self.K):
            try:
                A_inv = np.linalg.inv(self.A[k])
                std = np.sqrt(context_vec.T @ A_inv @ context_vec)
                uncertainty[k] = self.L/2 * std * np.sqrt(1 + 2*np.log(len(self.A[k])))
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self.A[k])
                std = np.sqrt(context_vec.T @ A_inv @ context_vec)
                uncertainty[k] = self.L/2 * std * np.sqrt(1 + 2*np.log(len(self.A[k])))

        return uncertainty

    def reset(self):
        """重置算法状态"""
        for k in range(self.K):
            self.A[k] = np.eye(self.context_dim) * self.lambda_reg
            self.b[k] = np.zeros(self.context_dim)
            self.θ[k] = np.zeros(self.context_dim)
        self.t = 0
        self.round_robin = True

    def save_state(self, filepath: str):
        """保存内部状态到文件"""
        state = {
            'A': self.A,
            'b': self.b,
            'θ': self.θ,
            't': self.t,
            'K': self.K,
            'L': self.L,
            'context_dim': self.context_dim,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'spec_lengths': self.spec_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """从文件加载内部状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.A = state['A']
        self.b = state['b']
        self.θ = state['θ']
        self.t = state['t']
        self.K = state['K']
        self.L = state['L']
        self.context_dim = state['context_dim']
        self.alpha = state['alpha']
        self.lambda_reg = state['lambda_reg']
        self.spec_lengths = state['spec_lengths']

class LinearThompsonSamplingSpec:
    def __init__(self,
                 K: int,                 # 候选投机长度数量
                 max_spec_length: int,   # 最大推测长度
                 context_dim: int = 1,   # 上下文维度（这里固定为1，表示请求数）
                 mu_prior: float = 0.0,  # 高斯分布先验均值
                 sigma_prior: float = 1.0, # 高斯分布先验标准差
                 sigma_noise: float = 1.0, # 观测噪声标准差
                ):
        """
        初始化Linear Thompson Sampling算法，基于上下文（请求数）决定投机长度
        
        使用线性高斯模型：
        - 每个arm维护一个线性高斯后验分布
        - 上下文是请求数（batch size）
        - 通过采样后验分布来选择最优arm
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_dim: 上下文维度，固定为1（请求数）
            mu_prior: 高斯分布先验均值
            sigma_prior: 高斯分布先验标准差
            sigma_noise: 观测噪声标准差
        """
        # === 状态空间 ===
        self.arm_stats = {
            'n': np.zeros(K),                    # 每个arm的选择次数
            'sum_rewards': np.zeros(K),          # 总奖励（吞吐量之和）
            'sum_contexts': np.zeros((K, context_dim)),  # 上下文向量之和
            'sum_context_rewards': np.zeros((K, context_dim)),  # 上下文与奖励的乘积之和
            'sum_context_squared': np.zeros((K, context_dim, context_dim)),  # 上下文平方和
        }

        # === 线性高斯后验分布参数 ===
        # 后验均值: mu_post = (mu_prior/sigma_prior^2 + sum_context_rewards/sigma_noise^2) / (1/sigma_prior^2 + sum_context_squared/sigma_noise^2)
        # 后验方差: sigma_post^2 = 1 / (1/sigma_prior^2 + sum_context_squared/sigma_noise^2)
        self.posterior_params = {
            'mu': np.full((K, context_dim), mu_prior),     # 后验均值
            'sigma_squared': np.full((K, context_dim, context_dim), sigma_prior**2)  # 后验协方差矩阵
        }

        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.context_dim = context_dim
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.sigma_noise = sigma_noise

        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)

        print(f"LinearThompsonSamplingSpec initialized: K={K}, L={max_spec_length}, context_dim={context_dim}")

    def _update_posterior(self, arm_idx: int, context: np.ndarray, reward: float):
        """
        更新指定arm的线性高斯后验分布参数
        
        Args:
            arm_idx: arm索引
            context: 上下文向量（请求数）
            reward: 观测到的奖励（吞吐量）
        """
        # 确保context是列向量
        context = context.reshape(-1, 1)

        # 更新统计量
        self.arm_stats['n'][arm_idx] += 1
        self.arm_stats['sum_rewards'][arm_idx] += reward
        self.arm_stats['sum_contexts'][arm_idx] += context.flatten()
        self.arm_stats['sum_context_rewards'][arm_idx] += reward * context.flatten()
        self.arm_stats['sum_context_squared'][arm_idx] += context @ context.T

        # 更新后验分布参数
        # 高斯-高斯共轭先验的更新公式
        mu_prev = self.posterior_params['mu'][arm_idx]
        sigma_squared_prev = self.posterior_params['sigma_squared'][arm_idx]

        # 新的后验参数
        precision_prior = 1.0 / (self.sigma_prior**2)
        precision_likelihood = 1.0 / (self.sigma_noise**2)

        # 更新协方差矩阵
        context_squared = context @ context.T
        precision_post = precision_prior + precision_likelihood * context_squared

        try:
            self.posterior_params['sigma_squared'][arm_idx] = 1.0 / precision_post
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            self.posterior_params['sigma_squared'][arm_idx] = np.linalg.pinv(precision_post)

        # 更新均值
        mu_post = (precision_prior * self.mu_prior +
                   precision_likelihood * self.arm_stats['sum_context_rewards'][arm_idx]) / precision_post
        self.posterior_params['mu'][arm_idx] = mu_post

    def select_arm(self, context: int) -> int:
        """
        使用Linear Thompson Sampling选择投机长度
        
        Args:
            context: 当前请求数（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K
        elif self.t < self.K:
            return self.t
        # 将context转换为向量
        context_vec = np.array([context])

        # 为每个arm采样线性后验分布
        sampled_values = np.zeros(self.K)
        for k in range(self.K):
            mu = self.posterior_params['mu'][k]
            sigma_squared = self.posterior_params['sigma_squared'][k]

            # 计算线性预测的均值和方差
            predicted_mean = mu @ context_vec
            predicted_variance = context_vec.T @ sigma_squared @ context_vec

            # 从高斯分布采样
            if predicted_variance > 0:
                sampled_values[k] = np.random.normal(predicted_mean, np.sqrt(predicted_variance))
            else:
                sampled_values[k] = predicted_mean

        # 选择采样值最大的arm
        selected_arm = np.argmax(sampled_values)

        print(f"Linear Thompson Sampling: context={context}, "
              f"sampled_values={sampled_values}, selected_arm={selected_arm}")

        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新Linear Thompson Sampling统计量
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求数（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        self.t += 1

        # 计算奖励（吞吐量）
        reward = generated_tokens / (elapsed_time + 1e-6)

        # 将context转换为向量
        context_vec = np.array([context])

        # 更新后验分布
        self._update_posterior(arm_idx, context_vec, reward)

        # print(f"Linear Thompson Update: arm={arm_idx}, context={context}, "
        #       f"reward={reward:.2f}, mu={self.posterior_params['mu'][arm_idx]}")

    def get_expected_rewards(self, context: int) -> np.ndarray:
        """
        获取每个arm在当前上下文下的期望奖励
        
        Args:
            context: 请求数
            
        Returns:
            每个arm的期望奖励数组
        """
        context_vec = np.array([context])
        expected_rewards = np.zeros(self.K)

        for k in range(self.K):
            mu = self.posterior_params['mu'][k]
            expected_rewards[k] = mu @ context_vec

        return expected_rewards

    def get_uncertainty(self, context: int) -> np.ndarray:
        """
        获取每个arm在当前上下文下的不确定性（预测方差）
        
        Args:
            context: 请求数
            
        Returns:
            每个arm的不确定性数组
        """
        context_vec = np.array([context])
        uncertainty = np.zeros(self.K)

        for k in range(self.K):
            sigma_squared = self.posterior_params['sigma_squared'][k]
            uncertainty[k] = context_vec.T @ sigma_squared @ context_vec

        return uncertainty

    def reset(self):
        """重置算法状态"""
        self.arm_stats['n'].fill(0)
        self.arm_stats['sum_rewards'].fill(0)
        self.arm_stats['sum_contexts'].fill(0)
        self.arm_stats['sum_context_rewards'].fill(0)
        self.arm_stats['sum_context_squared'].fill(0)

        for k in range(self.K):
            self.posterior_params['mu'][k].fill(self.mu_prior)
            self.posterior_params['sigma_squared'][k].fill(self.sigma_prior**2)

        self.t = 0
        self.round_robin = True

    def save_state(self, filepath: str):
        """保存内部状态到文件"""
        state = {
            'arm_stats': self.arm_stats,
            'posterior_params': self.posterior_params,
            't': self.t,
            'K': self.K,
            'L': self.L,
            'context_dim': self.context_dim,
            'mu_prior': self.mu_prior,
            'sigma_prior': self.sigma_prior,
            'sigma_noise': self.sigma_noise,
            'spec_lengths': self.spec_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """从文件加载内部状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.arm_stats = state['arm_stats']
        self.posterior_params = state['posterior_params']
        self.t = state['t']
        self.K = state['K']
        self.L = state['L']
        self.context_dim = state['context_dim']
        self.mu_prior = state['mu_prior']
        self.sigma_prior = state['sigma_prior']
        self.sigma_noise = state['sigma_noise']
        self.spec_lengths = state['spec_lengths']

class EpsilonGreedySpec:
    def __init__(self,
                 K: int,                 # 候选投机长度数量
                 max_spec_length: int,   # 最大推测长度
                 context_bins: int = 200, # 上下文分箱数量
                 epsilon: float = 0.1,   # 探索概率
                ):
        """
        初始化Epsilon-Greedy算法，基于上下文（请求量）决定投机长度
        
        使用epsilon-greedy策略：
        - 以概率epsilon随机探索
        - 以概率1-epsilon选择当前最优arm
        - 每个arm-context组合维护独立的统计量
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_bins: 上下文分箱数量，用于离散化请求量
            epsilon: 探索概率，控制探索-利用平衡
        """
        # === 状态空间 ===
        self.arm_stats = {
            'n': np.zeros((K, context_bins)),           # 每个arm-context组合的选择次数
            'sum_rewards': np.zeros((K, context_bins)), # 总奖励（吞吐量之和）
            'avg_rewards': np.zeros((K, context_bins)), # 平均奖励
        }

        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.context_bins = context_bins
        self.epsilon = epsilon
        self.has_explore_context = [[0 for _ in range(K)] for _ in range(200)]
        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0
        self.context_bounds = None  # 上下文边界，将在第一次使用时初始化

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)

        print(f"EpsilonGreedySpec initialized: K={K}, L={max_spec_length}, context_bins={context_bins}, epsilon={epsilon}")

    def _get_context_bin(self, context: int) -> int:
        """
        将连续上下文（请求量）映射到离散的bin索引
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            bin索引
        """
        if self.context_bounds is None:
            # 初始化上下文边界（假设请求量范围是1-100）
            self.context_bounds = np.linspace(1, 100, self.context_bins + 1)

        # 确保context在有效范围内
        context = max(1, min(context, 100))

        # 找到对应的bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _update_arm_stats(self, arm_idx: int, context_bin: int, reward: float):
        """
        更新指定arm-context组合的统计量
        
        Args:
            arm_idx: arm索引
            context_bin: 上下文bin索引
            reward: 观测到的奖励（吞吐量）
        """
        # 更新统计量
        self.arm_stats['n'][arm_idx, context_bin] += 1
        self.arm_stats['sum_rewards'][arm_idx, context_bin] += reward

        # 更新平均奖励
        n = self.arm_stats['n'][arm_idx, context_bin]
        self.arm_stats['avg_rewards'][arm_idx, context_bin] = (
            self.arm_stats['sum_rewards'][arm_idx, context_bin] / n
        )

    def select_arm(self, context: int) -> int:
        """
        使用Epsilon-Greedy选择投机长度
        
        Args:
            context: 当前请求量（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K
        # for arm_idx in range(self.K):
        #     if self.has_explore_context[context][arm_idx] == 0:
        #         return arm_idx
        context_bin = self._get_context_bin(context)

        # # Epsilon-greedy策略
        if np.random.random() < self.epsilon:
            # 探索：随机选择arm
            selected_arm = np.random.randint(0, self.K)
            print(f"Epsilon-Greedy (explore): context={context}, bin={context_bin}, selected_arm={selected_arm}")
        else:
            # 利用：选择当前最优arm
            avg_rewards = self.arm_stats['avg_rewards'][:, context_bin]

            # 处理未探索的arm（平均奖励为0）
            unexplored_mask = self.arm_stats['n'][:, context_bin] == 0
            if unexplored_mask.any():
                # 如果有未探索的arm，优先选择
                unexplored_arms = np.where(unexplored_mask)[0]
                selected_arm = np.random.choice(unexplored_arms)
            else:
                # 选择平均奖励最高的arm
                selected_arm = np.argmax(avg_rewards)

        # print(f"Epsilon-Greedy (exploit): context={context}, bin={context_bin}, "
        #         f"avg_rewards={avg_rewards}, selected_arm={selected_arm}")

        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新Epsilon-Greedy统计量
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求量（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        self.t += 1
        # self.has_explore_context[context][arm_idx] += 1
        # 计算奖励（吞吐量）
        reward = generated_tokens / (elapsed_time + 1e-6)

        context_bin = self._get_context_bin(context)

        # 更新统计量
        self._update_arm_stats(arm_idx, context_bin, reward)

        # print(f"Epsilon-Greedy Update: arm={arm_idx}, context={context}, bin={context_bin}, "
        #       f"reward={reward:.2f}, avg_reward={self.arm_stats['avg_rewards'][arm_idx, context_bin]:.2f}")

    def get_expected_rewards(self, context: int) -> np.ndarray:
        """
        获取每个arm在当前上下文下的期望奖励
        
        Args:
            context: 请求量
            
        Returns:
            每个arm的期望奖励数组
        """
        context_bin = self._get_context_bin(context)
        expected_rewards = np.zeros(self.K)

        for k in range(self.K):
            expected_rewards[k] = self.arm_stats['avg_rewards'][k, context_bin]

        return expected_rewards

    def get_uncertainty(self, context: int) -> np.ndarray:
        """
        获取每个arm在当前上下文下的不确定性（基于选择次数）
        
        Args:
            context: 请求量
            
        Returns:
            每个arm的不确定性数组
        """
        context_bin = self._get_context_bin(context)
        uncertainty = np.zeros(self.K)

        for k in range(self.K):
            n = self.arm_stats['n'][k, context_bin]
            # 不确定性基于选择次数的倒数，未选择的arm具有最高不确定性
            uncertainty[k] = 1.0 / (n + 1e-6)

        return uncertainty

    def reset(self):
        """重置算法状态"""
        self.arm_stats['n'].fill(0)
        self.arm_stats['sum_rewards'].fill(0)
        self.arm_stats['avg_rewards'].fill(0)
        self.t = 0
        self.round_robin = True

    def save_state(self, filepath: str):
        """保存内部状态到文件"""
        state = {
            'arm_stats': self.arm_stats,
            't': self.t,
            'K': self.K,
            'L': self.L,
            'context_bins': self.context_bins,
            'epsilon': self.epsilon,
            'context_bounds': self.context_bounds,
            'spec_lengths': self.spec_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """从文件加载内部状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.arm_stats = state['arm_stats']
        self.t = state['t']
        self.K = state['K']
        self.L = state['L']
        self.context_bins = state['context_bins']
        self.epsilon = state['epsilon']
        self.context_bounds = state['context_bounds']
        self.spec_lengths = state['spec_lengths']


class ADABinGreedy:
    def __init__(self, K: int, max_spec_length: int, num_log_bins: int = 12, ttft_diff_dict_path: Optional[str] = None):
        self.K = K
        self.num_log_bins = num_log_bins
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)
        self.have_disabled = False
        # === 状态存储 ===
        self.arm_stats = {
            'n': np.zeros((K, num_log_bins)),
            'avg_rewards': np.zeros((K, num_log_bins)),
        }

        # === 加载 TTFT 差值字典 ===
        self.ttft_diff_dict = {}
        if ttft_diff_dict_path is None:
            ttft_diff_dict_path = '/root/autodl-tmp/vllm_speculative/ttft_diff_dict.json'
        
        try:
            dict_file = Path(ttft_diff_dict_path)
            if dict_file.exists():
                with open(dict_file, 'r', encoding='utf-8') as f:
                    dict_data = json.load(f)
                    # 将字符串 key (如 "100_4") 转换为元组 key (100, 4)
                    self.ttft_diff_dict = {tuple(map(int, k.split('_'))): v for k, v in dict_data.items()}
                print(f"成功加载 TTFT 差值字典，包含 {len(self.ttft_diff_dict)} 个条目")
            else:
                print(f"警告: TTFT 差值字典文件不存在: {ttft_diff_dict_path}")
        except Exception as e:
            print(f"警告: 加载 TTFT 差值字典时出错: {e}")

        # === 先验权重初始化 ===
        # 使用 num_log_bins 存储，节省内存并加速索引
        self.prior_weights = np.ones((num_log_bins, K))
        self.prior_strength = 1  # 先验强度：相当于预设了 5 次实验的观察值
        self._init_prior_weights()

        # === 结构参数 ===
        self.context_stats = [{
            'current_block': 1,
            'current_bin_idx': 1,
            'bin_step_count': 0,
            'is_exploration_bin': True
        } for _ in range(num_log_bins)]

        self.total_rounds = 0

    def _init_prior_weights(self):
        """
        初始化先验权重。
        注意：每2个batch为一个bin。
        """
        for b_idx in range(self.num_log_bins):
            # 估算该 bin 代表的典型 Batch Size (取中值)
            # 例如 Bin 0 -> BS 1-2 (中值1.5), Bin 1 -> BS 3-4 (中值3.5), Bin 2 -> BS 5-6 (中值5.5)
            representative_context = b_idx * 2 + 1.5 
            
            for arm_idx in range(self.K):
                if representative_context < 50:  # 小/中流量
                    self.prior_weights[b_idx][arm_idx] = 0.5
                    self.prior_weights[b_idx][0] = 0     # 抑制不开启投机
                else: # 大流量
                    # 投机长度越短，初始权重越高
                    self.prior_weights[b_idx][arm_idx] = 1.0 + (self.K - arm_idx) / self.K


               

    def _get_context_bin(self, context: int) -> int:
        """每2个batch为一个bin的分箱逻辑 """
        # Bin 0: context 1-2, Bin 1: context 3-4, Bin 2: context 5-6, ...
        bin_idx = (context - 1) // 2
        return min(bin_idx, self.num_log_bins - 1)

    def select_arm(self, context: int, current_qps=None, skip_neural_net_proposer_step_nums: Optional[List[int]] = None) -> int:
        self.total_rounds += 1
        ctx_idx = self._get_context_bin(context)
        s = self.context_stats[ctx_idx]
        print("context:", context,"ctx_idx:", ctx_idx,"current_qps:", current_qps)
        if (context > 20 and current_qps > 2):
            self.have_disabled = True
            return 0
        # 如果提供了 skip_neural_net_proposer_step_nums 列表，可以在这里使用
        # 例如：根据跳过步数调整决策逻辑
        if skip_neural_net_proposer_step_nums is not None:
            # 可以计算平均跳过步数、最大跳过步数等统计信息用于决策
            max_skip_steps = np.max(skip_neural_net_proposer_step_nums) if skip_neural_net_proposer_step_nums else 0
            # 这里可以根据需要调整决策逻辑
            c_prefill = self.ttft_diff_dict[max_skip_steps][context] 
        
        # 预计算基于先验权重的概率分布（用于探索阶段）
        # 这样如果 prior_weights[ctx_idx][0] == 0，arm 0 就永远不会被选中
        p_weights = self.prior_weights[ctx_idx]
        p_dist = p_weights / np.sum(p_weights)
        
        # 1. 维护 Block 和 Bin 的级联结构
        block_len = 2 ** (s['current_block'] - 1)
        bin_len = max(1, int(math.sqrt(block_len)))

        if s['bin_step_count'] >= bin_len:
            s['bin_step_count'] = 0
            s['current_bin_idx'] += 1
            if (s['current_bin_idx'] - 1) * bin_len >= block_len:
                s['current_block'] += 1
                s['current_bin_idx'] = 1
                # 更新 block 后的 bin 长度
                block_len = 2 ** (s['current_block'] - 1)
                bin_len = max(1, int(math.sqrt(block_len)))

            # 决定新的 Bin 是否为探索分箱 [cite: 321]
            explore_prob = 1.0 / math.sqrt(s['current_bin_idx'])
            s['is_exploration_bin'] = (np.random.random() < explore_prob)

        s['bin_step_count'] += 1

        # 2. 决策逻辑
        if s['is_exploration_bin'] and not self.have_disabled:
            # === 修改点：探索阶段不再纯随机，而是加入先验权重 ===
            # 使用 np.random.choice 进行加权采样
            return np.random.choice(self.K, p=p_dist)
        else:
        
            # 正常的利用逻辑（Argmax）
            n = self.arm_stats['n'][:, ctx_idx]
            avg_r = self.arm_stats['avg_rewards'][:, ctx_idx]
            p_val = self.prior_weights[ctx_idx, :]
            if self.have_disabled:
                combined_scores[0] = 1/avg_r[arm_idx]
                for arm_idx in range(1, self.K):
                    combined_scores[arm_idx] = 1/avg_r[arm_idx] + c_prefill/ self.spec_lengths[arm_idx]
                arm = np.argmin(combined_scores)
                if arm > 0:
                    self.have_disabled = False
                return arm
            # p_val[0] += (current_qps - 10) * 0.5 # 随负载线性增加不投机的倾向
            # 贝叶斯平滑得分：(实测奖励*次数 + 先验权重*强度) / (总次数 + 强度)
            combined_scores = (avg_r * n + p_val * self.prior_strength) / (n + self.prior_strength)
            
            # 引入极小扰动打破平分，并取最大值
            return np.argmax(combined_scores)

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        ctx_idx = self._get_context_bin(context)
        reward = (generated_tokens/context) / (elapsed_time + 1e-6)
        
        # 增量更新经验平均奖励
        self.arm_stats['n'][arm_idx, ctx_idx] += 1
        n = self.arm_stats['n'][arm_idx, ctx_idx]
        old_avg = self.arm_stats['avg_rewards'][arm_idx, ctx_idx]
        self.arm_stats['avg_rewards'][arm_idx, ctx_idx] = old_avg + (reward - old_avg) / n

# class ADABinGreedy:

#     def __init__(
#             self,
#             K: int,  # 候选投机长度数量
#             max_spec_length: int,  # 最大推测长度
#             context_bins: int = 250,  # 上下文分箱数量
#             delta: float = 0.1,  # 非平稳性检测阈值
#     ):
#         """
#         ADA-BINGREEDY算法实现，基于：
#         1. 分箱(bin)结构的探索机制
#         2. 动态调整的探索概率
#         3. 非平稳性检测和重启机制
        
#         改进点：
#         - 将时间划分为epoch-block-bin三级结构
#         - 在bin级别实现探索/利用分离
#         - 动态调整探索概率μ_t
#         """
#         # === 状态空间 ===
#         self.arm_stats = {
#             'n': np.zeros((K, context_bins)),  # 每个arm-context组合的选择次数
#             'sum_rewards': np.zeros((K, context_bins)),  # 总奖励
#             'avg_rewards': np.zeros((K, context_bins)),  # 平均奖励
#             'last_reward': np.zeros((K, context_bins)),  # 上一次奖励
#         }

#         # === 算法参数 ===
#         self.K = K
#         self.L = max_spec_length
#         self.context_bins = context_bins
#         self.delta = delta

#         # === ADA-BINGREEDY特有参数 ===
#         self.current_epoch = 1
#         self.current_block = 1
#         self.current_bin = 1
#         self.epoch_start_time = 1
#         self.block_start_time = 1
#         self.bin_start_time = 1
#         self.total_rounds = 0

#         # 定义分箱结构（指数增长）
#         self.bin_length = 1  # 初始分箱长度
#         self.explore_prob = 1.0  # 初始探索概率
        
#         # 为每个context_bin维护独立的探索计数（用于动态流量场景）
#         self.context_bin_rounds = np.zeros(context_bins, dtype=int)  # 每个context_bin的访问次数
#         self.num_log_bins = 12
#         # 投机长度候选值
#         self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)

#         # 先验权重（与原始实现相同）
#         self.prior_weights = np.ones((context_bins, K))
#         self._init_prior_weights()

#     def _init_prior_weights(self):
#         """初始化上下文相关的先验权重"""
#         for context in range(self.context_bins):
#             for arm_idx in range(self.K):
#                 if context < 60:  # 小流量
#                     self.prior_weights[context][arm_idx] = 0.5  
#                     self.prior_weights[context][0] = 0
#                 elif context > 80:  # 大流量
#                     self.prior_weights[context][arm_idx] = 1.0 + (
#                         self.K - arm_idx) / self.K
#                 else:  # 中等流量
#                     self.prior_weights[context][arm_idx] = 0.5 
#                     self.prior_weights[context][0] = 0

#     def _get_context_bin(self, context: int) -> int:
#         """
#         优化点 2：对数分箱函数
#         将连续的 Batch Size 映射到稀疏的分箱中
#         """
#         if context <= 2:
#             return max(0, context - 1) # BS=1 -> 0, BS=2 -> 1
        
#         # 使用 log2 映射，BS=3~4 -> 2, BS=5~8 -> 3, BS=9~16 -> 4 ...
#         bin_idx = int(math.log2(context - 1)) + 1
#         return min(bin_idx, self.num_log_bins - 1)
#     # def _get_context_bin(self, context: int) -> int:
#     #     """将连续上下文映射到离散bin（简化版）"""
#     #     return min(context, self.context_bins - 1)

#     def _update_arm_stats(self, arm_idx: int, context_bin: int, reward: float):
#         """更新arm统计量"""
#         self.arm_stats['n'][arm_idx, context_bin] += 1
#         self.arm_stats['sum_rewards'][arm_idx, context_bin] += reward
#         self.arm_stats['avg_rewards'][arm_idx, context_bin] = (
#             self.arm_stats['sum_rewards'][arm_idx, context_bin] /
#             max(1, self.arm_stats['n'][arm_idx, context_bin]))
#         self.arm_stats['last_reward'][arm_idx, context_bin] = reward

    
#     def _nonstationarity_test(self, current_context: int) -> bool:
#         """
#         改进的非平稳性检测，基于论文中的统计测试
#         比较当前bin与历史block的性能差异
#         """
#         if self.current_block == 1:  # 第一个block无需检测
#             return False

#         context_bin = current_context
#         current_bin_size = self.current_bin
#         historical_block_size = 2 ** (self.current_block - 2)  # 上一个block的大小
        
#         # 获取当前bin和历史block的统计量
#         current_bin_rewards = []
#         historical_block_rewards = []
        
#         # 这里简化实现，实际应该存储历史数据
#         # 假设我们只比较最近的两个时间窗口
#         for arm_idx in range(self.K):
#             n_current = self.arm_stats['n'][arm_idx, context_bin]
#             sum_current = self.arm_stats['sum_rewards'][arm_idx, context_bin]
            
#             # 当前bin的统计量（简化：使用最近的部分数据）
#             current_avg = sum_current / max(1, n_current)
            
#             # 历史block的统计量（简化：使用较早的数据）
#             historical_avg = self.arm_stats['last_reward'][arm_idx, context_bin]
            
#             if n_current > 0 and not np.isnan(current_avg) and not np.isnan(historical_avg):
#                 current_bin_rewards.append(current_avg)
#                 historical_block_rewards.append(historical_avg)
        
#         if not current_bin_rewards or not historical_block_rewards:
#             return False
        
#         # 计算统计量（简化版，论文中使用更复杂的concentration inequality）
#         current_mean = np.mean(current_bin_rewards)
#         historical_mean = np.mean(historical_block_rewards)
        
#         # 计算方差（简化）
#         current_var = np.var(current_bin_rewards) if len(current_bin_rewards) > 1 else 0
#         historical_var = np.var(historical_block_rewards) if len(historical_block_rewards) > 1 else 0
        
#         # 计算统计显著性（简化版）
#         std_error = np.sqrt(current_var/len(current_bin_rewards) + historical_var/len(historical_block_rewards))
#         z_score = abs(current_mean - historical_mean) / (std_error + 1e-6)
        
#         # 阈值设置（论文中使用更复杂的公式）
#         threshold = 2 * np.sqrt(np.log(self.current_block) / min(len(current_bin_rewards), len(historical_block_rewards)))
        
#         return z_score > threshold

   

#     def _get_exploration_prob(self, t: int, context_bin: int = None) -> float:
#         """
#         动态调整探索概率μ_t ≈ t^(-1/3)
#         如果提供了context_bin，则基于该context_bin的访问次数计算（适用于动态流量）
#         """
#         if context_bin is not None and self.context_bin_rounds[context_bin] > 0:
#             # 基于context_bin的访问次数计算探索概率（每个batch_size独立）
#             context_t = self.context_bin_rounds[context_bin]
#             # 使用更慢的衰减，确保每个batch_size都有足够的探索
#             base_prob = (context_t + 1)**(-1 / 3)
#             # 增加最小探索概率，确保不会完全停止探索
#             min_explore_prob = 0  # 最小5%的探索概率
#             return max(min_explore_prob, base_prob)
#         else:
#             # 全局探索概率（向后兼容）
#             epoch_time = t - self.epoch_start_time + 1
#             base_prob = (epoch_time)**(-1 / 3)
#             min_explore_prob = 0.05
#             return max(min_explore_prob, base_prob)

#     def select_arm(self, context: int, current_qps: float = None) -> int:
#         """
#         ADA-BINGREEDY的核心选择逻辑：
#         1. 维护epoch-block-bin三级结构
#         2. 在bin级别实现探索/利用分离
#         3. 动态调整探索概率
#         4. 非平稳性检测触发重启
#         """
       
#         self.total_rounds += 1
#         t = self.total_rounds
#         context_bin = context #self._get_context_bin(context)
        
#         # 更新该context_bin的访问计数（用于动态流量场景）
#         self.context_bin_rounds[context_bin] += 1

#         # === 1. 检查是否需要重启epoch ===
#         # if self._nonstationarity_test(context):
#         #     self.current_epoch += 1
#         #     self.epoch_start_time = t
#         #     self.current_block = 1
#         #     self.current_bin = 1
#         #     self.bin_length = 1  # 重置bin长度
#         #     # 清空统计量（实际实现可能保留部分历史）
#         #     self.arm_stats['n'].fill(0)
#         #     self.arm_stats['sum_rewards'].fill(0)

#         # === 2. 检查是否需要进入新block（指数增长）===
#         if self.current_bin > self.bin_length:
#             self.current_block += 1
#             self.current_bin = 1
#             self.bin_length = 2 ** (self.current_block - 1)  # 指数增长

#         # === 2.5. 检查是否存在未探索的arm（优先探索）===
#         unexplored_arms = []
#         for arm_idx in range(self.K):
#             if self.prior_weights[context_bin][arm_idx] > 0:  # 只考虑有效的arm
#                 if self.arm_stats['n'][arm_idx, context_bin] == 0:
#                     unexplored_arms.append(arm_idx)
        
#         # 如果存在未探索的arm，优先选择它们（强制探索）
#         if unexplored_arms:
#             selected_arm = np.random.choice(unexplored_arms)
#             print(f"force_explore unexplored arm {selected_arm} for context_bin {context_bin}")
#             self.current_bin += 1
#             return selected_arm

#         # === 3. 分箱级别的探索/利用决策 ===
#         # 使用基于context_bin的探索概率（适用于动态流量）
#         explore_prob = self._get_exploration_prob(t, context_bin)
#         is_explore_bin = (np.random.random() < explore_prob)
#         # 如果存在某个arm在当前bin还未被探索（即self.arm_stats['n'][arm_idx, context_bin] <= 0），优先选此arm
       
#         if is_explore_bin:
#             # 探索阶段：随机选择，侧重未充分探索的arm
#             explore_probs = np.ones(self.K)
#             for arm_idx in range(self.K):
#                 base_prob = self.prior_weights[context_bin][arm_idx]
#                 # 未充分探索的arm获得额外概率
#                 if self.arm_stats['n'][arm_idx, context_bin] < 2:
#                     base_prob *= 10.0
#                 explore_probs[arm_idx] = base_prob

#             explore_probs /= np.sum(explore_probs)
#             print(f"explore_mode")
#             selected_arm = np.random.choice(self.K, p=explore_probs)
#             # print(f"explore_probs")
#         else:
#             # 利用阶段：选择历史表现最好的arm（带先验调整）
#             # 注意：由于前面已经检查了未探索的arm，这里所有arm都应该已被探索过

#             combined_scores = np.full(self.K, -np.inf)  # 初始化为负无穷，确保未探索的arm不会被选中
            
#             prior_strength = 1  # 相当于给每个 arm 预设了 5 次实验的信心
#             for arm_idx in range(self.K):
#                 if self.prior_weights[context_bin][arm_idx] > 0:
#                     n = self.arm_stats['n'][arm_idx, context_bin]
#                     empirical_avg = self.arm_stats['avg_rewards'][arm_idx, context_bin]
#                     prior_val = self.prior_weights[context_bin][arm_idx]
                    
#                     # 贝叶斯平滑得分公式
#                     combined_scores[arm_idx] = (empirical_avg * n + prior_val * prior_strength) / (n + prior_strength)
                        
           
            
#             # 取三位小数，如果相同则按索引从小到大选择
#             rounded_scores = np.round(combined_scores, 3)
#             print("exploit mode, combined_scores:", rounded_scores)
#             selected_arm = np.argmax(rounded_scores)

#         # === 4. 更新状态 ===
#         self.current_bin += 1
        
#         return selected_arm

#     def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
#         """
#         更新Epsilon-Greedy统计量
        
#         Args:
#             arm_idx: 选择的arm索引（投机长度）
#             context: 请求量（batch size）
#             generated_tokens: 生成的token总数（从scheduler传入的是 num_accepted_tokens + batch_size）
#             elapsed_time: 执行时间
#         """
        
       
#         reward = generated_tokens / (elapsed_time + 1e-6)
        
#         # print(f"reward: {reward:.2f}, generated_tokens: {generated_tokens}, num_accepted_est: {num_accepted_tokens if arm_idx > 0 else 'N/A'}, elapsed_time: {elapsed_time:.4f}, arm_idx: {arm_idx}")
#         context_bin = context #self._get_context_bin(context)
#         # 更新统计量
#         self._update_arm_stats(arm_idx, context_bin, reward)
        
class EpsilonGreedySpecSimple:
    def __init__(self,
                 K: int,                 # 候选投机长度数量
                 max_spec_length: int,   # 最大推测长度
                 context_bins: int = 200, # 上下文分箱数量
                 epsilon: float = 0.1,   # 探索概率
                ):
        """
        初始化Epsilon-Greedy算法，基于上下文（请求量）决定投机长度
        
        使用epsilon-greedy策略：
        - 以概率epsilon随机探索
        - 以概率1-epsilon选择当前最优arm
        - 每个arm-context组合维护独立的统计量
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_bins: 上下文分箱数量，用于离散化请求量
            epsilon: 探索概率，控制探索-利用平衡
        """
        # === 状态空间 ===
        self.arm_stats = {
            'n': np.zeros((K, context_bins)),           # 每个arm-context组合的选择次数
            'sum_rewards': np.zeros((K, context_bins)), # 总奖励（吞吐量之和）
            'avg_rewards': np.zeros((K, context_bins)), # 平均奖励
            'last_reward': np.zeros((K, context_bins)), # 上一次的奖励
        }

        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.context_bins = context_bins
        self.bin_pulls = np.zeros(context_bins)
        self.epsilon = epsilon
        self.has_explore_context = [[0 for _ in range(K)] for _ in range(200)]
        self.last_selected_arm = [None for _ in range(200)]
        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0
        self.context_bounds = None  # 上下文边界，将在第一次使用时初始化

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)
        self.prior_weights = np.ones((200,self.K))
        for context in range(200):
            for arm_idx in range(self.K):
                # 小context时，大臂权重更高；大context时，小臂权重更高
                if context < 60:  # 小流量
                    # 大臂权重更高：arm_idx越大，权重越大
                    self.prior_weights[context][arm_idx] = 1.0 + 0.5 #arm_idx / self.K
                    self.prior_weights[context][0] = 0
                elif context > 80:  # 大流量
                    # 小臂权重更高：arm_idx越小，权重越大
                    self.prior_weights[context][arm_idx] = 1.0 + (self.K - arm_idx) / self.K
                else:  # 中等流量，平衡策略
                    self.prior_weights[context][arm_idx] = 1.0 + 0.5
                    self.prior_weights[context][0] = 0

    def _get_context_bin(self, context: int) -> int:
        """
        将连续上下文（请求量）映射到离散的bin索引
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            bin索引
        """
        return context
        if self.context_bounds is None:
            # 初始化上下文边界（假设请求量范围是1-100）
            self.context_bounds = np.linspace(1, 100, self.context_bins + 1)

        # 确保context在有效范围内
        context = max(1, min(context, 100))

        # 找到对应的bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _update_arm_stats(self, arm_idx: int, context_bin: int, reward: float):
        """
        更新指定arm-context组合的统计量
        
        Args:
            arm_idx: arm索引
            context_bin: 上下文bin索引
            reward: 观测到的奖励（吞吐量）
        """
        # 更新统计量
        self.arm_stats['n'][arm_idx, context_bin] += 1
        self.arm_stats['sum_rewards'][arm_idx, context_bin] += reward
        self.arm_stats['last_reward'][arm_idx, context_bin] = reward
        # 更新平均奖励
        n = self.arm_stats['n'][arm_idx, context_bin]
        self.arm_stats['avg_rewards'][arm_idx, context_bin] = (
            self.arm_stats['sum_rewards'][arm_idx, context_bin] / n
        )


    def get_delta(self, context):
        # 使用sigmoid实现60附近的平滑过渡
        prob_pos = 1 / (1 + np.exp(0.1*(context-30)))  # 调节0.1可改变过渡陡峭度
        return 1 if np.random.random() < prob_pos else -1
    def select_arm(self, context: int, current_qps: float = None) -> int:
        """
        基于context动态调整探索策略的arm选择方法
        随着context增大，增加对更小臂的探索概率
        
        Args:
            context: 当前请求量（batch size）
            current_qps: 当前QPS（可选）
            
        Returns:
            选择的投机长度索引
        """

        context_bin = context #self._get_context_bin(context)
        epsilon = min(
            self.epsilon,
            1.0 / np.sqrt(self.bin_pulls[context_bin] + 1)
        )

        # 上下文敏感调整（小流量多探索，大流量少探索）
        epsilon = epsilon * 0.5


        # 3. 检查探索状态
        total_pulls =self.bin_pulls[context_bin]
        min_exploration = self.K  # 每个arm至少探索2次

        # 4. 结合先验知识和历史数据的策略
        if (np.random.random() < epsilon or total_pulls < min_exploration) and context < 80:
            # 探索阶段：结合先验知识和未探索程度
            exploration_probs = np.ones(self.K)
            for arm_idx in range(self.K):
                # 基础概率 = 先验权重
                base_prob = self.prior_weights[context_bin][arm_idx]
                # # 未充分探索的arm获得额外概率
                if self.arm_stats['n'][arm_idx, context_bin] < 2:
                    base_prob *= 2.0
                exploration_probs[arm_idx] = base_prob

            exploration_probs = exploration_probs / np.sum(exploration_probs)
            selected_arm = np.random.choice(self.K, p=exploration_probs)

        else:
            # 利用阶段：结合先验知识和历史奖励
            # time_start = time.time()  # 删除这行重复赋值
            avg_rewards = self.arm_stats['avg_rewards'][:, context_bin]

            # 计算综合得分：历史奖励 + 先验知识加成
            combined_scores = np.zeros(self.K)
            for arm_idx in range(self.K):
                if self.arm_stats['n'][arm_idx, context_bin] > 0:
                    # 有历史数据：历史奖励 + 先验知识加成
                    combined_scores[arm_idx] = avg_rewards[arm_idx] + self.prior_weights[context_bin][arm_idx] * 0.1
                else:
                    # 无历史数据：仅基于先验知识
                    combined_scores[arm_idx] = self.prior_weights[context_bin][arm_idx] * 0.5

            selected_arm = np.argmax(combined_scores)
            # print(f"Prior-Guided Exploitation time: {end_time - begin_time}")
            # print(f"Prior-Guided Exploitation: context={context}, "
            #       f"avg_rewards={avg_rewards}, "
            #       f"combined_scores={combined_scores}, "
            #       f"selected_arm={selected_arm}")

        # 4. 更新选择历史
        self.last_selected_arm[context] = selected_arm
        self.bin_pulls[context_bin] += 1

        return selected_arm
    def select_arm4(self, context: int, current_qps: float = None) -> int:
        """
        基于context动态调整探索策略的arm选择方法
        随着context增大，增加对更小臂的探索概率
        
        Args:
            context: 当前请求量（batch size）
            current_qps: 当前QPS（可选）
            
        Returns:
            选择的投机长度索引
        """
        context_bin = self._get_context_bin(context)


        # 1. 基于先验知识的多臂老虎机策略
        # 利用已知规律：流量越大，选择的臂要越小

        # 2. 计算基于先验知识的权重
        prior_weights = np.ones(self.K)
        for arm_idx in range(self.K):
            # 小context时，大臂权重更高；大context时，小臂权重更高
            if context < 20:  # 小流量
                # 大臂权重更高：arm_idx越大，权重越大
                prior_weights[arm_idx] = 1.0 + 0.5
                prior_weights[0] = 0
            elif context > 80:  # 大流量
                # 小臂权重更高：arm_idx越小，权重越大
                prior_weights[arm_idx] = 1.0 + (self.K - arm_idx) / self.K
            else:  # 中等流量，平衡策略
                prior_weights[arm_idx] = 1.0 + 0.5
                prior_weights[0] = 0

        # 3. 检查探索状态
        total_pulls = np.sum(self.arm_stats['n'][:, context_bin])
        min_exploration = self.K * 2  # 每个arm至少探索2次

        # 4. 结合先验知识和历史数据的策略
        if (np.random.random() < self.epsilon or total_pulls < min_exploration) and context < 100:
            # 探索阶段：结合先验知识和未探索程度
            exploration_probs = np.ones(self.K)
            for arm_idx in range(self.K):
                # 基础概率 = 先验权重
                base_prob = prior_weights[arm_idx]
                # 未充分探索的arm获得额外概率
                if self.arm_stats['n'][arm_idx, context_bin] < 2:
                    base_prob *= 2.0
                exploration_probs[arm_idx] = base_prob

            exploration_probs = exploration_probs / np.sum(exploration_probs)
            selected_arm = np.random.choice(self.K, p=exploration_probs)

            print(f"Prior-Guided Exploration: context={context}, "
                  f"prior_weights={prior_weights}, "
                  f"exploration_probs={exploration_probs}, "
                  f"selected_arm={selected_arm}")
        else:
            # 利用阶段：结合先验知识和历史奖励
            avg_rewards = self.arm_stats['avg_rewards'][:, context_bin]

            # 计算综合得分：历史奖励 + 先验知识加成
            combined_scores = np.zeros(self.K)
            for arm_idx in range(self.K):
                if self.arm_stats['n'][arm_idx, context_bin] > 0:
                    # 有历史数据：历史奖励 + 先验知识加成
                    combined_scores[arm_idx] = avg_rewards[arm_idx] + prior_weights[arm_idx] * 0.1
                else:
                    # 无历史数据：仅基于先验知识
                    combined_scores[arm_idx] = prior_weights[arm_idx] * 0.5

            selected_arm = np.argmax(combined_scores)

            print(f"Prior-Guided Exploitation: context={context}, "
                  f"avg_rewards={avg_rewards}, "
                  f"combined_scores={combined_scores}, "
                  f"selected_arm={selected_arm}")

        # 4. 更新选择历史
        self.last_selected_arm[context] = selected_arm

        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新Epsilon-Greedy统计量
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求量（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        self.t += 1
        # self.has_explore_context[context][arm_idx] += 1
        # 计算奖励（吞吐量）
        reward = generated_tokens / (elapsed_time + 1e-6)

        context_bin = self._get_context_bin(context)
        # 更新统计量
        self._update_arm_stats(arm_idx, context_bin, reward)

    def select_arm1(self, context: int, current_qps: float) -> int:
        """
        使用Epsilon-Greedy选择投机长度
        
        Args:
            context: 当前请求量（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if self.round_robin:
            return self.t % self.K
        # if current_qps > 10:
        #     self.last_selected_arm[context] = 0
        #     return 0
        middle_arm = 3
        context_bin = self._get_context_bin(context)
        for k in range(self.K):
            if self.arm_stats['avg_rewards'][k, context_bin] == 0:
                print(f"Epsilon-Greedy (explore1): context={context}, bin={context_bin}, selected_arm={k}")
                return k

        # # Epsilon-greedy策略
        if np.random.random() < self.epsilon:

            # 探索：随机选择arm
            delta = -1 #self.get_delta(context)
            if self.last_selected_arm[context] is not None:
                selected_arm = self.last_selected_arm[context] + delta
            else:
                selected_arm = middle_arm + delta
            if selected_arm == 4 or selected_arm == -1:
                selected_arm = middle_arm

            self.last_selected_arm[context] = selected_arm
            print(f"Epsilon-Greedy (explore2): context={context}, bin={context_bin}, selected_arm={selected_arm}")
        else:
            selected_arm = np.argmax(self.arm_stats['avg_rewards'][:, context_bin])
            self.last_selected_arm[context] = selected_arm

        print(f"Epsilon-Greedy (exploit): context={context}, bin={context_bin}, "
                f"selected_arm={selected_arm}")

        return selected_arm

    def select_arm3(self, context: int, current_qps: float) -> int:
        """
        使用Epsilon-Greedy选择投机长度
        
        Args:
            context: 当前请求量（batch size）
            
        Returns:
            选择的投机长度索引
        """

        context_bin = self._get_context_bin(context)
        # # Epsilon-greedy策略
        if np.random.random() < self.epsilon:

            # 探索：随机选择arm
            delta = -1 #self.get_delta(context)
            if self.last_selected_arm[context] is not None:
                selected_arm = self.last_selected_arm[context] + delta
            else:
                selected_arm = 3 + delta
            if selected_arm == 5 or selected_arm == -1:
                selected_arm = 3

            self.last_selected_arm[context] = selected_arm
            print(f"Epsilon-Greedy (explore): context={context}, bin={context_bin}, selected_arm={selected_arm}")
        else:
            # 判断 self.arm_stats['avg_rewards'][:, context_bin] 是否全为0
            if np.all(self.arm_stats['avg_rewards'][:, context_bin] == 0):
                selected_arm = 3
            else:
                selected_arm = np.argmax(self.arm_stats['avg_rewards'][:, context_bin])
            self.last_selected_arm[context] = selected_arm

        print(f"Epsilon-Greedy (exploit): context={context}, bin={context_bin}, "
                f"selected_arm={selected_arm}")

        return selected_arm

    def select_arm2(self, context: int, current_qps: float) -> int:
        """
        使用Epsilon-Greedy选择投机长度
        
        Args:
            context: 当前请求量（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if context > 60:
            self.last_selected_arm[context] = 0
            return 0
        context_bin = self._get_context_bin(context)
        middle_arm = 2
        # # Epsilon-greedy策略
        if np.random.random() < self.epsilon:

            # 探索：随机选择arm
            delta = -1 #self.get_delta(context)
            if self.last_selected_arm[context] is not None:
                selected_arm = self.last_selected_arm[context] + delta
            else:
                selected_arm = middle_arm + delta
            if selected_arm == 4 or selected_arm == -1:
                selected_arm = middle_arm

            self.last_selected_arm[context] = selected_arm
            print(f"Epsilon-Greedy (explore): context={context}, bin={context_bin}, selected_arm={selected_arm}")
        else:
            last_selected_arm = self.last_selected_arm[context]
            if last_selected_arm is not None:
                if last_selected_arm != middle_arm:
                    if self.arm_stats['avg_rewards'][middle_arm, context_bin] > 0 and self.arm_stats['last_reward'][last_selected_arm, context_bin] > self.arm_stats['avg_rewards'][middle_arm, context_bin]:
                        print("Epsilon-Greedy (exploit):",self.arm_stats['last_reward'][last_selected_arm, context_bin],self.arm_stats['avg_rewards'][3, context_bin])
                        selected_arm = last_selected_arm
                    else:
                        selected_arm = middle_arm
                else:
                    selected_arm = middle_arm
            else:
                selected_arm = middle_arm
            self.last_selected_arm[context] = selected_arm

        print(f"Epsilon-Greedy (exploit): context={context}, bin={context_bin}, "
                f"selected_arm={selected_arm}")

        return selected_arm


    def reset(self):
        """重置算法状态"""
        self.arm_stats['n'].fill(0)
        self.arm_stats['sum_rewards'].fill(0)
        self.arm_stats['avg_rewards'].fill(0)
        self.t = 0
        self.round_robin = True


    def save_state(self, filepath: str):
        pass

    def load_state(self, filepath: str):
        pass

class EpsilonGreedySpec3:
    def __init__(self,
                 K: int,                 # 候选投机长度数量
                 max_spec_length: int,   # 最大推测长度
                 context_bins: int = 50, # 上下文分箱数量
                 epsilon: float = 0.1,   # 探索概率
                ):
        """
        初始化Epsilon-Greedy算法，基于上下文（请求量）决定投机长度
        
        使用epsilon-greedy策略：
        - 以概率epsilon随机探索
        - 以概率1-epsilon选择当前最优arm
        - 每个arm-context组合维护独立的统计量
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_bins: 上下文分箱数量，用于离散化请求量
            epsilon: 探索概率，控制探索-利用平衡
        """
        # === 状态空间 ===
        self.arm_stats = {
            'n': np.zeros((K, context_bins)),           # 每个arm-context组合的选择次数
            'sum_rewards': np.zeros((K, context_bins)), # 总奖励（吞吐量之和）
            'avg_rewards': np.zeros((K, context_bins)), # 平均奖励
        }

        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.context_bins = context_bins
        self.epsilon = epsilon
        self.has_explore_context = [[0 for _ in range(K)] for _ in range(200)]
        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0
        self.context_bounds = None  # 上下文边界，将在第一次使用时初始化

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)

        print(f"EpsilonGreedySpec initialized: K={K}, L={max_spec_length}, context_bins={context_bins}, epsilon={epsilon}")

    def _get_context_bin(self, context: int) -> int:
        """
        将连续上下文（请求量）映射到离散的bin索引
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            bin索引
        """
        if self.context_bounds is None:
            # 初始化上下文边界（假设请求量范围是1-100）
            self.context_bounds = np.linspace(1, 100, self.context_bins + 1)

        # 确保context在有效范围内
        context = max(1, min(context, 100))

        # 找到对应的bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _update_arm_stats(self, arm_idx: int, context_bin: int, reward: float):
        """
        更新指定arm-context组合的统计量
        
        Args:
            arm_idx: arm索引
            context_bin: 上下文bin索引
            reward: 观测到的奖励（吞吐量）
        """
        # 更新统计量
        self.arm_stats['n'][arm_idx, context_bin] += 1
        self.arm_stats['sum_rewards'][arm_idx, context_bin] += reward
        self.has_explore_context[context_bin][arm_idx] += 1
        # 更新平均奖励
        n = self.arm_stats['n'][arm_idx, context_bin]
        self.arm_stats['avg_rewards'][arm_idx, context_bin] = (
            self.arm_stats['sum_rewards'][arm_idx, context_bin] / n
        )

    def select_arm(self, context: int) -> int:
        """
        使用Epsilon-Greedy选择投机长度
        
        Args:
            context: 当前请求量（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K
        # for arm_idx in range(self.K):
        #     if self.has_explore_context[context][arm_idx] == 0:
        #         return arm_idx
        context_bin = self._get_context_bin(context)
        n = sum(self.arm_stats['n'][:, context_bin])
        avg_rewards = self.arm_stats['avg_rewards'][:, context_bin]
        if n > 0 and np.random.random() < 1 / n:  # epsilon值随时间衰减
            print("explore!")
            # selected_arm = np.random.randint(0, self.K)
            # Find arms that haven't been tried yet for this context bin
            untried_arms = np.where(self.arm_stats['n'][:, context_bin] == 0)[0]
            if len(untried_arms) > 0:
                # Randomly select one of the untried arms
                selected_arm = np.random.choice(untried_arms)
            else:
                # If all arms have been tried, select randomly
                selected_arm = np.random.randint(0, self.K)
        else:
            selected_arm = np.argmax(avg_rewards)


        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新Epsilon-Greedy统计量
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求量（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        self.t += 1
        # self.has_explore_context[context][arm_idx] += 1
        # 计算奖励（吞吐量）
        reward = generated_tokens / (elapsed_time + 1e-6)

        context_bin = self._get_context_bin(context)

        # 更新统计量
        self._update_arm_stats(arm_idx, context_bin, reward)

        # print(f"Epsilon-Greedy Update: arm={arm_idx}, context={context}, bin={context_bin}, "
        #       f"reward={reward:.2f}, avg_reward={self.arm_stats['avg_rewards'][arm_idx, context_bin]:.2f}")



    def reset(self):
        """重置算法状态"""
        self.arm_stats['n'].fill(0)
        self.arm_stats['sum_rewards'].fill(0)
        self.arm_stats['avg_rewards'].fill(0)
        self.t = 0
        # self.round_robin = True

    def save_state(self, filepath: str):
        """保存内部状态到文件"""
        state = {
            'arm_stats': self.arm_stats,
            't': self.t,
            'K': self.K,
            'L': self.L,
            'context_bins': self.context_bins,
            'epsilon': self.epsilon,
            'context_bounds': self.context_bounds,
            'spec_lengths': self.spec_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """从文件加载内部状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.arm_stats = state['arm_stats']
        self.t = state['t']
        self.K = state['K']
        self.L = state['L']
        self.context_bins = state['context_bins']
        self.epsilon = state['epsilon']
        self.context_bounds = state['context_bounds']
        self.spec_lengths = state['spec_lengths']


class EpsilonGreedySpecSlidingWindow:
    def __init__(self,
                 K: int,                 # 候选投机长度数量
                 max_spec_length: int,   # 最大推测长度
                 context_bins: int = 20, # 上下文分箱数量
                 epsilon: float = 0.1,   # 探索概率
                 window_size: int = 500, # 滑动窗口大小
                ):
        """
        初始化Epsilon-Greedy算法，基于上下文（请求量）决定投机长度，使用滑动窗口
        
        使用epsilon-greedy策略：
        - 以概率epsilon随机探索
        - 以概率1-epsilon选择当前最优arm
        - 每个arm-context组合维护一个滑动窗口队列
        - 滑动窗口只保留最近的window_size个观测值
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_bins: 上下文分箱数量，用于离散化请求量
            epsilon: 探索概率，控制探索-利用平衡
            window_size: 滑动窗口大小
        """
        # === 状态空间 ===
        # 使用字典存储每个(arm, context_bin)组合的观测历史
        self.arm_stats = {
            'observations': {},  # key: (arm_idx, context_bin), value: list of rewards
            'total_observation': [],
        }

        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.context_bins = context_bins
        self.epsilon = epsilon
        self.window_size = window_size

        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0
        self.context_bounds = None  # 上下文边界，将在第一次使用时初始化

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)
        self.last_total_n = [-1 for _ in range(context_bins)]
        print(f"EpsilonGreedySpecSlidingWindow initialized: K={K}, L={max_spec_length}, context_bins={context_bins}, epsilon={epsilon}, window_size={window_size}")

    def _get_context_bin(self, context: int) -> int:
        """
        将连续上下文（请求量）映射到离散的bin索引
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            bin索引
        """
        if self.context_bounds is None:
            # 初始化上下文边界（假设请求量范围是1-100）
            self.context_bounds = np.linspace(1, 100, self.context_bins + 1)

        # 确保context在有效范围内
        context = max(1, min(context, 100))

        # 找到对应的bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _get_window_stats(self, arm_idx: int, context_bin: int):
        """
        获取滑动窗口中指定arm-context组合的统计量
        
        Args:
            arm_idx: arm索引
            context_bin: 上下文bin索引
            
        Returns:
            tuple: (total_count, total_reward, mean_reward)
        """
        key = (arm_idx, context_bin)
        if key not in self.arm_stats['observations']:
            return 0, 0.0, 0.0

        observations = self.arm_stats['observations'][key]
        if len(observations) == 0:
            return 0, 0.0, 0.0

        total_count = len(observations)
        total_reward = sum(observations)
        mean_reward = total_reward / total_count

        return total_count, total_reward, mean_reward

    def _update_arm_stats(self, arm_idx: int, context_bin: int, reward: float):
        """
        更新指定arm-context组合的统计量（使用滑动窗口）
        
        Args:
            arm_idx: arm索引
            context_bin: 上下文bin索引
            reward: 观测到的奖励（吞吐量）
        """
        key = (arm_idx, context_bin)

        # 如果这是第一次观测该arm-context组合，初始化队列
        if key not in self.arm_stats['observations']:
            self.arm_stats['observations'][key] = []

        # 添加新的观测值
        self.arm_stats['observations'][key].append(reward)
        self.arm_stats['total_observation'].append(key)
        if len(self.arm_stats['total_observation']) > self.window_size:
            key = self.arm_stats['total_observation'].pop(0)
            self.arm_stats['observations'][key].pop(0)
            print(f"EpsilonGreedySpecSlidingWindow: Remove old observation {key},{len(self.arm_stats['observations'][key])}")

    def select_arm(self, context: int) -> int:
        """
        使用Epsilon-Greedy选择投机长度（基于滑动窗口统计）
        
        Args:
            context: 当前请求量（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K

        context_bin = self._get_context_bin(context)

        # 计算当前上下文bin的总计数
        total_n = 0
        arm_stats = []
        total_counts = []
        for k in range(self.K):
            total_count, _, mean_reward = self._get_window_stats(k, context_bin)
            total_counts.append(total_count)
            total_n += total_count
            arm_stats.append((k, total_count, mean_reward))
        # # 因为滑动窗口最多100，那么当某个数连续选了100次之后，继续选还是100，因为滑动窗口已经踢掉之前的值 所以total_n 不变了
        if self.last_total_n[context_bin] == total_n:
            unselected_arm = np.argmax(total_counts)
            selected_arm = np.random.choice([k for k, count, _ in arm_stats if k != unselected_arm])
            return selected_arm
        self.last_total_n[context_bin] = total_n

        # 动态调整epsilon值（基于总计数）
        if total_n > 0:
            dynamic_epsilon = min(self.epsilon, 1.0 / total_n)  # epsilon值随时间衰减，但不超过初始值
        else:
            dynamic_epsilon = self.epsilon

        print(f"EpsilonGreedySpecSlidingWindow: context={context}, bin={context_bin}, "
              f"total_n={total_n}, dynamic_epsilon={dynamic_epsilon:.4f}")
        untried_arms = [k for k, count, _ in arm_stats if count == 0]
        if len(untried_arms) > 0:
            # 随机选择一个未尝试的arm
            selected_arm = np.random.choice(untried_arms)
            print(f"EpsilonGreedySpecSlidingWindow: Exploring untried arm {selected_arm}")
        # Epsilon-greedy策略
        elif np.random.random() < dynamic_epsilon:
            # 探索：优先选择未尝试的arm
            if context > 60:
                selected_arm = np.random.randint(0, 3)
            elif context < 20:
                selected_arm = np.random.randint(1, self.K)
            else:
                selected_arm = np.random.randint(0, self.K)
            print(f"EpsilonGreedySpecSlidingWindow: Random exploration, selected arm {selected_arm}")
        else:
            # 利用：选择当前最优arm（基于滑动窗口平均奖励）
            best_reward = -float('inf')
            selected_arm = 0

            for k, count, mean_reward in arm_stats:
                if count > 0 and mean_reward > best_reward:
                    best_reward = mean_reward
                    selected_arm = k

            print(f"EpsilonGreedySpecSlidingWindow: Exploiting best arm {selected_arm} "
                  f"with reward {best_reward:.4f}")

        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新Epsilon-Greedy统计量（使用滑动窗口）
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求量（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        # 计算奖励（吞吐量）
        reward = generated_tokens / (elapsed_time + 1e-6)

        context_bin = self._get_context_bin(context)
        # 更新统计量（滑动窗口会自动处理旧数据的移除）
        self._update_arm_stats(arm_idx, context_bin, reward)

        # 增加时间步
        self.t += 1


    def reset(self):
        """重置所有统计量"""
        self.arm_stats['observations'].clear()
        self.t = 0
        self.round_robin = True
        self.context_bounds = None
        print("EpsilonGreedySpecSlidingWindow: Reset completed")

    def get_window_summary(self, context: int) -> dict:
        """
        获取指定上下文的滑动窗口统计摘要（用于调试）
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            dict: 包含各arm统计信息的字典
        """
        context_bin = self._get_context_bin(context)
        summary = {}

        for k in range(self.K):
            total_count, total_reward, mean_reward = self._get_window_stats(k, context_bin)
            key = (k, context_bin)
            observations = self.arm_stats['observations'].get(key, [])
            summary[f'arm_{k}'] = {
                'total_count': total_count,
                'total_reward': total_reward,
                'mean_reward': mean_reward,
                'recent_observations': observations[-5:] if len(observations) > 5 else observations  # 显示最近5个观测值
            }

        return summary

    def save_state(self, filepath: str):
        """保存内部状态到文件"""
        state = {
            'arm_stats': self.arm_stats,
            't': self.t,
            'K': self.K,
            'L': self.L,
            'context_bins': self.context_bins,
            'epsilon': self.epsilon,
            'window_size': self.window_size,
            'round_robin': self.round_robin,
            'context_bounds': self.context_bounds,
            'spec_lengths': self.spec_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """从文件加载内部状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.arm_stats = state['arm_stats']
        self.t = state['t']
        self.K = state['K']
        self.L = state['L']
        self.context_bins = state['context_bins']
        self.epsilon = state['epsilon']
        self.window_size = state['window_size']
        self.round_robin = state['round_robin']
        self.context_bounds = state['context_bounds']
        self.spec_lengths = state['spec_lengths']

class UCBBinSpec:
    def __init__(self, num_arms, max_spec_length=4, confidence_param=0.01, context_bins=120):
        """
        Initialize the UCBBinSpec algorithm with context bins.
        Args:
            num_arms (int): Number of hyperparameter configurations (arms)
            max_spec_length (int): Maximum speculation length (L in the paper)
            confidence_param (float): Confidence parameter (δ in the paper)
            context_bins (int): Number of context bins for request counts (up to 120)
        """
        self.K = num_arms
        self.delta = confidence_param
        self.L = max_spec_length
        self.context_bins = context_bins

        # Initialize statistics for each arm-context combination
        self.arm_counts = np.zeros((num_arms, context_bins))  # n_i,t for each arm-context
        self.arm_rewards = np.zeros((num_arms, context_bins))  # Sum of rewards for each arm-context
        self.ucb_values = np.zeros((num_arms, context_bins))   # UCB values for each arm-context

        # Context binning parameters
        self.context_bounds = np.linspace(1, 120, context_bins + 1)

        # Track the total number of rounds
        self.t = 0
        self.round_robin = True

        print(f"UCBBinSpec initialized: K={num_arms}, L={max_spec_length}, context_bins={context_bins}")

    def _get_context_bin(self, context: int) -> int:
        """
        Map continuous context (request count) to discrete bin index.
        
        Args:
            context: Request count (batch size)
            
        Returns:
            bin index
        """
        # Ensure context is within valid range
        context = max(1, min(context, 120))

        # Find corresponding bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def select_arm(self, context):
        """
        Select an arm according to the UCBBinSpec algorithm.
        
        Args:
            context: Current request count (batch size)
        
        Returns:
            int: Index of the selected arm
        """
        context_bin = self._get_context_bin(context)

        if self.round_robin:
            # Round-robin for the first K rounds
            return self.t % self.K
        elif self.t < self.K:
            return self.t
        else:
            # Select arm with highest UCB value for this context
            ucb_values_for_context = self.ucb_values[:, context_bin]

            # Handle arms that haven't been explored in this context
            unexplored_mask = self.arm_counts[:, context_bin] == 0
            if unexplored_mask.any():
                # Prioritize unexplored arms for this context
                unexplored_arms = np.where(unexplored_mask)[0]
                selected_arm = unexplored_arms[0]
            else:
                # Select arm with highest UCB value
                # Update UCB values for all arms in this context
                for i in range(self.K):
                    if self.arm_counts[i, context_bin] > 0:
                        # Calculate empirical mean
                        mu_hat = self.arm_rewards[i, context_bin] / self.arm_counts[i, context_bin]

                        # Calculate confidence radius
                        n = self.arm_counts[i, context_bin]
                        log_term = math.log((self.K * (self.t**2) * math.sqrt(1 + n)) / self.delta)
                        cr = (self.L / 2) * math.sqrt((1 + n) / (n**2) * (1 + 2 * log_term))
                        # cr = np.sqrt(
                        #     2 * np.log(self.t / self.delta ) / (n + 1e-6)
                        # )
                        # cr = np.sqrt(
                        #     2 * np.log(self.t**2 / self.delta) / (n + 1e-6)
                        # )

                        # Update UCB value
                        self.ucb_values[i, context_bin] = mu_hat + cr
                    else:
                        # If arm hasn't been pulled yet in this context, set UCB to infinity
                        self.ucb_values[i, context_bin] = float('inf')
                selected_arm = np.argmax(ucb_values_for_context)

            print(f"UCBBinSpec: context={context}, bin={context_bin}, "
                  f"ucb_values={ucb_values_for_context}, selected_arm={selected_arm}")

            return selected_arm

    def update(self, chosen_arm, context, generated_tokens: int, elapsed_time: float):
        """
        Update the statistics after observing a reward.
        
        Args:
            chosen_arm (int): Index of the arm that was pulled
            context: Request count (batch size)
            generated_tokens (int): Number of generated tokens
            elapsed_time (float): Time taken for execution
        """
        self.t += 1
        context_bin = self._get_context_bin(context)

        # Calculate reward (throughput)
        reward = generated_tokens / (elapsed_time + 1e-6)

        # Update counts and rewards for the chosen arm-context combination
        self.arm_counts[chosen_arm, context_bin] += 1
        self.arm_rewards[chosen_arm, context_bin] += reward

    def reset(self):
        """Reset the algorithm's state."""
        self.arm_counts = np.zeros((self.K, self.context_bins))
        self.arm_rewards = np.zeros((self.K, self.context_bins))
        self.ucb_values = np.zeros((self.K, self.context_bins))
        self.t = 0
        self.round_robin = True

    def save_state(self, filepath):
        """Save the internal state to a file."""
        state = {
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'ucb_values': self.ucb_values,
            't': self.t,
            'K': self.K,
            'delta': self.delta,
            'L': self.L,
            'context_bins': self.context_bins,
            'context_bounds': self.context_bounds,
        }
        print("save_state", state, filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath):
        """Load the internal state from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print("load_state", filepath, state)
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.t = state['t']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']
        self.context_bins = state['context_bins']
        self.context_bounds = state['context_bounds']


class UCBBinSpecSlidingWindow:
    def __init__(self, num_arms, max_spec_length=4, confidence_param=0.01, context_bins=10, window_size=100):
        """
        Initialize the UCBBinSpec algorithm with context bins and sliding window.
        Args:
            num_arms (int): Number of hyperparameter configurations (arms)
            max_spec_length (int): Maximum speculation length (L in the paper)
            confidence_param (float): Confidence parameter (δ in the paper)
            context_bins (int): Number of context bins for request counts (up to 120)
            window_size (int): Size of the sliding window for maintaining recent statistics
        """
        self.K = num_arms
        self.delta = confidence_param
        self.L = max_spec_length
        self.context_bins = context_bins
        self.window_size = window_size

        # Initialize sliding window statistics for each arm-context combination
        # Shape: (num_arms, context_bins, window_size)
        self.arm_counts = np.zeros((num_arms, context_bins, window_size))  # n_i,t for each arm-context
        self.arm_rewards = np.zeros((num_arms, context_bins, window_size))  # Sum of rewards for each arm-context
        self.ucb_values = np.zeros((num_arms, context_bins, window_size))   # UCB values for each arm-context

        # Current position in the sliding window
        self.window_pos = 0

        # Context binning parameters
        self.context_bounds = np.linspace(1, 120, context_bins + 1)

        # Track the total number of rounds
        self.t = 0
        self.round_robin = True

        print(f"UCBBinSpecSlidingWindow initialized: K={num_arms}, L={max_spec_length}, context_bins={context_bins}, window_size={window_size}")

    def _get_context_bin(self, context: int) -> int:
        """
        Map continuous context (request count) to discrete bin index.
        
        Args:
            context: Request count (batch size)
            
        Returns:
            bin index
        """
        # Ensure context is within valid range
        context = max(1, min(context, 120))

        # Find corresponding bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _get_window_stats(self, arm_idx: int, context_bin: int):
        """
        Get aggregated statistics from the sliding window for a specific arm-context combination.
        
        Args:
            arm_idx: Index of the arm
            context_bin: Index of the context bin
            
        Returns:
            tuple: (total_count, total_reward, mean_reward)
        """
        # Get the current window data
        counts = self.arm_counts[arm_idx, context_bin, :]
        rewards = self.arm_rewards[arm_idx, context_bin, :]

        # Calculate total count and reward in the window
        total_count = np.sum(counts)
        total_reward = np.sum(rewards)

        # Calculate mean reward
        mean_reward = total_reward / (total_count + 1e-6)

        return total_count, total_reward, mean_reward

    def select_arm(self, context):
        """
        Select an arm according to the UCBBinSpec algorithm with sliding window.
        
        Args:
            context: Current request count (batch size)
        
        Returns:
            int: Index of the selected arm
        """
        context_bin = self._get_context_bin(context)

        if self.round_robin:
            # Round-robin for the first K rounds
            return self.t % self.K
        elif self.t < self.K:
            return self.t
        else:
            # Select arm with highest UCB value for this context
            ucb_values_for_context = np.zeros(self.K)

            # Handle arms that haven't been explored in this context
            unexplored_mask = np.zeros(self.K, dtype=bool)

            for i in range(self.K):
                total_count, _, _ = self._get_window_stats(i, context_bin)
                unexplored_mask[i] = (total_count == 0)

                if total_count > 0:
                    # Calculate empirical mean from sliding window
                    _, _, mean_reward = self._get_window_stats(i, context_bin)

                    # Calculate confidence radius using sliding window data
                    n = total_count
                    # log_term = math.log((self.K * (self.t**2) * math.sqrt(1 + n)) / self.delta)
                    # cr = (self.L / 2) * math.sqrt((1 + n) / (n**2) * (1 + 2 * log_term))
                    cr = np.sqrt(
                        2 * np.log(self.t / self.delta) / (total_count + 1e-6)
                    )

                    # Update UCB value
                    ucb_values_for_context[i] = mean_reward + cr
                else:
                    # If arm hasn't been pulled yet in this context, set UCB to infinity
                    ucb_values_for_context[i] = float('inf')

            if unexplored_mask.any():
                # Prioritize unexplored arms for this context
                unexplored_arms = np.where(unexplored_mask)[0]
                selected_arm = unexplored_arms[0]
            else:
                # Select arm with highest UCB value
                selected_arm = np.argmax(ucb_values_for_context)

            print(f"UCBBinSpecSlidingWindow: context={context}, bin={context_bin}, "
                  f"ucb_values={ucb_values_for_context}, selected_arm={selected_arm}")

            return selected_arm

    def update(self, chosen_arm, context, generated_tokens: int, elapsed_time: float):
        """
        Update the statistics after observing a reward using sliding window.
        
        Args:
            chosen_arm (int): Index of the arm that was pulled
            context: Request count (batch size)
            generated_tokens (int): Number of generated tokens
            elapsed_time (float): Time taken for execution
        """
        self.t += 1
        context_bin = self._get_context_bin(context)

        # Calculate reward (throughput)
        reward = generated_tokens / (elapsed_time + 1e-6)

        # Update counts and rewards for the chosen arm-context combination in the sliding window
        self.arm_counts[chosen_arm, context_bin, self.window_pos] += 1
        self.arm_rewards[chosen_arm, context_bin, self.window_pos] += reward

        # Update UCB value for the chosen arm-context combination
        total_count, _, mean_reward = self._get_window_stats(chosen_arm, context_bin)
        if total_count > 0:
            cr = np.sqrt(
                2 * np.log(self.t / self.delta) / (total_count + 1e-6)
            )
            self.ucb_values[chosen_arm, context_bin, self.window_pos] = mean_reward + cr

        # Move to next position in sliding window
        self.window_pos = (self.window_pos + 1) % self.window_size

    def reset(self):
        """Reset the algorithm's state."""
        self.arm_counts = np.zeros((self.K, self.context_bins, self.window_size))
        self.arm_rewards = np.zeros((self.K, self.context_bins, self.window_size))
        self.ucb_values = np.zeros((self.K, self.context_bins, self.window_size))
        self.window_pos = 0
        self.t = 0
        self.round_robin = True

    def save_state(self, filepath):
        """Save the internal state to a file."""
        state = {
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'ucb_values': self.ucb_values,
            'window_pos': self.window_pos,
            't': self.t,
            'K': self.K,
            'delta': self.delta,
            'L': self.L,
            'context_bins': self.context_bins,
            'context_bounds': self.context_bounds,
            'window_size': self.window_size,
        }
        print("save_state", state, filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath):
        """Load the internal state from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print("load_state", filepath, state)
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.window_pos = state.get('window_pos', 0)  # Backward compatibility
        self.t = state['t']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']
        self.context_bins = state['context_bins']
        self.context_bounds = state['context_bounds']
        self.window_size = state.get('window_size', 100)  # Backward compatibility

class LinUCBSlidingWindow:
    def __init__(self,
                 K: int,                 # 候选投机长度数量
                 max_spec_length: int,   # 最大推测长度
                 context_dim: int = 1,   # 上下文维度（这里固定为1，表示请求数）
                 alpha: float = 1.0,     # 探索参数
                 lambda_reg: float = 1.0, # 正则化参数
                 window_size: int = 500, # 滑动窗口大小
                ):
        """
        初始化LinUCB算法的滑动窗口版本，基于上下文（请求数）决定投机长度
        
        使用线性模型和滑动窗口：
        - 每个arm维护一个滑动窗口的观测历史
        - 上下文是请求数（batch size）
        - 通过UCB公式选择最优arm
        - 只使用最近的window_size个观测值来更新模型
        
        Args:
            K: 候选投机长度数量（0到max_spec_length）
            max_spec_length: 最大推测长度
            context_dim: 上下文维度，固定为1（请求数）
            alpha: 探索参数，控制探索-利用平衡
            lambda_reg: 正则化参数，防止过拟合
            window_size: 滑动窗口大小
        """
        # === 状态空间 ===
        # 使用字典存储每个arm的观测历史
        self.arm_observations = {
            'contexts': [[] for _ in range(K)],  # 每个arm的上下文历史
            'rewards': [[] for _ in range(K)],   # 每个arm的奖励历史
        }

        # 每个arm的线性模型参数
        self.arm_params = {
            'A': [np.eye(context_dim) * lambda_reg for _ in range(K)],  # 每个arm的A矩阵
            'b': [np.zeros(context_dim) for _ in range(K)],             # 每个arm的b向量
            'theta': [np.zeros(context_dim) for _ in range(K)],         # 每个arm的线性参数
        }

        # === 超参数 ===
        self.K = K
        self.L = max_spec_length
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.window_size = window_size

        # === 运行时状态 ===
        self.round_robin = True
        self.t = 0

        # 定义投机长度候选值
        self.spec_lengths = np.linspace(0, max_spec_length, K, dtype=int)

        print(f"LinUCBSlidingWindow initialized: K={K}, L={max_spec_length}, context_dim={context_dim}, alpha={alpha}, window_size={window_size}")

    def _update_arm_params(self, arm_idx: int):
        """
        基于滑动窗口中的观测值更新指定arm的线性模型参数
        
        Args:
            arm_idx: arm索引
        """
        contexts = self.arm_observations['contexts'][arm_idx]
        rewards = self.arm_observations['rewards'][arm_idx]

        if len(contexts) == 0:
            return

        # 重置A矩阵和b向量
        self.arm_params['A'][arm_idx] = np.eye(self.context_dim) * self.lambda_reg
        self.arm_params['b'][arm_idx] = np.zeros(self.context_dim)

        # 基于滑动窗口中的所有观测值重新计算
        for context, reward in zip(contexts, rewards):
            context = context.reshape(-1, 1)
            self.arm_params['A'][arm_idx] += context @ context.T
            self.arm_params['b'][arm_idx] += reward * context.flatten()

        # 更新theta参数（线性回归解）
        try:
            A_inv = np.linalg.inv(self.arm_params['A'][arm_idx])
            self.arm_params['theta'][arm_idx] = A_inv @ self.arm_params['b'][arm_idx]
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            A_inv = np.linalg.pinv(self.arm_params['A'][arm_idx])
            self.arm_params['theta'][arm_idx] = A_inv @ self.arm_params['b'][arm_idx]

    def _add_observation(self, arm_idx: int, context: np.ndarray, reward: float):
        """
        添加新的观测值到滑动窗口
        
        Args:
            arm_idx: arm索引
            context: 上下文向量（请求数）
            reward: 观测到的奖励（吞吐量）
        """
        # 添加新的观测值
        self.arm_observations['contexts'][arm_idx].append(context)
        self.arm_observations['rewards'][arm_idx].append(reward)

        # 如果超过窗口大小，移除最旧的观测值
        if len(self.arm_observations['contexts'][arm_idx]) > self.window_size:
            self.arm_observations['contexts'][arm_idx].pop(0)
            self.arm_observations['rewards'][arm_idx].pop(0)
            print(f"LinUCBSlidingWindow: Removed old observation for arm {arm_idx}, "
                  f"window size now: {len(self.arm_observations['contexts'][arm_idx])}")

    def select_arm(self, context: int) -> int:
        """
        使用LinUCB选择投机长度（基于滑动窗口）
        
        Args:
            context: 当前请求数（batch size）
            
        Returns:
            选择的投机长度索引
        """
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K
        elif self.t < self.K:
            return self.t

        # 将context转换为向量
        context_vec = np.array([context])

        # 计算每个arm的UCB值
        ucb_values = np.zeros(self.K)
        for k in range(self.K):
            # 检查该arm是否有观测值
            if len(self.arm_observations['contexts'][k]) == 0:
                # 如果没有观测值，设置UCB为无穷大以鼓励探索
                ucb_values[k] = float('inf')
                continue

            # 预测奖励
            predicted_reward = self.arm_params['theta'][k] @ context_vec

            # 计算置信区间
            try:
                A_inv = np.linalg.inv(self.arm_params['A'][k])
                context_vec_col = context_vec.reshape(-1, 1)
                confidence = self.alpha * np.sqrt(context_vec_col.T @ A_inv @ context_vec_col)
                ucb_values[k] = predicted_reward + confidence
            except np.linalg.LinAlgError:
                # 如果矩阵不可逆，使用伪逆
                A_inv = np.linalg.pinv(self.arm_params['A'][k])
                context_vec_col = context_vec.reshape(-1, 1)
                confidence = self.alpha * np.sqrt(context_vec_col.T @ A_inv @ context_vec_col)
                ucb_values[k] = predicted_reward + confidence

        # 选择UCB值最大的arm
        selected_arm = np.argmax(ucb_values)

        print(f"LinUCBSlidingWindow: context={context}, ucb_values={ucb_values}, selected_arm={selected_arm}")

        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新LinUCB统计量（使用滑动窗口）
        
        Args:
            arm_idx: 选择的arm索引
            context: 请求数（batch size）
            generated_tokens: 生成的token总数
            elapsed_time: 执行时间
        """
        self.t += 1

        # 计算奖励（吞吐量）
        reward = generated_tokens / (elapsed_time + 1e-6)

        # 将context转换为向量
        context_vec = np.array([context])

        # 添加新的观测值到滑动窗口
        self._add_observation(arm_idx, context_vec, reward)

        # 重新计算线性模型参数
        self._update_arm_params(arm_idx)

        # print(f"LinUCBSlidingWindow Update: arm={arm_idx}, context={context}, "
        #       f"reward={reward:.2f}, window_size={len(self.arm_observations['contexts'][arm_idx])}")

    def get_window_stats(self, arm_idx: int) -> dict:
        """
        获取指定arm的滑动窗口统计信息（用于调试）
        
        Args:
            arm_idx: arm索引
            
        Returns:
            dict: 包含窗口统计信息的字典
        """
        contexts = self.arm_observations['contexts'][arm_idx]
        rewards = self.arm_observations['rewards'][arm_idx]

        if len(contexts) == 0:
            return {
                'window_size': 0,
                'mean_reward': 0.0,
                'mean_context': 0.0,
                'recent_rewards': [],
                'theta': self.arm_params['theta'][arm_idx].tolist()
            }

        mean_reward = np.mean(rewards)
        mean_context = np.mean([c[0] for c in contexts])  # 假设context_dim=1

        return {
            'window_size': len(contexts),
            'mean_reward': mean_reward,
            'mean_context': mean_context,
            'recent_rewards': rewards[-5:] if len(rewards) > 5 else rewards,
            'theta': self.arm_params['theta'][arm_idx].tolist()
        }

    def reset(self):
        """重置算法状态"""
        # 清空观测历史
        for k in range(self.K):
            self.arm_observations['contexts'][k].clear()
            self.arm_observations['rewards'][k].clear()

        # 重置线性模型参数
        for k in range(self.K):
            self.arm_params['A'][k] = np.eye(self.context_dim) * self.lambda_reg
            self.arm_params['b'][k] = np.zeros(self.context_dim)
            self.arm_params['theta'][k] = np.zeros(self.context_dim)

        self.t = 0
        self.round_robin = True
        print("LinUCBSlidingWindow: Reset completed")

    def save_state(self, filepath: str):
        """保存内部状态到文件"""
        state = {
            'arm_observations': self.arm_observations,
            'arm_params': self.arm_params,
            't': self.t,
            'K': self.K,
            'L': self.L,
            'context_dim': self.context_dim,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'window_size': self.window_size,
            'round_robin': self.round_robin,
            'spec_lengths': self.spec_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """从文件加载内部状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.arm_observations = state['arm_observations']
        self.arm_params = state['arm_params']
        self.t = state['t']
        self.K = state['K']
        self.L = state['L']
        self.context_dim = state['context_dim']
        self.alpha = state['alpha']
        self.lambda_reg = state['lambda_reg']
        self.window_size = state['window_size']
        self.round_robin = state['round_robin']
        self.spec_lengths = state['spec_lengths']

class QlearningSpec:
    def __init__(self, num_arms, confidence_param=0.5, max_spec_length=4,
                 learning_rate=0.1, discount_factor=0.95, epsilon=0.1,
                 context_bins=20, context_min=1, context_max=100):
        """
        Initialize the Q-learning algorithm for speculation length selection.
        
        Args:
            num_arms (int): Number of hyperparameter configurations (arms) 
            confidence_param (float): Not used in Q-learning, kept for interface compatibility
            max_spec_length (int): Maximum speculation length (L in the paper)
            learning_rate (float): Learning rate α for Q-learning updates
            discount_factor (float): Discount factor γ for future rewards
            epsilon (float): Epsilon for epsilon-greedy policy
            context_bins (int): Number of bins to discretize context space
            context_min (int): Minimum context value (batch size)
            context_max (int): Maximum context value (batch size)
        """
        self.K = num_arms
        self.delta = confidence_param  # Kept for interface compatibility
        self.L = max_spec_length
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.context_bins = context_bins
        self.context_min = context_min
        self.context_max = context_max

        # Initialize Q-table: Q[state][action]
        # States are discretized context bins, actions are speculation lengths
        self.Q = np.zeros((context_bins, num_arms))

        # State transition tracking for MDP
        self.last_state = None
        self.last_action = None

        # Statistics for interface compatibility
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards = np.zeros(num_arms)
        self.ucb_values = np.zeros(num_arms)  # Not used but kept for compatibility

        # Episode and time tracking
        self.t = 0
        self.episode_count = 0

        # Context binning
        self.context_bounds = np.linspace(context_min, context_max, context_bins + 1)

        print(f"QlearningSpec initialized: K={num_arms}, L={max_spec_length}, "
              f"α={learning_rate}, γ={discount_factor}, ε={epsilon}, "
              f"context_bins={context_bins}")

    def _get_state(self, context):
        """
        Map continuous context to discrete state.
        
        Args:
            context: Current context (batch size)
            
        Returns:
            int: Discrete state index
        """
        # return context
        # Ensure context is within bounds
        context = max(self.context_min, min(context, self.context_max))

        # Find the bin index
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _epsilon_greedy_policy(self, state):
        """
        Epsilon-greedy policy for action selection.
        
        Args:
            state (int): Current state index
            
        Returns:
            int: Selected action (arm index)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.K)
        else:
            # Exploit: greedy action
            q_values = self.Q[state]
            max_q = np.max(q_values)
            # 如果多个动作有相同的最大Q值，随机选择一个
            max_indices = np.where(q_values == max_q)[0]
            return np.random.choice(max_indices)

    def select_arm(self, context):
        """
        Select an arm according to the Q-learning algorithm.
        
        Args:
            context: Current context (batch size)
        
        Returns:
            int: Index of the selected arm
        """
        # Get current state
        current_state = self._get_state(context)

        # Select action using epsilon-greedy policy
        action = self._epsilon_greedy_policy(current_state)

        # Store current state and action for next update
        self.last_state = current_state
        self.last_action = action

        print(f"Q-learning: context={context}, state={current_state}, "
              f"Q_values={self.Q[current_state]}, selected_action={action}")

        return action

    def update(self, chosen_arm, context, generated_tokens: int, elapsed_time: float):
        """
        Update the Q-table after observing a reward.
        
        Args:
            chosen_arm (int): Index of the arm that was pulled
            context: Current context (batch size)
            generated_tokens (int): Number of generated tokens
            elapsed_time (float): Time taken for execution
        """
        self.t += 1

        # Calculate reward (throughput)
        reward = generated_tokens / (elapsed_time + 1e-6)

        # Update statistics for compatibility
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward

        # Get current state
        current_state = self._get_state(context)

        # Q-learning update
        if self.last_state is not None and self.last_action is not None:
            # Q-learning update formula:
            # Q(s,a) = Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]

            old_q_value = self.Q[self.last_state, self.last_action]
            max_next_q = np.max(self.Q[current_state])

            # TD error
            td_error = reward + self.gamma * max_next_q - old_q_value

            # Update Q-value
            self.Q[self.last_state, self.last_action] += self.alpha * td_error

            print(f"Q-learning Update: s={self.last_state}, a={self.last_action}, "
                  f"r={reward:.3f}, s'={current_state}, "
                  f"old_Q={old_q_value:.3f}, new_Q={self.Q[self.last_state, self.last_action]:.3f}, "
                  f"td_error={td_error:.3f}")

        # Update for next iteration
        self.last_state = current_state
        self.last_action = chosen_arm

    def reset(self):
        """Reset the algorithm's state."""
        self.Q = np.zeros((self.context_bins, self.K))
        self.arm_counts = np.zeros(self.K)
        self.arm_rewards = np.zeros(self.K)
        self.ucb_values = np.zeros(self.K)
        self.last_state = None
        self.last_action = None
        self.t = 0
        self.episode_count = 0
        print("Q-learning: Reset completed")

    def save_state(self, filepath):
        """Save the internal state to a file."""
        state = {
            'Q': self.Q,
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'ucb_values': self.ucb_values,
            'last_state': self.last_state,
            'last_action': self.last_action,
            't': self.t,
            'episode_count': self.episode_count,
            'K': self.K,
            'delta': self.delta,
            'L': self.L,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'context_bins': self.context_bins,
            'context_min': self.context_min,
            'context_max': self.context_max,
            'context_bounds': self.context_bounds,
        }
        print("save_state (Q-learning)", state, filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath):
        """Load the internal state from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print("load_state (Q-learning)", filepath, state)

        self.Q = state['Q']
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.last_state = state['last_state']
        self.last_action = state['last_action']
        self.t = state['t']
        self.episode_count = state['episode_count']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']
        self.alpha = state['alpha']
        self.gamma = state['gamma']
        self.epsilon = state['epsilon']
        self.context_bins = state['context_bins']
        self.context_min = state['context_min']
        self.context_max = state['context_max']
        self.context_bounds = state['context_bounds']

    def get_policy(self, context):
        """
        Get the current policy for a given context.
        
        Args:
            context: Context (batch size)
            
        Returns:
            dict: Policy information including Q-values and optimal action
        """
        state = self._get_state(context)
        q_values = self.Q[state]
        optimal_action = np.argmax(q_values)

        return {
            'state': state,
            'q_values': q_values.tolist(),
            'optimal_action': optimal_action,
            'epsilon': self.epsilon
        }

    def decay_epsilon(self, decay_rate=0.99, min_epsilon=0.01):
        """
        Decay epsilon for exploration-exploitation balance.
        
        Args:
            decay_rate (float): Rate of epsilon decay
            min_epsilon (float): Minimum epsilon value
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        print(f"Epsilon decayed to: {self.epsilon:.4f}")

    def get_q_table_summary(self):
        """
        Get a summary of the Q-table for debugging.
        
        Returns:
            dict: Q-table statistics
        """
        return {
            'q_table_shape': self.Q.shape,
            'q_table_mean': np.mean(self.Q),
            'q_table_max': np.max(self.Q),
            'q_table_min': np.min(self.Q),
            'non_zero_entries': np.count_nonzero(self.Q),
            'total_entries': self.Q.size
        }


class SarsaSpec:
    def __init__(self, num_arms, confidence_param=0.5, max_spec_length=4,
                 learning_rate=0.1, discount_factor=0.95, epsilon=0.1,
                 context_bins=20, context_min=1, context_max=100):
        """
        Initialize the SARSA algorithm for speculation length selection.
        
        SARSA is an on-policy TD learning algorithm that updates Q(s,a) using
        the actual next action taken, not the maximum Q-value.
        
        Args:
            num_arms (int): Number of hyperparameter configurations (arms)
            confidence_param (float): Not used in SARSA, kept for interface compatibility
            max_spec_length (int): Maximum speculation length (L in the paper)
            learning_rate (float): Learning rate α for SARSA updates
            discount_factor (float): Discount factor γ for future rewards
            epsilon (float): Epsilon for epsilon-greedy policy
            context_bins (int): Number of bins to discretize context space
            context_min (int): Minimum context value (batch size)
            context_max (int): Maximum context value (batch size)
        """
        self.K = num_arms
        self.delta = confidence_param  # Kept for interface compatibility
        self.L = max_spec_length
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.context_bins = context_bins
        self.context_min = context_min
        self.context_max = context_max

        # Initialize Q-table: Q[state][action]
        self.Q = np.zeros((context_bins, num_arms))

        # SARSA requires tracking current and next state-action pairs
        self.current_state = None
        self.current_action = None
        self.next_state = None
        self.next_action = None

        # Statistics for interface compatibility
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards = np.zeros(num_arms)
        self.ucb_values = np.zeros(num_arms)  # Not used but kept for compatibility

        # Episode and time tracking
        self.t = 0
        self.episode_count = 0

        # Context binning
        self.context_bounds = np.linspace(context_min, context_max, context_bins + 1)

        print(f"SarsaSpec initialized: K={num_arms}, L={max_spec_length}, "
              f"α={learning_rate}, γ={discount_factor}, ε={epsilon}, "
              f"context_bins={context_bins}")

    def _get_state(self, context):
        """
        Map continuous context to discrete state.
        
        Args:
            context: Current context (batch size)
            
        Returns:
            int: Discrete state index
        """
        # Ensure context is within bounds
        context = max(self.context_min, min(context, self.context_max))

        # Find the bin index
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _epsilon_greedy_policy(self, state):
        """
        Epsilon-greedy policy for action selection.
        
        Args:
            state (int): Current state index
            
        Returns:
            int: Selected action (arm index)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.K)
        else:
            # Exploit: greedy action
            return np.argmax(self.Q[state])

    def select_arm(self, context):
        """
        Select an arm according to the SARSA algorithm.
        
        Args:
            context: Current context (batch size)
        
        Returns:
            int: Index of the selected arm
        """
        # Get current state
        current_state = self._get_state(context)

        # Select action using epsilon-greedy policy
        action = self._epsilon_greedy_policy(current_state)

        # For SARSA, we need to track the sequence of states and actions
        # Shift the state-action pairs
        self.current_state = self.next_state
        self.current_action = self.next_action
        self.next_state = current_state
        self.next_action = action

        print(f"SARSA: context={context}, state={current_state}, "
              f"Q_values={self.Q[current_state]}, selected_action={action}")

        return action

    def update(self, chosen_arm, context, generated_tokens: int, elapsed_time: float):
        """
        Update the Q-table after observing a reward using SARSA.
        
        Args:
            chosen_arm (int): Index of the arm that was pulled
            context: Current context (batch size)
            generated_tokens (int): Number of generated tokens
            elapsed_time (float): Time taken for execution
        """
        self.t += 1

        # Calculate reward (throughput)
        reward = generated_tokens / (elapsed_time + 1e-6)

        # Update statistics for compatibility
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward

        # SARSA update
        if (self.current_state is not None and self.current_action is not None and
            self.next_state is not None and self.next_action is not None):

            # SARSA update formula:
            # Q(s,a) = Q(s,a) + α[r + γ * Q(s',a') - Q(s,a)]
            # where s' and a' are the actual next state and action

            old_q_value = self.Q[self.current_state, self.current_action]
            next_q_value = self.Q[self.next_state, self.next_action]

            # TD error
            td_error = reward + self.gamma * next_q_value - old_q_value

            # Update Q-value
            self.Q[self.current_state, self.current_action] += self.alpha * td_error

            print(f"SARSA Update: s={self.current_state}, a={self.current_action}, "
                  f"r={reward:.3f}, s'={self.next_state}, a'={self.next_action}, "
                  f"old_Q={old_q_value:.3f}, new_Q={self.Q[self.current_state, self.current_action]:.3f}, "
                  f"td_error={td_error:.3f}")

    def reset(self):
        """Reset the algorithm's state."""
        self.Q = np.zeros((self.context_bins, self.K))
        self.arm_counts = np.zeros(self.K)
        self.arm_rewards = np.zeros(self.K)
        self.ucb_values = np.zeros(self.K)
        self.current_state = None
        self.current_action = None
        self.next_state = None
        self.next_action = None
        self.t = 0
        self.episode_count = 0
        print("SARSA: Reset completed")

    def save_state(self, filepath):
        """Save the internal state to a file."""
        state = {
            'Q': self.Q,
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'ucb_values': self.ucb_values,
            'current_state': self.current_state,
            'current_action': self.current_action,
            'next_state': self.next_state,
            'next_action': self.next_action,
            't': self.t,
            'episode_count': self.episode_count,
            'K': self.K,
            'delta': self.delta,
            'L': self.L,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'context_bins': self.context_bins,
            'context_min': self.context_min,
            'context_max': self.context_max,
            'context_bounds': self.context_bounds,
        }
        print("save_state (SARSA)", state, filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath):
        """Load the internal state from a file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print("load_state (SARSA)", filepath, state)

        self.Q = state['Q']
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.current_state = state['current_state']
        self.current_action = state['current_action']
        self.next_state = state['next_state']
        self.next_action = state['next_action']
        self.t = state['t']
        self.episode_count = state['episode_count']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']
        self.alpha = state['alpha']
        self.gamma = state['gamma']
        self.epsilon = state['epsilon']
        self.context_bins = state['context_bins']
        self.context_min = state['context_min']
        self.context_max = state['context_max']
        self.context_bounds = state['context_bounds']

    def get_policy(self, context):
        """
        Get the current policy for a given context.
        
        Args:
            context: Context (batch size)
            
        Returns:
            dict: Policy information including Q-values and optimal action
        """
        state = self._get_state(context)
        q_values = self.Q[state]
        optimal_action = np.argmax(q_values)

        return {
            'state': state,
            'q_values': q_values.tolist(),
            'optimal_action': optimal_action,
            'epsilon': self.epsilon
        }

    def decay_epsilon(self, decay_rate=0.99, min_epsilon=0.01):
        """
        Decay epsilon for exploration-exploitation balance.
        
        Args:
            decay_rate (float): Rate of epsilon decay
            min_epsilon (float): Minimum epsilon value
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        print(f"Epsilon decayed to: {self.epsilon:.4f}")

    def get_q_table_summary(self):
        """
        Get a summary of the Q-table for debugging.
        
        Returns:
            dict: Q-table statistics
        """
        return {
            'q_table_shape': self.Q.shape,
            'q_table_mean': np.mean(self.Q),
            'q_table_max': np.max(self.Q),
            'q_table_min': np.min(self.Q),
            'non_zero_entries': np.count_nonzero(self.Q),
            'total_entries': self.Q.size
        }


class DQNSpec:
    def __init__(self, num_arms, confidence_param=0.5, max_spec_length=4,
                 learning_rate=0.001, discount_factor=0.95, epsilon=0.1,
                 context_bins=20, context_min=1, context_max=100,
                 hidden_size=64, memory_size=10000, batch_size=32, target_update=100):
        """
        Initialize the DQN algorithm for speculation length selection.
        
        Args:
            num_arms (int): Number of hyperparameter configurations (arms)
            confidence_param (float): Not used in DQN, kept for interface compatibility
            max_spec_length (int): Maximum speculation length (L in the paper)
            learning_rate (float): Learning rate for neural network
            discount_factor (float): Discount factor γ for future rewards
            epsilon (float): Epsilon for epsilon-greedy policy
            context_bins (int): Number of bins to discretize context space
            context_min (int): Minimum context value (batch size)
            context_max (int): Maximum context value (batch size)
            hidden_size (int): Size of hidden layers in neural network
            memory_size (int): Size of replay memory
            batch_size (int): Batch size for training
            target_update (int): Frequency of target network updates
        """
        self.K = num_arms
        self.delta = confidence_param  # Kept for interface compatibility
        self.L = max_spec_length
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.context_bins = context_bins
        self.context_min = context_min
        self.context_max = context_max
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update

        # Neural network for Q-function approximation
        self.q_network = QNetwork(context_bins, num_arms, hidden_size)
        self.target_network = QNetwork(context_bins, num_arms, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = deque(maxlen=memory_size)

        # State transition tracking
        self.last_state = None
        self.last_action = None

        # Statistics for interface compatibility
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards = np.zeros(num_arms)
        self.ucb_values = np.zeros(num_arms)  # Not used but kept for compatibility

        # Episode and time tracking
        self.t = 0
        self.episode_count = 0

        # Context binning
        self.context_bounds = np.linspace(context_min, context_max, context_bins + 1)

        print(f"DQNSpec initialized: K={num_arms}, L={max_spec_length}, "
              f"α={learning_rate}, γ={discount_factor}, ε={epsilon}, "
              f"hidden_size={hidden_size}, memory_size={memory_size}")

    def _get_state(self, context):
        """
        Map continuous context to discrete state.
        
        Args:
            context: Current context (batch size)
            
        Returns:
            int: Discrete state index
        """
        # Ensure context is within bounds
        context = max(self.context_min, min(context, self.context_max))

        # Find the bin index
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _epsilon_greedy_policy(self, state):
        """
        Epsilon-greedy policy for action selection.
        
        Args:
            state (int): Current state index
            
        Returns:
            int: Selected action (arm index)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.K)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                state_tensor = torch.LongTensor([state])
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def _store_transition(self, state, action, reward, next_state):
        """
        Store transition in replay memory.
        
        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state
        """
        self.memory.append((state, action, reward, next_state))

    def _train_network(self):
        """
        Train the Q-network using experience replay.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from replay memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors
        states = torch.LongTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.LongTensor(next_states)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]

        # Target Q values
        target_q_values = rewards + self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def select_arm(self, context):
        """
        Select an arm according to the DQN algorithm.
        
        Args:
            context: Current context (batch size)
        
        Returns:
            int: Index of the selected arm
        """
        # Get current state
        begin_time = time.time()
        current_state = self._get_state(context)

        # Select action using epsilon-greedy policy
        action = self._epsilon_greedy_policy(current_state)

        # Store current state and action for next update
        self.last_state = current_state
        self.last_action = action
        end_time = time.time()
        print(f"DQN: context={context}, state={current_state}, selected_action={action}, "
              f"time={end_time - begin_time:.6f}s")

        return action

    def update(self, chosen_arm, context, generated_tokens: int, elapsed_time: float):
        """
        Update the DQN after observing a reward.
        
        Args:
            chosen_arm (int): Index of the arm that was pulled
            context: Current context (batch size)
            generated_tokens (int): Number of generated tokens
            elapsed_time (float): Time taken for execution
        """
        self.t += 1

        # Calculate reward (throughput)
        reward = generated_tokens / (elapsed_time + 1e-6)
        time_start = time.time()
        # Update statistics for compatibility
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward

        # Get current state
        current_state = self._get_state(context)

        # Store transition in replay memory
        if self.last_state is not None and self.last_action is not None:
            self._store_transition(self.last_state, self.last_action, reward, current_state)

        # Train the network
        loss = self._train_network()

        # Update target network periodically
        if self.t % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"DQN: Target network updated at step {self.t}")
        time_end = time.time()
        # Update for next iteration
        self.last_state = current_state
        self.last_action = chosen_arm

        if loss is not None:
            print(f"DQN Update: s={self.last_state}, a={chosen_arm}, "
                  f"r={reward:.3f}, s'={current_state}, loss={loss:.6f}, "
                  f"time={time_end - time_start:.6f}s")

    def reset(self):
        """Reset the algorithm's state."""
        self.q_network = QNetwork(self.context_bins, self.K, self.hidden_size)
        self.target_network = QNetwork(self.context_bins, self.K, self.hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.memory.clear()
        self.arm_counts = np.zeros(self.K)
        self.arm_rewards = np.zeros(self.K)
        self.ucb_values = np.zeros(self.K)
        self.last_state = None
        self.last_action = None
        self.t = 0
        self.episode_count = 0
        print("DQN: Reset completed")

    def save_state(self, filepath):
        """Save the internal state to a file."""
        state = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': list(self.memory),
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'ucb_values': self.ucb_values,
            'last_state': self.last_state,
            'last_action': self.last_action,
            't': self.t,
            'episode_count': self.episode_count,
            'K': self.K,
            'delta': self.delta,
            'L': self.L,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'context_bins': self.context_bins,
            'context_min': self.context_min,
            'context_max': self.context_max,
            'context_bounds': self.context_bounds,
            'hidden_size': self.hidden_size,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'target_update': self.target_update,
        }
        print("save_state (DQN)", filepath)
        torch.save(state, filepath)

    def load_state(self, filepath):
        """Load the internal state from a file."""
        state = torch.load(filepath, weights_only=False)
        print("load_state (DQN)", filepath)

        self.q_network.load_state_dict(state['q_network_state_dict'])
        self.target_network.load_state_dict(state['target_network_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.memory = deque(state['memory'], maxlen=self.memory_size)
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.last_state = state['last_state']
        self.last_action = state['last_action']
        self.t = state['t']
        self.episode_count = state['episode_count']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']
        self.alpha = state['alpha']
        self.gamma = state['gamma']
        self.epsilon = state['epsilon']
        self.context_bins = state['context_bins']
        self.context_min = state['context_min']
        self.context_max = state['context_max']
        self.context_bounds = state['context_bounds']
        self.hidden_size = state['hidden_size']
        self.memory_size = state['memory_size']
        self.batch_size = state['batch_size']
        self.target_update = state['target_update']

    def get_policy(self, context):
        """
        Get the current policy for a given context.
        
        Args:
            context: Context (batch size)
            
        Returns:
            dict: Policy information including Q-values and optimal action
        """
        state = self._get_state(context)
        with torch.no_grad():
            state_tensor = torch.LongTensor([state])
            q_values = self.q_network(state_tensor)
            optimal_action = q_values.argmax().item()

        return {
            'state': state,
            'q_values': q_values.numpy().tolist(),
            'optimal_action': optimal_action,
            'epsilon': self.epsilon
        }

    def decay_epsilon(self, decay_rate=0.99, min_epsilon=0.01):
        """
        Decay epsilon for exploration-exploitation balance.
        
        Args:
            decay_rate (float): Rate of epsilon decay
            min_epsilon (float): Minimum epsilon value
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        print(f"Epsilon decayed to: {self.epsilon:.4f}")


class QNetwork(nn.Module):
    """
    Neural network for Q-function approximation.
    """
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.embedding = nn.Embedding(state_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.embedding(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PPOSpec:
    def __init__(self, num_arms, confidence_param=0.5, max_spec_length=4,
                 learning_rate=0.0003, discount_factor=0.95, epsilon=0.2,
                 context_bins=200, context_min=1, context_max=200,
                 hidden_size=64, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        """
        Initialize the PPO algorithm for speculation length selection.
        
        Args:
            num_arms (int): Number of hyperparameter configurations (arms)
            confidence_param (float): Not used in PPO, kept for interface compatibility
            max_spec_length (int): Maximum speculation length (L in the paper)
            learning_rate (float): Learning rate for neural network
            discount_factor (float): Discount factor γ for future rewards
            epsilon (float): Not used in PPO, kept for interface compatibility
            context_bins (int): Number of bins to discretize context space
            context_min (int): Minimum context value (batch size)
            context_max (int): Maximum context value (batch size)
            hidden_size (int): Size of hidden layers in neural network
            clip_ratio (float): PPO clip ratio
            value_coef (float): Value function coefficient
            entropy_coef (float): Entropy coefficient for exploration
        """
        self.K = num_arms
        self.delta = confidence_param  # Kept for interface compatibility
        self.L = max_spec_length
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # Not used in PPO
        self.context_bins = context_bins
        self.context_min = context_min
        self.context_max = context_max
        self.hidden_size = hidden_size
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Neural networks for policy and value function
        self.policy_network = PolicyNetwork(context_bins, num_arms, hidden_size)
        self.value_network = ValueNetwork(context_bins, hidden_size)

        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.policy_network.parameters(), 'lr': learning_rate},
            {'params': self.value_network.parameters(), 'lr': learning_rate}
        ])

        # Experience buffer
        self.experience_buffer = []

        # Statistics for interface compatibility
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards = np.zeros(num_arms)
        self.ucb_values = np.zeros(num_arms)  # Not used but kept for compatibility

        # Episode and time tracking
        self.t = 0
        self.episode_count = 0

        # Context binning
        self.context_bounds = np.linspace(context_min, context_max, context_bins + 1)

        print(f"PPOSpec initialized: K={num_arms}, L={max_spec_length}, "
              f"α={learning_rate}, γ={discount_factor}, "
              f"hidden_size={hidden_size}, clip_ratio={clip_ratio}")

    def _get_state(self, context):
        """
        Map continuous context to discrete state.
        
        Args:
            context: Current context (batch size)
            
        Returns:
            int: Discrete state index
        """
        # return context
        # Ensure context is within bounds
        context = max(self.context_min, min(context, self.context_max))

        # Find the bin index
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def select_arm(self, context):
        """
        Select an arm according to the PPO algorithm.
        
        Args:
            context: Current context (batch size)
        
        Returns:
            int: Index of the selected arm
        """
        # Get current state
        time_start = time.time()
        current_state = self._get_state(context)

        # Get action probabilities from policy network
        with torch.no_grad():
            state_tensor = torch.LongTensor([current_state])
            action_probs = F.softmax(self.policy_network(state_tensor), dim=1)
            # action = torch.argmax(action_probs, dim=1).item()  # 利用
            # 对于小batch size，排除action 0
            if context < 100:
                # 创建掩码，将action 0的概率设为0
                mask = torch.ones_like(action_probs)
                mask[0, 0] = 0  # 将第一个action的概率设为0

                # 应用掩码并重新归一化
                masked_probs = action_probs * mask
                masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)

                action = torch.multinomial(masked_probs, 1).item()
            else:
                action = torch.multinomial(action_probs, 1).item()

        time_end = time.time()
        print(f"PPO: context={context}, state={current_state}, "
              f"action_probs={action_probs.numpy()}, selected_action={action}, "
              f"time={time_end - time_start:.6f}s")

        return action

    def update(self, chosen_arm, context, generated_tokens: int, elapsed_time: float):
        """
        Update the PPO after observing a reward.
        
        Args:
            chosen_arm (int): Index of the arm that was pulled
            context: Current context (batch size)
            generated_tokens (int): Number of generated tokens
            elapsed_time (float): Time taken for execution
        """
        self.t += 1

        # Calculate reward (throughput)
        reward = generated_tokens / (elapsed_time + 1e-6)

        # Update statistics for compatibility
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward

        # Get current state
        current_state = self._get_state(context)

        # Store experience
        self.experience_buffer.append({
            'state': current_state,
            'action': chosen_arm,
            'reward': reward,
            'next_state': current_state  # In this case, state doesn't change
        })

        # Train periodically (every 10 steps)
        if len(self.experience_buffer) >= 10:
            time_start = time.time()
            self._train_network()
            time_end = time.time()
            print(f"PPO Update: s={current_state}, a={chosen_arm}, r={reward:.3f}, "
                  f"time={time_end - time_start:.6f}s")
            self.experience_buffer.clear()

        print(f"PPO Update: s={current_state}, a={chosen_arm}, r={reward:.3f}")

    def _train_network(self):
        """
        Train the policy and value networks using PPO.
        """
        if len(self.experience_buffer) == 0:
            return

        # Prepare batch data
        states = torch.LongTensor([exp['state'] for exp in self.experience_buffer])
        actions = torch.LongTensor([exp['action'] for exp in self.experience_buffer])
        rewards = torch.FloatTensor([exp['reward'] for exp in self.experience_buffer])

        # Compute returns
        returns = self._compute_returns(rewards)

        # Get current policy and value
        with torch.no_grad():
            action_probs = F.softmax(self.policy_network(states), dim=1)
            values = self.value_network(states).squeeze()

        # Get action probabilities for taken actions
        action_probs_taken = action_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute advantages
        advantages = returns - values.detach()

        # PPO update
        for _ in range(3):  # Multiple epochs
            # Get new action probabilities
            new_action_probs = F.softmax(self.policy_network(states), dim=1)
            new_action_probs_taken = new_action_probs.gather(1, actions.unsqueeze(1)).squeeze()

            # Compute ratio
            ratio = new_action_probs_taken / (action_probs_taken + 1e-8)

            # Compute clipped ratio
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            # Policy loss
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # Value loss
            new_values = self.value_network(states).squeeze()
            value_loss = F.mse_loss(new_values, returns)

            # Entropy loss for exploration
            entropy = -(new_action_probs * torch.log(new_action_probs + 1e-8)).sum(dim=1).mean()

            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            # Print loss values
            print(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, "
                  f"Entropy: {entropy.item():.4f}, Total Loss: {total_loss.item():.4f}")

    def _compute_returns(self, rewards):
        """
        Compute discounted returns.
        
        Args:
            rewards (torch.Tensor): Rewards tensor
            
        Returns:
            torch.Tensor: Discounted returns
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def reset(self):
        """Reset the algorithm's state."""
        self.policy_network = PolicyNetwork(self.context_bins, self.K, self.hidden_size)
        self.value_network = ValueNetwork(self.context_bins, self.hidden_size)
        self.optimizer = optim.Adam([
            {'params': self.policy_network.parameters(), 'lr': self.alpha},
            {'params': self.value_network.parameters(), 'lr': self.alpha}
        ])
        self.experience_buffer.clear()
        self.arm_counts = np.zeros(self.K)
        self.arm_rewards = np.zeros(self.K)
        self.ucb_values = np.zeros(self.K)
        self.t = 0
        self.episode_count = 0
        print("PPO: Reset completed")

    def save_state(self, filepath):
        """Save the internal state to a file."""
        state = {
            'policy_network_state_dict': self.policy_network.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'experience_buffer': self.experience_buffer,
            'arm_counts': self.arm_counts,
            'arm_rewards': self.arm_rewards,
            'ucb_values': self.ucb_values,
            't': self.t,
            'episode_count': self.episode_count,
            'K': self.K,
            'delta': self.delta,
            'L': self.L,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'context_bins': self.context_bins,
            'context_min': self.context_min,
            'context_max': self.context_max,
            'context_bounds': self.context_bounds,
            'hidden_size': self.hidden_size,
            'clip_ratio': self.clip_ratio,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
        }
        print("save_state (PPO)", filepath)
        torch.save(state, filepath)

    def load_state(self, filepath):
        """Load the internal state from a file."""
        state = torch.load(filepath, weights_only=False)
        print("load_state (PPO)", filepath)

        self.policy_network.load_state_dict(state['policy_network_state_dict'])
        self.value_network.load_state_dict(state['value_network_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.experience_buffer = state['experience_buffer']
        self.arm_counts = state['arm_counts']
        self.arm_rewards = state['arm_rewards']
        self.ucb_values = state['ucb_values']
        self.t = state['t']
        self.episode_count = state['episode_count']
        self.K = state['K']
        self.delta = state['delta']
        self.L = state['L']
        self.alpha = state['alpha']
        self.gamma = state['gamma']
        self.epsilon = state['epsilon']
        self.context_bins = state['context_bins']
        self.context_min = state['context_min']
        self.context_max = state['context_max']
        self.context_bounds = state['context_bounds']
        self.hidden_size = state['hidden_size']
        self.clip_ratio = state['clip_ratio']
        self.value_coef = state['value_coef']
        self.entropy_coef = state['entropy_coef']

    def get_policy(self, context):
        """
        Get the current policy for a given context.
        
        Args:
            context: Context (batch size)
            
        Returns:
            dict: Policy information including action probabilities and optimal action
        """
        state = self._get_state(context)
        with torch.no_grad():
            state_tensor = torch.LongTensor([state])
            action_logits = self.policy_network(state_tensor)
            action_probs = F.softmax(action_logits, dim=1)
            optimal_action = action_probs.argmax().item()

        return {
            'state': state,
            'action_probs': action_probs.numpy().tolist(),
            'optimal_action': optimal_action,
            'epsilon': self.epsilon
        }

    def decay_epsilon(self, decay_rate=0.99, min_epsilon=0.01):
        """
        Decay epsilon for exploration-exploitation balance.
        Note: PPO uses entropy regularization for exploration, not epsilon-greedy.
        
        Args:
            decay_rate (float): Rate of epsilon decay
            min_epsilon (float): Minimum epsilon value
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        print(f"Epsilon decayed to: {self.epsilon:.4f} (Note: PPO uses entropy regularization)")


class PolicyNetwork(nn.Module):
    """
    Neural network for policy function.
    """
    def __init__(self, state_size, action_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(state_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.embedding(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ValueNetwork(nn.Module):
    """
    Neural network for value function.
    """
    def __init__(self, state_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.embedding = nn.Embedding(state_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.embedding(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DASpecWithExploration:
    def __init__(self, model, draft_model, max_proposed_length=5,
                 exploration_rate=0.1, exploration_decay=0.99, min_exploration_rate=0.01,
                 context_bins=200, context_min=1, context_max=200,
                 consecutive_threshold=5, forced_exploration_arms=[2]):
        """
        初始化带探索机制的DASpec包装器
        
        Args:
            model: 模型路径，用于计算执行时间
            draft_model: 草稿模型路径
            max_proposed_length: 最大推测长度
            exploration_rate: 初始探索概率
            exploration_decay: 探索概率衰减率
            min_exploration_rate: 最小探索概率
            context_bins: 上下文分箱数量
            context_min: 最小上下文值
            context_max: 最大上下文值
            consecutive_threshold: 连续选择同一arm的阈值
            forced_exploration_arms: 强制探索的候选arm列表
        """
        # 初始化原始的DASpec
        self.daspec = DASpec(model, draft_model, max_proposed_length)

        # 探索参数
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # 上下文分箱
        self.context_bins = context_bins
        self.context_min = context_min
        self.context_max = context_max
        self.context_bounds = np.linspace(context_min, context_max, context_bins + 1)

        # 每个arm-context组合的统计信息
        self.arm_stats = {
            'counts': np.zeros((max_proposed_length + 1, context_bins)),  # 选择次数
            'rewards': np.zeros((max_proposed_length + 1, context_bins)),  # 累积奖励
            'ucb_values': np.zeros((max_proposed_length + 1, context_bins)),  # UCB值
        }

        # 连续选择检测
        self.consecutive_threshold = consecutive_threshold
        self.forced_exploration_arms = forced_exploration_arms
        self.consecutive_selections = {}  # {context_bin: {'last_arm': arm, 'count': count}}
        self.performance_history = {
            'avg_rewards': np.zeros((max_proposed_length + 1, context_bins)),  # 平均奖励
            'forced_exploration_results': {}  # 强制探索结果记录
        }

        # 时间步计数
        self.t = 0

        print(f"DASpecWithExploration initialized: max_proposed_length={max_proposed_length}, "
              f"exploration_rate={exploration_rate}, context_bins={context_bins}, "
              f"consecutive_threshold={consecutive_threshold}, forced_exploration_arms={forced_exploration_arms}")

    def change_prev_alphas(self,prev_alphas):
        self.daspec.change_prev_alphas(prev_alphas)
    def online_correction_factor(self,k,batch_size,actual_accepted_tokens):
        return self.daspec.online_correction_factor(k,batch_size,actual_accepted_tokens)
    def chage_table(self,k,batch_size,new):
        self.daspec.chage_table(k,batch_size,new)
    def _get_context_bin(self, context: int) -> int:
        """
        将连续上下文映射到离散的bin索引
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            bin索引
        """
        # 确保context在有效范围内
        context = max(self.context_min, min(context, self.context_max))

        # 找到对应的bin
        bin_idx = np.digitize(context, self.context_bounds) - 1
        return max(0, min(bin_idx, self.context_bins - 1))

    def _update_ucb_values(self, context_bin: int):
        """
        更新指定上下文bin的UCB值
        
        Args:
            context_bin: 上下文bin索引
        """
        for arm in range(self.daspec.max_proposed_length + 1):
            count = self.arm_stats['counts'][arm, context_bin]
            if count > 0:
                # 计算经验均值
                mean_reward = self.arm_stats['rewards'][arm, context_bin] / count

                # 计算置信区间（使用UCB1公式）
                confidence_radius = np.sqrt(2 * np.log(self.t + 1) / count)

                # 更新UCB值
                self.arm_stats['ucb_values'][arm, context_bin] = mean_reward + confidence_radius
            else:
                # 如果arm还没有被选择过，设置UCB为无穷大
                self.arm_stats['ucb_values'][arm, context_bin] = float('inf')

    def optimize_proposed_length(self, context_length, batch_size, speculative_metrics=None):
        """
        优化推测长度，结合DASpec的预测和探索机制
        
        Args:
            context_length: 上下文长度
            batch_size: 批处理大小
            speculative_metrics: 推测指标
            
        Returns:
            最优的推测长度
        """
        self.t += 1
        context_bin = self._get_context_bin(batch_size)

        # 更新UCB值
        self._update_ucb_values(context_bin)

        # 检查是否需要强制探索
        forced_exploration_needed = self._check_forced_exploration(context_bin)

        if forced_exploration_needed:
            # 强制探索：选择强制探索候选arm中的一个
            selected_arm = self._select_forced_exploration_arm(context_bin)
            print(f"DASpecWithExploration (forced_exploration): context={batch_size}, bin={context_bin}, "
                  f"selected_arm={selected_arm}, consecutive_count={self.consecutive_selections[context_bin]['count']}")
        elif np.random.random() < self.exploration_rate:
            # 探索：优先选择未尝试的arm
            unexplored_arms = np.where(self.arm_stats['counts'][:, context_bin] == 0)[0]
            if len(unexplored_arms) > 0:
                # 随机选择一个未尝试的arm
                selected_arm = np.random.choice(unexplored_arms)
                print(f"DASpecWithExploration (explore): context={batch_size}, bin={context_bin}, "
                      f"selected_unexplored_arm={selected_arm}")
            else:
                # 如果所有arm都尝试过了，随机选择一个
                selected_arm = np.random.randint(0, self.daspec.max_proposed_length + 1)
                print(f"DASpecWithExploration (explore): context={batch_size}, bin={context_bin}, "
                      f"selected_random_arm={selected_arm}")
        else:
            # 利用：结合DASpec预测和UCB值
            # 获取DASpec的预测结果
            daspec_prediction = self.daspec.optimize_proposed_length(
                context_length, batch_size, speculative_metrics
            )

            # 获取当前上下文bin的UCB值
            ucb_values = self.arm_stats['ucb_values'][:, context_bin]

            # 检查是否应该使用强制探索的结果
            selected_arm = self._check_forced_exploration_performance(context_bin, daspec_prediction, ucb_values)

            if selected_arm is None:
                # 如果DASpec预测的arm的UCB值不是最低的，选择UCB值最高的arm
                if ucb_values[daspec_prediction] < np.max(ucb_values):
                    # 选择UCB值最高的arm
                    selected_arm = np.argmax(ucb_values)
                    print(f"DASpecWithExploration (exploit): context={batch_size}, bin={context_bin}, "
                          f"daspec_prediction={daspec_prediction}, ucb_best={selected_arm}, "
                          f"ucb_values={ucb_values}")
                else:
                    # 使用DASpec的预测
                    selected_arm = daspec_prediction
                    print(f"DASpecWithExploration (exploit): context={batch_size}, bin={context_bin}, "
                          f"using_daspec_prediction={selected_arm}")
            else:
                print(f"DASpecWithExploration (exploit): context={batch_size}, bin={context_bin}, "
                      f"using_forced_exploration_result={selected_arm}")

        # 更新连续选择计数
        self._update_consecutive_selections(context_bin, selected_arm)

        return selected_arm

    def update_with_feedback(self, context_length, batch_size, proposed_length,
                           actual_draft_time, actual_verify_time, num_accepted_tokens):
        """
        更新反馈信息
        
        Args:
            context_length: 上下文长度
            batch_size: 批处理大小
            proposed_length: 推测长度
            actual_draft_time: 实际草稿时间
            actual_verify_time: 实际验证时间
            num_accepted_tokens: 接受的token数量
        """

        # 计算奖励（吞吐量）
        total_time = actual_draft_time + actual_verify_time
        reward = num_accepted_tokens / (total_time + 1e-6)

        # 更新统计信息
        context_bin = self._get_context_bin(batch_size)
        self.arm_stats['counts'][proposed_length, context_bin] += 1
        self.arm_stats['rewards'][proposed_length, context_bin] += reward

        # 更新平均奖励
        count = self.arm_stats['counts'][proposed_length, context_bin]
        self.performance_history['avg_rewards'][proposed_length, context_bin] = (
            self.arm_stats['rewards'][proposed_length, context_bin] / count
        )

        # 如果是强制探索的arm，记录其性能
        if proposed_length in self.forced_exploration_arms:
            key = f"{context_bin}"
            if key not in self.performance_history['forced_exploration_results']:
                self.performance_history['forced_exploration_results'][key] = {}

            self.performance_history['forced_exploration_results'][key][proposed_length] = (
                self.performance_history['avg_rewards'][proposed_length, context_bin]
            )

            print(f"DASpecWithExploration: Forced exploration arm {proposed_length} performance updated: "
                  f"context={batch_size}, reward={reward:.3f}, avg_reward={self.performance_history['avg_rewards'][proposed_length, context_bin]:.3f}")

        # 衰减探索概率
        # self.exploration_rate = max(self.min_exploration_rate,
        #                            self.exploration_rate * self.exploration_decay)

        # print(f"DASpecWithExploration Update: arm={proposed_length}, context={batch_size}, "
        #       f"reward={reward:.3f}, exploration_rate={self.exploration_rate:.4f}")

    def get_exploration_stats(self, context: int) -> dict:
        """
        获取指定上下文的探索统计信息
        
        Args:
            context: 请求量（batch size）
            
        Returns:
            dict: 包含统计信息的字典
        """
        context_bin = self._get_context_bin(context)
        stats = {}

        for arm in range(self.daspec.max_proposed_length + 1):
            count = self.arm_stats['counts'][arm, context_bin]
            if count > 0:
                mean_reward = self.arm_stats['rewards'][arm, context_bin] / count
                ucb_value = self.arm_stats['ucb_values'][arm, context_bin]
            else:
                mean_reward = 0.0
                ucb_value = float('inf')

            stats[f'arm_{arm}'] = {
                'count': int(count),
                'mean_reward': mean_reward,
                'ucb_value': ucb_value
            }

        return {
            'context': context,
            'context_bin': context_bin,
            'exploration_rate': self.exploration_rate,
            'arm_stats': stats
        }
    def reset(self):
        """重置算法状态"""
        # 重置DASpec
        self.daspec = DASpec(self.daspec.model, self.daspec.draft_model, self.daspec.max_proposed_length)

        # 重置探索统计
        self.arm_stats['counts'].fill(0)
        self.arm_stats['rewards'].fill(0)
        self.arm_stats['ucb_values'].fill(0)

        # 重置时间步和探索概率
        self.t = 0
        self.exploration_rate = 0.1  # 重置为初始值

        print("DASpecWithExploration: Reset completed")

    def save_state(self, filepath: str):
        pass

    def load_state(self, filepath: str):
        pass

    def _check_forced_exploration(self, context_bin):
        """
        检查是否需要强制探索
        
        Args:
            context_bin: 上下文bin索引
            
        Returns:
            bool: 是否需要强制探索
        """
        if context_bin not in self.consecutive_selections:
            return False

        consecutive_info = self.consecutive_selections[context_bin]
        return consecutive_info['count'] >= self.consecutive_threshold

    def _select_forced_exploration_arm(self, context_bin):
        """
        选择强制探索的arm
        
        Args:
            context_bin: 上下文bin索引
            
        Returns:
            int: 选择的arm索引
        """
        # 优先选择强制探索候选arm中未尝试或尝试较少的
        best_arm = None
        min_count = float('inf')

        for arm in self.forced_exploration_arms:
            if arm <= self.daspec.max_proposed_length:
                count = self.arm_stats['counts'][arm, context_bin]
                if count < min_count:
                    min_count = count
                    best_arm = arm

        # 如果没有找到合适的arm，随机选择一个
        if best_arm is None:
            best_arm = np.random.choice(self.forced_exploration_arms)

        return best_arm

    def _check_forced_exploration_performance(self, context_bin, daspec_prediction, ucb_values):
        """
        检查强制探索的性能，如果表现更好则优先选择
        
        Args:
            context_bin: 上下文bin索引
            daspec_prediction: DASpec预测的arm
            ucb_values: UCB值数组
            
        Returns:
            int or None: 如果有更好的强制探索结果返回arm索引，否则返回None
        """
        # 检查强制探索历史结果
        key = f"{context_bin}"
        if key in self.performance_history['forced_exploration_results']:
            forced_results = self.performance_history['forced_exploration_results'][key]

            # 获取当前预测arm的平均奖励
            current_avg_reward = self.performance_history['avg_rewards'][daspec_prediction, context_bin]

            # 检查是否有强制探索的arm表现更好
            best_forced_arm = None
            best_forced_reward = current_avg_reward

            for arm, avg_reward in forced_results.items():
                if avg_reward > best_forced_reward:
                    best_forced_reward = avg_reward
                    best_forced_arm = arm

            if best_forced_arm is not None:
                return best_forced_arm

        return None

    def _update_consecutive_selections(self, context_bin, selected_arm):
        """
        更新连续选择计数
        
        Args:
            context_bin: 上下文bin索引
            selected_arm: 选择的arm
        """
        if context_bin not in self.consecutive_selections:
            self.consecutive_selections[context_bin] = {'last_arm': selected_arm, 'count': 1}
        else:
            if self.consecutive_selections[context_bin]['last_arm'] == selected_arm:
                self.consecutive_selections[context_bin]['count'] += 1
            else:
                self.consecutive_selections[context_bin] = {'last_arm': selected_arm, 'count': 1}