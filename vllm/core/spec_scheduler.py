import numpy as np
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple
from collections import deque
from joblib import load
import math

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

        # start = time.time()
        # results = list(process_pool.map(goodput_estimation_unpack, args_list))
        # print("并行总耗时", time.time() - start)
        # results = list(process_pool.map(goodput_estimation_unpack, args_list))
        for k, goodput in results:
            if goodput > best_goodput:
                best_goodput = goodput
                best_length = k
        return best_length

class ContextualUCBSpec:
    def __init__(self, num_arms, max_spec_length=4, delta=0.01):
        self.K = num_arms
        self.L = max_spec_length
        self.delta = delta
        self.t = 0
        # 每个arm维护一个context->(sum_reward, count)的字典
        self.arm_models = [{} for _ in range(num_arms)]
        self.round_robin = True
    
    def select_arm(self, context):
        # context: 当前请求量（如batch size）
        ucb_values = []
        for i in range(self.K):
            model = self.arm_models[i]
            if context in model and model[context][1] > 0:
                mean = model[context][0] / model[context][1]
                n = model[context][1]
                cr = self.L/2 * np.sqrt((1+n)/(n**2) * (1 + 2*np.log(self.K*self.t**2*(1+n)**0.5/self.delta)))
                ucb = mean + cr
            else:
                ucb = float('inf')  # 未探索过的context-arm组合
            ucb_values.append(ucb)
        print("ucb_values",ucb_values)
        return np.argmax(ucb_values)
    
    def update(self, arm_idx, context, generated_tokens, elapsed_time):
        self.t += 1
        reward = generated_tokens / (elapsed_time + 1e-6)
        model = self.arm_models[arm_idx]
        if context not in model:
            model[context] = [0.0, 0]
        model[context][0] += reward
        model[context][1] += 1

class UCBSpec:
    def __init__(self, num_arms, confidence_param=0.01, max_spec_length=4):
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
        self.round_robin = True
        # Track the total number of rounds
        self.t = 0
        
    def select_arm(self, context):
        """
        Select an arm according to the UCBSPEC algorithm.
        
        Returns:
            int: Index of the selected arm
        """
        if self.round_robin:
            # Round-robin for the first K rounds
            return  self.t % self.K
        else:
            # Select arm with highest UCB value
            print("self.ucb_values",self.ucb_values,self.arm_counts,self.arm_rewards)
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
        self.arm_rewards[chosen_arm] += reward
        
        # Update UCB values for all arms
        for i in range(self.K):
            if self.arm_counts[i] > 0:
                # Calculate empirical mean
                mu_hat = self.arm_rewards[i] / self.arm_counts[i]
                
                # Calculate confidence radius
                n = self.arm_counts[i]
                log_term = math.log((self.K * (self.t**2) * math.sqrt(1 + n)) / self.delta)
                cr = (self.L / 2) * math.sqrt((1 + n) / (n**2) * (1 + 2 * log_term))
                exploration_bonus = max(0.1, 1.0 / math.sqrt(self.t))
                # Update UCB value
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
        if self.round_robin:  # 初始轮次：Round-Robin
            return self.t % self.K
        
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
        
        return selected_arm

    def update(self, arm_idx: int, context: int, generated_tokens: int, elapsed_time: float):
        """
        更新LinUCB统计量
        
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
        
        print(f"LinUCB Update: arm={arm_idx}, context={context}, "
              f"reward={reward:.2f}, theta={self.arm_params['theta'][arm_idx]}")

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
            expected_rewards[k] = self.arm_params['theta'][k] @ context_vec
        
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
                A_inv = np.linalg.inv(self.arm_params['A'][k])
                context_vec_col = context_vec.reshape(-1, 1)
                uncertainty[k] = self.alpha * np.sqrt(context_vec_col.T @ A_inv @ context_vec_col)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self.arm_params['A'][k])
                context_vec_col = context_vec.reshape(-1, 1)
                uncertainty[k] = self.alpha * np.sqrt(context_vec_col.T @ A_inv @ context_vec_col)
        
        return uncertainty

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
        
        print(f"Linear Thompson Update: arm={arm_idx}, context={context}, "
              f"reward={reward:.2f}, mu={self.posterior_params['mu'][arm_idx]}")

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


