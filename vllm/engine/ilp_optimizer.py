import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pulp  # For linear programming
from sklearn.linear_model import LinearRegression
import joblib  # For saving/loading models
import os
import torch
from collections import defaultdict
import json
import pickle
logger = logging.getLogger(__name__)

class ILPAction(Enum):
    """Actions that the ILP optimizer can take"""
    USE_SMALL_MODEL_1 = 0 # neural_model
    USE_SMALL_MODEL_2 = 1 # ngram
    DISABLE_SPEC_DECODING = 2

class ILPOptimizer:
    """An ILP-based optimizer for dynamic model switching.
    
    This optimizer uses integer linear programming to select the optimal
    speculative sampling method (SSM) from M choices including M-1 small models
    and the option to disable speculative decoding.
    
    The optimization problem is formulated as:
    
    max_x  g_x * x
    s.t.   sum_i x_i = 1, for all i in M
           x_i in {0, 1}, for all i in M
           sum_i v_i <= Mem
           sum_i g_i * x_i >= g_0
    
    where:
    - x is a one-hot vector representing the chosen SSM
    - g_x is the throughput under selection x
    - v_i is the memory usage of each model
    - Mem is the total available memory
    - g_0 is the minimum acceptable throughput
    """
    
    def __init__(self, 
                 llm_engine,
                 reward_window_size: int = 50,
                 min_samples_per_model: int = 3):  # tokens/s
        self.llm_engine = llm_engine
        # Create actions list based on number of small models
        self.actions = []
        for i in range(len(ILPAction)):
            self.actions.append(ILPAction(i))
        
        # Initialize counts for each model
        self.counts = {action: 0 for action in self.actions}
        
        # Initialize throughput history for each model
        self.throughput_history = {action: [] for action in self.actions}
        
        # Running metrics
        self.request_load_history = []
        self.memory_usage_history = []
        self.acceptance_rate_history = []
        self.spec_length_history = []
        
        # Linear models for T_D and T_V prediction using sklearn
        self.t_d_model = LinearRegression()
        self.t_v_model = LinearRegression()
        self.t_d_model_trained = False
        self.t_v_model_trained = False
        
        # Memory usage for each model
        self.model_memory_usage = {action: 0.0 for action in self.actions}
        
        # State tracking
        self.last_action = ILPAction.USE_SMALL_MODEL_1
        self.last_action_time = 0.0
        self.current_state = None
        
        # Configuration
        self.reward_window_size = reward_window_size # history window size
        self.min_samples_per_model = min_samples_per_model # minimum samples per model
        
        # Current model state
        self.current_model_index = 0  # -1 means no speculative decoding
        
        # Added for the new record_metrics method
        self.action_metrics_history = {action.value: {
            "acceptance_rates": [],
            "spec_lengths": [],
            "proposal_time": [],
            "scoring_time": [],
        } for action in self.actions}

        self._init_action_time_history()

    def _init_action_time_history(self):
        """初始化三维action_time_history字典"""
        self.action_time_history = {}
        for action in self.actions:
            self.action_time_history[action.value] = {}
            # 预先初始化一些可能的batch size
            for batch_size in range(1, 3):  # 假设batch size从1到300
                self.action_time_history[action.value][batch_size] = {}
                for i in range(2):
                    self.action_time_history[action.value][batch_size][i] = {
                        "proposal_time": [],
                        "scoring_time": [],
                        "verification_time": [],
                        "acceptance_rate": [],
                        "total_num": 0,
                        "total_latency": [],
                        "accepted_tokens_length": [],
                        "prefill_time": [],
                        "prefill_total_num": 0,
                        "context_length": []
                }

    def save_action_time_history(self,file_name=None):
        """Save action time history to file"""
        # for action in self.actions:
        #     action = action.value
        #     for batch_size in self.action_time_history[action]:
        #         decode_total_num = self.action_time_history[action][batch_size][1]["total_num"] 
        #         prefill_total_num = self.action_time_history[action][batch_size][0]["prefill_total_num"]
        #         if decode_total_num > 0:
        #             self.action_time_history[action][batch_size][1]["proposal_time"]/=decode_total_num
        #             self.action_time_history[action][batch_size][1]["scoring_time"]/=decode_total_num
        #             self.action_time_history[action][batch_size][1]["verification_time"]/=decode_total_num
        #             self.action_time_history[action][batch_size][1]["acceptance_rate"]/=decode_total_num
        #             self.action_time_history[action][batch_size][1]["total_latency"]/=decode_total_num
        #             self.action_time_history[action][batch_size][1]["accepted_tokens_length"]/=decode_total_num
        #             self.action_time_history[action][batch_size][1]["total_num"] = 1
        #         if prefill_total_num > 0:
        #             self.action_time_history[action][batch_size][0]["prefill_time"]/=prefill_total_num
        #             self.action_time_history[action][batch_size][0]["context_length"]/=prefill_total_num
        #             self.action_time_history[action][batch_size][0]["prefill_total_num"] = 1
                    
        if file_name is not None:
            logger.info(f"save action_time_history to file {file_name}")
            with open(file_name, "w") as f:
                json.dump(self.action_time_history, f)
        else:
            with open("action_time_history.json", "w") as f:
                json.dump(self.action_time_history, f)
            logger.info(f"save action_time_history to file action_time_history.json")

    def load_action_time_history(self):
        def read_json(file_path):
            with open(file_path, "r") as f:
                action_time_history = json.load(f)
                # 将字符串key转换为int
                converted_history = {}
                for action_str in   action_time_history:
                    action_int = int(action_str)
                    converted_history[action_int] = {}
                    for batch_str in action_time_history[action_str]:
                        batch_int = int(batch_str)
                        converted_history[action_int][batch_int] = action_time_history[action_str][batch_str]
                        for i in range(2):
                            converted_history[action_int][batch_int][int(i)] = action_time_history[action_str][batch_str][str(i)]
                action_time_history = converted_history
            return action_time_history
        action_time_history = {}
        ngram_action_time_history = read_json("./ngram_300.json")
        nospec_action_time_history = read_json("./nospec_300.json")
        deep_action_time_history = read_json("./deep_05b_300.json")
        # ngram_action_time_history = read_json("ngram1.json")
        # nospec_action_time_history = read_json("nospec1.json")
        # deep_action_time_history = read_json("deep_05b1.json")
        action_time_history[1] = ngram_action_time_history[1]
        action_time_history[2] = nospec_action_time_history[2]
        action_time_history[0] = deep_action_time_history[0]
        self.action_time_history = action_time_history
        #print("load action_time_history success",self.action_time_history)


    def record_metrics(self, metrics) -> None:
        """Record current system metrics"""
        # Record speculative decoding metrics if available
        proposal_time = metrics["proposal_time"]
        acceptance_rate = metrics["acceptance_rate"]
        spec_length = metrics["spec_length"]
        #print(self.last_action,metrics["batch_size"],metrics["stage"],metrics)
        if self.last_action == ILPAction.DISABLE_SPEC_DECODING:
            if metrics["stage"] == 2: 
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["total_num"]+=1
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["scoring_time"].append(metrics["scoring_time"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["context_length"].append(metrics["context_length"])
            else:
                self.action_time_history[self.last_action.value][metrics["batch_size"]][0]["prefill_time"].append(metrics["scoring_time"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][0]["context_length"].append(metrics["context_length"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][0]["prefill_total_num"]+=1
        else: 
            if metrics["stage"] == 2: #SequenceStage.DECODE.value:
                # decode 而不包含prefill 
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["proposal_time"].append(proposal_time)
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["total_latency"].append(metrics["total_latency"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["accepted_tokens_length"].append(metrics["accepted_tokens_length"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["total_num"]+=1
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["scoring_time"].append(metrics["scoring_time"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["verification_time"].append(metrics["verification_time"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["acceptance_rate"].append(acceptance_rate)
                self.action_time_history[self.last_action.value][metrics["batch_size"]][1]["context_length"].append(metrics["context_length"])
            else:
                self.action_time_history[self.last_action.value][metrics["batch_size"]][0]["prefill_time"].append(proposal_time + metrics["scoring_time"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][0]["context_length"].append(metrics["context_length"])
                self.action_time_history[self.last_action.value][metrics["batch_size"]][0]["prefill_total_num"]+=1
        
    def load_model(self, model_path_d: str, model_path_v: str) -> None:
        if self.t_d_model_trained and self.t_v_model_trained:
            return
        self.t_d_model = joblib.load(model_path_d)
        self.t_v_model = joblib.load(model_path_v)
        self.t_d_model_trained = True
        self.t_v_model_trained = True
    
    # def _predict_lantency(self, 
    #                        action: ILPAction, 
    #                        acceptance_rate: float,
    #                        spec_length: int,
    #                        metrics,scheduler_outputs) -> float:
    #     """
        
    #     g_x = (B * (1 - α^(γ+1))/(1 - α)) / 
    #           (γ * T_D(B, ∑S_b, 1) + T_V(B, ∑S_b, γ) + C_swap)
    #     """
        
    #     request_load = len(scheduler_outputs.scheduled_seq_groups) - scheduler_outputs.num_prefill_groups
    #     context_tokens = scheduler_outputs.num_cached_tokens
    #     # batch_tokens = scheduler_outputs.num_batched_tokens 
    #     B = max(request_load, 1)
    #     # print("B: ",B,"metrics['batch_size']: ",metrics["batch_size"])

    #     # Calculate C_swap (cost of switching models)
    #     # If we're already using this model, no swap cost
    #     # Otherwise, assume some fixed cost
    #     c_swap = 0.0  # 切换草稿模型还要prefill
    #     if action.value != self.current_model_index and self.current_model_index == ILPAction.USE_SMALL_MODEL_1.value:
    #         c_swap = 0.1004  # FIXME : offload cost
    #     if action.value != self.current_model_index and action.value == ILPAction.USE_SMALL_MODEL_1.value:
    #         c_swap = 0.2004  # FIXME : load cost
    #     # t_v = 0.023
    #     # feature_vector = np.array([[B,context_tokens,B*1]])
    #     if self.action_time_history[action.value][B]["scoring_time"] > 0:
    #         t_s = self.action_time_history[action.value][B]["scoring_time"] /self.action_time_history[action.value][B]["total_num"]
    #         logger.info(f"use action_time_history as t_s, {t_s}")
    #     elif metrics is not None and "scoring_time" in metrics and metrics["scoring_time"] is not None:
    #         t_s = metrics["scoring_time"]
    #         logger.info(f"use metrics as t_s, {t_s}")
    #     else:
    #         logger.info("no scoring time, use 1.0")
    #         t_s = 1.0

        
    #     if action == ILPAction.DISABLE_SPEC_DECODING:
    #         print("disable speculative decoding",B,t_s,c_swap)
    #         throughput = t_s / B * 10  
    #         throughput += c_swap
    #         print("throughput: ",throughput,c_swap)
    #         return throughput
        
    #     if self.action_time_history[action.value][B]["verification_time"] > 0:
    #         t_v = self.action_time_history[action.value][B]["verification_time"] /self.action_time_history[action.value][B]["total_num"]
    #         logger.info(f"use action_time_history as t_v, {t_v}")
    #     elif metrics is not None and "verification_time" in metrics and metrics["verification_time"] is not None:
    #         t_v = metrics["verification_time"]
    #         logger.info(f"use metrics as t_v, {t_v}")
    #     else:
    #         logger.info("no verification time, use 1.0")
    #         t_v = 1.0
    #     # If we have historical data for this action, use the average as baseline
    #     # if self.throughput_history[action] and len(self.throughput_history[action]) >= self.min_samples_per_model:
    #     #     return np.mean(self.throughput_history[action])
        
    #     # Otherwise, predict using the formula
    #     # B is the batch size (number of requests)

    #     # if self.action_time_history[action.value][B]["acceptance_rate"] > 0:

    #     #     alpha1 = self.action_time_history[action.value][B]["acceptance_rate"]/self.action_time_history[action.value][B]["total_num"]
    #     #     a = self.action_time_history[action.value][B]["acceptance_rate"]
    #     #     b = self.action_time_history[action.value][B]["total_num"]
    #     #     logger.info(f"use action_time_history as alpha, {a},{b},{alpha1},history acceptance_rate {acceptance_rate}")
    #     # else:
    #     #     raise ValueError(f"no acceptance_rate in action_time_history, {self.action_time_history[action.value][B]}")
    #     alpha = acceptance_rate
    #     # γ is the speculation length
    #     gamma = spec_length if spec_length > 0 else 3  # default if unknown
    #     # Calculate numerator: number of tokens processed
    #     # B * (1 - α^(γ+1))/(1 - α)
    #     # tokens_processed = B * (gamma * alpha + 1)
    #     if abs(1 - alpha) < 1e-6:  # alpha is close to 1
    #         tokens_processed = B * (gamma + 1)
    #     else:
    #         tokens_processed = B * (1 - alpha**(gamma+1)) / (1 - alpha)
        
    #     if  self.action_time_history[action.value][B]["total_num"] > 0:
    #         t_d = self.action_time_history[action.value][B]["proposal_time"]/self.action_time_history[action.value][B]["total_num"]
    #     elif len(self.action_metrics_history[action.value]["proposal_time"]) > 0:
    #         logger.info(f"use proposal throughput as t_d, {self.action_metrics_history[action.value]['proposal_time']}")
    #         t_d = self.action_metrics_history[action.value]["proposal_time"][-1]
    #     else:
    #         logger.info("no t_d, use 0")
    #         t_d = 0
    #     # if self.t_d_model_trained and action == ILPAction.USE_SMALL_MODEL_1:
    #     #     feature_vector = np.array([[B,context_tokens,B]])
    #     #     # t_d =   float(self.t_d_model.predict(feature_vector)[0])
    #     #     t_d = metrics["proposal_time"] #4.7
    #     # elif action == ILPAction.USE_SMALL_MODEL_2:
    #     #     t_d = metrics["proposal_time"] #0.22
        
    #     # Predict T_V (verification time) using sklearn model
    #     # if self.t_v_model_trained:
    #     #     # Using sklearn's predict method
    #     #     feature_vector = np.array([[B,context_tokens,B*gamma]])
    #     #     t_v = float(self.t_v_model.predict(feature_vector)[0])
    #     # else:
    #     #     # Default model if not trained
    #     #     t_v = 0.005 * B
        
        
        
    #     # Calculate denominator: total time
    #     # γ * T_D(B, ∑S_b, 1) + T_V(B, ∑S_b, γ) + C_swap
    #     total_time =  (gamma * t_d + t_s + t_v) 
        
    #     # Calculate throughput: tokens / time
    #     throughput = total_time / tokens_processed   * 10 
    #     throughput += c_swap
    #     print("action: ",action,"accept rate: ",alpha, "t_d: ",t_d, "t_s: ",t_s, "t_v: ",t_v, 
    #            "total_time: ",total_time,"tokens_processed: ",tokens_processed,"c_swap: ",c_swap,"throughput: ",throughput)
      
        
    #     return throughput
    
    def get_latency(self,b,action):
        need_record = True
        if action == ILPAction.USE_SMALL_MODEL_1.value or action == ILPAction.USE_SMALL_MODEL_2.value:
            latency_decode = self.action_time_history[action][b][1]["total_latency"] # decode
            if latency_decode - 0 < 1e-6:
                logger.info(f"latency_decode is 0, action: {action}, batch_size: {b}")
                need_record = False
                return need_record,float('inf')
            latency_prefill = self.action_time_history[action][1][0]["prefill_time"] # prefill
            latency2 = latency_prefill #/ (action_time_history[action][1][0]["context_length"])
            if self.action_time_history[action][b][1]["accepted_tokens_length"] == 0:
                decode_latency = latency_decode / (b) *4
            else:
                decode_latency = latency_decode / (self.action_time_history[action][b][1]["accepted_tokens_length"]) *4
            latency = decode_latency #+ latency2
            return need_record,latency
            #latency = (latency_prefill+ latency_decode*4) / (action_time_history[action][1][0]["context_length"]+(action_time_history[action][b][1]["accepted_tokens_length"]+b)*4)
        elif action == ILPAction.DISABLE_SPEC_DECODING.value and self.action_time_history[action][b][1]["scoring_time"] > 0:
            latency_prefill = self.action_time_history[action][1][0]["prefill_time"] # prefill
            latency_decode = self.action_time_history[action][b][1]["scoring_time"]*4 # decode
            if latency_decode - 0 < 1e-6:
                logger.info(f"latency_decode is 0, action: {action}, batch_size: {b}")
                need_record = False
                return need_record,float('inf')
            latency2 = latency_prefill #/ (action_time_history[action][1][0]["context_length"])
            decode_latency = latency_decode / b 
            latency = decode_latency  #+ latency2
            return need_record,latency
            # latency  = (latency_prefill + action_time_history[action][b][1]["scoring_time"]*4) / (action_time_history[action][1][0]["context_length"]+ b * 4)
        else:
            need_record = False
            logger.info(f"action: {action}, batch_size: {b}, no record")
            return need_record,float('inf')
    
    def _has_min_samples(self) -> bool:
        """Check if all models have been tried the minimum number of times"""
        return all(self.counts[action] >= self.min_samples_per_model for action in self.actions)
    
    def solve_ilp(self,request_load) -> Tuple[Optional[ILPAction], float]:
        """Solve the ILP optimization problem to select the best model"""
        # # FIXME: 这里需要修改 try_scheduler 里面的block number
        # virtual_engine = 0
        # scheduler_outputs, can_increase_space, can_decrease_space = self.llm_engine.try_scheduler()
        # end_time = time.time()
        # print("try_scheduler time: ",end_time - time_start)
        # logger.info(f"scheduled_seq_groups: {len(scheduler_outputs.scheduled_seq_groups)}, running: {len(self.llm_engine.scheduler[virtual_engine].running)}, waiting: {len(self.llm_engine.scheduler[virtual_engine].waiting)}")
        # request_load = len(scheduler_outputs.scheduled_seq_groups) - scheduler_outputs.num_prefill_groups
        min_lantency = float('inf')
        selected_action = None
        for action in self.actions:
            action_value = action.value
            need_record,lantency = self.get_latency(request_load,action_value)
            if  lantency < min_lantency:
                min_lantency = lantency
                selected_action = action
        
        if selected_action is not None: # 否则保持不变
            self.last_action_time = time.time()
        
        return selected_action
    
    def select_action(self,request_load) -> Optional[ILPAction]:
        """Select the best action based on current metrics"""
        
        # output_dir = "/home/hello/lirui/vllm_speculative/perf_results"
        # big_model_save_path = os.path.join(output_dir, f"vllm_prefill_model_DeepSeek-R1-DRAFT-Qwen2.5-0.5B.pkl")
        # small_model_save_path = os.path.join(output_dir, f"vllm_prefill_model_alamios_DeepSeek-R1-DRAFT-Qwen2.5-0.5B.pkl")
        # self.load_model(big_model_save_path, small_model_save_path)
        
        # Solve the ILP problem
        action = self.solve_ilp(request_load)
        
        # Check if we should actually switch
        if action is not None and action.value == self.current_model_index:
            # We're already using this model, so no need to switch
            logger.info(f"request_load: {request_load}, ILP optimizer selected action: {action}, but we're already using this model")
            return None
        
        logger.info(f"request_load: {request_load}, ILP optimizer selected action: {action}")
        return action
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the optimizer"""
        if self.current_state is None:
            return {"status": "not_initialized"}
        
        # avg_throughput = {}
        # for action in self.actions:
        #     if self.throughput_history[action]:
        #         avg_throughput[action.name] = np.mean(self.throughput_history[action])
        #     else:
        #         avg_throughput[action.name] = 0.0
        
        return {
            "current_model_index": self.current_model_index,
            "last_action": self.last_action.name if self.last_action else None,
            "counts": {action.name: count for action, count in self.counts.items()},
            # "average_throughput": avg_throughput,
            "current_metrics": self.current_state,
            "model_memory_usage": {action.name: usage for action, usage in self.model_memory_usage.items()},
            "t_d_model_trained": self.t_d_model_trained,
            "t_v_model_trained": self.t_v_model_trained
        }
        
            
    def load_models(self, t_d_model_path: str, t_v_model_path: str) -> bool:
        """Load trained linear regression models from files
        
        Args:
            t_d_model_path: Path to load the draft model from
            t_v_model_path: Path to load the verification model from
            
        Returns:
            bool: Whether the models were successfully loaded
        """
        try:
            # Load the sklearn models
            self.t_d_model = joblib.load(t_d_model_path)
            self.t_v_model = joblib.load(t_v_model_path)
            
            # Mark models as trained
            self.t_d_model_trained = True
            self.t_v_model_trained = True
            
            logger.info(f"Models loaded from {t_d_model_path} and {t_v_model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False