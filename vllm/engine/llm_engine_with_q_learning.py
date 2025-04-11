from typing import Dict, List, Optional, Union, Any, Tuple
import time
import asyncio
import logging
import argparse
import numpy as np
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.q_learning_integration import QLearningOptimizationManager
from vllm.engine.q_learning_optimizer import QLearningAction
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# Define a request class for training
@dataclass
class TrainingRequest:
    """Simple request class for training the Q-learning optimizer"""
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Dict] = None


class LLMEngineWithQLearning(LLMEngine):
    """LLMEngine with integrated Q-learning optimization.
    
    This class extends LLMEngine to automatically apply Q-learning
    optimization for dynamic model switching between n-gram and neural drafts,
    as well as enabling/disabling speculative decoding based on system load.
    
    It also provides benchmark functionality to continuously train the optimizer
    and test it with different request rates.
    """
    
    def __init__(self, *args, 
                 q_learning_monitoring_interval: float = 5.0,
                 q_learning_cooldown_period: float = 30.0,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 0.2,
                 exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01,
                 memory_penalty_coefficient: float = 0.3,
                 load_q_table_path: Optional[str] = None,
                 **kwargs):
        """Initialize LLMEngine with Q-learning optimization.
        
        Args:
            q_learning_monitoring_interval: How often to check metrics and potentially
                switch models (seconds)
            q_learning_cooldown_period: Minimum time between model switches (seconds)
            learning_rate: Learning rate for Q-learning
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate for epsilon-greedy policy
            exploration_decay: Rate at which exploration decays
            min_exploration_rate: Minimum exploration rate
            memory_penalty_coefficient: Weight for memory usage penalty
            load_q_table_path: Path to load pre-trained Q-table from
            *args, **kwargs: Arguments passed to LLMEngine.__init__
        """
        super().__init__(*args, **kwargs)
        
        # Create Q-learning optimization manager
        self.q_learning_manager = QLearningOptimizationManager(
            llm_engine=self,
            monitoring_interval=q_learning_monitoring_interval,
            cooldown_period=q_learning_cooldown_period,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate,
            memory_penalty_coefficient=memory_penalty_coefficient
        )
        
        # Load pre-trained model if provided
        if load_q_table_path:
            self.q_learning_manager.load_model(load_q_table_path)
        
        # Benchmark metrics
        self.benchmark_metrics = {
            "throughput_history": [],
            "request_load_history": [],
            "memory_usage_history": [],
            "action_history": [],
            "reward_history": [],
            "testing_results": []
        }
        
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        q_learning_monitoring_interval: float = 5.0,
        q_learning_cooldown_period: float = 30.0,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        memory_penalty_coefficient: float = 0.3,
        load_q_table_path: Optional[str] = None,
    ) -> "LLMEngineWithQLearning":
        """Creates an LLM engine with Q-learning optimization from the engine arguments."""
        # Create the engine configs
        vllm_config = engine_args.create_engine_config(usage_context)
        
        # Create the engine
        engine = cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            log_stats=(not engine_args.disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            q_learning_monitoring_interval=q_learning_monitoring_interval,
            q_learning_cooldown_period=q_learning_cooldown_period,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate,
            memory_penalty_coefficient=memory_penalty_coefficient,
            load_q_table_path=load_q_table_path,
        )
        
        return engine
    
    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Perform one decoding iteration and run Q-learning optimization."""
        # Call the parent step method to process requests
        outputs = super().step()
        
        # Record token generation for throughput calculation
        tokens_generated = 0
        for output in outputs:
            for o in output.outputs:
                tokens_generated += len(o.token_ids)
        
        # Update token counter in Q-learning manager
        self.q_learning_manager.record_tokens(tokens_generated)
        
        # Run optimization step if needed
        action = self.q_learning_manager.step()
        
        # Record action for benchmarking
        if action:
            status = self.q_learning_manager.get_status()
            self.benchmark_metrics["action_history"].append({
                "action": action.name,
                "time": time.time(),
                "metrics": status["current_metrics"]
            })
        
        return outputs
    
    def get_optimization_status(self) -> Dict:
        """Get current status of the Q-learning optimization."""
        return self.q_learning_manager.get_status()
    
    def save_q_table(self, file_path: str) -> None:
        """Save the current Q-table to a file for later use."""
        self.q_learning_manager.save_model(file_path)
    
    async def run_q_learning_training(self, 
                              training_requests: List[TrainingRequest],
                              sampling_params_list: List[SamplingParams],
                              num_iterations: int = 10,
                              log_interval: int = 1):
        """Run continuous training of the Q-learning optimizer with requests.
        
        Args:
            training_requests: List of training requests to cycle through
            sampling_params_list: List of sampling parameters for each request
            num_iterations: Number of training iterations
            log_interval: How often to log progress
        """
        logger.info(f"Starting Q-learning training with {len(training_requests)} requests for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            # Track episode rewards
            episode_reward = 0.0
            
            # Cycle through requests
            for i, (request, params) in enumerate(zip(training_requests, sampling_params_list)):
                if i % 10 == 0:
                    logger.info(f"Training iteration {iteration+1}/{num_iterations}, request {i+1}/{len(training_requests)}")
                
                # Add request to engine
                request_id = f"train_{iteration}_{i}"
                self.add_request(request_id, request.prompt, params)
                
                # Process until request is finished
                finished = False
                while not finished:
                    request_outputs = self.step()
                    for index,output in enumerate(request_outputs):
                        if output.request_id == request_id and output.finished:
                            finished = True
                            break
                    
                    # Small delay to simulate realistic request processing
                    await asyncio.sleep(0.1)
                
                # Record current status for metrics tracking
                if iteration % log_interval == 0:
                    status = self.q_learning_manager.get_status()
                    self.benchmark_metrics["throughput_history"].append(status["current_metrics"]["throughput"])
                    self.benchmark_metrics["request_load_history"].append(status["current_metrics"]["request_load"])
                    self.benchmark_metrics["memory_usage_history"].append(status["current_metrics"]["memory_usage"])
                    
                    # Record reward if available
                    if status["optimizer_status"]["last_reward"] is not None:
                        reward = status["optimizer_status"]["last_reward"]
                        self.benchmark_metrics["reward_history"].append(reward)
                        episode_reward += reward
            
            # End of episode
            self.q_learning_manager.optimizer.episode_count += 1
            logger.info(f"Completed training iteration {iteration+1}/{num_iterations}, "
                       f"episode reward: {episode_reward:.2f}")
        
        logger.info("Q-learning training completed")
        return self.benchmark_metrics
    
    async def run_benchmark_test(self,
                         test_requests: List[TrainingRequest],
                         sampling_params_list: List[SamplingParams],
                         request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                         test_duration: float = 60.0):
        """Run benchmark test with different request rates to evaluate Q-learning performance.
        
        Args:
            test_requests: List of test requests to use
            sampling_params_list: List of sampling parameters for each request
            request_rates: List of request rates to test (requests per second)
            test_duration: Duration of each test in seconds
        
        Returns:
            Dict containing test results for each request rate
        """
        test_results = []
        
        # Save exploration rate to restore after testing
        original_exploration_rate = self.q_learning_manager.optimizer.exploration_rate
        
        # Use minimal exploration during testing to get more consistent results
        self.q_learning_manager.optimizer.exploration_rate = 0.05
        
        try:
            for request_rate in request_rates:
                logger.info(f"Starting benchmark test with request rate: {request_rate} req/s")
                
                # Reset metrics for this test
                test_metrics = {
                    "request_rate": request_rate,
                    "completed_requests": 0,
                    "total_tokens_generated": 0,
                    "avg_latency": 0.0,
                    "throughput": 0.0,
                    "q_learning_actions": []
                }
                
                # Track completion times, latencies, and tokens for all requests
                completion_times = []
                latencies = []
                tokens_generated = []
                
                # Generate request intervals using Poisson process
                request_intervals = np.random.exponential(1.0 / request_rate, len(test_requests))
                
                # Remember initial Q-learning state
                initial_q_learning_status = self.q_learning_manager.get_status()
                test_metrics["initial_state"] = {
                    "using_ngram_model": self.q_learning_manager.optimizer.using_ngram_model,
                    "using_neural_model": self.q_learning_manager.optimizer.using_neural_model,
                    "spec_decoding_disabled": self.q_learning_manager.optimizer.spec_decoding_disabled
                }
                
                start_time = time.time()
                request_idx = 0
                next_request_time = start_time
                active_requests = {}  # request_id -> start_time
                
                # Run the test for specified duration or until all requests are processed
                while (time.time() - start_time < test_duration and 
                       (request_idx < len(test_requests) or active_requests)):
                    
                    current_time = time.time()
                    
                    # Add new requests according to the Poisson process
                    if request_idx < len(test_requests) and current_time >= next_request_time:
                        request_id = f"test_{request_rate}_{request_idx}"
                        request = test_requests[request_idx]
                        params = sampling_params_list[request_idx % len(sampling_params_list)]
                        
                        self.add_request(request_id, request.prompt, params)
                        
                        # Record start time
                        active_requests[request_id] = current_time
                        
                        # Schedule next request
                        request_idx += 1
                        if request_idx < len(test_requests):
                            next_request_time = start_time + sum(request_intervals[:request_idx])
                    
                    # Process engine step
                    request_outputs = self.step()
                    
                    # Track Q-learning actions
                    status = self.q_learning_manager.get_status()
                    if status["last_action"]:
                        test_metrics["q_learning_actions"].append({
                            "action": status["last_action"],
                            "time": time.time() - start_time
                        })
                    
                    # Process completed requests
                    for output in request_outputs:
                        if output.request_id in active_requests and output.finished:
                            # Calculate latency and token count
                            start_time_req = active_requests[output.request_id]
                            latency = current_time - start_time_req
                            token_count = sum(len(o.token_ids) for o in output.outputs)
                            
                            # Record metrics
                            completion_times.append(current_time)
                            latencies.append(latency)
                            tokens_generated.append(token_count)
                            
                            # Remove from active requests
                            del active_requests[output.request_id]
                            
                            # Update test metrics
                            test_metrics["completed_requests"] += 1
                            test_metrics["total_tokens_generated"] += token_count
                    
                    # Small delay to not overload the CPU
                    await asyncio.sleep(0.01)
                
                # Calculate final metrics
                test_duration_actual = time.time() - start_time
                if test_metrics["completed_requests"] > 0:
                    test_metrics["avg_latency"] = sum(latencies) / len(latencies)
                    test_metrics["throughput"] = test_metrics["total_tokens_generated"] / test_duration_actual
                
                # Get final Q-learning state
                test_metrics["final_state"] = {
                    "using_ngram_model": self.q_learning_manager.optimizer.using_ngram_model,
                    "using_neural_model": self.q_learning_manager.optimizer.using_neural_model,
                    "spec_decoding_disabled": self.q_learning_manager.optimizer.spec_decoding_disabled
                }
                
                # Record detailed metrics
                test_metrics["latencies"] = latencies
                test_metrics["tokens_generated"] = tokens_generated
                test_metrics["test_duration"] = test_duration_actual
                
                # Save to overall results
                test_results.append(test_metrics)
                logger.info(f"Benchmark test completed for rate {request_rate} req/s: "
                          f"{test_metrics['completed_requests']} requests completed, "
                          f"throughput: {test_metrics['throughput']:.2f} tokens/s")
        finally:
            # Restore original exploration rate
            self.q_learning_manager.optimizer.exploration_rate = original_exploration_rate
        
        # Save results to benchmark metrics
        self.benchmark_metrics["testing_results"] = test_results
        
        return test_results
    
    async def train_and_evaluate_q_learning(self,
                                     training_requests: List[TrainingRequest],
                                     test_requests: List[TrainingRequest],
                                     sampling_params: List[SamplingParams],
                                     training_iterations: int = 100,
                                     request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                                     test_duration: float = 60.0,
                                     save_results: bool = True,
                                     save_q_table: bool = True,
                                     results_dir: str = "q_learning_results"):
        """Complete end-to-end training and evaluation of the Q-learning optimizer.
        
        This method:
        1. Trains the Q-learning optimizer through multiple iterations
        2. Evaluates the trained optimizer with different request rates
        3. Optionally saves the results and Q-table
        
        Args:
            training_requests: Requests used for training
            test_requests: Requests used for evaluation
            sampling_params: Sampling parameters for requests
            training_iterations: Number of training iterations
            request_rates: List of request rates to test
            test_duration: Duration of each test in seconds
            save_results: Whether to save results to file
            save_q_table: Whether to save the Q-table
            results_dir: Directory to save results
        
        Returns:
            Dict with complete training and evaluation results
        """
        logger.info("Starting Q-learning optimizer training and evaluation")
        
        # Step 1: Train the Q-learning optimizer
        logger.info(f"Training Q-learning optimizer for {training_iterations} iterations")
        await self.run_q_learning_training(
            training_requests=training_requests,
            sampling_params_list=sampling_params,
            num_iterations=training_iterations
        )
        
        # Step 2: Evaluate the trained optimizer with different request rates
        logger.info(f"Evaluating Q-learning optimizer with request rates: {request_rates}")
        test_results = await self.run_benchmark_test(
            test_requests=test_requests,
            sampling_params_list=sampling_params,
            request_rates=request_rates,
            test_duration=test_duration
        )
        
        # Step 3: Analyze results
        best_rate = 0
        best_throughput = 0
        
        for result in test_results:
            rate = result["request_rate"]
            throughput = result["throughput"]
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_rate = rate
            
            logger.info(f"Request rate {rate} req/s: {throughput:.2f} tokens/s")
        
        logger.info(f"Best throughput: {best_throughput:.2f} tokens/s at {best_rate} req/s")
        
        # Step 4: Save results if requested
        if save_results or save_q_table:
            import os
            import json
            from datetime import datetime
            
            # Create results directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            
            if save_results:
                # Save results to file
                results_filename = f"{results_dir}/q_learning_results_{timestamp}.json"
                
                with open(results_filename, "w") as f:
                    json.dump({
                        "training_metrics": {
                            "throughput_history": self.benchmark_metrics["throughput_history"],
                            "request_load_history": self.benchmark_metrics["request_load_history"],
                            "memory_usage_history": self.benchmark_metrics["memory_usage_history"],
                            "action_history": self.benchmark_metrics["action_history"],
                            "reward_history": self.benchmark_metrics["reward_history"]
                        },
                        "testing_results": test_results,
                        "best_rate": best_rate,
                        "best_throughput": best_throughput,
                        "q_learning_stats": self.q_learning_manager.optimizer.get_status()
                    }, f, indent=2)
                
                logger.info(f"Results saved to {results_filename}")
            
            if save_q_table:
                # Save Q-table to file
                q_table_filename = f"{results_dir}/q_table_{timestamp}.json"
                self.save_q_table(q_table_filename)
                logger.info(f"Q-table saved to {q_table_filename}")
        
        return {
            "training_metrics": self.benchmark_metrics,
            "testing_results": test_results,
            "best_rate": best_rate,
            "best_throughput": best_throughput,
            "q_learning_stats": self.q_learning_manager.optimizer.get_status()
        } 