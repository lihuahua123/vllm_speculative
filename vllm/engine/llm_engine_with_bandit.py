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
from vllm.engine.bandit_integration import BanditOptimizationManager
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.engine.bandit_optimizer import BanditAction
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# Define a request class similar to SampleRequest from benchmark_serving.py
@dataclass
class TrainingRequest:
    """Simple request class for training the bandit optimizer"""
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Dict] = None


class LLMEngineWithBandit(LLMEngine):
    """LLMEngine with integrated multi-armed bandit optimization.
    
    This class extends LLMEngine to automatically apply the multi-armed bandit
    optimization for dynamic model switching between n-gram and neural drafts,
    as well as enabling/disabling speculative decoding based on system load.
    
    It also provides benchmark functionality to continuously train the bandit optimizer
    and test it with different request rates.
    """
    
    def __init__(self, *args, 
                 bandit_monitoring_interval: float = 5.0,
                 bandit_cooldown_period: float = 30.0,
                 **kwargs):
        """Initialize LLMEngine with bandit optimization.
        
        Args:
            bandit_monitoring_interval: How often to check metrics and potentially
                switch models (seconds)
            bandit_cooldown_period: Minimum time between model switches (seconds)
            *args, **kwargs: Arguments passed to LLMEngine.__init__
        """
        super().__init__(*args, **kwargs)
        
        # Create bandit optimization manager
        self.bandit_manager = BanditOptimizationManager(
            llm_engine=self,
            monitoring_interval=bandit_monitoring_interval,
            cooldown_period=bandit_cooldown_period
        )
        
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
        bandit_monitoring_interval: float = 5.0,
        bandit_cooldown_period: float = 30.0,
    ) -> "LLMEngineWithBandit":
        """Creates an LLM engine with bandit optimization from the engine arguments."""
        # Create the engine configs
        vllm_config = engine_args.create_engine_config(usage_context)
        
        # Create the engine
        engine = cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            log_stats=(not engine_args.disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            bandit_monitoring_interval=bandit_monitoring_interval,
            bandit_cooldown_period=bandit_cooldown_period,
        )
        
        return engine
    
    def _bandit_update_request_load(self) -> None:
        """Use bandit optimization to manage model switching based on load."""
        # This replaces the original update_request_load method
        # The bandit manager will handle all the decision making
        self.bandit_manager.step()
    
    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Perform one decoding iteration and run bandit optimization."""
        # Call the parent step method to process requests
        outputs = super().step()
        
        # Record token generation for throughput calculation
        tokens_generated = 0
        for output in outputs:
            for o in output.outputs:
                tokens_generated += len(o.token_ids)
        
        # Update token counter in bandit manager
        self.bandit_manager.record_tokens(tokens_generated)
        
        # Run optimization step if needed
        action = self.bandit_manager.step()
        
        # Record action for benchmarking
        if action:
            status = self.bandit_manager.get_status()
            self.benchmark_metrics["action_history"].append({
                "action": action.name,
                "time": time.time(),
                "metrics": status["current_metrics"]
            })
        
        return outputs
    
    def get_optimization_status(self) -> Dict:
        """Get current status of the bandit optimization."""
        return self.bandit_manager.get_status()
    
    async def run_bandit_training(self, 
                          training_requests: List[TrainingRequest],
                          sampling_params_list: List[SamplingParams],
                          num_iterations: int = 10,
                          log_interval: int = 1):
        """Run continuous training of the bandit optimizer with requests.
        
        Args:
            training_requests: List of training requests to cycle through
            sampling_params_list: List of sampling parameters for each request
            num_iterations: Number of training iterations
            log_interval: How often to log progress
        """
        logger.info(f"Starting bandit training with {len(training_requests)} requests for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
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
                    for output in request_outputs:
                        if output.request_id == request_id and output.finished:
                            finished = True
                            break
                    
                    # Small delay to simulate realistic request processing
                    await asyncio.sleep(0.1)
                
                # Record current status for metrics tracking
                if iteration % log_interval == 0:
                    status = self.bandit_manager.get_status()
                    self.benchmark_metrics["throughput_history"].append(status["current_metrics"]["throughput"])
                    self.benchmark_metrics["request_load_history"].append(status["current_metrics"]["request_load"])
                    self.benchmark_metrics["memory_usage_history"].append(status["current_metrics"]["memory_usage"])
                    
                    # Record reward if available
                    if self.bandit_manager.optimizer.last_action:
                        reward_history = self.bandit_manager.optimizer.reward_history[self.bandit_manager.optimizer.last_action]
                        if reward_history:
                            self.benchmark_metrics["reward_history"].append(reward_history[-1])
            
            logger.info(f"Completed training iteration {iteration+1}/{num_iterations}")
        
        logger.info("Bandit training completed")
        return self.benchmark_metrics
    
    async def run_benchmark_test(self,
                         test_requests: List[TrainingRequest],
                         sampling_params_list: List[SamplingParams],
                         request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                         test_duration: float = 60.0):
        """Run benchmark test with different request rates to evaluate bandit performance.
        
        Args:
            test_requests: List of test requests to use
            sampling_params_list: List of sampling parameters for each request
            request_rates: List of request rates to test (requests per second)
            test_duration: Duration of each test in seconds
        
        Returns:
            Dict containing test results for each request rate
        """
        test_results = []
        
        for request_rate in request_rates:
            logger.info(f"Starting benchmark test with request rate: {request_rate} req/s")
            
            # Reset metrics for this test
            test_metrics = {
                "request_rate": request_rate,
                "completed_requests": 0,
                "total_tokens_generated": 0,
                "avg_latency": 0.0,
                "throughput": 0.0,
                "bandit_actions": []
            }
            
            # Track completion times, latencies, and tokens for all requests
            completion_times = []
            latencies = []
            tokens_generated = []
            
            # Generate request intervals using Poisson process
            request_intervals = np.random.exponential(1.0 / request_rate, len(test_requests))
            
            # Remember initial bandit state
            initial_bandit_status = self.bandit_manager.get_status()
            test_metrics["initial_bandit_state"] = {
                "using_ngram_model": self.bandit_manager.optimizer.using_ngram_model,
                "using_neural_model": self.bandit_manager.optimizer.using_neural_model,
                "spec_decoding_disabled": self.bandit_manager.optimizer.spec_decoding_disabled
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
                
                # Track bandit actions
                action = self.bandit_manager.last_action
                if action:
                    test_metrics["bandit_actions"].append({
                        "action": action.name,
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
            
            # Get final bandit state
            test_metrics["final_bandit_state"] = {
                "using_ngram_model": self.bandit_manager.optimizer.using_ngram_model,
                "using_neural_model": self.bandit_manager.optimizer.using_neural_model,
                "spec_decoding_disabled": self.bandit_manager.optimizer.spec_decoding_disabled
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
        
        # Save results to benchmark metrics
        self.benchmark_metrics["testing_results"] = test_results
        
        return test_results
    
    async def train_and_evaluate_bandit(self,
                                training_requests: List[TrainingRequest],
                                test_requests: List[TrainingRequest],
                                sampling_params: List[Dict[str, Any]],
                                training_iterations: int = 100,
                                request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                                test_duration: float = 60.0,
                                save_results: bool = True,
                                results_dir: str = "bandit_results"):
        """Complete end-to-end training and evaluation of the bandit optimizer.
        
        This method:
        1. Trains the bandit optimizer through multiple iterations
        2. Evaluates the trained optimizer with different request rates
        3. Optionally saves the results
        
        Args:
            training_requests: Requests used for training
            test_requests: Requests used for evaluation
            sampling_params: Sampling parameters for requests
            training_iterations: Number of training iterations
            request_rates: List of request rates to test
            test_duration: Duration of each test in seconds
            save_results: Whether to save results to file
            results_dir: Directory to save results
        
        Returns:
            Dict with complete training and evaluation results
        """
        logger.info("Starting bandit optimizer training and evaluation")
        
        # Step 1: Train the bandit optimizer
        logger.info(f"Training bandit optimizer for {training_iterations} iterations")
        await self.run_bandit_training(
            training_requests=training_requests,
            sampling_params_list=sampling_params,
            num_iterations=training_iterations
        )
        
        # Step 2: Evaluate the trained optimizer with different request rates
        logger.info(f"Evaluating bandit optimizer with request rates: {request_rates}")
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
        if save_results:
            import os
            import json
            from datetime import datetime
            
            # Create results directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{results_dir}/bandit_results_{timestamp}.json"
            
            # Save results to file
            with open(filename, "w") as f:
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
                    "best_throughput": best_throughput
                }, f, indent=2)
            
            logger.info(f"Results saved to {filename}")
        
        return {
            "training_metrics": self.benchmark_metrics,
            "testing_results": test_results,
            "best_rate": best_rate,
            "best_throughput": best_throughput
        }