#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Example script demonstrating how to use the multi-armed bandit optimizer
with vLLM's LLMEngine for dynamic adaptation.
"""
import sys
import os
sys.path.append('/home/nudt/lirui/vllm_speculative/')
import time
import argparse
import logging
from typing import List, Tuple
import sys
import os

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.engine.bandit_integration import BanditOptimizationManager
from vllm.engine.bandit_optimizer import BanditAction
from vllm.engine.llm_engine_with_bandit import LLMEngineWithBandit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("Explain the concept of machine learning in simple terms.",
         SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)),
        ("What is the capital of France?",
         SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)),
        ("Write a short story about a robot learning to feel emotions.",
         SamplingParams(temperature=0.7, top_p=0.95, max_tokens=200)),
        ("List five benefits of regular exercise.",
         SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)),
        ("Explain how transformers work in deep learning.",
         SamplingParams(temperature=0.7, top_p=0.95, max_tokens=150)),
    ]

def process_requests(engine: LLMEngine,
                     optimizer: BanditOptimizationManager,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Process requests and monitor optimization decisions."""
    request_id = 0
    
    logger.info(f"Processing {len(test_prompts)} requests...")
    start_time = time.time()
    
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1
            
            # Add a small delay between requests to simulate realistic load
            time.sleep(0.2)

        request_outputs = engine.step()
        
        # Process outputs
        tokens_generated = 0
        for request_output in request_outputs:
            for output in request_output.outputs:
                tokens_generated += len(output.token_ids)
            
            if request_output.finished:
                generation_time = time.time() - start_time
                logger.info(f"Request {request_output.request_id} finished in {generation_time:.2f}s")
        
        # Update the optimizer with token generation information
        optimizer.record_tokens(tokens_generated)
        
        # Run optimizer step
        action = optimizer.step()
        if action:
            logger.info(f"Optimizer selected action: {action.name}")

def initialize_engine(args: argparse.Namespace) -> Tuple[LLMEngine, BanditOptimizationManager]:
    """Initialize the LLMEngine and BanditOptimizationManager."""
    # Check if using bandit-integrated engine
    if args.use_bandit_engine:
        engine_args = EngineArgs.from_cli_args(args)
        
        # Create engine with built-in bandit optimization
        engine = LLMEngineWithBandit.from_engine_args(
            engine_args,
            bandit_monitoring_interval=args.monitoring_interval,
            bandit_cooldown_period=args.cooldown_period
        )
        # Get the bandit manager from the engine
        optimizer = engine.bandit_manager
    else:
        # Regular engine with separate bandit optimization
        engine_args = EngineArgs.from_cli_args(args)
        engine = LLMEngine.from_engine_args(engine_args)
        
        # Create bandit optimization manager
        optimizer = BanditOptimizationManager(
            llm_engine=engine,
            monitoring_interval=args.monitoring_interval,
            cooldown_period=args.cooldown_period
        )
    
    return engine, optimizer

def simulate_load_pattern(engine: LLMEngine, 
                          optimizer: BanditOptimizationManager):
    """Simulate different load patterns to test the optimizer."""
    # First batch: low load (5 requests with 1s delay)
    logger.info("=== Starting LOW LOAD test ===")
    low_load_prompts = create_test_prompts()[:3]
    process_requests(engine, optimizer, low_load_prompts)
    logger.info("Low load test completed")
    
    # Wait a bit to let the optimizer stabilize
    time.sleep(5)
    
    # Second batch: high load (20 requests with 0.2s delay)
    logger.info("=== Starting HIGH LOAD test ===")
    high_load_prompts = create_test_prompts() * 5  # Create more prompts
    process_requests(engine, optimizer, high_load_prompts)
    logger.info("High load test completed")
    
    # Wait a bit to let the optimizer stabilize
    time.sleep(5)
    
    # Third batch: medium load (10 requests with 0.5s delay)
    logger.info("=== Starting MEDIUM LOAD test ===")
    medium_load_prompts = create_test_prompts() * 2
    process_requests(engine, optimizer, medium_load_prompts)
    logger.info("Medium load test completed")

def main(args: argparse.Namespace):
    """Main function that sets up and runs the bandit optimization test."""
    # Initialize engine and optimizer
    engine, optimizer = initialize_engine(args)
    
    if args.run_load_pattern:
        simulate_load_pattern(engine, optimizer)
    else:
        # Just process a single batch of requests
        test_prompts = create_test_prompts()
        process_requests(engine, optimizer, test_prompts)
    
    # Display final optimizer status
    status = optimizer.get_status()
    logger.info("Final optimizer status:")
    logger.info(f"  Throughput: {status['current_metrics']['throughput']:.2f} tokens/s")
    logger.info(f"  Memory usage: {status['current_metrics']['memory_usage']:.2%}")
    logger.info(f"  Request load: {status['current_metrics']['request_load']}")
    logger.info(f"  Action counts: {status['optimizer_status']['counts']}")

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo using the multi-armed bandit optimizer with vLLM')
    parser = EngineArgs.add_cli_args(parser)
    
    # Add bandit optimization arguments
    parser.add_argument("--monitoring-interval", type=float, default=5.0,
                      help="How often to check metrics and potentially switch models (seconds)")
    parser.add_argument("--cooldown-period", type=float, default=30.0,
                      help="Minimum time between model switches (seconds)")
    parser.add_argument("--use-bandit-engine", action="store_true",
                      help="Use LLMEngineWithBandit instead of regular LLMEngine")
    parser.add_argument("--run-load-pattern", action="store_true",
                      help="Run a simulated load pattern test (low, high, medium)")
    
    args = parser.parse_args()
    
   
        
    main(args) 