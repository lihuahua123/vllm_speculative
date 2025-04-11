#!/usr/bin/env python3
# Tool for training and evaluating the multi-armed bandit optimizer in vLLM

import os
import sys
import time
import json
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional

# Add the repository root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine_with_bandit import LLMEngineWithBandit, TrainingRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sampling_params(temperature: float = 0.7, 
                           top_p: float = 0.95, 
                           max_tokens: int = 100) -> Dict[str, Any]:
    """Create sampling parameters for text generation."""
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

def load_requests_from_file(file_path: str, 
                           tokenizer_id: str, 
                           num_requests: int = 20,
                           max_output_tokens: Optional[int] = None) -> List[TrainingRequest]:
    """Load requests from a dataset file (such as ShareGPT)."""
    
    logger.info(f"Loading requests from {file_path}")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.warning("Using approximate token counting method")
        tokenizer = None
        
    # Load dataset file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset file: {e}")
        return []
    
    # Process dataset entries
    requests = []
    
    if isinstance(data, list):
        # ShareGPT/conversational format
        for entry in data[:num_requests]:
            if 'conversations' in entry and isinstance(entry['conversations'], list):
                # Extract prompt from the first message
                prompt = ""
                for msg in entry['conversations']:
                    if isinstance(msg, dict) and 'value' in msg:
                        prompt += msg['value'] + "\n"
                        break
                
                if prompt:
                    # Get prompt length in tokens
                    prompt_len = len(tokenizer.encode(prompt)) if tokenizer else len(prompt.split())
                    
                    # Estimate output length (or use provided value)
                    output_len = max_output_tokens if max_output_tokens else 100
                    
                    requests.append(TrainingRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                        multi_modal_data=None
                    ))
    
    # If no requests were created, generate some simple ones
    if not requests:
        logger.warning("No valid requests found in the dataset. Creating simple ones.")
        
        simple_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What is the capital of France?",
            "Write a short story about a robot learning to feel emotions.",
            "List five benefits of regular exercise.",
            "Explain how transformers work in deep learning.",
            "Summarize the plot of Romeo and Juliet.",
            "What are the main challenges in climate change?",
            "Give me a recipe for chocolate chip cookies.",
            "Discuss the impact of social media on society.",
            "Explain the difference between renewable and non-renewable energy."
        ]
        
        for prompt in simple_prompts[:num_requests]:
            prompt_len = len(tokenizer.encode(prompt)) if tokenizer else len(prompt.split())
            output_len = max_output_tokens if max_output_tokens else 100
            
            requests.append(TrainingRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                expected_output_len=output_len,
                multi_modal_data=None
            ))
            
    # Ensure we have enough requests
    while len(requests) < num_requests:
        # Duplicate the last request to make up the numbers
        requests.append(requests[-1])
    
    logger.info(f"Loaded {len(requests)} requests")
    return requests

async def run_bandit_training_and_evaluation(args):
    """Run the bandit training and evaluation process."""
    
    # Get engine arguments directly from the args object
    engine_args = EngineArgs.from_cli_args(args)
    
    # Create LLM Engine with Bandit
    engine = LLMEngineWithBandit.from_engine_args(
        engine_args,
        bandit_monitoring_interval=args.monitoring_interval,
        bandit_cooldown_period=args.cooldown_period
    )
    
    logger.info(f"Created LLM Engine with model {args.model}")
    
    # Load requests for training and testing
    training_requests = load_requests_from_file(
        file_path=args.dataset_path,
        tokenizer_id=args.model,
        num_requests=args.num_training_requests,
        max_output_tokens=args.max_output_tokens
    )
    
    # Use same requests for testing (or could load different ones)
    test_requests = training_requests[:args.num_test_requests]
    
    # Create sampling parameters
    sampling_params = [
        SamplingParams(
            temperature=0.0, 
            max_tokens=args.max_output_tokens
        )
        for _ in range(len(training_requests))
    ]
    
    # Run training and evaluation
    results = await engine.train_and_evaluate_bandit(
        training_requests=training_requests,
        test_requests=test_requests,
        sampling_params=sampling_params,
        training_iterations=args.training_iterations,
        request_rates=args.request_rates,
        test_duration=args.test_duration,
        save_results=args.save_results,
        results_dir=args.results_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("BANDIT OPTIMIZATION RESULTS SUMMARY")
    print("="*50)
    print(f"Best throughput: {results['best_throughput']:.2f} tokens/s at request rate {results['best_rate']} req/s")
    
    # Print final configuration 
    bandit_status = engine.get_optimization_status()
    print("\nFinal bandit configuration:")
    print(f"  Using n-gram model: {engine.bandit_manager.optimizer.using_ngram_model}")
    print(f"  Using neural model: {engine.bandit_manager.optimizer.using_neural_model}")
    print(f"  Speculative decoding disabled: {engine.bandit_manager.optimizer.spec_decoding_disabled}")
    
    print("\nAction counts during training:")
    for action, count in bandit_status["optimizer_status"]["counts"].items():
        print(f"  {action}: {count}")
    
    print("\nTest results by request rate:")
    for result in results["testing_results"]:
        print(f"  {result['request_rate']} req/s: {result['throughput']:.2f} tokens/s, "
              f"{result['completed_requests']} requests completed")
        
    return results

def parse_args():
    try:
        from vllm.utils import FlexibleArgumentParser
    except ImportError:
        from argparse import ArgumentParser as FlexibleArgumentParser
    
    parser = FlexibleArgumentParser(
        description="Train and evaluate the multi-armed bandit optimizer for vLLM"
    )
    
    # Add all the original vLLM arguments
    from vllm.engine.arg_utils import EngineArgs
    parser = EngineArgs.add_cli_args(parser)
    
    # Add bandit-specific arguments
    bandit_group = parser.add_argument_group("Bandit Optimization")
    bandit_group.add_argument("--monitoring-interval", type=float, default=5.0,
                      help="How often to check metrics and potentially switch models (seconds)")
    bandit_group.add_argument("--cooldown-period", type=float, default=30.0,
                      help="Minimum time between model switches (seconds)")
    
    # Training and testing parameters
    training_group = parser.add_argument_group("Training and Evaluation")
    training_group.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset file")
    training_group.add_argument("--num-training-requests", type=int, default=20,
                        help="Number of requests to use for training")
    training_group.add_argument("--num-test-requests", type=int, default=20,
                        help="Number of requests to use for testing")
    training_group.add_argument("--max-output-tokens", type=int, default=100,
                        help="Maximum number of output tokens per request")
    training_group.add_argument("--training-iterations", type=int, default=50,
                        help="Number of training iterations")
    training_group.add_argument("--request-rates", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0],
                        help="Request rates to test (requests per second)")
    training_group.add_argument("--test-duration", type=float, default=60.0,
                        help="Duration of each test in seconds")
    
    # Output parameters
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--save-results", action="store_true",
                        help="Save results to file")
    output_group.add_argument("--results-dir", type=str, default="bandit_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Set appropriate defaults
    if args.num_gpu_blocks_override is None:
        # This will allow the engine to automatically determine the number of blocks
        args.num_gpu_blocks_override = 0
    
    return args

def main():
    args = parse_args()
    
    # Run the training and evaluation
    asyncio.run(run_bandit_training_and_evaluation(args))

if __name__ == "__main__":
    main() 