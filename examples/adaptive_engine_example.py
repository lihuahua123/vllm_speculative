#!/usr/bin/env python3
"""
Example script showing how to use the adaptive engine with both bandit and Q-learning approaches.

This script:
1. Creates sample requests for training and testing
2. Sets up both bandit and Q-learning engines
3. Trains both optimizers
4. Evaluates their performance under different request rates
5. Compares results

Usage:
    python adaptive_engine_example.py --model <model_name> --strategy <bandit|q_learning>
"""

import os
import time
import asyncio
import argparse
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.adaptive_engine_factory import create_adaptive_engine, OptimizationStrategy
from vllm.engine.llm_engine_with_bandit import TrainingRequest as BanditTrainingRequest
from vllm.engine.llm_engine_with_q_learning import TrainingRequest as QLearningTrainingRequest
from vllm.sampling_params import SamplingParams

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample prompts for training and evaluation
SAMPLE_PROMPTS = [
    "Explain the concept of reinforcement learning in simple terms.",
    "Write a short story about a robot that learns to paint.",
    "What are the key differences between Q-learning and deep Q-learning?",
    "Summarize the principles of transfer learning in AI.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "Explain how transformer models work in natural language processing.",
    "What are the ethical considerations in deploying AI systems?",
    "Compare and contrast supervised and unsupervised learning.",
    "Write a short poem about artificial intelligence.",
    "Explain the concept of backpropagation in neural networks."
]

def create_training_requests(
    num_requests: int,
    strategy: OptimizationStrategy
) -> Tuple[List, List[SamplingParams]]:
    """Create training requests for the specified strategy.
    
    Args:
        num_requests: Number of requests to create
        strategy: The optimization strategy
        
    Returns:
        Tuple of (training_requests, sampling_params)
    """
    # Create enough prompts by repeating the sample prompts
    prompts = []
    for _ in range((num_requests + len(SAMPLE_PROMPTS) - 1) // len(SAMPLE_PROMPTS)):
        prompts.extend(SAMPLE_PROMPTS)
    prompts = prompts[:num_requests]
    
    # Create sampling parameters with different settings to simulate variety
    sampling_params = []
    for i in range(num_requests):
        temperature = 0.7 + (i % 5) * 0.1  # 0.7, 0.8, 0.9, 1.0, 1.1
        max_tokens = 128 + (i % 3) * 64    # 128, 192, 256
        
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        sampling_params.append(params)
    
    # Create the appropriate training request objects
    if strategy == "bandit":
        training_requests = [
            BanditTrainingRequest(
                prompt=prompt,
                prompt_len=len(prompt.split()),
                expected_output_len=params.max_tokens
            )
            for prompt, params in zip(prompts, sampling_params)
        ]
    elif strategy == "q_learning":
        training_requests = [
            QLearningTrainingRequest(
                prompt=prompt,
                prompt_len=len(prompt.split()),
                expected_output_len=params.max_tokens
            )
            for prompt, params in zip(prompts, sampling_params)
        ]
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    return training_requests, sampling_params

async def train_and_evaluate(
    engine_args: EngineArgs,
    strategy: OptimizationStrategy,
    training_iterations: int = 20,
    num_training_requests: int = 100,
    num_test_requests: int = 30,
    request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
    test_duration: float = 60.0,
    results_dir: str = "adaptive_results"
) -> Dict:
    """Train and evaluate an adaptive engine with the specified strategy.
    
    Args:
        engine_args: Engine configuration
        strategy: Optimization strategy to use
        training_iterations: Number of training iterations
        num_training_requests: Number of requests for training
        num_test_requests: Number of requests for testing
        request_rates: List of request rates to test
        test_duration: Duration of each test in seconds
        results_dir: Directory to save results
    
    Returns:
        Dictionary with evaluation results
    """
    # Create an adaptive engine with the specified strategy
    engine = create_adaptive_engine(
        engine_args=engine_args,
        optimization_strategy=strategy,
        monitoring_interval=3.0,
        cooldown_period=15.0,
        memory_penalty_coefficient=0.3,
        # Q-learning specific parameters
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.3,
        exploration_decay=0.995,
        min_exploration_rate=0.05
    )
    
    # Create training and test requests
    training_requests, training_params = create_training_requests(
        num_requests=num_training_requests,
        strategy=strategy
    )
    
    test_requests, test_params = create_training_requests(
        num_requests=num_test_requests,
        strategy=strategy
    )
    
    # Train and evaluate based on strategy
    if strategy == "bandit":
        results = await engine.train_and_evaluate_bandit(
            training_requests=training_requests,
            test_requests=test_requests,
            sampling_params=training_params,
            training_iterations=training_iterations,
            request_rates=request_rates,
            test_duration=test_duration,
            save_results=True,
            results_dir=f"{results_dir}/{strategy}"
        )
    elif strategy == "q_learning":
        results = await engine.train_and_evaluate_q_learning(
            training_requests=training_requests,
            test_requests=test_requests,
            sampling_params=training_params,
            training_iterations=training_iterations,
            request_rates=request_rates,
            test_duration=test_duration,
            save_results=True,
            save_q_table=True,
            results_dir=f"{results_dir}/{strategy}"
        )
    elif strategy == "ilp":
        results = await engine.train_and_evaluate_ilp(
            training_requests=training_requests,
            test_requests=test_requests,
            sampling_params=training_params,
            training_iterations=training_iterations,
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    return results

def plot_results(results: Dict, strategy: str, output_dir: str):
    """Plot results from the training and evaluation.
    
    Args:
        results: Results dictionary from training and evaluation
        strategy: Strategy used (for titles and filenames)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Throughput over training
    plt.figure(figsize=(10, 6))
    plt.plot(results["training_metrics"]["throughput_history"])
    plt.title(f"{strategy.capitalize()} - Throughput During Training")
    plt.xlabel("Training Step")
    plt.ylabel("Throughput (tokens/s)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/{strategy}_throughput.png")
    
    # Plot 2: Rewards over training (if available)
    if "reward_history" in results["training_metrics"] and results["training_metrics"]["reward_history"]:
        plt.figure(figsize=(10, 6))
        plt.plot(results["training_metrics"]["reward_history"])
        plt.title(f"{strategy.capitalize()} - Rewards During Training")
        plt.xlabel("Training Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(f"{output_dir}/{strategy}_rewards.png")
    
    # Plot 3: Request Load vs Memory Usage
    if (results["training_metrics"]["request_load_history"] and 
        results["training_metrics"]["memory_usage_history"]):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            results["training_metrics"]["request_load_history"],
            results["training_metrics"]["memory_usage_history"],
            alpha=0.6
        )
        plt.title(f"{strategy.capitalize()} - Request Load vs Memory Usage")
        plt.xlabel("Request Load")
        plt.ylabel("Memory Usage")
        plt.grid(True)
        plt.savefig(f"{output_dir}/{strategy}_load_vs_memory.png")
    
    # Plot 4: Throughput vs Request Rate
    if results["testing_results"]:
        rates = [result["request_rate"] for result in results["testing_results"]]
        throughputs = [result["throughput"] for result in results["testing_results"]]
        
        plt.figure(figsize=(10, 6))
        plt.bar(rates, throughputs)
        plt.title(f"{strategy.capitalize()} - Throughput vs Request Rate")
        plt.xlabel("Request Rate (req/s)")
        plt.ylabel("Throughput (tokens/s)")
        plt.grid(True, axis='y')
        plt.savefig(f"{output_dir}/{strategy}_request_rate_throughput.png")
    
    # Plot 5: Actions Taken During Test
    if results["testing_results"]:
        # Find test with highest throughput
        best_test = max(results["testing_results"], key=lambda x: x["throughput"])
        
        if "q_learning_actions" in best_test:
            actions = best_test["q_learning_actions"]
        elif "bandit_actions" in best_test:
            actions = best_test["bandit_actions"]
        else:
            actions = []
            
        if actions:
            action_times = [a["time"] for a in actions]
            action_names = [a["action"] for a in actions]
            
            plt.figure(figsize=(12, 6))
            for i, (time, action) in enumerate(zip(action_times, action_names)):
                plt.axvline(x=time, color='r', linestyle='--', alpha=0.6)
                plt.text(time, i % 5, action, rotation=90, alpha=0.8)
            
            plt.title(f"{strategy.capitalize()} - Actions During Best Test (Rate: {best_test['request_rate']} req/s)")
            plt.xlabel("Time (s)")
            plt.yticks([])
            plt.xlim(0, best_test["test_duration"])
            plt.grid(True)
            plt.savefig(f"{output_dir}/{strategy}_actions.png")

def compare_strategies(bandit_results: Dict, q_learning_results: Dict, output_dir: str):
    """Create plots comparing bandit and Q-learning results.
    
    Args:
        bandit_results: Results from bandit evaluation
        q_learning_results: Results from Q-learning evaluation
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Compare throughput vs request rate
    if (bandit_results["testing_results"] and q_learning_results["testing_results"]):
        bandit_rates = [result["request_rate"] for result in bandit_results["testing_results"]]
        bandit_throughputs = [result["throughput"] for result in bandit_results["testing_results"]]
        
        q_rates = [result["request_rate"] for result in q_learning_results["testing_results"]]
        q_throughputs = [result["throughput"] for result in q_learning_results["testing_results"]]
        
        # Set up bar positions
        x = np.arange(len(bandit_rates))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bandit_bars = ax.bar(x - width/2, bandit_throughputs, width, label='Bandit')
        q_bars = ax.bar(x + width/2, q_throughputs, width, label='Q-Learning')
        
        ax.set_title('Throughput Comparison by Request Rate')
        ax.set_xlabel('Request Rate (req/s)')
        ax.set_ylabel('Throughput (tokens/s)')
        ax.set_xticks(x)
        ax.set_xticklabels(bandit_rates)
        ax.legend()
        ax.grid(True, axis='y')
        
        # Add value labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
                
        add_labels(bandit_bars)
        add_labels(q_bars)
        
        plt.savefig(f"{output_dir}/comparison_throughput.png")
    
    # Save summary comparison as JSON
    comparison = {
        "bandit": {
            "best_throughput": bandit_results.get("best_throughput", 0),
            "best_rate": bandit_results.get("best_rate", 0),
            "total_rewards": sum(bandit_results["training_metrics"].get("reward_history", [0]))
        },
        "q_learning": {
            "best_throughput": q_learning_results.get("best_throughput", 0),
            "best_rate": q_learning_results.get("best_rate", 0),
            "total_rewards": sum(q_learning_results["training_metrics"].get("reward_history", [0]))
        }
    }
    
    with open(f"{output_dir}/comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
        
    return comparison

async def main():
    try:
        from vllm.utils import FlexibleArgumentParser
    except ImportError:
        from argparse import ArgumentParser as FlexibleArgumentParser
    
    parser = FlexibleArgumentParser(
        description="Train and evaluate the LLM adaptive engine for vLLM"
    )
    
    # Add all the original vLLM arguments
    from vllm.engine.arg_utils import EngineArgs
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument("--strategy", type=str, choices=["bandit", "q_learning", "ilp", "both"],
                        default="both", help="Optimization strategy to use")
    parser.add_argument("--training-iterations", type=int, default=5,
                        help="Number of training iterations")
    parser.add_argument("--results-dir", type=str, default="adaptive_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    

    
    if args.strategy == "both" or args.strategy == "bandit":
        logger.info("Training and evaluating bandit strategy")
        bandit_results = await train_and_evaluate(
            engine_args=engine_args,
            strategy="bandit",
            training_iterations=args.training_iterations,
            results_dir=args.results_dir
        )
        plot_results(bandit_results, "bandit", f"{args.results_dir}/plots")
    
    if args.strategy == "both" or args.strategy == "q_learning":
        logger.info("Training and evaluating Q-learning strategy")
        q_learning_results = await train_and_evaluate(
            engine_args=engine_args,
            strategy="q_learning",
            training_iterations=args.training_iterations,
            results_dir=args.results_dir
        )
        plot_results(q_learning_results, "q_learning", f"{args.results_dir}/plots")
    if args.strategy == "both" or args.strategy == "ilp":
        logger.info("Training and evaluating ILP strategy")
        ilp_results = await train_and_evaluate(
            engine_args=engine_args,
            strategy="ilp",
            training_iterations=args.training_iterations,
            results_dir=args.results_dir
        )
        plot_results(ilp_results, "ilp", f"{args.results_dir}/plots")
    if args.strategy == "both":
        logger.info("Comparing strategies")
        comparison = compare_strategies(
            bandit_results=bandit_results,
            q_learning_results=q_learning_results,
            ilp_results=ilp_results,
            output_dir=f"{args.results_dir}/comparison"
        )
        
        logger.info("Comparison summary:")
        logger.info(f"Bandit best throughput: {comparison['bandit']['best_throughput']:.2f} tokens/s at {comparison['bandit']['best_rate']} req/s")
        logger.info(f"Q-learning best throughput: {comparison['q_learning']['best_throughput']:.2f} tokens/s at {comparison['q_learning']['best_rate']} req/s")
        
        # Determine winner
        if comparison['bandit']['best_throughput'] > comparison['q_learning']['best_throughput']:
            logger.info("Bandit strategy performed better")
        elif comparison['q_learning']['best_throughput'] > comparison['bandit']['best_throughput']:
            logger.info("Q-learning strategy performed better")
        else:
            logger.info("Both strategies performed similarly")

if __name__ == "__main__":
    asyncio.run(main()) 