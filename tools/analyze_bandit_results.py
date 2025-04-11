#!/usr/bin/env python3
# Utility to analyze and visualize bandit optimization results

import os
import sys
import json
import argparse
import glob
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_results(results_dir: str, bandit_filename: Optional[str] = None, standard_filename: str = "standard_benchmark.json") -> Dict[str, Any]:
    """Load results from bandit training/evaluation and standard benchmark."""
    results = {
        "bandit": None,
        "standard": None
    }
    
    # Load bandit results
    if bandit_filename:
        bandit_path = os.path.join(results_dir, bandit_filename)
        if os.path.exists(bandit_path):
            with open(bandit_path, 'r') as f:
                results["bandit"] = json.load(f)
    else:
        # Find the most recent bandit results file
        bandit_files = glob.glob(os.path.join(results_dir, "bandit_results_*.json"))
        if bandit_files:
            # Sort by modification time (most recent first)
            most_recent = max(bandit_files, key=os.path.getmtime)
            with open(most_recent, 'r') as f:
                results["bandit"] = json.load(f)
                print(f"Loaded bandit results from {most_recent}")
    
    # Load standard benchmark results
    standard_path = os.path.join(results_dir, standard_filename)
    if os.path.exists(standard_path):
        with open(standard_path, 'r') as f:
            # Parse the JSON - handle both array and object formats
            content = f.read()
            try:
                data = json.loads(content)
                # Check if it's a list or object
                if isinstance(data, list):
                    results["standard"] = data
                else:
                    results["standard"] = [data]
            except json.JSONDecodeError:
                # Some benchmark files might have multiple JSON objects separated by commas
                # Try to parse as a list
                try:
                    data = json.loads(f"[{content}]")
                    results["standard"] = data
                except json.JSONDecodeError:
                    print(f"Error: Could not parse {standard_path} as valid JSON")
                    results["standard"] = None
    
    return results

def extract_throughput_by_request_rate(bandit_results: Dict[str, Any], standard_results: List[Dict[str, Any]]) -> Dict[float, Dict[str, float]]:
    """Extract throughput values for each request rate from both result sets."""
    throughput_by_rate = {}
    
    # Extract from bandit results
    if bandit_results and "testing_results" in bandit_results:
        for test_result in bandit_results["testing_results"]:
            rate = test_result["request_rate"]
            throughput = test_result["throughput"]
            throughput_by_rate[rate] = {"bandit": throughput}
    
    # Extract from standard benchmark results
    if standard_results:
        for result in standard_results:
            if "request_rate" in result and result["request_rate"] != "inf":
                rate = float(result["request_rate"])
                if "output_throughput" in result:
                    if rate in throughput_by_rate:
                        throughput_by_rate[rate]["standard"] = result["output_throughput"]
                    else:
                        throughput_by_rate[rate] = {"standard": result["output_throughput"]}
    
    return throughput_by_rate

def plot_training_progression(bandit_results: Dict[str, Any], output_path: Optional[str] = None):
    """Plot the progression of metrics during bandit training."""
    if not bandit_results or "training_metrics" not in bandit_results:
        print("Error: No training metrics found in bandit results")
        return
    
    metrics = bandit_results["training_metrics"]
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot throughput history
    if "throughput_history" in metrics and metrics["throughput_history"]:
        axs[0].plot(metrics["throughput_history"], 'b-', linewidth=2)
        axs[0].set_title('Throughput During Training')
        axs[0].set_xlabel('Training Step')
        axs[0].set_ylabel('Throughput (tokens/s)')
        axs[0].grid(True)
    
    # Plot request load history
    if "request_load_history" in metrics and metrics["request_load_history"]:
        axs[1].plot(metrics["request_load_history"], 'g-', linewidth=2)
        axs[1].set_title('Request Load During Training')
        axs[1].set_xlabel('Training Step')
        axs[1].set_ylabel('Number of Requests')
        axs[1].grid(True)
    
    # Plot reward history
    if "reward_history" in metrics and metrics["reward_history"]:
        axs[2].plot(metrics["reward_history"], 'r-', linewidth=2)
        axs[2].set_title('Reward History')
        axs[2].set_xlabel('Training Step')
        axs[2].set_ylabel('Reward')
        axs[2].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Training progression plot saved to {output_path}")
    else:
        plt.show()

def plot_action_history(bandit_results: Dict[str, Any], output_path: Optional[str] = None):
    """Plot the history of actions taken by the bandit optimizer."""
    if not bandit_results or "training_metrics" not in bandit_results or "action_history" not in bandit_results["training_metrics"]:
        print("Error: No action history found in bandit results")
        return
    
    action_history = bandit_results["training_metrics"]["action_history"]
    
    if not action_history:
        print("No actions were taken during training")
        return
    
    # Extract action types and timestamps
    actions = [entry["action"] for entry in action_history]
    timestamps = [entry["time"] - action_history[0]["time"] for entry in action_history]
    
    # Count occurrences of each action type
    action_counts = {}
    for action in actions:
        if action in action_counts:
            action_counts[action] += 1
        else:
            action_counts[action] = 1
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot action timeline
    unique_actions = list(set(actions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_actions)))
    
    # Create a scatter plot with different colors for each action type
    for i, action_type in enumerate(unique_actions):
        action_indices = [j for j, a in enumerate(actions) if a == action_type]
        action_times = [timestamps[j] for j in action_indices]
        ax1.scatter(action_times, [1] * len(action_times), label=action_type, 
                   s=100, color=colors[i], marker='o')
    
    ax1.set_title('Action Timeline')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_yticks([])
    ax1.legend()
    ax1.grid(True)
    
    # Plot action counts
    ax2.bar(action_counts.keys(), action_counts.values(), color=colors[:len(action_counts)])
    ax2.set_title('Action Counts')
    ax2.set_xlabel('Action Type')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Action history plot saved to {output_path}")
    else:
        plt.show()

def plot_throughput_comparison(throughput_by_rate: Dict[float, Dict[str, float]], output_path: Optional[str] = None):
    """Plot throughput comparison between bandit and standard approaches."""
    if not throughput_by_rate:
        print("Error: No throughput data available for comparison")
        return
    
    # Extract data for plotting
    rates = sorted(throughput_by_rate.keys())
    bandit_throughput = [throughput_by_rate[rate].get("bandit", 0) for rate in rates]
    standard_throughput = [throughput_by_rate[rate].get("standard", 0) for rate in rates]
    
    # Create bar plot
    x = np.arange(len(rates))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, bandit_throughput, width, label='Bandit Optimization')
    rects2 = ax.bar(x + width/2, standard_throughput, width, label='Standard Approach')
    
    ax.set_title('Throughput Comparison: Bandit vs Standard')
    ax.set_xlabel('Request Rate (req/s)')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_xticks(x)
    ax.set_xticklabels(rates)
    ax.legend()
    
    # Add throughput improvement percentages
    for i in range(len(rates)):
        if standard_throughput[i] > 0:
            improvement = ((bandit_throughput[i] / standard_throughput[i]) - 1) * 100
            if improvement > 0:
                label = f"+{improvement:.1f}%"
                color = 'green'
            else:
                label = f"{improvement:.1f}%"
                color = 'red'
                
            ax.annotate(label,
                        xy=(x[i], max(bandit_throughput[i], standard_throughput[i]) + 5),
                        ha='center', va='bottom',
                        color=color, fontweight='bold')
    
    # Add throughput values
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Throughput comparison plot saved to {output_path}")
    else:
        plt.show()

def analyze_results(results_dir: str, bandit_filename: Optional[str] = None, standard_filename: str = "standard_benchmark.json", save_plots: bool = False):
    """Analyze bandit optimization results and compare with standard benchmark."""
    # Load results
    results = load_results(results_dir, bandit_filename, standard_filename)
    
    if not results["bandit"]:
        print("Error: No bandit results found")
        return
    
    # Create a directory for plots if saving
    plots_dir = None
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plots_dir = os.path.join(results_dir, f"plots_{timestamp}")
        os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training progression
    if plots_dir:
        plot_training_progression(results["bandit"], os.path.join(plots_dir, "training_progression.png"))
    else:
        plot_training_progression(results["bandit"])
    
    # Plot action history
    if plots_dir:
        plot_action_history(results["bandit"], os.path.join(plots_dir, "action_history.png"))
    else:
        plot_action_history(results["bandit"])
    
    # Compare throughput with standard benchmark if available
    if results["standard"]:
        throughput_by_rate = extract_throughput_by_request_rate(results["bandit"], results["standard"])
        
        # Print throughput comparison
        print("\nThroughput Comparison:")
        print("=====================")
        print(f"{'Request Rate':<15} {'Bandit':<15} {'Standard':<15} {'Improvement':<15}")
        print("-" * 60)
        
        for rate in sorted(throughput_by_rate.keys()):
            bandit_tp = throughput_by_rate[rate].get("bandit", 0)
            standard_tp = throughput_by_rate[rate].get("standard", 0)
            
            if standard_tp > 0:
                improvement = ((bandit_tp / standard_tp) - 1) * 100
                imp_str = f"{improvement:+.2f}%"
            else:
                imp_str = "N/A"
                
            print(f"{rate:<15.1f} {bandit_tp:<15.2f} {standard_tp:<15.2f} {imp_str:<15}")
        
        # Plot throughput comparison
        if plots_dir:
            plot_throughput_comparison(throughput_by_rate, os.path.join(plots_dir, "throughput_comparison.png"))
        else:
            plot_throughput_comparison(throughput_by_rate)
    else:
        print("No standard benchmark results available for comparison")
    
    # Print summary of bandit actions
    if "training_metrics" in results["bandit"] and "action_history" in results["bandit"]["training_metrics"]:
        action_history = results["bandit"]["training_metrics"]["action_history"]
        action_counts = {}
        
        for entry in action_history:
            action = entry["action"]
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 1
        
        print("\nBandit Action Summary:")
        print("=====================")
        for action, count in action_counts.items():
            print(f"{action}: {count}")
    
    # Print best throughput from testing
    if "best_throughput" in results["bandit"] and "best_rate" in results["bandit"]:
        print("\nBest Performance:")
        print("===============")
        print(f"Best throughput: {results['bandit']['best_throughput']:.2f} tokens/s at request rate {results['bandit']['best_rate']} req/s")
    
    if plots_dir:
        print(f"\nAll plots saved to {plots_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and visualize bandit optimization results")
    parser.add_argument("--results-dir", type=str, default="bandit_results",
                       help="Directory containing results files")
    parser.add_argument("--bandit-file", type=str, default=None,
                       help="Specific bandit results file (if not provided, most recent will be used)")
    parser.add_argument("--standard-file", type=str, default="standard_benchmark.json",
                       help="Standard benchmark results file")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save plots to files instead of displaying them")
    return parser.parse_args()

def main():
    args = parse_args()
    analyze_results(args.results_dir, args.bandit_file, args.standard_file, args.save_plots)

if __name__ == "__main__":
    main() 