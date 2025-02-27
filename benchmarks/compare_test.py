import pandas as pd

my = pd.read_csv("benchmark_results/benchmark_results_my_llama3instruct_20250221-112116.csv")
org = pd.read_csv("benchmark_results/benchmark_results_org_llama3instruct_20250221-101202.csv")

import matplotlib.pyplot as plt
import datetime
def compare_metrics(df1, df2, metrics):
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(df1['request_rate'], df1[metric], label='My', marker='o')
        plt.plot(df2['request_rate'], df2[metric], label='Org', marker='x')
        plt.title(f'Comparison of {metric}')
        plt.xlabel('Request Rate (req/s)')
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        plt.show()
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(f'./benchmark_results/comparison_{metric}_{timestamp}.png')

metrics_to_compare = ['output_throughput', 'total_token_throughput', 'p99_ttft', 'p99_tpot']
compare_metrics(my, org, metrics_to_compare)


