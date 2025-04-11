import os
import argparse
from collect_perf_stats import collect_prefill_performance_stats
from perf_model import PrefillPerformanceModel

def main():
    # VLLM_USE_V1=0  python simple_spec/run_perf_analysis.py --big_model_path /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B --small_model_path alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B --dataset_path /data/sharegpt.json --output_dir ./perf_results --collect_data --train_model
    parser = argparse.ArgumentParser(description="VLLM Prefill性能分析工具")
    parser.add_argument("--big_model_path", type=str, required=True, help="大模型路径")
    parser.add_argument("--small_model_path", type=str, required=True, help="小模型路径")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="./perf_results", help="输出目录")
    parser.add_argument("--collect_data", action="store_true", help="是否收集新的性能数据")
    parser.add_argument("--train_model", action="store_true", help="是否训练预测模型")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128], 
                        help="要测试的batch大小列表")
    parser.add_argument("--num_samples", type=int, default=200, help="从数据集中采样的请求数量")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取模型名称用于命名文件
    big_model_name = os.path.basename(args.big_model_path.rstrip("/"))
    small_model_name = os.path.basename(args.small_model_path.rstrip("/"))
    
    # 设置文件路径
    big_perf_stats_path = os.path.join(args.output_dir, f"vllm_prefill_perf_stats_{big_model_name}.json")
    small_perf_stats_path = os.path.join(args.output_dir, f"vllm_prefill_perf_stats_{small_model_name}.json")
    big_model_save_path = os.path.join(args.output_dir, f"vllm_prefill_model_{big_model_name}.pkl")
    small_model_save_path = os.path.join(args.output_dir, f"vllm_prefill_model_{small_model_name}.pkl")
    
    # 收集性能数据
    if args.collect_data:
        print(f"开始收集VLLM prefill性能数据，使用模型: {args.big_model_path}")
        collect_prefill_performance_stats(
            model_path=args.big_model_path,
            dataset_path=args.dataset_path,
            output_path=big_perf_stats_path,
            batch_sizes=args.batch_sizes,
            num_samples=args.num_samples,
            is_small_model=False
        )
        # 如果是ngram，通常就是0.1s
        collect_prefill_performance_stats(
            model_path=args.small_model_path,
            dataset_path=args.dataset_path,
            output_path=small_perf_stats_path,
            batch_sizes=args.batch_sizes,
            num_samples=args.num_samples,
            is_small_model=True
        )
    
    # 训练预测模型
    if args.train_model:
        print(f"开始训练prefill性能预测模型")
        model = PrefillPerformanceModel()
        
        try:
            r2_score = model.train(big_perf_stats_path, big_model_save_path)
            print(f"big模型训练完成，R²得分: {r2_score:.3f}")
            print(f"big模型保存到: {big_model_save_path}")
            r2_score = model.train(small_perf_stats_path, small_model_save_path)
            print(f"small模型训练完成，R²得分: {r2_score:.3f}")
            print(f"small模型保存到: {small_model_save_path}")
        except FileNotFoundError:
            print(f"找不到性能数据文件: {big_perf_stats_path},{small_perf_stats_path}")
            print("请先使用--collect_data参数收集性能数据")

if __name__ == "__main__":
    main() 