import json
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, Any
import joblib  # 添加 joblib 用于保存/加载模型
import matplotlib.pyplot as plt

class PrefillPerformanceModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = ["batch_size", "total_prompt_tokens", "num_new_tokens"]
        
    def train(self, perf_stats_path: str, save_path: str = None, plot: bool = True):
        # 加载性能数据
        with open(perf_stats_path) as f:
            perf_stats = json.load(f)
        
        # 准备训练数据
        X = np.array([[d['batch_size'], d['total_prompt_tokens'], d['num_new_tokens']] for d in perf_stats])
        y = np.array([d['prefill_time'] for d in perf_stats])
        
        # 训练模型
        self.model.fit(X, y)
        r2_score = self.model.score(X, y)
        print(f"模型 R² 得分: {r2_score:.3f}")
        
        # 可视化结果
        if plot:
            self._plot_results(X, y, perf_stats)
        
        # 保存模型
        if save_path:
            joblib.dump(self.model, save_path)
            print(f"模型已保存到 {save_path}")
            
        return r2_score
    
    def _plot_results(self, X, y, perf_stats):
        """绘制预测结果与实际结果的对比图"""
        y_pred = self.model.predict(X)
        
        # 绘制实际vs预测时间
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred)
        plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
        plt.xlabel('实际时间 (秒)')
        plt.ylabel('预测时间 (秒)')
        plt.title('预测时间 vs 实际时间')
        plt.savefig('prefill_time_prediction.png')
        
        # 绘制batch大小vs时间关系
        no_cache_points = [(d['batch_size'], d['prefill_time']) 
                          for d in perf_stats if not d['has_cache'] and d['num_new_tokens'] == 1]
        no_cache_points.sort(key=lambda x: x[0])
        
        with_cache_points = [(d['batch_size'], d['prefill_time']) 
                          for d in perf_stats if d['has_cache'] and d['num_new_tokens'] == 4]
        with_cache_points.sort(key=lambda x: x[0])
        
        if no_cache_points and with_cache_points:
            plt.figure(figsize=(10, 6))
            plt.plot([p[0] for p in no_cache_points], [p[1] for p in no_cache_points], 'o-', label='无缓存 (1 token)')
            plt.plot([p[0] for p in with_cache_points], [p[1] for p in with_cache_points], 'o-', label='有缓存 (4 tokens)')
            plt.xlabel('Batch 大小')
            plt.ylabel('Prefill 时间 (秒)')
            plt.title('Batch 大小对 Prefill 时间的影响')
            plt.legend()
            plt.savefig('batch_vs_time.png')
    
    @classmethod
    def load_model(cls, model_path: str):
        """从文件加载预训练模型"""
        instance = cls()
        instance.model = joblib.load(model_path)
        return instance

    def predict_time(self, batch_size: int, total_prompt_tokens: int, num_new_tokens: int) -> float:
        """预测给定条件下的prefill时间"""
        return float(self.model.predict([[batch_size, total_prompt_tokens, num_new_tokens]])[0])
    
    def optimize_batch_size(self, 
                          avg_prompt_length: int,
                          num_new_tokens: int = 1,
                          max_batch_size: int = 128,
                          min_batch_size: int = 1) -> Tuple[int, float]:
        """为给定的提示长度找到最佳batch大小"""
        best_throughput = 0
        best_batch = min_batch_size
        
        for batch in range(min_batch_size, max_batch_size + 1):
            total_tokens = batch * avg_prompt_length
            time = self.predict_time(batch, total_tokens, num_new_tokens)
            throughput = batch / time  # 每秒处理的序列数
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch = batch
                
        return best_batch, best_throughput

def main():
    # 训练模型
    model = PrefillPerformanceModel()
    perf_stats_path = "vllm_prefill_perf_stats.json"
    model_save_path = "vllm_prefill_performance_model.pkl"
    
    try:
        model.train(perf_stats_path, model_save_path)
        
        # 打印一些预测结果
        print("\n不同提示长度和token数的最佳batch大小:")
        for avg_length in [32, 64, 128, 256, 512, 1024]:
            for num_tokens in [1, 4]:
                best_batch, throughput = model.optimize_batch_size(
                    avg_prompt_length=avg_length,
                    num_new_tokens=num_tokens
                )
                cache_status = "有缓存" if num_tokens == 4 else "无缓存"
                print(f"提示长度 {avg_length}, {cache_status} ({num_tokens} tokens): "
                      f"最佳batch = {best_batch}, 吞吐量 = {throughput:.1f} seq/s")
    except FileNotFoundError:
        print(f"找不到性能数据文件: {perf_stats_path}")
        print("请先运行 collect_perf_stats.py 收集性能数据")

if __name__ == "__main__":
    main() 