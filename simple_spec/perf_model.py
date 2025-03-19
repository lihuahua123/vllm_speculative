import json
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple
import joblib  # 添加 joblib 用于保存/加载模型

class PrefillPerformanceModel:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, perf_stats_path: str, save_path: str = None):
        # Load performance data
        with open(perf_stats_path) as f:
            perf_stats = json.load(f)
        
        # Prepare training data
        X = np.array([[d['batch_size'], d['avg_prompt_len']] for d in perf_stats])
        y = np.array([d['prefill_time'] for d in perf_stats])
        
        # Train model
        self.model.fit(X, y)
        print(f"Model R² score: {self.model.score(X, y):.3f}")
        
        # 保存模型
        if save_path:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, model_path: str):
        """从文件加载预训练模型"""
        instance = cls()
        instance.model = joblib.load(model_path)
        return instance

    def predict_time(self, batch_size: int, prompt_length: int) -> float:
        return float(self.model.predict([[batch_size, prompt_length]])[0])
    
    def optimize_batch_size(self, 
                          prompt_length: int,
                          max_batch_size: int = 128,
                          min_batch_size: int = 1) -> Tuple[int, float]:
        """Find optimal batch size for given prompt length"""
        best_throughput = 0
        best_batch = min_batch_size
        
        for batch in range(min_batch_size, max_batch_size + 1):
            time = self.predict_time(batch, prompt_length)
            throughput = batch / time  # sequences per second
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch = batch
                
        return best_batch, best_throughput

def main():
    # Train model
    model = PrefillPerformanceModel()
    model.train('perf_stats_deepseek-aiDeepSeek-R1-Distill-Qwen-7B.json','deepseek-aiDeepSeek-R1-Distill-Qwen-7B.pkl')
    model = PrefillPerformanceModel()
    model.train('perf_stats_DeepSeek-R1-DRAFT-Qwen2.5-0.5B.json','DeepSeek-R1-DRAFT-Qwen2.5-0.5B.pkl')
    # Print some predictions
    test_lengths = [32, 64, 128, 256, 512, 1024]
    print("\nOptimal batch sizes for different prompt lengths:")
    for length in test_lengths:
        best_batch, throughput = model.optimize_batch_size(length)
        print(f"Prompt length {length}: optimal batch = {best_batch}, "
              f"throughput = {throughput:.1f} seq/s")

if __name__ == "__main__":
    main() 