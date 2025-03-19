from joblib import load

def train_and_save():
    # 训练并保存模型
    from .perf_model import PrefillPerformanceModel
    model = PrefillPerformanceModel()
    model.train('perf_stats.json', save_path='trained_model.joblib')

def load_and_predict():
    # 直接使用joblib加载模型
    model = load('trained_model.joblib')
    
    # 使用加载的模型进行预测
    test_lengths = [32, 64, 128, 256, 512, 1024]
    print("\n使用加载的模型进行预测:")
    for length in test_lengths:
        time =  float(model.predict([[10, length]])[0])
        print(f"提示长度 {length}: 批次大小 = {10}, "
              f"时间 = {time:.1f} s")

if __name__ == "__main__":
    # 首次运行时取消注释下面这行来训练和保存模型
    #train_and_save()
    
    # 之后可以直接加载保存的模型使用
    load_and_predict() 