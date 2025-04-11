# VLLM Prefill 性能分析工具

这个工具集用于收集VLLM引擎在prefill阶段的性能数据，并训练一个预测模型来预测不同场景下的prefill时间。

## 组件

1. `collect_perf_stats.py` - 收集VLLM引擎在不同batch大小下的prefill性能数据
2. `perf_model.py` - 基于收集的性能数据训练线性回归模型，并提供预测功能
3. `run_perf_analysis.py` - 整合上述功能的运行脚本

## 安装依赖

确保已安装以下Python包：
```bash
pip install vllm torch numpy scikit-learn matplotlib tqdm joblib
```

## 使用方法

### 1. 一键运行（收集数据并训练模型）

```bash
python run_perf_analysis.py --model_path /path/to/model --dataset_path /path/to/sharegpt.json --collect_data --train_model
```

### 2. 仅收集性能数据

```bash
python run_perf_analysis.py --model_path /path/to/model --dataset_path /path/to/sharegpt.json --collect_data
```

### 3. 仅训练预测模型（使用已有的性能数据）

```bash
python run_perf_analysis.py --model_path /path/to/model --dataset_path /path/to/sharegpt.json --train_model
```

### 4. 自定义参数

```bash
python run_perf_analysis.py --model_path /path/to/model --dataset_path /path/to/sharegpt.json --collect_data --train_model --batch_sizes 1 2 4 8 16 32 --num_samples 100 --output_dir ./my_results
```

## 参数说明

- `--model_path`: 模型路径（必需）
- `--dataset_path`: 数据集路径（必需）
- `--output_dir`: 输出目录，默认为 `./perf_results`
- `--collect_data`: 是否收集新的性能数据
- `--train_model`: 是否训练预测模型
- `--batch_sizes`: 要测试的batch大小列表，默认为 `[1, 2, 4, 8, 16, 32, 64, 128]`
- `--num_samples`: 从数据集中采样的请求数量，默认为 200

## 收集的性能数据

性能数据收集过程会测试两种情况：
1. **初始prefill（无缓存）**: 对每个batch生成1个token的时间
2. **有缓存下的prefill**: 对已有缓存的情况下生成4个token的时间

测试结果将保存为JSON文件，包含以下字段：
- `batch_size`: 批处理大小
- `total_prompt_tokens`: 所有prompt的token总数
- `avg_prompt_len`: 平均prompt长度
- `num_new_tokens`: 新生成的token数（1或4）
- `prefill_time`: prefill阶段花费的时间（秒）
- `has_cache`: 是否使用了KV缓存

## 预测模型

预测模型使用线性回归算法，基于以下特征：
- batch大小
- prompt token总数
- 新生成token数

模型会自动生成可视化图表：
1. 预测时间与实际时间的对比
2. batch大小与prefill时间的关系

## 示例输出

性能模型训练完成后，会输出不同prompt长度和不同场景（有缓存/无缓存）下的最佳batch大小和预计吞吐量。例如：

```
不同提示长度和token数的最佳batch大小:
提示长度 32, 无缓存 (1 tokens): 最佳batch = 64, 吞吐量 = 128.5 seq/s
提示长度 32, 有缓存 (4 tokens): 最佳batch = 128, 吞吐量 = 320.7 seq/s
提示长度 64, 无缓存 (1 tokens): 最佳batch = 32, 吞吐量 = 96.4 seq/s
...
```

## 注意事项

- 收集性能数据需要GPU资源，且可能需要较长时间
- 对于大型模型，可能需要减小测试的batch大小范围以避免OOM错误 