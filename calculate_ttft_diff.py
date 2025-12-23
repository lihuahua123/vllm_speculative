#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 prefill_benchmark_results_deep 和 prefill_benchmark_results 中对应文件的 mean_ttft_ms 差值
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

def extract_file_key(filename):
    """从文件名中提取 key (input_len, batch_size, rate)"""
    # 匹配格式: prefill_benchmark_input{input_len}_batch{batch_size}_rate{rate}_{timestamp}.json
    # 时间戳格式可能是 20251218_110617 或 20251218_112408
    pattern = r'prefill_benchmark_input(\d+)_batch(\d+)_rate(\d+)_\d+_\d+\.json'
    match = re.match(pattern, filename)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None

def load_mean_ttft_ms(filepath):
    """从 JSON 文件中加载 mean_ttft_ms"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'mean_ttft_ms' in data:
                return data['mean_ttft_ms']
            else:
                print(f"警告: {filepath} 中没有找到 mean_ttft_ms")
                return None
    except Exception as e:
        print(f"错误: 读取 {filepath} 时出错: {e}")
        return None

def main():
    deep_dir = Path('/root/autodl-tmp/vllm_speculative/prefill_benchmark_results_deep')
    normal_dir = Path('/root/autodl-tmp/vllm_speculative/prefill_benchmark_results_nospec')
    
    # 读取 deep 文件夹中的文件
    deep_files = {}
    for file in deep_dir.glob('prefill_benchmark_*.json'):
        if file.name == 'summary.json':
            continue
        key = extract_file_key(file.name)
        if key:
            # 如果有多个文件匹配同一个 key，使用最新的（文件名中时间戳最大的）
            if key not in deep_files or file.name > deep_files[key][0]:
                mean_ttft = load_mean_ttft_ms(file)
                if mean_ttft is not None:
                    deep_files[key] = (file.name, mean_ttft)
    
    # 读取 normal 文件夹中的文件
    normal_files = {}
    for file in normal_dir.glob('prefill_benchmark_*.json'):
        if file.name == 'summary.json':
            continue
        key = extract_file_key(file.name)
        if key:
            if key not in normal_files or file.name > normal_files[key][0]:
                mean_ttft = load_mean_ttft_ms(file)
                if mean_ttft is not None:
                    normal_files[key] = (file.name, mean_ttft)
    
    # 计算差值
    results = []
    all_keys = set(deep_files.keys()) & set(normal_files.keys())
    
    if not all_keys:
        print("错误: 没有找到匹配的文件对")
        return
    
    # 按 input_len, batch_size, rate 排序
    sorted_keys = sorted(all_keys, key=lambda x: (x[0], x[1], x[2]))
    
    print("=" * 100)
    print(f"{'输入长度':<10} {'批次大小':<10} {'请求速率':<10} {'Deep TTFT (ms)':<18} {'Normal TTFT (ms)':<18} {'差值 (ms)':<15} {'Deep文件':<50}")
    print("=" * 100)
    
    for key in sorted_keys:
        input_len, batch_size, rate = key
        deep_filename, deep_ttft = deep_files[key]
        normal_filename, normal_ttft = normal_files[key]
        diff = deep_ttft - normal_ttft
        
        results.append({
            'input_len': input_len,
            'batch_size': batch_size,
            'rate': rate,
            'deep_filename': deep_filename,
            'normal_filename': normal_filename,
            'deep_mean_ttft_ms': deep_ttft,
            'normal_mean_ttft_ms': normal_ttft,
            'diff_mean_ttft_ms': diff
        })
        
        print(f"{input_len:<10} {batch_size:<10} {rate:<10} {deep_ttft:<18.4f} {normal_ttft:<18.4f} {diff:<15.4f} {deep_filename:<50}")
    
    print("=" * 100)
    print(f"\n总共找到 {len(results)} 对匹配的文件")
    
    # 保存结果到 JSON 文件
    output_file = Path('/root/autodl-tmp/vllm_speculative/ttft_diff_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 创建以 (input_len, batch_size) 为 key 的字典
    # 对于同一个 (input_len, batch_size)，如果有多个 rate，取平均差值
    diff_dict = defaultdict(list)
    for result in results:
        key = (result['input_len'], result['batch_size'])
        diff_dict[key].append(result['diff_mean_ttft_ms'])
    
    # 计算平均值
    diff_dict_avg = {}
    for key, values in diff_dict.items():
        diff_dict_avg[key] = sum(values) / len(values)
    
    # 保存字典到文件（使用字符串 key 以便 JSON 序列化）
    dict_output_file = Path('/root/autodl-tmp/vllm_speculative/ttft_diff_dict.json')
    dict_output = {f"{k[0]}_{k[1]}": v for k, v in diff_dict_avg.items()}
    with open(dict_output_file, 'w', encoding='utf-8') as f:
        json.dump(dict_output, f, indent=2, ensure_ascii=False)
    
    print(f"差值字典已保存到: {dict_output_file}")
    print(f"字典包含 {len(dict_output)} 个 (input_len, batch_size) 组合")
    
    # 检查是否有只在 deep 或 normal 中存在的文件
    only_deep = set(deep_files.keys()) - set(normal_files.keys())
    only_normal = set(normal_files.keys()) - set(deep_files.keys())
    
    if only_deep:
        print(f"\n警告: 以下配置只在 deep 文件夹中存在:")
        for key in sorted(only_deep):
            print(f"  input={key[0]}, batch={key[1]}, rate={key[2]}: {deep_files[key][0]}")
    
    if only_normal:
        print(f"\n警告: 以下配置只在 normal 文件夹中存在:")
        for key in sorted(only_normal):
            print(f"  input={key[0]}, batch={key[1]}, rate={key[2]}: {normal_files[key][0]}")

if __name__ == '__main__':
    main()

