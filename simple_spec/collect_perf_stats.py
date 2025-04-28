import torch
import time
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import sys
import gc
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm import LLM, SamplingParams
from vllm.core.scheduler import Device
from inference import sample_sharegpt_requests
from transformers import AutoTokenizer

def collect_prefill_performance_stats(
    model_path: str,
    dataset_path: str,
    output_path: str,
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128],
    num_samples: int = 100,
    device: str = "cuda",
    is_small_model: bool = False
):
    """收集VLLM引擎prefill的性能数据"""
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 初始化VLLM引擎
    llm = LLM(model=model_path, 
              tensor_parallel_size=1,
              gpu_memory_utilization=0.9,
              max_model_len=8432)
    
    # 从数据集采样请求
    print(f"从数据集 {dataset_path} 采样 {num_samples} 个请求")
    spec_len = 4
    for i in range(num_samples//128):
        begin_index = i*128
        requests, requests2 = sample_sharegpt_requests(
            dataset_path=dataset_path,
            num_requests=128,
            tokenizer=tokenizer,
            spec_len=spec_len,
            begin_index=begin_index
        )
        print(f"成功采样 {len(requests)} 个请求")
        
        performance_stats = []
        
        # 对每个batch大小进行测试
        for batch_size in tqdm(batch_sizes, desc="测试不同batch大小"):
            # 如果请求数量不足batch_size，则跳过
            if len(requests) < batch_size:
                continue
            llm.reset_prefix_cache(Device.GPU)
            # 第一步：收集初始prefill (只生成1个token) 的性能数据
            prompts = [req[0] for req in requests[:batch_size]]
            # 记录总token数
            total_prompt_tokens1 = sum([req[1] for req in requests[:batch_size]])
            
            # 设置采样参数 - 只生成1个token
            sampling_params = SamplingParams(temperature=0, max_tokens=1)
            for index,prompt in enumerate(prompts):
                llm.llm_engine.add_request(f"test_{index}",prompt, sampling_params)
            llm.llm_engine.step()  # Prefill cache
            for index in range(batch_size):
                llm.llm_engine.abort_request([f"test_{index}"])
            
            if not is_small_model:
                prompts = [req[0] for req in requests2[:batch_size]] # 比requests1 多出spec_len个token
                # 记录总token数
                for index,prompt in enumerate(prompts):
                    llm.llm_engine.add_request(f"test_{index}",prompt, sampling_params)
                torch.cuda.synchronize()
                start_time = time.time()
                llm.llm_engine.step()  # Prefill spec_len
                torch.cuda.synchronize()
                prefill_time = time.time() - start_time

                performance_stats.append({
                    "batch_size": batch_size,
                    "spec_len": spec_len,
                    "total_prompt_tokens": total_prompt_tokens1,
                    "num_new_tokens": spec_len * batch_size,
                    "prefill_time": prefill_time,
                    "has_cache": True
                })
                for index in range(batch_size):
                    llm.llm_engine.abort_request([f"test_{index}"])
            else:
                start_time = time.time()
                llm.llm_engine.step()  # Decode
                torch.cuda.synchronize()
                decode_time = time.time() - start_time
                print(f"{batch_size} decode_time: {decode_time}")
                performance_stats.append({
                    "batch_size": batch_size,
                    "spec_len": spec_len,
                    "total_prompt_tokens": total_prompt_tokens1,
                    "num_new_tokens": 1 * batch_size,
                    "prefill_time": decode_time,
                    "has_cache": True
                })
                for index in range(batch_size):
                    llm.llm_engine.abort_request([f"test_{index}"])
            
    
    # 保存性能数据
    with open(output_path, "w") as f:
        json.dump(performance_stats, f, indent=2)
    
    print(f"性能数据已保存到 {output_path}")
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return performance_stats

