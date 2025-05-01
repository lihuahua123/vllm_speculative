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
    speculative_model = "/data/model/alamios_DeepSeek-R1-DRAFT-Qwen2.5-0.5B"
    num_speculative_tokens = 1
    # 初始化VLLM引擎
    llm = LLM(model=model_path, 
              tensor_parallel_size=1,
              gpu_memory_utilization=0.9,
              speculative_model=speculative_model,
              num_speculative_tokens=num_speculative_tokens,
              max_model_len=8432)
    
    
   
            
