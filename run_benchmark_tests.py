#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import time
import os
import signal
import argparse
import sys
import requests

def parse_args():
    parser = argparse.ArgumentParser(description="运行vLLM服务器并执行多个请求率的基准测试")
    
    # 服务器参数
    parser.add_argument("--model", type=str, required=True, help="要测试的模型名称或路径")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8010, help="服务器端口")
    
    # 基准测试参数
    parser.add_argument("--dataset-name", type=str, default="sharegpt", 
                        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "alpaca"],
                        help="基准测试数据集名称")
    parser.add_argument("--dataset-path", type=str, required=True, help="数据集路径")
    parser.add_argument("--num-prompts", type=int, default=100, help="测试的提示数量")
    parser.add_argument("--result-dir", type=str, default="benchmark_results", help="结果保存目录")
    
    # 请求率参数
    parser.add_argument("--request-rates", type=float, nargs="+", default=[8], 
                       help="要测试的请求率列表")
    
    parser.add_argument("--strategy", type=str, default="baseline", 
                        choices=["baseline", "ilp", "no-spec"], help="策略名称")
    parser.add_argument("--sub-strategy", type=str, default="ngram", 
                        choices=["ngram", "deep", "nospec", "daspec", "smart_spec"], help="子策略名称")
    parser.add_argument("--speculative-len", type=int, default=1, help="speculative长度")
    parser.add_argument("--draft-model", type=str, default="", help="draft模型")
    parser.add_argument("--profile",action="store_true", help="是否开启profile")
    
    return parser.parse_args()

def start_server(model, host, port, strategy,sub_strategy,draft_model,speculative_len=1):
    """启动vLLM服务器"""
    print(f"正在启动vLLM服务器，模型: {model}, 地址: {host}:{port}...")
    
    num_gpu_blocks_override = 2140 #4800 #28845

    # 设置环境变量
    my_env = os.environ.copy()
    my_env["VLLM_USE_V1"] = "0"

    exec_cmd =[
        sys.executable,
        # "-m", "vllm.entrypoints.openai.api_server",
        "examples/adaptive_engine_example.py",
        "--host", host,
        "--port", str(port),
        "--model", model,
        "--gpu-memory-utilization", "0.75", # 0.65 跑不起来
        # "--ngram_prompt_lookup_max", "4",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "3000",#"27432",
        # "--enable-chunked-prefill",
        # "--max_num_batched_tokens", "256",
        "--strategy", strategy,
    ]
    if strategy != "no-spec":
        exec_cmd.append("--speculative-model")
        exec_cmd.append(draft_model)
        exec_cmd.append("--num-speculative-tokens")
        if sub_strategy == "ngram":
            exec_cmd.append("1")
        else:
            exec_cmd.append(str(speculative_len))
    if strategy == "ilp":
        exec_cmd.append("--num_gpu_blocks_override")
        exec_cmd.append(str(num_gpu_blocks_override))
    print(f"exec_cmd: {exec_cmd}")
    server_process = subprocess.Popen(exec_cmd, env=my_env)
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(40)
   
    return server_process

def run_benchmark(host, port, model, dataset_name, dataset_path, num_prompts, 
                 request_rate, result_dir, strategy, text):
    """运行单个请求率的基准测试"""
    print(f"正在运行基准测试，strategy: {strategy}, 请求率: {request_rate} QPS...")
    
    os.makedirs(result_dir, exist_ok=True)
    result_filename = f"benchmark_rate_{text}_{num_prompts}_{request_rate}.json"
    # 检查结果文件是否存在,不存在则创建
    result_file = os.path.join(result_dir, result_filename)
    if not os.path.exists(result_file):
        with open(result_file, 'w') as f:
            f.write('[')
    benchmark_cmd = [
        sys.executable,
        "benchmarks/benchmark_serving.py",
        "--backend", "generate",# "vllm",#
        "--endpoint", "/generate", # "/v1/completions"
        "--host", host,
        "--port", str(port),
        "--model", model,
        "--dataset-name", dataset_name,
        "--dataset-path", dataset_path,
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_filename,
        "--enable-trace"
    ]
    
    subprocess.run(benchmark_cmd)
    print(f"完成请求率为 {request_rate} QPS 的基准测试，结果保存在 {os.path.join(result_dir, result_filename)}")
    
def send_speculative_action(host, port, action,strategy="ilp",save_action_time_history=False, profile=False,file_name=None):
    """向服务器发送speculative_action请求"""
    url = f"http://{host}:{port}/speculative_action"
    data = {"action": action,"strategy":strategy,"save_action_time_history":save_action_time_history, "profile":profile,"file_name":file_name}
    
    # 创建一个会话对象，显式禁用所有代理
    session = requests.Session()
    session.trust_env = False  # 不使用环境变量中的代理设置
    
    try:
        print(f"正在向 {url} 发送请求，action={action}...")
        # 使用会话发送请求，并明确设置proxies为空字典
        response = session.post(url, json=data, proxies={})
        
        if response.status_code == 200:
            print(f"请求成功！响应状态码: {response.status_code}")
        else:
            print(f"请求失败。响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"发送请求时出错: {e}")
        return False
    
    return response.status_code == 200

def main():
    args = parse_args()
    
    # 启动服务器
    server_process = start_server(args.model, args.host, args.port, args.strategy,args.sub_strategy,args.draft_model,args.speculative_len)
    sub_strategy = args.sub_strategy
    try:
        num_prompts = args.num_prompts
        if args.profile:
            profile = True
            save_action_time_history = True
        else:
            profile = False
            save_action_time_history = False
        file_name = f"300_new{args.speculative_len}"
        # args.dataset_name = "alpaca"
        # args.dataset_path = "tatsu-lab/alpaca"
        for rate in args.request_rates:
            request_rate = rate
            if sub_strategy == "ngram":
                send_speculative_action(args.host, args.port, 1,strategy=args.sub_strategy,profile=profile)
                run_benchmark(
                        host=args.host,
                        port=args.port,
                        model=args.model,
                        dataset_name=args.dataset_name,
                        dataset_path=args.dataset_path,
                        num_prompts=num_prompts,
                        request_rate=request_rate,
                        result_dir=args.result_dir,
                        strategy=args.strategy,
                        text="Ngram"
                    )
                send_speculative_action(args.host, args.port, -1,save_action_time_history=save_action_time_history,profile=profile,file_name=f"ngram_{file_name}.json")
            if sub_strategy == "nospec":
                send_speculative_action(args.host, args.port, 2,strategy=args.sub_strategy,profile=profile)

                run_benchmark(
                        host=args.host,
                        port=args.port,
                        model=args.model,
                        dataset_name=args.dataset_name,
                        dataset_path=args.dataset_path,
                        num_prompts=num_prompts,
                        request_rate=request_rate,
                        result_dir=args.result_dir,
                        strategy=args.strategy,
                        text="Nospec"
                    )
                send_speculative_action(args.host, args.port, -1,save_action_time_history=save_action_time_history,profile=profile,file_name=f"nospec_{file_name}.json")
            if sub_strategy == "deep":
                if not send_speculative_action(args.host, args.port, 0,strategy=args.sub_strategy,profile=profile):
                    print("发送speculative_action请求失败")
                    return
                run_benchmark(
                        host=args.host,
                        port=args.port,
                        model=args.model,
                        dataset_name=args.dataset_name,
                        dataset_path=args.dataset_path,
                        num_prompts=num_prompts,
                        request_rate=request_rate,
                        result_dir=args.result_dir,
                        strategy=args.strategy,
                        text="Deep_05b"
                    )
                send_speculative_action(args.host, args.port, -1,save_action_time_history=save_action_time_history,profile=profile,file_name=f"deep_05b_{file_name}.json")
            if sub_strategy == "daspec":
                send_speculative_action(args.host, args.port, -1,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"daspec_{file_name}.json")
                run_benchmark(
                    host=args.host,
                    port=args.port,
                    model=args.model,
                    dataset_name=args.dataset_name,
                    dataset_path=args.dataset_path,
                    num_prompts=args.num_prompts,
                    request_rate=rate,
                    result_dir=args.result_dir,
                    strategy=args.strategy,
                    text="DASpec"
                )
            if sub_strategy == "smart_spec":
                send_speculative_action(args.host, args.port, -1,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"smart_spec_{file_name}.json")
                run_benchmark(
                    host=args.host,
                    port=args.port,
                    model=args.model,
                    dataset_name=args.dataset_name,
                    dataset_path=args.dataset_path,
                    num_prompts=args.num_prompts,
                    request_rate=rate,
                    result_dir=args.result_dir,
                    strategy=args.strategy,
                    text="Smart_Spec"
                )
        
        
    finally:
        # 确保服务器被正确关闭
        print("正在关闭服务器...")
        server_process.send_signal(signal.SIGINT)
        server_process.wait()
        print("服务器已关闭")

if __name__ == "__main__":
    main() 
#  python run_benchmark_tests.py --strategy ilp --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json         --num-prompts 100 --request-rate 1