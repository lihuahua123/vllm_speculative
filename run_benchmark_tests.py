#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import time
import os
import signal
import argparse
import sys
import requests
import psutil
import json
from typing import List, Tuple


def check_server_health(host: str,
                        port: int,
                        server_process: subprocess.Popen,
                        max_retries: int = 30,
                        retry_interval: int = 5) -> bool:
    """检查服务器是否健康运行"""
    url = f"http://{host}:{port}/health"
    for i in range(max_retries):
        # 首先检查进程是否还在运行
        if not check_server_process_alive(server_process):
            return False

        try:
            # 创建一个会话对象，显式禁用所有代理
            session = requests.Session()
            session.trust_env = False  # 不使用环境变量中的代理设置
            response = session.get(url, timeout=10, proxies={})
            if response.status_code == 200:
                print(f"服务器健康检查通过，尝试次数: {i+1}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"服务器健康检查失败 (尝试 {i+1}/{max_retries}): {e}")
        time.sleep(retry_interval)
    return False

def check_server_process_alive(process: subprocess.Popen) -> bool:
    """检查服务器进程是否还在运行"""
    if process.poll() is not None:
        # 进程已经结束
        return_code = process.returncode
        print(f"服务器进程已结束，返回码: {return_code}")
        return False
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="运行vLLM服务器并执行多个请求率的基准测试")

    # 服务器参数
    parser.add_argument("--model", type=str, required=True, help="要测试的模型名称或路径")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8010, help="服务器端口")

    # 基准测试参数
    parser.add_argument("--dataset-name", type=str, default="sharegpt",
                        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "alpaca","specbench"],
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
                        choices=["ngram", "deep", "nospec", "daspec", "smart_spec", "threshold","ucb","ucb-offload","epsilon_greedy","epsilon_greedy_with_offload","epsilon_greedy_with_c_prefill","ada_bin_greedy","ada_bin_greedy_simple","epsilon_greedy_simple","epsilon_greedy_context_bin","lin_ucb"], help="子策略名称")
    parser.add_argument("--speculative-len", type=int, default=1, help="speculative长度")
    parser.add_argument("--draft-model", type=str, default="", help="draft模型")
    parser.add_argument("--profile",action="store_true", help="是否开启profile")
    parser.add_argument("--start-index", type=int, default=0, help="benchmark 数据集开始索引")
    parser.add_argument("--output-len", type=int, default=-1, help="hf数据集输出长度")
    parser.add_argument("--num-gpu-blocks-override", type=int, default=28845, help="gpu blocks override")
    parser.add_argument("--enable-trace", type=str, default="False", help="是否开启trace")
    parser.add_argument("--burstiness", type=float, default=1.0, help="burstiness")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50, help="gpu memory utilization")
    parser.add_argument("--explore", type=str, default="False", help="是否开启explore")
    parser.add_argument("--select-strategy", type=str, default=None, help="select策略")
    parser.add_argument("--save-trace", type=str, default="False", help="是否保存trace")
    parser.add_argument("--increase-block-threshold", type=int, default=150, help="Threshold for free GPU blocks to trigger block number increase")
    parser.add_argument("--decrease-block-threshold", type=int, default=100, help="Threshold offset for free GPU blocks to trigger block number decrease")
    return parser.parse_args()


def start_server(model, host, port, strategy,sub_strategy,draft_model,speculative_len=1,num_gpu_blocks_override=28845,gpu_memory_utilization=0.85,increase_block_threshold=150,decrease_block_threshold=100):
    """启动vLLM服务器"""
    print(f"正在启动vLLM服务器，模型: {model}, 地址: {host}:{port}...")

    num_gpu_blocks_override = num_gpu_blocks_override #2140 4090 0.75 mem #4800 4090 0.85 mem #28845 a6000

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
        "--gpu-memory-utilization", str(gpu_memory_utilization), # 0.65 跑不起来
        # "--ngram_prompt_lookup_max", "4",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "2048",#"27432",
        # "--enable-chunked-prefill",
        # "--max_num_batched_tokens", "256",
        "--strategy", strategy,
        "--tensor-parallel-size", "1",
        "--speculative-draft-tensor-parallel-size", "1",
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
    exec_cmd.append("--increase-block-threshold")
    exec_cmd.append(str(increase_block_threshold))
    exec_cmd.append("--decrease-block-threshold")
    exec_cmd.append(str(decrease_block_threshold))
    print(f"exec_cmd: {exec_cmd}")
    server_process = subprocess.Popen(exec_cmd, env=my_env)
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(60)
    # 使用健康检查而不是固定等待时间
    if check_server_health(host, port, server_process):
        print("服务器启动成功！")
    else:
        print("服务器启动失败或超时")
        server_process.terminate()
        raise RuntimeError("服务器启动失败")

    return server_process

def run_benchmark(host, port, model, dataset_name, dataset_path, num_prompts,
                 request_rate, result_dir, strategy, text, start_index=0,output_len=-1,enable_trace="False",burstiness=1.0, strategy_name=None, increase_block_threshold=150, decrease_block_threshold=100):
    """运行单个请求率的基准测试"""
    print(f"正在运行基准测试，strategy: {strategy}, 请求率: {request_rate} QPS...")

    os.makedirs(result_dir, exist_ok=True)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    result_filename = f"benchmark_{text}_{dataset_name}_{num_prompts}_{request_rate}_{time_str}.json"
    # 检查结果文件是否存在,不存在则创建
    # result_file = os.path.join(result_dir, result_filename)
    # if not os.path.exists(result_file):
    #     with open(result_file, 'w') as f:
    #         f.write('[')
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
        "--start-index", str(start_index),
        "--burstiness", str(burstiness),
        "--ignore-eos",
        "--increase-block-threshold", str(increase_block_threshold),
        "--decrease-block-threshold", str(decrease_block_threshold),
    ]
    if strategy_name:
        benchmark_cmd.extend(["--strategy-name", strategy_name])
    if enable_trace.lower() == "true":
        benchmark_cmd.append("--enable-trace")
    if output_len != -1:
        benchmark_cmd.append("--hf-output-len")
        benchmark_cmd.append(str(output_len))
    print(f"benchmark_cmd: {benchmark_cmd}")
    subprocess.run(benchmark_cmd)
    print(f"完成请求率为 {request_rate} QPS 的基准测试，结果保存在 {os.path.join(result_dir, result_filename)}")

def send_speculative_action(host, port, action,strategy="ilp",save_action_time_history=False, profile=False,file_name=None, offload=False,ucb_file_name=None,select_strategy=None):
    """向服务器发送speculative_action请求"""
    url = f"http://{host}:{port}/speculative_action"
    data = {"action": action,"strategy":strategy,"save_action_time_history":save_action_time_history, "profile":profile,"file_name":file_name,"offload":offload,"ucb_file_name":ucb_file_name,"select_strategy":select_strategy}
    print(f"data: {data}")
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

def kill_child_processes(parent_pid):
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            print(f"Killing child process {child.pid}")
            child.terminate()
        gone, alive = psutil.wait_procs(children, timeout=5)
        for p in alive:
            print(f"Force killing child process {p.pid}")
            p.kill()
    except Exception as e:
        print(f"Error killing child processes: {e}")

def get_strategy_name(sub_strategy, speculative_len, select_strategy=None):
    """根据 sub_strategy 和 speculative_len 生成策略名称"""
    strategy_mapping = {
        "epsilon_greedy": "Nightjar",
        "epsilon_greedy_with_offload": "Nightjar-Offload",
        "epsilon_greedy_with_c_prefill": "Nightjar-CPrefill",
        "ucb": "ucb",
        "threshold": "threshold",
        "nospec": "nospec",
        "smart_spec": "smart_spec",
        "daspec": "daspec",
        "ngram": "ngram",
        "ada_bin_greedy": "ADABinGreedy",
        "ada_bin_greedy_simple": "ADABinGreedySimple",
        "epsilon_greedy_simple": "EpsilonGreedySimple",
        "epsilon_greedy_context_bin": "EpsilonGreedyContextBin",
        "lin_ucb": "LinUCBSpec",
    }
    
    if sub_strategy == "deep":
        if select_strategy == "capacity":
            return "capacity"
        else:
            return f"deep-{speculative_len}"
    elif sub_strategy in strategy_mapping:
        return strategy_mapping[sub_strategy]
    else:
        return sub_strategy

def main():
    args = parse_args()
    print(f"args: {args}")
    # 启动服务器
    server_process = start_server(args.model, args.host, args.port, args.strategy,args.sub_strategy,args.draft_model,args.speculative_len,args.num_gpu_blocks_override,args.gpu_memory_utilization,args.increase_block_threshold,args.decrease_block_threshold)
    sub_strategy = args.sub_strategy
    start_index = args.start_index
    try:
        num_prompts = args.num_prompts
        if args.profile:
            profile = True
            save_action_time_history = True
        else:
            profile = False
            save_action_time_history = False
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if not os.path.exists("./profile_log"):
            os.makedirs("./profile_log")
        profile_file_name = f"./profile_log/300_new_llama_{args.speculative_len}_{time_str}"
        benchmark_file_name = sub_strategy+"_"+str(args.speculative_len)
        # args.dataset_name = "alpaca"
        # args.dataset_path = "tatsu-lab/alpaca"
        for rate in args.request_rates:
            request_rate = rate
            if sub_strategy == "ngram":
                send_speculative_action(args.host, args.port, 1,strategy=args.sub_strategy,profile=profile)
                time.sleep(5)
                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=sub_strategy+"_"+str(args.speculative_len),
                        start_index=start_index,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_ngram.json")
                send_speculative_action(args.host, args.port, -1,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_ngram.json")
            if sub_strategy == "nospec":
                send_speculative_action(args.host, args.port, 2,strategy=args.sub_strategy,profile=profile)
                time.sleep(5)
                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=start_index,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_nospec.json")
                send_speculative_action(args.host, args.port, -1,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_nospec.json")
            if sub_strategy == "deep":
                if not send_speculative_action(args.host, args.port, 0,strategy=args.sub_strategy,profile=profile):
                    print("发送speculative_action请求失败")
                    return
                # action 15 设置选择的策略
                send_speculative_action(args.host, args.port, 15,strategy="deep",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_deep.json",select_strategy=args.select_strategy)

                time.sleep(5)
                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=start_index,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_deep.json")
                send_speculative_action(args.host, args.port, -1,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_deep.json")
            if sub_strategy == "daspec":
                send_speculative_action(args.host, args.port, -1,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_daspec.json")
                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                    text=benchmark_file_name,
                    start_index=start_index,
                    output_len=args.output_len,
                    enable_trace=args.enable_trace,
                    burstiness=args.burstiness,
                    strategy_name=strategy_name
                )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_daspec.json")
            if sub_strategy == "smart_spec":
                send_speculative_action(args.host, args.port, -1,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_smart_spec.json")
                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                    text=benchmark_file_name,
                    start_index=start_index,
                    output_len=args.output_len,
                    enable_trace=args.enable_trace,
                    burstiness=args.burstiness,
                    strategy_name=strategy_name
                )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_smart_spec.json")
            if sub_strategy == "threshold":
                send_speculative_action(args.host, args.port, 50,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_threshold.json")
                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                    text=benchmark_file_name,
                    start_index=start_index,
                    output_len=args.output_len,
                    enable_trace=args.enable_trace,
                    burstiness=args.burstiness,
                    strategy_name=strategy_name
                )
                # 保存trace 文件
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_threshold.json")
            if sub_strategy == "ucb":
                # 设置sub_strategy为ucb
                send_speculative_action(args.host, args.port, -1,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_ucb.json")
                # # action 为 12 设置为 round_robin
                if args.explore == "True":
                    send_speculative_action(args.host, args.port, 12,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_ucb.json",ucb_file_name=f"explore_ucb")

                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
                    run_benchmark(
                        host=args.host,
                        port=args.port,
                        model=args.model,
                        dataset_name=args.dataset_name,
                        dataset_path=args.dataset_path,
                        num_prompts=200,
                        request_rate=30,
                        result_dir=args.result_dir,
                        strategy=args.strategy,
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace="False",
                        burstiness=args.burstiness,
                        strategy_name=strategy_name
                    )
                # action 为 11 设置round_robin为False
                send_speculative_action(args.host, args.port, 11,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_ucb.json",ucb_file_name=f"explore_ucb")

                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                    text=benchmark_file_name,
                    start_index=start_index,
                    output_len=args.output_len,
                    enable_trace=args.enable_trace,
                    burstiness=args.burstiness,
                    strategy_name=strategy_name
                )
                # 保存trace 文件
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_ucb.json")
                # send_speculative_action(args.host, args.port, 10,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_ucb.json",ucb_file_name=f"explore_ucb")
            if sub_strategy == "epsilon_greedy":
                # 设置sub_strategy为ucb
                send_speculative_action(args.host, args.port, -1,strategy="epsilon_greedy",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"explore_epsilon_greedy")
                # action 15 设置选择的策略
                send_speculative_action(args.host, args.port, 15,strategy="epsilon_greedy",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"explore_epsilon_greedy",select_strategy=args.select_strategy)

                if args.explore == "True":
                    print("explore True")
                    send_speculative_action(args.host, args.port, 12,strategy="epsilon_greedy",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"explore_epsilon_greedy")
                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                    if args.save_trace == "True":
                        send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy1.json")
                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                    if args.save_trace == "True":
                        send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy2.json")
                    # action 为 13 设置save explore_ucb
                    send_speculative_action(args.host, args.port, 13,strategy="epsilon_greedy",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"epsilon_greedy")
                    # action 为 14 设置load explore_ucb
                    send_speculative_action(args.host, args.port, 14,strategy="epsilon_greedy",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"epsilon_greedy")
                    print("explore True 2")
                # action 为 11 设置round robin为False
                send_speculative_action(args.host, args.port, 11,strategy="epsilon_greedy",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"epsilon_greedy",select_strategy=args.select_strategy)

                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                    text=benchmark_file_name,
                    start_index=args.start_index,
                    output_len=args.output_len,
                    enable_trace=args.enable_trace,
                    burstiness=args.burstiness,
                    strategy_name=strategy_name
                )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy3.json")
            if sub_strategy == "epsilon_greedy_with_offload":
                # 设置sub_strategy为epsilon_greedy_with_offload
                send_speculative_action(args.host, args.port, -1,strategy="epsilon_greedy_with_offload",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload.json",offload=True,ucb_file_name=f"explore_epsilon_greedy_with_offload")
                # action 15 设置选择的策略
                send_speculative_action(args.host, args.port, 15,strategy="epsilon_greedy_with_offload",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload.json",offload=True,ucb_file_name=f"explore_epsilon_greedy_with_offload",select_strategy=args.select_strategy)

                if args.explore == "True":
                    print("explore True")
                    send_speculative_action(args.host, args.port, 12,strategy="epsilon_greedy_with_offload",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload.json",offload=True,ucb_file_name=f"explore_epsilon_greedy_with_offload")
                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                    if args.save_trace == "True":
                        send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload1.json")
                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                    if args.save_trace == "True":
                        send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload2.json")
                    # action 为 13 设置save explore_epsilon_greedy_with_offload
                    send_speculative_action(args.host, args.port, 13,strategy="epsilon_greedy_with_offload",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload.json",offload=True,ucb_file_name=f"epsilon_greedy_with_offload")
                    # action 为 14 设置load explore_epsilon_greedy_with_offload
                    send_speculative_action(args.host, args.port, 14,strategy="epsilon_greedy_with_offload",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload.json",offload=True,ucb_file_name=f"epsilon_greedy_with_offload")
                    print("explore True 2")
                # action 为 11 设置round robin为False
                send_speculative_action(args.host, args.port, 11,strategy="epsilon_greedy_with_offload",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload.json",offload=True,ucb_file_name=f"epsilon_greedy_with_offload",select_strategy=args.select_strategy)

                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                    text=benchmark_file_name,
                    start_index=args.start_index,
                    output_len=args.output_len,
                    enable_trace=args.enable_trace,
                    burstiness=args.burstiness,
                    strategy_name=strategy_name,
                    increase_block_threshold=args.increase_block_threshold,
                    decrease_block_threshold=args.decrease_block_threshold
                )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy_with_offload3.json")
            if sub_strategy == "epsilon_greedy_with_c_prefill":
                # 设置sub_strategy为epsilon_greedy_with_c_prefill（与epsilon_greedy完全一样的过程，只是need_c_prefill=True）
                send_speculative_action(args.host, args.port, -1,strategy="epsilon_greedy_with_c_prefill",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"explore_epsilon_greedy")
                # action 15 设置选择的策略
                send_speculative_action(args.host, args.port, 15,strategy="epsilon_greedy_with_c_prefill",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"explore_epsilon_greedy",select_strategy=args.select_strategy)

                if args.explore == "True":
                    print("explore True")
                    send_speculative_action(args.host, args.port, 12,strategy="epsilon_greedy_with_c_prefill",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"explore_epsilon_greedy")
                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                    if args.save_trace == "True":
                        send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy1.json")
                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                    if args.save_trace == "True":
                        send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy2.json")
                    # action 为 13 设置save explore_epsilon_greedy
                    send_speculative_action(args.host, args.port, 13,strategy="epsilon_greedy_with_c_prefill",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"epsilon_greedy")
                    # action 为 14 设置load explore_epsilon_greedy
                    send_speculative_action(args.host, args.port, 14,strategy="epsilon_greedy_with_c_prefill",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"epsilon_greedy")
                    print("explore True 2")
                # action 为 11 设置round robin为False
                send_speculative_action(args.host, args.port, 11,strategy="epsilon_greedy_with_c_prefill",save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy.json",offload=True,ucb_file_name=f"epsilon_greedy",select_strategy=args.select_strategy)

                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                    text=benchmark_file_name,
                    start_index=args.start_index,
                    output_len=args.output_len,
                    enable_trace=args.enable_trace,
                    burstiness=args.burstiness,
                    strategy_name=strategy_name
                )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9,strategy=args.sub_strategy,save_action_time_history=save_action_time_history,profile=profile,file_name=f"{profile_file_name}_epsilon_greedy3.json")
            
            # 新策略：ada_bin_greedy, ada_bin_greedy_simple, epsilon_greedy_simple, epsilon_greedy_context_bin, lin_ucb
            # 这些策略都使用 epsilon_greedy_spec 属性，但实例化不同的类
            if sub_strategy in ["ada_bin_greedy", "ada_bin_greedy_simple", "epsilon_greedy_simple", "epsilon_greedy_context_bin", "lin_ucb"]:
                # 设置sub_strategy
                send_speculative_action(args.host, args.port, -1, strategy=sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}.json", ucb_file_name=f"explore_{sub_strategy}")
                
                # action 15 设置选择的策略
                send_speculative_action(args.host, args.port, 15, strategy=sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}.json", ucb_file_name=f"explore_{sub_strategy}", select_strategy=args.select_strategy)
                
                if args.explore == "True":
                    print(f"explore True for {sub_strategy}")
                    send_speculative_action(args.host, args.port, 12, strategy=sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}.json", ucb_file_name=f"explore_{sub_strategy}")
                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                    if args.save_trace == "True":
                        send_speculative_action(args.host, args.port, 9, strategy=args.sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}1.json")
                    strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                        text=benchmark_file_name,
                        start_index=0,
                        output_len=args.output_len,
                        enable_trace=args.enable_trace,
                        burstiness=args.burstiness,
                        strategy_name=strategy_name,
                        increase_block_threshold=args.increase_block_threshold,
                        decrease_block_threshold=args.decrease_block_threshold
                    )
                    if args.save_trace == "True":
                        send_speculative_action(args.host, args.port, 9, strategy=args.sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}2.json")
                    # action 为 13 设置save
                    send_speculative_action(args.host, args.port, 13, strategy=sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}.json", ucb_file_name=f"{sub_strategy}")
                    # action 为 14 设置load
                    send_speculative_action(args.host, args.port, 14, strategy=sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}.json", ucb_file_name=f"{sub_strategy}")
                    # action 为 11 设置round robin为False
                    send_speculative_action(args.host, args.port, 11, strategy=sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}.json", ucb_file_name=f"{sub_strategy}", select_strategy=args.select_strategy)
                
                strategy_name = get_strategy_name(sub_strategy, args.speculative_len, args.select_strategy)
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
                    text=benchmark_file_name,
                    start_index=start_index,
                    output_len=args.output_len,
                    enable_trace=args.enable_trace,
                    burstiness=args.burstiness,
                    strategy_name=strategy_name,
                    increase_block_threshold=args.increase_block_threshold,
                    decrease_block_threshold=args.decrease_block_threshold
                )
                if args.save_trace == "True":
                    send_speculative_action(args.host, args.port, 9, strategy=args.sub_strategy, save_action_time_history=save_action_time_history, profile=profile, file_name=f"{profile_file_name}_{sub_strategy}3.json")
    finally:
        # 在 finally 里
        server_process.send_signal(signal.SIGINT)
        try:
            server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Timeout, killing all child processes...")
            kill_child_processes(server_process.pid)
            server_process.kill()

if __name__ == "__main__":
    main()
#  python run_benchmark_tests.py --strategy ilp --model /data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B         --dataset-name sharegpt         --dataset-path /data/sharegpt.json         --num-prompts 100 --request-rate 1
