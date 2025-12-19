#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prefill 性能测试脚本
支持不同的 input length 和 batch size (通过 request-rate 控制)
"""

import subprocess
import time
import os
import signal
import argparse
import sys
import json
import requests
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description="Prefill 性能测试脚本")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--draft-model", type=str, default="", help="Draft 模型路径（可选）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8010, help="服务器端口")
    
    # 测试参数
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048],
                       help="要测试的 input length 列表（token 数）")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                       help="要测试的 batch size 列表（通过 request-rate 控制并发）")
    parser.add_argument("--num-prompts", type=int, default=100, help="每个配置的 prompts 数量")
    parser.add_argument("--output-len", type=int, default=1, help="输出长度（prefill 测试通常设为 1）")
    
    # 服务器配置
    parser.add_argument("--strategy", type=str, default="ilp", choices=["baseline", "ilp", "no-spec"], help="策略名称")
    parser.add_argument("--sub-strategy", type=str, default="nospec", 
                       choices=["ngram", "deep", "nospec", "daspec", "smart_spec", "threshold", "ucb", "ucb-offload", "epsilon_greedy"], 
                       help="子策略名称")
    parser.add_argument("--speculative-len", type=int, default=1, help="speculative 长度")
    parser.add_argument("--num-gpu-blocks-override", type=int, default=717, help="GPU blocks override")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU 内存利用率")
    parser.add_argument("--enable-trace", type=str, default="False", help="是否开启 trace")
    parser.add_argument("--burstiness", type=float, default=1.0, help="burstiness")
    
    # 结果保存
    parser.add_argument("--result-dir", type=str, default="prefill_benchmark_results", help="结果保存目录")
    parser.add_argument("--log-file", type=str, default="prefill_benchmark.log", help="日志文件")
    
    return parser.parse_args()


def check_server_process_alive(process: subprocess.Popen) -> bool:
    """检查服务器进程是否还在运行"""
    if process.poll() is not None:
        # 进程已经结束
        return_code = process.returncode
        print(f"服务器进程已结束，返回码: {return_code}")
        return False
    return True


def check_server_health(host, port, server_process, max_retries=30, retry_interval=5):
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


def start_server(args):
    """启动 vLLM 服务器（使用与 run_benchmark_tests.py 相同的方式）"""
    print(f"正在启动 vLLM 服务器，模型: {args.model}, 地址: {args.host}:{args.port}...")
    
    # 设置环境变量
    my_env = os.environ.copy()
    my_env["VLLM_USE_V1"] = "0"
    
    exec_cmd = [
        sys.executable,
        "examples/adaptive_engine_example.py",
        "--host", args.host,
        "--port", str(args.port),
        "--model", args.model,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "2048",
        "--strategy", args.strategy,
        "--tensor-parallel-size", "1",
        "--speculative-draft-tensor-parallel-size", "1",
    ]
    
    if args.strategy != "no-spec":
        exec_cmd.append("--speculative-model")
        exec_cmd.append(args.draft_model)
        exec_cmd.append("--num-speculative-tokens")
        if args.sub_strategy == "ngram":
            exec_cmd.append("1")
        else:
            exec_cmd.append(str(args.speculative_len))
    
    if args.strategy == "ilp":
        exec_cmd.append("--num_gpu_blocks_override")
        exec_cmd.append(str(args.num_gpu_blocks_override))
    
    print(f"启动命令: {' '.join(exec_cmd)}")
    server_process = subprocess.Popen(exec_cmd, env=my_env)
    
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(60)
    
    # 使用健康检查而不是固定等待时间
    if check_server_health(args.host, args.port, server_process):
        print("服务器启动成功！")
    else:
        print("服务器启动失败或超时")
        server_process.terminate()
        raise RuntimeError("服务器启动失败")
    
    return server_process


def run_benchmark_for_config(args, input_len, batch_size, request_rate):
    """运行单个配置的 benchmark"""
    print(f"\n{'='*60}")
    print(f"测试配置: Input Length={input_len}, Batch Size={batch_size}, Request Rate={request_rate}")
    print(f"{'='*60}")
    
    os.makedirs(args.result_dir, exist_ok=True)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    result_filename = f"prefill_benchmark_input{input_len}_batch{batch_size}_rate{request_rate}_{time_str}.json"
    
    benchmark_cmd = [
        sys.executable,
        "benchmarks/benchmark_serving.py",
        "--backend", "generate",
        "--endpoint", "/generate",
        "--host", args.host,
        "--port", str(args.port),
        "--model", args.model,
        "--dataset-name", "random",
        "--dataset-path", "",
        "--random-input-len", str(input_len),
        "--random-output-len", str(args.output_len),
        "--num-prompts", str(args.num_prompts),
        "--request-rate", str(request_rate),
        "--save-result",
        "--result-dir", args.result_dir,
        "--result-filename", result_filename,
        "--burstiness", str(args.burstiness),
        "--ignore-eos"
    ]
    
    if args.enable_trace.lower() == "true":
        benchmark_cmd.append("--enable-trace")
    
    print(f"运行命令: {' '.join(benchmark_cmd)}")
    result = subprocess.run(benchmark_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ 测试完成，结果保存在 {os.path.join(args.result_dir, result_filename)}")
    else:
        print(f"✗ 测试失败，错误信息:")
        print(result.stderr)
    
    return result.returncode == 0


def send_speculative_action(host, port, action, strategy="ilp", save_action_time_history=False, profile=False, file_name=None, offload=False, ucb_file_name=None, select_strategy=None):
    """向服务器发送speculative_action请求"""
    url = f"http://{host}:{port}/speculative_action"
    data = {
        "action": action,
        "strategy": strategy,
        "save_action_time_history": save_action_time_history,
        "profile": profile,
        "file_name": file_name,
        "offload": offload,
        "ucb_file_name": ucb_file_name,
        "select_strategy": select_strategy
    }
    print(f"发送 speculative_action 请求: {data}")
    
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post(url, json=data, proxies={}, timeout=10)
        
        if response.status_code == 200:
            print(f"✓ speculative_action 请求成功 (action={action}, strategy={strategy})")
            return True
        else:
            print(f"✗ speculative_action 请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"✗ 发送 speculative_action 请求时出错: {e}")
        return False


def kill_child_processes(parent_pid):
    """杀死子进程"""
    try:
        import psutil
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


def main():
    args = parse_args()
    
    # 验证参数
    if args.strategy != "no-spec" and not args.draft_model:
        print("错误: 当 strategy 不是 'no-spec' 时，必须提供 --draft-model 参数")
        sys.exit(1)
    
    # 打开日志文件
    log_file = open(args.log_file, 'w')
    
    print("="*60)
    print("Prefill 性能测试")
    print("="*60)
    print(f"模型: {args.model}")
    if args.draft_model:
        print(f"Draft 模型: {args.draft_model}")
    print(f"策略: {args.strategy}, 子策略: {args.sub_strategy}")
    print(f"测试的 Input Lengths: {args.input_lengths}")
    print(f"测试的 Batch Sizes: {args.batch_sizes}")
    print(f"每个配置的 Prompts 数: {args.num_prompts}")
    print(f"输出长度: {args.output_len}")
    print("="*60)
    
    # 启动服务器
    server_process = start_server(args)
    
    try:
        # 根据 sub_strategy 设置相应的策略
        # 这很重要，因为 scheduler 初始化时会创建 smart_spec，需要显式禁用不需要的策略
        print(f"\n设置策略: {args.sub_strategy}")
        if args.sub_strategy == "deep":
            # 对于 deep 策略，需要先发送 action=0，然后 action=15
            if not send_speculative_action(args.host, args.port, 0, strategy=args.sub_strategy, profile=False):
                print("警告: 设置 deep 策略失败，但继续执行测试")
            time.sleep(2)
            send_speculative_action(args.host, args.port, 15, strategy="deep", profile=False)
            time.sleep(2)
        elif args.sub_strategy == "smart_spec":
            # 对于 smart_spec，发送 action=-1
            send_speculative_action(args.host, args.port, -1, strategy=args.sub_strategy, profile=False)
            time.sleep(2)
        elif args.sub_strategy == "daspec":
            # 对于 daspec，发送 action=-1
            send_speculative_action(args.host, args.port, -1, strategy=args.sub_strategy, profile=False)
            time.sleep(2)
        elif args.sub_strategy == "nospec":
            # 对于 nospec，发送 action=2
            send_speculative_action(args.host, args.port, 2, strategy=args.sub_strategy, profile=False)
            time.sleep(2)
        elif args.sub_strategy == "ngram":
            # 对于 ngram，发送 action=1
            send_speculative_action(args.host, args.port, 1, strategy=args.sub_strategy, profile=False)
            time.sleep(2)
        elif args.sub_strategy in ["ucb", "epsilon_greedy"]:
            # 对于 ucb 和 epsilon_greedy，发送 action=-1
            send_speculative_action(args.host, args.port, -1, strategy=args.sub_strategy, profile=False)
            time.sleep(2)
        else:
            # 对于其他策略，发送 action=-1 并传入 strategy，这会禁用其他策略
            send_speculative_action(args.host, args.port, -1, strategy=args.sub_strategy, profile=False)
            time.sleep(2)
        
        # 遍历不同的 input length 和 batch size
        results_summary = []
        
        for input_len in args.input_lengths:
            for batch_size in args.batch_sizes:
                # 使用 batch_size 作为 request_rate
                request_rate = batch_size
                
                success = run_benchmark_for_config(args, input_len, batch_size, request_rate)
                
                results_summary.append({
                    "input_len": input_len,
                    "batch_size": batch_size,
                    "request_rate": request_rate,
                    "success": success
                })
                
                # 等待一段时间确保请求完成
                time.sleep(3)
        
        # 保存结果摘要
        summary_file = os.path.join(args.result_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n结果摘要保存在: {summary_file}")
        
    finally:
        # 关闭服务器
        print("\n正在关闭服务器...")
        server_process.send_signal(signal.SIGINT)
        try:
            server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Timeout, killing all child processes...")
            kill_child_processes(server_process.pid)
            server_process.kill()
        
        log_file.close()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()

