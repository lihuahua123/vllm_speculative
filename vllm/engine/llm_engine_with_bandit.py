from typing import Dict, List, Optional, Union, Any, Tuple
import time
import asyncio
import logging
import argparse
import numpy as np
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.bandit_integration import BanditOptimizationManager
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.engine.bandit_optimizer import BanditAction
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# Define a request class similar to SampleRequest from benchmark_serving.py
@dataclass
class TrainingRequest:
    """Simple request class for training the bandit optimizer"""
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Dict] = None


class LLMEngineWithBandit(LLMEngine):
    """LLMEngine with integrated multi-armed bandit optimization.
    
    This class extends LLMEngine to automatically apply the multi-armed bandit
    optimization for dynamic model switching between n-gram and neural drafts,
    as well as enabling/disabling speculative decoding based on system load.
    
    It also provides benchmark functionality to continuously train the bandit optimizer
    and test it with different request rates.
    """
    
    def __init__(self, *args, 
                 bandit_monitoring_interval: float = 5.0,
                 memory_penalty_coefficient: float = 0.3,
                 **kwargs):
        """Initialize LLMEngine with bandit optimization.
        
        Args:
            bandit_monitoring_interval: How often to check metrics and potentially
                switch models (seconds)
            memory_penalty_coefficient: Weight given to memory efficiency in reward calculation
            *args, **kwargs: Arguments passed to LLMEngine.__init__
        """
        super().__init__(*args, **kwargs)
        
        # Create bandit optimization manager
        self.bandit_manager = BanditOptimizationManager(
            llm_engine=self,
            monitoring_interval=bandit_monitoring_interval,
        )
        
        # Store the memory penalty coefficient for reward calculation
        self.memory_penalty_coefficient = memory_penalty_coefficient
        
        # Benchmark metrics
        self.benchmark_metrics = {
            "throughput_history": [],
            "request_load_history": [],
            "memory_usage_history": [],
            "action_history": [],
            "reward_history": [],
            "testing_results": []
        }
        
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        bandit_monitoring_interval: float = 5.0,
        memory_penalty_coefficient: float = 0.3,
    ) -> "LLMEngineWithBandit":
        """Creates an LLM engine with bandit optimization from the engine arguments."""
        # Create the engine configs
        vllm_config = engine_args.create_engine_config(usage_context)
        
        # Create the engine
        engine = cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            log_stats=(not engine_args.disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            bandit_monitoring_interval=bandit_monitoring_interval,
            memory_penalty_coefficient=memory_penalty_coefficient,
        )
        
        return engine

    
    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Perform one decoding iteration and run bandit optimization."""
        # Call the parent step method to process requests
        outputs = super().step()
        
        # Record token generation for throughput calculation
        tokens_generated = 0
        for output in outputs:
            for o in output.outputs:
                tokens_generated += len(o.token_ids)
        
        # Update token counter in bandit manager
        self.bandit_manager.record_tokens(tokens_generated)
        
        # Run optimization step if needed
        action = self.bandit_manager.step()
        
        # Record action for benchmarking
        if action:
            status = self.bandit_manager.get_status()
            self.benchmark_metrics["action_history"].append({
                "action": action.name,
                "time": time.time(),
                "metrics": status["current_metrics"]
            })
        
        return outputs
    
    def get_optimization_status(self) -> Dict:
        """Get current status of the bandit optimization."""
        return self.bandit_manager.get_status()
    
    async def run_benchmark_test(self,
                         test_requests: List[TrainingRequest],
                         sampling_params_list: List[SamplingParams],
                         request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                         test_duration: float = 60.0):
        """运行基准测试，使用不同的请求速率来评估Bandit性能。
        
        Args:
            test_requests: 用于测试的请求列表
            sampling_params_list: 每个请求的采样参数列表
            request_rates: 要测试的请求速率列表（每秒请求数）
            test_duration: 每次测试持续时间（秒）
        
        Returns:
            包含每个请求速率测试结果的字典
        """
        import asyncio
        import aiohttp
        import numpy as np
        import time
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
        from uvicorn.config import Config
        from uvicorn.server import Server
        from contextlib import asynccontextmanager
        
        logger.info("开始准备基准测试服务器")
        
        # 设置全局测试结果容器
        test_results = []
        
        # 创建服务器应用
        app = FastAPI()
        llm_engine = self  # 使用当前引擎实例
        
        # 配置应用启动和关闭事件
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # 启动时的操作
            logger.info("服务器启动")
            yield
            # 关闭时的操作
            logger.info("服务器关闭")
        
        app.router.lifespan_context = lifespan
        
        # 设置请求处理端点
        @app.post("/generate")
        async def generate(request_data: dict):
            request_id = request_data.get("request_id")
            prompt = request_data.get("prompt")
            sampling_params_json = request_data.get("sampling_params")
            # 从JSON数据恢复SamplingParams对象
            sampling_params = SamplingParams(**sampling_params_json)
            
            # 记录请求开始时间
            start_time = time.time()
            
            # 添加请求到引擎
            llm_engine.add_request(request_id, prompt, sampling_params)
            
            # 处理引擎输出，直到请求完成
            finished = False
            result = None
            tokens_generated = 0
            
            while not finished:
                outputs = llm_engine.step()
                for output in outputs:
                    if output.request_id == request_id:
                        if output.finished:
                            finished = True
                            result = output
                            tokens_generated = sum(len(o.token_ids) for o in output.outputs)
                            break
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            return JSONResponse({
                "success": True,
                "latency": process_time,
                "tokens_generated": tokens_generated,
                "text": [o.text for o in result.outputs] if result else []
            })
            
        
        # 定义异步客户端函数，按照指定的请求率发送请求
        async def send_requests(session, api_url, requests, params_list, request_rate, test_metrics):
            # 请求计数器
            completed_requests = 0
            total_tokens = 0
            latencies = []
            
            # 计算请求间隔时间（泊松分布）
            request_intervals = np.random.exponential(1.0 / request_rate, len(requests))
            
            # 记录测试开始时间
            start_time = time.time()
            next_request_time = start_time
            
            # 启动测试循环
            for i, request in enumerate(requests):
                current_time = time.time()
                
                # 如果测试时间已超过设定的测试持续时间，停止发送新请求
                if current_time - start_time >= test_duration:
                    break
                
                # 使用泊松过程等待下一个请求的时间
                if current_time < next_request_time:
                    await asyncio.sleep(next_request_time - current_time)
                
                # 准备请求数据
                request_id = f"test_{request_rate}_{i}"
                prompt = request.prompt
                params = params_list[i % len(params_list)]
                
                # 记录请求开始时间
                req_start_time = time.time()
                
                # 发送请求
                try:
                    async with session.post(
                        api_url,
                        json={
                            "request_id": request_id,
                            "prompt": prompt,
                            "sampling_params": params.to_json()
                        }
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if result.get("success", False):
                                # 记录请求完成情况
                                latency = result.get("latency", 0)
                                token_count = result.get("tokens_generated", 0)
                                
                                # 收集指标
                                completed_requests += 1
                                total_tokens += token_count
                                latencies.append(latency)
                        else:
                            logger.error(f"请求失败，状态码: {response.status}")
                except Exception as e:
                    logger.error(f"发送请求时出错: {str(e)}")
                
                # 计算下一个请求的时间
                if i < len(requests) - 1:
                    next_request_time = start_time + sum(request_intervals[:i+1])
            
            # 更新测试指标
            test_duration_actual = time.time() - start_time
            if completed_requests > 0:
                test_metrics["completed_requests"] = completed_requests
                test_metrics["total_tokens_generated"] = total_tokens
                test_metrics["avg_latency"] = sum(latencies) / len(latencies)
                test_metrics["throughput"] = total_tokens / test_duration_actual
            
            return test_metrics
        
        # 启动服务器函数
        async def start_server():
            config = Config(app=app, host="127.0.0.1", port=8000, log_level="info")
            server = Server(config=config)
            await server.serve()
            return server
        
        # 主测试循环
        # 启动服务器并等待它准备好
        server_task = asyncio.create_task(start_server())
        await asyncio.sleep(2)  # 等待服务器启动

        # 添加预热阶段
        logger.info("开始预热阶段: 发送50个请求")
        warm_up_rate = 2.0  # 使用适中的请求速率进行预热
        warm_up_requests = test_requests[:50] if len(test_requests) > 50 else test_requests
        if len(warm_up_requests) < 50:
            repeat_times = 50 // len(warm_up_requests) + 1
            warm_up_requests = (warm_up_requests * repeat_times)[:50]
        warm_up_metrics = {
            "request_rate": warm_up_rate,
            "completed_requests": 0,
            "total_tokens_generated": 0,
            "avg_latency": 0.0,
            "throughput": 0.0,
            "bandit_actions": []
        }

        # 创建HTTP会话进行预热
        async with aiohttp.ClientSession() as session:
            
            # 运行预热
            await send_requests(
                session,
                "http://127.0.0.1:8000/generate",
                warm_up_requests,
                sampling_params_list,
                warm_up_rate,
                warm_up_metrics
            )
            
            logger.info(f"预热阶段完成: {warm_up_metrics['completed_requests']} 个请求已处理")
        
        # 重置前缀缓存，以确保测试公平
        self.reset_prefix_cache()
        
        # 现在开始正式测试循环
        for request_rate in request_rates:
            logger.info(f"开始测试请求率: {request_rate} req/s")
            # 初始化测试指标
            test_metrics = {
                "request_rate": request_rate,
                "completed_requests": 0,
                "total_tokens_generated": 0,
                "avg_latency": 0.0,
                "throughput": 0.0,
                "bandit_actions": []
            }
            
            # 记录初始Bandit状态
            test_metrics["initial_bandit_state"] = {
                "using_ngram_model": self.bandit_manager.optimizer.using_ngram_model,
                "using_neural_model": self.bandit_manager.optimizer.using_neural_model,
                "spec_decoding_disabled": self.bandit_manager.optimizer.spec_decoding_disabled
            }
            
            # 创建HTTP会话
            async with aiohttp.ClientSession() as session:
                # 运行客户端测试
                result_metrics = await send_requests(
                    session,
                    "http://127.0.0.1:8000/generate",
                    test_requests,
                    sampling_params_list,
                    request_rate,
                    test_metrics
                )
                
                # 更新最终Bandit状态
                result_metrics["final_bandit_state"] = {
                    "using_ngram_model": self.bandit_manager.optimizer.using_ngram_model,
                    "using_neural_model": self.bandit_manager.optimizer.using_neural_model,
                    "spec_decoding_disabled": self.bandit_manager.optimizer.spec_decoding_disabled
                }
                
                # 保存测试结果
                test_results.append(result_metrics)
                
                logger.info(f"请求率 {request_rate} req/s 测试完成: "
                        f"{result_metrics['completed_requests']} 个请求已完成, "
                        f"吞吐量: {result_metrics['throughput']:.2f} tokens/s")
        
        # 保存结果到基准测试指标
        self.benchmark_metrics["testing_results"] = test_results
        
        return test_results
    
    async def run_bandit_training(self, 
                          training_requests: List[TrainingRequest],
                          sampling_params_list: List[SamplingParams],
                          num_iterations: int = 10,
                          log_interval: int = 1):
        """运行连续训练，以优化bandit算法的参数。
        
        Args:
            training_requests: 用于训练的请求列表
            sampling_params_list: 每个请求的采样参数列表
            num_iterations: 训练迭代次数
            log_interval: 记录进度的间隔
        """
        logger.info(f"开始Bandit优化器训练，共{len(training_requests)}个请求，{num_iterations}次迭代")
        
        for iteration in range(num_iterations):
            # 循环处理请求
            for i, (request, params) in enumerate(zip(training_requests, sampling_params_list)):
                if i % 10 == 0:
                    logger.info(f"训练迭代 {iteration+1}/{num_iterations}, 请求 {i+1}/{len(training_requests)}")
                
                # 添加请求到引擎
                request_id = f"train_{iteration}_{i}"
                self.add_request(request_id, request.prompt, params)
                
                # 处理直到请求完成
                finished = False
                while not finished:
                    request_outputs = self.step()
                    for output in request_outputs:
                        if output.request_id == request_id and output.finished:
                            finished = True
                            break
                    
                    # 小延迟以模拟真实的请求处理
                    await asyncio.sleep(0.1)
                
                # 记录当前状态以跟踪指标
                if iteration % log_interval == 0:
                    status = self.bandit_manager.get_status()
                    self.benchmark_metrics["throughput_history"].append(status["current_metrics"]["throughput"])
                    self.benchmark_metrics["request_load_history"].append(status["current_metrics"]["request_load"])
                    self.benchmark_metrics["memory_usage_history"].append(status["current_metrics"]["memory_usage"])
                    
                    # 记录奖励（如果有）
                    if self.bandit_manager.optimizer.last_action:
                        reward_history = self.bandit_manager.optimizer.reward_history[self.bandit_manager.optimizer.last_action]
                        if reward_history:
                            self.benchmark_metrics["reward_history"].append(reward_history[-1])
            
            logger.info(f"完成训练迭代 {iteration+1}/{num_iterations}")
        
        logger.info("Bandit训练完成")
        return self.benchmark_metrics

    async def train_and_evaluate_bandit(self,
                                training_requests: List[TrainingRequest],
                                test_requests: List[TrainingRequest],
                                sampling_params: List[SamplingParams],
                                training_iterations: int = 10,
                                request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                                test_duration: float = 60.0,
                                save_results: bool = True,
                                results_dir: str = "bandit_results"):
        """对Bandit优化器进行完整的训练和评估。
        
        该方法:
        1. 通过多次迭代训练Bandit优化器
        2. 评估经过训练的优化器在不同请求速率下的性能
        3. 可选地保存结果
        
        Args:
            training_requests: 用于训练的请求
            test_requests: 用于评估的请求
            sampling_params: 请求的采样参数
            training_iterations: 训练迭代次数
            request_rates: 要测试的请求速率列表
            test_duration: 每次测试的持续时间（秒）
            save_results: 是否将结果保存到文件
            results_dir: 保存结果的目录
        
        Returns:
            包含完整训练和评估结果的字典
        """
        logger.info("开始Bandit优化器训练和评估")
        
        # 步骤1: 训练Bandit优化器
        # logger.info(f"训练Bandit优化器，迭代次数: {training_iterations}")
        # await self.run_bandit_training(
        #     training_requests=training_requests,
        #     sampling_params_list=sampling_params,
        #     num_iterations=training_iterations
        # )
        
        # 步骤2: 用不同的请求速率评估经过训练的优化器
        logger.info(f"评估Bandit优化器，请求速率: {request_rates}")
        test_results = await self.run_benchmark_test(
            test_requests=test_requests,
            sampling_params_list=sampling_params,
            request_rates=request_rates,
            test_duration=test_duration
        )
        
        # 步骤3: 分析结果
        best_rate = 0
        best_throughput = 0
        
        for result in test_results:
            rate = result["request_rate"]
            throughput = result["throughput"]
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_rate = rate
            
            logger.info(f"请求速率 {rate} req/s: {throughput:.2f} tokens/s")
        
        logger.info(f"最佳吞吐量: {best_throughput:.2f} tokens/s，请求速率: {best_rate} req/s")
        
        # 步骤4: 如果请求，保存结果
        if save_results:
            import os
            import json
            from datetime import datetime
            
            # 如果结果目录不存在，创建它
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{results_dir}/bandit_results_{timestamp}.json"
            
            # # 将结果保存到文件
            # with open(filename, "w") as f:
            #     json.dump({
            #         "training_metrics": {
            #             "throughput_history": self.benchmark_metrics["throughput_history"],
            #             "request_load_history": self.benchmark_metrics["request_load_history"],
            #             "memory_usage_history": self.benchmark_metrics["memory_usage_history"],
            #             "action_history": self.benchmark_metrics["action_history"],
            #             "reward_history": self.benchmark_metrics["reward_history"]
            #         },
            #         "testing_results": test_results,
            #         "best_rate": best_rate,
            #         "best_throughput": best_throughput
            #     }, f, indent=2)
            
            # logger.info(f"结果已保存到 {filename}")
        
        return {
            "training_metrics": self.benchmark_metrics,
            "testing_results": test_results,
            "best_rate": best_rate,
            "best_throughput": best_throughput
        }