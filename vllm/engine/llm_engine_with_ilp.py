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
from vllm.engine.ilp_integration import ILPOptimizationManager
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.engine.ilp_optimizer import ILPAction
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# Define a request class similar to SampleRequest from benchmark_serving.py
@dataclass
class TrainingRequest:
    """Simple request class for training the ILP optimizer"""
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Dict] = None


class LLMEngineWithILP(LLMEngine):
    """LLMEngine with integrated integer linear programming optimization.
    
    This class extends LLMEngine to automatically apply ILP optimization
    for dynamic model selection among multiple small models for speculative
    decoding, as well as enabling/disabling speculative decoding based on
    memory constraints.
    
    The ILP optimization problem is formulated to maximize throughput while
    respecting memory constraints and ensuring minimum acceptable performance.
    """
    
    def __init__(self, *args, 
                 ilp_monitoring_interval: float = 5.0,
                 ilp_cooldown_period: float = 30.0,
                 num_small_models: int = 3,
                 min_acceptable_throughput: float = 10.0,
                 **kwargs):
        """Initialize LLMEngine with ILP optimization.
        
        Args:
            ilp_monitoring_interval: How often to check metrics and potentially
                switch models (seconds)
            ilp_cooldown_period: Minimum time between model switches (seconds)
            num_small_models: Number of small models available for selection
            min_acceptable_throughput: Minimum acceptable throughput (tokens/s)
            *args, **kwargs: Arguments passed to LLMEngine.__init__
        """
        super().__init__(*args, **kwargs)
        
        # Create ILP optimization manager
        self.ilp_manager = ILPOptimizationManager(
            llm_engine=self,
            monitoring_interval=ilp_monitoring_interval,
            cooldown_period=ilp_cooldown_period,
            num_small_models=num_small_models,
            min_acceptable_throughput=min_acceptable_throughput
        )
        
        # Benchmark metrics
        self.benchmark_metrics = {
            "throughput_history": [],
            "request_load_history": [],
            "memory_usage_history": [],
            "action_history": [],
            "acceptance_rate_history": [],
            "spec_length_history": [],
            "testing_results": []
        }
        
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        ilp_monitoring_interval: float = 5.0,
        ilp_cooldown_period: float = 30.0,
        num_small_models: int = 3,
        min_acceptable_throughput: float = 10.0,
    ) -> "LLMEngineWithILP":
        """Creates an LLM engine with ILP optimization from the engine arguments."""
        # Create the engine configs
        vllm_config = engine_args.create_engine_config(usage_context)
        
        # Create the engine
        engine = cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            log_stats=(not engine_args.disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            ilp_monitoring_interval=ilp_monitoring_interval,
            ilp_cooldown_period=ilp_cooldown_period,
            num_small_models=num_small_models,
            min_acceptable_throughput=min_acceptable_throughput,
        )
        
        return engine
    
    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Perform one decoding iteration and run ILP optimization."""
        # Call the parent step method to process requests
        outputs = super().step()
        
        # Record token generation for throughput calculation
        tokens_generated = 0
        for output in outputs:
            for o in output.outputs:
                tokens_generated += len(o.token_ids)
        
        # Update token counter in ILP manager
        self.ilp_manager.record_tokens(tokens_generated)
        
        # Run optimization step if needed
        action = self.ilp_manager.step()
        
        # Record action for benchmarking
        if action:
            status = self.ilp_manager.get_status()
            self.benchmark_metrics["action_history"].append({
                "action": action.name,
                "time": time.time(),
                "metrics": status["current_metrics"]
            })
        
        return outputs
    
    def get_optimization_status(self) -> Dict:
        """Get current status of the ILP optimization."""
        return self.ilp_manager.get_status()
    

    
    async def run_benchmark_test(self,
                         test_requests: List[TrainingRequest],
                         sampling_params_list: List[SamplingParams],
                         request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                         test_duration: float = 60.0):
        """运行基准测试，使用不同的请求速率来评估ILP性能。
        
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
        
        logger.info("准备启动基准测试服务器")
        
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
            
            try:
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
                    
                    # 避免过度占用CPU
                    if not finished:
                        await asyncio.sleep(0.01)
                
                # 计算处理时间
                process_time = time.time() - start_time
                
                return JSONResponse({
                    "success": True,
                    "latency": process_time,
                    "tokens_generated": tokens_generated,
                    "text": [o.text for o in result.outputs] if result else []
                })
            except Exception as e:
                logger.error(f"处理请求时出错: {str(e)}")
                return JSONResponse({
                    "success": False,
                    "error": str(e)
                }, status_code=500)
        
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
        for request_rate in request_rates:
            logger.info(f"开始测试请求率: {request_rate} req/s")
            
            # 初始化测试指标
            test_metrics = {
                "request_rate": request_rate,
                "completed_requests": 0,
                "total_tokens_generated": 0,
                "avg_latency": 0.0,
                "throughput": 0.0,
                "ilp_actions": []
            }
            
            # 记录初始ILP状态
            test_metrics["initial_ilp_state"] = {
                "current_model_index": self.ilp_manager.optimizer.current_model_index,
            }
            
            # 启动服务器并等待它准备好
            server_task = asyncio.create_task(start_server())
            await asyncio.sleep(2)  # 等待服务器启动
            
            try:
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
                    
                    # 更新最终ILP状态
                    result_metrics["final_ilp_state"] = {
                        "current_model_index": self.ilp_manager.optimizer.current_model_index,
                    }
                    
                    # 保存测试结果
                    test_results.append(result_metrics)
                    
                    logger.info(f"请求率 {request_rate} req/s 测试完成: "
                            f"{result_metrics['completed_requests']} 个请求已完成, "
                            f"吞吐量: {result_metrics['throughput']:.2f} tokens/s")
            finally:
                # 关闭服务器
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
        
        # 保存结果到基准测试指标
        self.benchmark_metrics["testing_results"] = test_results
        
        return test_results
    
    async def train_and_evaluate_ilp(self,
                                training_requests: List[TrainingRequest],
                                test_requests: List[TrainingRequest],
                                sampling_params: List[Dict[str, Any]],
                                training_iterations: int = 100,
                                request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                                test_duration: float = 60.0,
                                save_results: bool = True,
                                results_dir: str = "ilp_results"):
        """Complete end-to-end training and evaluation of the ILP optimizer.
        
        This method:
        1. Trains the ILP optimizer through multiple iterations
        2. Evaluates the trained optimizer with different request rates
        3. Optionally saves the results
        
        Args:
            training_requests: Requests used for training
            test_requests: Requests used for evaluation
            sampling_params: Sampling parameters for requests
            training_iterations: Number of training iterations
            request_rates: List of request rates to test
            test_duration: Duration of each test in seconds
            save_results: Whether to save results to file
            results_dir: Directory to save results
        
        Returns:
            Dict with complete training and evaluation results
        """
        logger.info("Starting ILP optimizer training and evaluation")
        logger.info(f"Evaluating ILP optimizer with request rates: {request_rates}")
        test_results = await self.run_benchmark_test(
            test_requests=test_requests,
            sampling_params_list=sampling_params,
            request_rates=request_rates,
            test_duration=test_duration
        )
        
        # Step 3: Analyze results
        best_rate = 0
        best_throughput = 0
        
        for result in test_results:
            rate = result["request_rate"]
            throughput = result["throughput"]
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_rate = rate
            
            logger.info(f"Request rate {rate} req/s: {throughput:.2f} tokens/s")
        
        logger.info(f"Best throughput: {best_throughput:.2f} tokens/s at {best_rate} req/s")
        
        # Step 4: Save results if requested
        if save_results:
            import os
            import json
            from datetime import datetime
            
            # Create results directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{results_dir}/ilp_results_{timestamp}.json"
            
            # Save results to file
            with open(filename, "w") as f:
                json.dump({
                    "training_metrics": {
                        "throughput_history": self.benchmark_metrics["throughput_history"],
                        "request_load_history": self.benchmark_metrics["request_load_history"],
                        "memory_usage_history": self.benchmark_metrics["memory_usage_history"],
                        "action_history": self.benchmark_metrics["action_history"],
                        "acceptance_rate_history": self.benchmark_metrics["acceptance_rate_history"] if self.benchmark_metrics["acceptance_rate_history"] else [],
                        "spec_length_history": self.benchmark_metrics["spec_length_history"] if self.benchmark_metrics["spec_length_history"] else []
                    },
                    "testing_results": test_results,
                    "best_rate": best_rate,
                    "best_throughput": best_throughput
                }, f, indent=2)
            
            logger.info(f"Results saved to {filename}")
        
        return {
            "training_metrics": self.benchmark_metrics,
            "testing_results": test_results,
            "best_rate": best_rate,
            "best_throughput": best_throughput
        } 