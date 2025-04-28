from typing import Dict, List, Optional, Union, Any, Tuple
import time
import asyncio
import logging
import torch
import numpy as np
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.dqn_integration import DQNOptimizationManager
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.engine.dqn_optimizer import DQNAction
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# 定义一个请求类，与ILP版本保持一致
@dataclass
class TrainingRequest:
    """用于训练DQN优化器的简单请求类"""
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Dict] = None


class LLMEngineWithDQN(LLMEngine):
    """集成深度Q网络优化的LLMEngine
    
    此类扩展了LLMEngine，使用DQN和强化学习自动选择最佳的推测解码策略：
    1. 在神经模型和n-gram模型之间切换
    2. 必要时禁用推测解码
    
    DQN优化器将环境建模为马尔科夫决策过程，考虑当前的状态（吞吐量、
    请求负载、接受率等）来做出决策，并通过强化学习不断改进其策略。
    """
    
    def __init__(self, *args, 
                 dqn_monitoring_interval: float = 30.0,
                 dqn_cooldown_period: float = 10.0,
                 min_acceptable_throughput: float = 10.0,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 hidden_dim: int = 64,
                 batch_size: int = 64,
                 target_update: int = 10,
                 model_path: Optional[str] = None,
                 **kwargs):
        """初始化带有DQN优化的LLMEngine
        
        参数:
            dqn_monitoring_interval: 检查指标并可能切换模型的频率（秒）
            dqn_cooldown_period: 模型切换之间的最小时间（秒）
            min_acceptable_throughput: 最低可接受的吞吐量（tokens/s）
            learning_rate: DQN学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            hidden_dim: 神经网络隐藏层的维度
            batch_size: DQN训练的批大小
            target_update: 目标网络更新频率
            model_path: 预训练模型的路径（如果有）
            *args, **kwargs: 传递给LLMEngine.__init__的参数
        """
        super().__init__(*args, **kwargs)
        
        # 创建DQN优化管理器
        self.dqn_manager = DQNOptimizationManager(
            llm_engine=self,
            monitoring_interval=dqn_monitoring_interval,
            cooldown_period=dqn_cooldown_period,
            min_acceptable_throughput=min_acceptable_throughput,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            target_update=target_update,
            model_path=model_path
        )
        
        # 基准指标
        self.benchmark_metrics = {
            "throughput_history": [],
            "request_load_history": [],
            "memory_usage_history": [],
            "action_history": [],
            "acceptance_rate_history": [],
            "spec_length_history": [],
            "reward_history": [],
            "epsilon_history": [],
            "q_values_history": [],
            "testing_results": []
        }
        
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        dqn_monitoring_interval: float = 30.0,
        dqn_cooldown_period: float = 10.0,
        min_acceptable_throughput: float = 10.0,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        hidden_dim: int = 64,
        batch_size: int = 64,
        target_update: int = 10,
        model_path: Optional[str] = None,
    ) -> "LLMEngineWithDQN":
        """从引擎参数创建带有DQN优化的LLM引擎"""
        # 创建引擎配置
        vllm_config = engine_args.create_engine_config(usage_context)
        
        # 创建引擎
        engine = cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            log_stats=(not engine_args.disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            dqn_monitoring_interval=dqn_monitoring_interval,
            dqn_cooldown_period=dqn_cooldown_period,
            min_acceptable_throughput=min_acceptable_throughput,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            target_update=target_update,
            model_path=model_path
        )
        
        return engine
    
    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """执行一次解码迭代并运行DQN优化"""
        # 调用父类的step方法处理请求
        outputs = super().step()
        
        # 记录标记生成以计算吞吐量
        tokens_generated = 0
        for output in outputs:
            for o in output.outputs:
                tokens_generated += len(o.token_ids)
        
        # 更新DQN管理器中的标记计数
        self.dqn_manager.record_tokens(tokens_generated)
        
        # 必要时运行优化步骤
        action = self.dqn_manager.step()
        
        # 记录动作用于基准测试
        if action:
            status = self.dqn_manager.get_status()
            self.benchmark_metrics["action_history"].append({
                "action": action.name,
                "time": time.time(),
                "metrics": status["current_metrics"]
            })
            
            # 记录ε和奖励
            optimizer_status = status["optimizer_status"]
            self.benchmark_metrics["epsilon_history"].append(optimizer_status.get("epsilon", 0.0))
            
            for action_name, reward in optimizer_status.get("reward_means", {}).items():
                self.benchmark_metrics["reward_history"].append({
                    "action": action_name,
                    "reward": reward,
                    "time": time.time()
                })
        
        return outputs
    
    def get_optimization_status(self) -> Dict:
        """获取DQN优化的当前状态"""
        return self.dqn_manager.get_status()
    
    def save_dqn_model(self, path: str) -> bool:
        """保存DQN模型到指定路径"""
        return self.dqn_manager.save_model(path)
    
    async def run_benchmark_test(self,
                         test_requests: List[TrainingRequest],
                         sampling_params_list: List[SamplingParams],
                         request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                         test_duration: float = 60.0) -> List[Dict]:
        """运行基准测试，使用不同的请求速率来评估DQN性能
        
        参数:
            test_requests: 用于测试的请求列表
            sampling_params_list: 每个请求的采样参数列表
            request_rates: 要测试的请求速率列表（每秒请求数）
            test_duration: 每次测试持续时间（秒）
        
        返回:
            包含每个请求速率测试结果的列表
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
            "dqn_actions": []
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
        
        # 重置前缀缓存
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
                "dqn_actions": []
            }
            
            # 记录初始DQN状态
            test_metrics["initial_dqn_state"] = {
                "current_model": "neural" if self.dqn_manager.optimizer.using_neural_model else 
                               "ngram" if self.dqn_manager.optimizer.using_ngram_model else 
                               "disabled",
                "epsilon": self.dqn_manager.optimizer.epsilon
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
                
                # 更新最终DQN状态
                result_metrics["final_dqn_state"] = {
                    "current_model": "neural" if self.dqn_manager.optimizer.using_neural_model else 
                                   "ngram" if self.dqn_manager.optimizer.using_ngram_model else 
                                   "disabled",
                    "epsilon": self.dqn_manager.optimizer.epsilon
                }
                
                # 保存测试结果
                test_results.append(result_metrics)
                
                logger.info(f"请求率 {request_rate} req/s 测试完成: "
                        f"{result_metrics['completed_requests']} 个请求已完成, "
                        f"吞吐量: {result_metrics['throughput']:.2f} tokens/s")
        
        # 保存结果到基准测试指标
        self.benchmark_metrics["testing_results"] = test_results
        
        return test_results
    
    async def train_and_evaluate_dqn(self,
                                test_requests: List[TrainingRequest],
                                sampling_params: List[SamplingParams],
                                training_iterations: int = 100,
                                request_rates: List[float] = [1.0, 2.0, 4.0, 8.0],
                                test_duration: float = 60.0,
                                save_results: bool = True,
                                save_model: bool = True,
                                results_dir: str = "dqn_results",
                                model_path: Optional[str] = "dqn_model.pt"):
        """完整的端到端DQN优化器训练和评估
        
        此方法:
        1. 训练DQN优化器
        2. 使用不同的请求率评估训练后的优化器
        3. 可选地保存结果和模型
        
        参数:
            test_requests: 用于评估的请求
            sampling_params: 请求的采样参数
            training_iterations: 训练迭代次数
            request_rates: 要测试的请求率列表
            test_duration: 每次测试的持续时间（秒）
            save_results: 是否将结果保存到文件
            save_model: 是否保存训练好的模型
            results_dir: 保存结果的目录
            model_path: 保存模型的路径
        
        返回:
            包含完整训练和评估结果的字典
        """
        logger.info("开始DQN优化器训练和评估")
        logger.info(f"使用请求率评估DQN优化器: {request_rates}")
        
        # 评估DQN优化器
        test_results = await self.run_benchmark_test(
            test_requests=test_requests,
            sampling_params_list=sampling_params,
            request_rates=request_rates,
            test_duration=test_duration
        )
        
        # 分析结果
        best_rate = 0
        best_throughput = 0
        
        for result in test_results:
            rate = result["request_rate"]
            throughput = result["throughput"]
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_rate = rate
            
            logger.info(f"请求率 {rate} req/s: {throughput:.2f} tokens/s")
        
        logger.info(f"最佳吞吐量: {best_throughput:.2f} tokens/s，请求率: {best_rate} req/s")
        
        # 如果需要保存模型
        if save_model and model_path:
            success = self.save_dqn_model(model_path)
            if success:
                logger.info(f"DQN模型已保存到 {model_path}")
            else:
                logger.error("保存DQN模型失败")
        
        # 如果需要保存结果
        if save_results:
            import os
            import json
            from datetime import datetime
            
            # 创建结果目录（如果不存在）
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成带有时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{results_dir}/dqn_results_{timestamp}.json"
            
            # TODO: 保存结果到文件，暂时注释掉防止错误
            '''
            # 保存结果到文件
            with open(filename, "w") as f:
                json.dump({
                    "training_metrics": {
                        "throughput_history": self.benchmark_metrics["throughput_history"],
                        "request_load_history": self.benchmark_metrics["request_load_history"],
                        "memory_usage_history": self.benchmark_metrics["memory_usage_history"],
                        "action_history": [{**item, "time": str(item["time"])} for item in self.benchmark_metrics["action_history"]],
                        "acceptance_rate_history": self.benchmark_metrics["acceptance_rate_history"] if "acceptance_rate_history" in self.benchmark_metrics else [],
                        "spec_length_history": self.benchmark_metrics["spec_length_history"] if "spec_length_history" in self.benchmark_metrics else [],
                        "reward_history": [{**item, "time": str(item["time"])} for item in self.benchmark_metrics["reward_history"]],
                        "epsilon_history": self.benchmark_metrics["epsilon_history"]
                    },
                    "testing_results": test_results,
                    "best_rate": best_rate,
                    "best_throughput": best_throughput
                }, f, indent=2)
            
            logger.info(f"结果已保存到 {filename}")
            '''
        
        return {
            "training_metrics": self.benchmark_metrics,
            "testing_results": test_results,
            "best_rate": best_rate,
            "best_throughput": best_throughput
        } 