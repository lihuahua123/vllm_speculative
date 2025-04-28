from typing import Optional, Dict, Literal, Union, Type, Any
import logging

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.llm_engine_with_bandit import LLMEngineWithBandit
from vllm.engine.llm_engine_with_ilp import LLMEngineWithILP
from vllm.engine.llm_engine_with_dqn import LLMEngineWithDQN
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

logger = logging.getLogger(__name__)

# Define valid optimization strategies
OptimizationStrategy = Literal["bandit", "none", "ilp", "dqn"]

def create_adaptive_engine(
    engine_args: EngineArgs,
    optimization_strategy: OptimizationStrategy = "bandit",
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    # Common parameters for both strategies
    monitoring_interval: float = 5.0,
    memory_penalty_coefficient: float = 0.3,
    # Q-learning specific parameters
    learning_rate: float = 0.001,
    discount_factor: float = 0.99,
    exploration_rate: float = 0.3,
    exploration_decay: float = 0.995,
    min_exploration_rate: float = 0.05,
    load_model_path: Optional[str] = None,
    hidden_dim: int = 64,
    batch_size: int = 64,
    target_update: int = 10,
) -> Union[LLMEngine, LLMEngineWithBandit, LLMEngineWithILP, LLMEngineWithDQN]:
    """Creates an LLM engine with the specified optimization strategy.
    
    Args:
        engine_args: Engine configuration arguments
        optimization_strategy: Which optimization strategy to use
            - "bandit": Use Multi-Armed Bandit optimization
            - "none": Use standard LLMEngine with no optimization
            - "ilp": Use ILP optimization
            - "dqn": Use DQN optimization
        usage_context: Context for usage tracking
        stat_loggers: Optional stat loggers
        monitoring_interval: How often to check metrics (seconds)
        memory_penalty_coefficient: Weight for memory usage penalty
        
        # Q-learning specific parameters
        learning_rate: Learning rate for Q-learning
        discount_factor: Discount factor for future rewards
        exploration_rate: Initial exploration rate
        exploration_decay: Rate of exploration decay
        min_exploration_rate: Minimum exploration rate
        load_model_path: Optional path to pre-trained model
        hidden_dim: Dimension of the neural network hidden layer
        batch_size: Batch size for DQN
        target_update: Target network update frequency for DQN
        
    Returns:
        An LLM engine with the selected optimization strategy
    """
    if optimization_strategy == "none":
        logger.info("Creating standard LLMEngine without adaptive optimization")
        return LLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=usage_context,
            stat_loggers=stat_loggers
        )
    
    elif optimization_strategy == "bandit":
        logger.info("Creating LLMEngine with Multi-Armed Bandit optimization")
        return LLMEngineWithBandit.from_engine_args(
            engine_args=engine_args,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            bandit_monitoring_interval=monitoring_interval,
        )
    
    elif optimization_strategy == "ilp":
        logger.info("Creating LLMEngine with ILP optimization")
        return LLMEngineWithILP.from_engine_args(
            engine_args=engine_args,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            num_small_models=2,
        )
    
    elif optimization_strategy == "dqn":
        logger.info("Creating LLMEngine with DQN optimization")
        return LLMEngineWithDQN.from_engine_args(
            engine_args=engine_args,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            dqn_monitoring_interval=monitoring_interval,
            learning_rate=learning_rate,
            gamma=discount_factor,
            epsilon_start=exploration_rate,
            epsilon_end=min_exploration_rate,
            epsilon_decay=exploration_decay,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            target_update=target_update,
            model_path=load_model_path
        )
    
    else:
        raise ValueError(f"Unknown optimization strategy: {optimization_strategy}. "
                        f"Valid options are: 'bandit', 'ilp', 'dqn', or 'none'")

def create_async_adaptive_engine(
    engine_args: AsyncEngineArgs,
    optimization_strategy: OptimizationStrategy = "none",
    monitoring_interval: float = 30.0,
    memory_penalty_coefficient: float = 0.5,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
) -> AsyncLLMEngine:
    """创建一个异步自适应LLM引擎，根据指定的优化策略。
    
    Args:
        engine_args: 引擎配置参数
        optimization_strategy: 优化策略（"bandit", "ilp", 或 "none"）
        monitoring_interval: 监控指标的时间间隔（秒）
        memory_penalty_coefficient: 内存使用的惩罚系数
        usage_context: 使用上下文
        stat_loggers: 统计日志记录器
        
    Returns:
        配置了指定优化策略的异步LLM引擎实例
    """
    if optimization_strategy == "none":
        # 返回标准异步LLM引擎
        return AsyncLLMEngine.from_engine_args(
            engine_args, 
            usage_context=usage_context,
            stat_loggers=stat_loggers
        )
    elif optimization_strategy == "ilp":
        # 返回启用ILP优化的异步引擎
        return LLMEngineWithILP.from_engine_args(
            engine_args,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            ilp_monitoring_interval=monitoring_interval,
            memory_penalty_coefficient=memory_penalty_coefficient
        )
    elif optimization_strategy == "bandit":
        # 目前使用ILP引擎，后续可实现专门的Bandit异步引擎
        return LLMEngineWithILP.from_engine_args(
            engine_args,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            ilp_monitoring_interval=monitoring_interval,
            memory_penalty_coefficient=memory_penalty_coefficient
        )
    else:
        raise ValueError(f"不支持的优化策略: {optimization_strategy}") 