from typing import Optional, Dict, Literal, Union, Type, Any
import logging

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.llm_engine_with_bandit import LLMEngineWithBandit
from vllm.engine.llm_engine_with_q_learning import LLMEngineWithQLearning
from vllm.engine.llm_engine_with_ilp import LLMEngineWithILP
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext

logger = logging.getLogger(__name__)

# Define valid optimization strategies
OptimizationStrategy = Literal["bandit", "q_learning", "none", "ilp"]

def create_adaptive_engine(
    engine_args: EngineArgs,
    optimization_strategy: OptimizationStrategy = "bandit",
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    # Common parameters for both strategies
    monitoring_interval: float = 5.0,
    cooldown_period: float = 30.0,
    memory_penalty_coefficient: float = 0.3,
    # Q-learning specific parameters
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    exploration_rate: float = 0.2,
    exploration_decay: float = 0.995,
    min_exploration_rate: float = 0.01,
    load_model_path: Optional[str] = None,
) -> Union[LLMEngine, LLMEngineWithBandit, LLMEngineWithQLearning, LLMEngineWithILP]:
    """Creates an LLM engine with the specified optimization strategy.
    
    Args:
        engine_args: Engine configuration arguments
        optimization_strategy: Which optimization strategy to use
            - "bandit": Use Multi-Armed Bandit optimization
            - "q_learning": Use Q-learning optimization
            - "none": Use standard LLMEngine with no optimization
        usage_context: Context for usage tracking
        stat_loggers: Optional stat loggers
        monitoring_interval: How often to check metrics (seconds)
        cooldown_period: Minimum time between actions (seconds)
        memory_penalty_coefficient: Weight for memory usage penalty
        
        # Q-learning specific parameters
        learning_rate: Learning rate for Q-learning
        discount_factor: Discount factor for future rewards
        exploration_rate: Initial exploration rate
        exploration_decay: Rate of exploration decay
        min_exploration_rate: Minimum exploration rate
        load_model_path: Optional path to pre-trained model
        
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
            bandit_cooldown_period=cooldown_period,
        )
    
    elif optimization_strategy == "q_learning":
        logger.info("Creating LLMEngine with Q-learning optimization")
        return LLMEngineWithQLearning.from_engine_args(
            engine_args=engine_args,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            q_learning_monitoring_interval=monitoring_interval,
            q_learning_cooldown_period=cooldown_period,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate,
            memory_penalty_coefficient=memory_penalty_coefficient,
            load_q_table_path=load_model_path
        )
    elif optimization_strategy == "ilp":
        logger.info("Creating LLMEngine with ILP optimization")
        return LLMEngineWithILP.from_engine_args(
            engine_args=engine_args,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            num_small_models=2,
        )
    
    else:
        raise ValueError(f"Unknown optimization strategy: {optimization_strategy}. "
                        f"Valid options are: 'bandit', 'q_learning', or 'none'") 