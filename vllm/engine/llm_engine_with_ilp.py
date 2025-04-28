from typing import Dict, List, Optional, Union, Any, Tuple
import time
import asyncio
import logging
import argparse
import numpy as np
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.ilp_integration import ILPOptimizationManager
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.engine.ilp_optimizer import ILPAction
from vllm.sampling_params import SamplingParams
from vllm.v1.executor.abstract import Executor
logger = logging.getLogger(__name__)

# Define a request class similar to SampleRequest from benchmark_serving.py
@dataclass
class TrainingRequest:
    """Simple request class for training the ILP optimizer"""
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Dict] = None


class LLMEngineWithILP(AsyncLLMEngine):
    """LLMEngine with integrated integer linear programming optimization.
    
    This class extends AsyncLLMEngine to automatically apply ILP optimization
    for dynamic model selection among multiple small models for speculative
    decoding, as well as enabling/disabling speculative decoding based on
    memory constraints.
    
    The ILP optimization problem is formulated to maximize throughput while
    respecting memory constraints and ensuring minimum acceptable performance.
    """
    
    def __init__(self, *args, 
                 ilp_monitoring_interval: float = 5.0,
                 ilp_cooldown_period: float = 30.0,
                 **kwargs):
        """Initialize AsyncLLMEngine with ILP optimization.
        
        Args:
            ilp_monitoring_interval: How often to check metrics and potentially
                switch models (seconds)
            ilp_cooldown_period: Minimum time between model switches (seconds)
            *args, **kwargs: Arguments passed to AsyncLLMEngine.__init__
        """
        super().__init__(*args, **kwargs)
        # Create ILP optimization manager
        self.ilp_manager = ILPOptimizationManager(
            engine=self,
            monitoring_interval=ilp_monitoring_interval
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
        engine_args: AsyncEngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        ilp_monitoring_interval: float = 15.0,
        ilp_cooldown_period: float = 30.0,
    ) -> "LLMEngineWithILP":
        """Creates an LLM engine with ILP optimization from the engine arguments."""
        # Create the engine configs
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)
        # Create the engine
        engine = cls(
            executor_class=executor_class,
            vllm_config=vllm_config,
            log_stats=(not engine_args.disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            ilp_monitoring_interval=ilp_monitoring_interval,
            ilp_cooldown_period=ilp_cooldown_period,
        )
        
        return engine
    
    async def engine_step(self, virtual_engine: int) -> Tuple[bool, List[Union[RequestOutput, PoolingRequestOutput]]]:
        """Perform one decoding iteration and run ILP optimization."""
        # Call the parent step method to process requests (now async)
        begin_time = time.time()
        all_finished, outputs = await super().engine_step(virtual_engine)
        end_time = time.time()

        # # Record token generation for throughput calculation
        tokens_generated = 0
        for output in outputs:
            for o in output.outputs:
                tokens_generated += len(o.token_ids)
        tokens_throughput_per_step = tokens_generated / (end_time - begin_time)
        
        # # Update token counter in ILP manager
        self.ilp_manager.record_tokens(tokens_generated, tokens_throughput_per_step)
        
        # # Run optimization step if needed
        action = self.ilp_manager.step()
        
        
        return all_finished, outputs
    
    def get_optimization_status(self) -> Dict:
        """Get current status of the ILP optimization."""
        return self.ilp_manager.get_status()