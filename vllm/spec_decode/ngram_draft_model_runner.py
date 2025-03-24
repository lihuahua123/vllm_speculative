# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Dict, Any, Union, Type
import torch
import numpy as np
import collections

from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.worker.model_runner_base import (ModelRunnerBase,
                                          ModelRunnerInputBase,
                                          ModelRunnerWrapperBase)
from vllm.logger import init_logger
from vllm.multimodal import MultiModalKwargs

logger = init_logger(__name__)

class NGramDraftModelRunner(ModelRunnerWrapperBase):
    """A lightweight n-gram model to use as a draft model for speculative decoding.
    It's much more efficient than a neural model in high traffic situations.
    """

    def __init__(self, model_runner: ModelRunnerBase, n_gram: int = 3):
        super().__init__(model_runner)
        
        self.n_gram = n_gram
        self.ngram_cache: Dict[tuple, List[int]] = collections.defaultdict(list)
        self.indices_of_seq_with_bonus_tokens = None
        self.is_trained = False
        self.fallback_token_id = 0  # Default token ID when n-gram prediction fails
        
    def train_from_samples(self, samples: List[List[int]]) -> None:
        """Train the n-gram model from sample text."""
        self.ngram_cache.clear()
        
        for sample in samples:
            if len(sample) < self.n_gram + 1:
                continue
                
            for i in range(len(sample) - self.n_gram):
                prefix = tuple(sample[i:i+self.n_gram])
                next_token = sample[i+self.n_gram]
                self.ngram_cache[prefix].append(next_token)
        
        self.is_trained = len(self.ngram_cache) > 0
        logger.info(f"N-gram model trained with {len(self.ngram_cache)} unique contexts")
        
    def predict_next_token(self, context: List[int]) -> int:
        """Predict the next token based on the last n tokens in the context."""
        if len(context) < self.n_gram or not self.is_trained:
            return self.fallback_token_id
            
        prefix = tuple(context[-self.n_gram:])
        if prefix in self.ngram_cache and self.ngram_cache[prefix]:
            # Get most common next token
            next_tokens = self.ngram_cache[prefix]
            # Count frequencies and return the most common token
            counter = collections.Counter(next_tokens)
            return counter.most_common(1)[0][0]
            
        # If we don't have this prefix, back off to a shorter n-gram
        for k in range(self.n_gram - 1, 0, -1):
            shorter_prefix = tuple(context[-(k):])
            if shorter_prefix in self.ngram_cache and self.ngram_cache[shorter_prefix]:
                counter = collections.Counter(self.ngram_cache[shorter_prefix])
                return counter.most_common(1)[0][0]
                
        return self.fallback_token_id
        
    def set_indices_of_seq_with_bonus_tokens(self, indices_of_seq_with_bonus_tokens):
        self.indices_of_seq_with_bonus_tokens = indices_of_seq_with_bonus_tokens
    
    def supports_gpu_multi_step(self, execute_model_req: ExecuteModelRequest):
        """NGram model always supports multi-step as it's very lightweight."""
        return True
        
    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelRunnerInputBase,
        kv_caches: List[torch.Tensor],
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[List[SamplerOutput]]:
        """Execute the n-gram model to generate draft tokens.
        
        This implementation is optimized for speed instead of accuracy, as this
        is meant to be used only as a draft model during high traffic periods.
        """
        # Create an empty list to store our output
        outputs: List[SamplerOutput] = []
        
        # Extract input sequences from the model_input
        input_tokens = model_input.input_tokens.cpu().numpy()
        
        # Generate n different outputs (for n steps)
        for step in range(num_steps):
            # For each sequence in the batch
            next_token_ids = []
            for i in range(len(input_tokens)):
                # Convert to list so we can use it as context
                context = input_tokens[i].tolist()
                # Filter out padding (0) tokens if present
                context = [token for token in context if token != 0]
                # Predict next token
                next_token = self.predict_next_token(context)
                next_token_ids.append(next_token)
                
                # Update context for next step prediction
                if step < num_steps - 1:
                    input_tokens[i] = np.append(input_tokens[i], next_token)
            
            # Create tensor for sampled token IDs
            sampled_tokens = torch.tensor(next_token_ids, 
                                         dtype=torch.int64, 
                                         device=model_input.input_tokens.device)
            sampled_tokens = sampled_tokens.unsqueeze(-1)  # Add dimension to match expected shape
            
            # Create a dummy tensor for token probabilities
            # In real models this would contain the actual probabilities
            token_probs = torch.ones_like(sampled_tokens, dtype=torch.float32)
            
            # Handle the bonus tokens if needed
            if model_input.attn_metadata.num_prefills == 0 \
                and self.indices_of_seq_with_bonus_tokens is not None:
                nums_seqs = sampled_tokens.shape[0]
                count = 0
                for i in range(nums_seqs):
                    bonus_seq_idx = self.indices_of_seq_with_bonus_tokens[count]
                    if i != bonus_seq_idx:
                        sampled_tokens[i, :] = model_input.input_tokens[bonus_seq_idx]
                    else:
                        count += 1
            
            # Create SamplerOutput for this step
            output = SamplerOutput(
                sampled_token_ids=sampled_tokens,
                sampled_token_probs=token_probs,
                model_forward_time=0.001,  # Dummy value, n-gram is very fast
                model_execute_time=0.001,  # Dummy value
            )
            outputs.append(output)
        
        return outputs 