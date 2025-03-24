# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Dict, Any, Union
import torch

from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest

logger = init_logger(__name__)

class ModelExecutor:
    """Base class for model executors with support for switching draft models."""
    
    def __init__(self):
        self.neural_draft_model = None
        self.ngram_draft_model = None
        self.current_draft_model_type = "neural"
    
    def register_ngram_draft_model(self, training_samples: List[List[int]]) -> None:
        """Register an n-gram draft model and train it with sample data."""
        from vllm.spec_decode.ngram_draft_model_runner import NGramDraftModelRunner
        
        # Create and initialize the n-gram model
        if self.neural_draft_model is not None:
            # Use the same base model runner
            self.ngram_draft_model = NGramDraftModelRunner(self.neural_draft_model.model_runner)
            # Train the n-gram model with provided samples
            self.ngram_draft_model.train_from_samples(training_samples)
            # Register with the neural model for model switching
            self.neural_draft_model.register_ngram_model(self.ngram_draft_model)
            
            logger.info("N-gram draft model registered and trained")
            return True
        else:
            logger.warning("Neural draft model not initialized, can't register n-gram model")
            return False
    
    def switch_draft_model_to_ngram(self) -> bool:
        """Switch to using the n-gram draft model."""
        if self.neural_draft_model is None:
            logger.warning("Neural draft model not initialized")
            return False
            
        if self.ngram_draft_model is None:
            logger.warning("N-gram draft model not initialized")
            return False
            
        # Save the state of the current model
        self.neural_draft_model.save_model_state()
        
        # Transfer necessary state to the n-gram model
        self._transfer_draft_model_state(self.neural_draft_model, self.ngram_draft_model)
        
        # Switch the model in the draft model runner
        success = self.neural_draft_model.switch_to_ngram_model()
        
        if success:
            self.current_draft_model_type = "ngram"
            logger.info("Successfully switched to n-gram draft model")
        
        return success
    
    def switch_draft_model_to_neural(self) -> bool:
        """Switch back to using the neural draft model."""
        if self.neural_draft_model is None:
            logger.warning("Neural draft model not initialized")
            return False
            
        if self.ngram_draft_model is None:
            logger.warning("N-gram draft model not initialized")
            return False
            
        # Save the state of the current model
        self.ngram_draft_model.save_model_state()
        
        # Transfer necessary state to the neural model
        self._transfer_draft_model_state(self.ngram_draft_model, self.neural_draft_model)
        
        # Switch the model in the draft model runner
        success = self.neural_draft_model.switch_to_neural_model()
        
        if success:
            self.current_draft_model_type = "neural"
            logger.info("Successfully switched to neural draft model")
        
        return success
    
    def _transfer_draft_model_state(self, from_model, to_model) -> None:
        """Transfer necessary state between draft models during switching."""
        # Transfer any critical state between models
        # For example, indices_of_seq_with_bonus_tokens
        if hasattr(from_model, 'indices_of_seq_with_bonus_tokens'):
            to_model.set_indices_of_seq_with_bonus_tokens(from_model.indices_of_seq_with_bonus_tokens)
            
        # Load any previously saved state
        to_model.load_model_state(self.current_draft_model_type)
        
    def execute_model(self, execute_model_req: ExecuteModelRequest) -> Optional[List]:
        """Execute model with draft model, providing current load info."""
        # This is implemented in the specific model executor subclasses
        # We just provide the method signature here
        raise NotImplementedError("Subclasses must implement execute_model") 