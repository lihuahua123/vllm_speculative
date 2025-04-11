import unittest
import time
from unittest.mock import MagicMock, patch

from vllm.engine.bandit_optimizer import MultiArmedBanditOptimizer, BanditAction, ThresholdSwitcher
from vllm.engine.bandit_integration import BanditOptimizationManager


class TestThresholdSwitcher(unittest.TestCase):
    
    def test_empty_history(self):
        """Test that empty history returns zero throughput"""
        switcher = ThresholdSwitcher()
        self.assertEqual(switcher.get_current_throughput(), 0.0)
    
    def test_throughput_calculation(self):
        """Test that throughput is correctly calculated"""
        switcher = ThresholdSwitcher()
        
        # Mock time
        with patch('time.time') as mock_time:
            # First record at t=0
            mock_time.return_value = 0.0
            switcher.record_tokens(100)
            
            # Second record at t=2
            mock_time.return_value = 2.0
            switcher.record_tokens(100)
            
            # Should be 100 tokens per second (200 tokens / 2 seconds)
            self.assertEqual(switcher.get_current_throughput(), 100.0)
    
    def test_action_selection(self):
        """Test that actions are correctly selected based on throughput"""
        switcher = ThresholdSwitcher(
            high_threshold=80.0,
            low_threshold=20.0,
            cooldown_period=0.0  # No cooldown for testing
        )
        
        # Mock time
        with patch('time.time') as mock_time:
            # First record at t=0
            mock_time.return_value = 0.0
            switcher.record_tokens(100)
            
            # Second record at t=1 (high throughput: 100 tokens/s)
            mock_time.return_value = 1.0
            switcher.record_tokens(100)
            
            # Should recommend ngram model for high throughput
            self.assertEqual(switcher.check_and_update(), BanditAction.SWITCH_TO_NGRAM)
            
            # Reset for next test
            switcher = ThresholdSwitcher(
                high_threshold=80.0,
                low_threshold=20.0,
                cooldown_period=0.0
            )
            
            # First record at t=0
            mock_time.return_value = 0.0
            switcher.record_tokens(10)
            
            # Second record at t=1 (low throughput: 10 tokens/s)
            mock_time.return_value = 1.0
            switcher.record_tokens(10)
            
            # Should recommend neural model for low throughput
            self.assertEqual(switcher.check_and_update(), BanditAction.SWITCH_TO_NEURAL)
    
    def test_cooldown_period(self):
        """Test that cooldown period prevents frequent switching"""
        switcher = ThresholdSwitcher(
            high_threshold=80.0,
            low_threshold=20.0,
            cooldown_period=5.0
        )
        
        # Mock time
        with patch('time.time') as mock_time:
            # First record at t=0
            mock_time.return_value = 0.0
            switcher.record_tokens(100)
            
            # Second record at t=1 (high throughput: 100 tokens/s)
            mock_time.return_value = 1.0
            switcher.record_tokens(100)
            
            # Should recommend ngram model for high throughput
            self.assertEqual(switcher.check_and_update(), BanditAction.SWITCH_TO_NGRAM)
            
            # Still in cooldown period, should return None
            mock_time.return_value = 3.0
            self.assertIsNone(switcher.check_and_update())
            
            # After cooldown period, should allow switching again
            mock_time.return_value = 7.0
            self.assertIsNotNone(switcher.check_and_update())


class TestMultiArmedBanditOptimizer(unittest.TestCase):
    
    def test_initialization(self):
        """Test that optimizer initializes correctly"""
        optimizer = MultiArmedBanditOptimizer()
        
        # Check initial state
        self.assertEqual(optimizer.counts[BanditAction.SWITCH_TO_NGRAM], 0)
        self.assertEqual(optimizer.counts[BanditAction.SWITCH_TO_NEURAL], 0)
        self.assertEqual(optimizer.counts[BanditAction.DISABLE_SPEC_DECODING], 0)
        
        self.assertEqual(len(optimizer.reward_history[BanditAction.SWITCH_TO_NGRAM]), 0)
        self.assertIsNone(optimizer.baseline_throughput)
        self.assertFalse(optimizer.using_ngram_model)
    
    def test_early_action_selection(self):
        """Test action selection before min samples are collected"""
        optimizer = MultiArmedBanditOptimizer(min_samples_per_arm=5)
        
        # For high request load, should select ngram
        action = optimizer.select_action(throughput=50.0, request_load=20, memory_usage=0.5)
        self.assertEqual(action, BanditAction.SWITCH_TO_NGRAM)
        
        # Update state to reflect the action was taken
        optimizer.using_ngram_model = True
        
        # For high memory usage, should select disable spec decoding
        action = optimizer.select_action(throughput=50.0, request_load=10, memory_usage=0.95)
        self.assertEqual(action, BanditAction.DISABLE_SPEC_DECODING)
        
        # Update state to reflect the action was taken
        optimizer.spec_decoding_disabled = True
        
        # For low request load and memory, should select neural
        optimizer.using_ngram_model = False  # Reset first
        action = optimizer.select_action(throughput=50.0, request_load=5, memory_usage=0.5)
        self.assertEqual(action, BanditAction.SWITCH_TO_NEURAL)
    
    def test_reward_calculation(self):
        """Test that rewards are calculated correctly"""
        optimizer = MultiArmedBanditOptimizer()
        
        # Set baseline
        optimizer.baseline_throughput = 100.0
        
        # Calculate reward for improved throughput
        reward = optimizer._calculate_reward(current_throughput=150.0, memory_usage=0.5)
        self.assertGreater(reward, 0)
        
        # Calculate reward for worse throughput
        reward = optimizer._calculate_reward(current_throughput=50.0, memory_usage=0.5)
        self.assertLess(reward, 0)
        
        # Calculate reward for same throughput but different memory usage
        high_mem_reward = optimizer._calculate_reward(current_throughput=100.0, memory_usage=0.9)
        low_mem_reward = optimizer._calculate_reward(current_throughput=100.0, memory_usage=0.1)
        self.assertGreater(low_mem_reward, high_mem_reward)
    
    def test_ucb_values(self):
        """Test UCB value calculation"""
        optimizer = MultiArmedBanditOptimizer(exploration_weight=1.0)
        
        # Set up some rewards
        optimizer.reward_history[BanditAction.SWITCH_TO_NGRAM] = [0.1, 0.2, 0.3]
        optimizer.reward_history[BanditAction.SWITCH_TO_NEURAL] = [0.4, 0.5, 0.6]
        optimizer.counts[BanditAction.SWITCH_TO_NGRAM] = 3
        optimizer.counts[BanditAction.SWITCH_TO_NEURAL] = 3
        
        # Calculate UCB values
        total_count = 6
        ngram_ucb = optimizer._get_ucb_value(BanditAction.SWITCH_TO_NGRAM, total_count)
        neural_ucb = optimizer._get_ucb_value(BanditAction.SWITCH_TO_NEURAL, total_count)
        
        # Neural should have higher value due to higher rewards
        self.assertGreater(neural_ucb, ngram_ucb)
        
        # Untried arm should have infinite UCB
        disable_ucb = optimizer._get_ucb_value(BanditAction.DISABLE_SPEC_DECODING, total_count)
        self.assertEqual(disable_ucb, float('inf'))


class TestBanditOptimizationManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        # Mock LLMEngine
        self.mock_engine = MagicMock()
        self.mock_engine.get_num_running_requests.return_value = 10
        self.mock_engine.using_ngram_draft_model = False
        
        # Set up cache_config with num_gpu_blocks
        self.mock_engine.cache_config.num_gpu_blocks = 100
        
        # Set up scheduler with block_manager
        mock_scheduler = MagicMock()
        mock_block_manager = MagicMock()
        mock_block_manager.get_num_free_gpu_blocks.return_value = 50
        mock_scheduler.block_manager = mock_block_manager
        self.mock_engine.scheduler = [mock_scheduler]
        
        # Set up model_executor
        self.mock_engine.model_executor.get_speculative_decoding_disabled = MagicMock(return_value=False)
        self.mock_engine.model_executor.set_disable_speculative_decoding = MagicMock()
        
        # Create manager
        self.manager = BanditOptimizationManager(
            llm_engine=self.mock_engine,
            monitoring_interval=0.0,  # No delay for testing
            cooldown_period=0.0  # No cooldown for testing
        )
    
    def test_get_current_metrics(self):
        """Test metrics collection"""
        # Record some tokens
        with patch('time.time') as mock_time:
            mock_time.return_value = 0.0
            self.manager.last_check_time = 0.0
            
            # Record tokens at t=1
            mock_time.return_value = 1.0
            self.manager.tokens_generated_since_last_check = 100
            
            # Get metrics
            metrics = self.manager._get_current_metrics()
            
            # Check calculated throughput (100 tokens / 1 second)
            self.assertEqual(metrics["throughput"], 100.0)
            
            # Check memory usage (50 free out of 100 total = 50% used)
            self.assertEqual(metrics["memory_usage"], 0.5)
            
            # Check request load
            self.assertEqual(metrics["request_load"], 10)
    
    def test_execute_action_ngram(self):
        """Test executing ngram action"""
        # Engine not using ngram
        self.mock_engine.using_ngram_draft_model = False
        
        # Execute action
        result = self.manager._execute_action(BanditAction.SWITCH_TO_NGRAM)
        
        # Check engine method was called
        self.mock_engine.switch_to_ngram_draft_model.assert_called_once()
        self.assertTrue(result)
        
        # Already using ngram
        self.mock_engine.using_ngram_draft_model = True
        self.mock_engine.switch_to_ngram_draft_model.reset_mock()
        
        # Execute action again
        result = self.manager._execute_action(BanditAction.SWITCH_TO_NGRAM)
        
        # Should not call method again
        self.mock_engine.switch_to_ngram_draft_model.assert_not_called()
        self.assertFalse(result)
    
    def test_execute_action_neural(self):
        """Test executing neural action"""
        # Engine using ngram and neural model loaded
        self.mock_engine.using_ngram_draft_model = True
        self.mock_engine.has_loaded_neural_model = True
        
        # Execute action
        result = self.manager._execute_action(BanditAction.SWITCH_TO_NEURAL)
        
        # Check engine method was called
        self.mock_engine.switch_to_neural_draft_model.assert_called_once()
        self.assertTrue(result)
        
        # Not using ngram anymore
        self.mock_engine.using_ngram_draft_model = False
        self.mock_engine.switch_to_neural_draft_model.reset_mock()
        
        # Execute action again
        result = self.manager._execute_action(BanditAction.SWITCH_TO_NEURAL)
        
        # Should not call method again
        self.mock_engine.switch_to_neural_draft_model.assert_not_called()
        self.assertFalse(result)
    
    def test_execute_action_disable_spec(self):
        """Test executing disable spec decoding action"""
        # Spec decoding not disabled
        self.mock_engine.model_executor.get_speculative_decoding_disabled.return_value = False
        
        # Execute action
        result = self.manager._execute_action(BanditAction.DISABLE_SPEC_DECODING)
        
        # Check engine method was called
        self.mock_engine.model_executor.set_disable_speculative_decoding.assert_called_once_with(True)
        self.assertTrue(result)
        
        # Already disabled
        self.mock_engine.model_executor.get_speculative_decoding_disabled.return_value = True
        self.mock_engine.model_executor.set_disable_speculative_decoding.reset_mock()
        
        # Execute action again
        result = self.manager._execute_action(BanditAction.DISABLE_SPEC_DECODING)
        
        # Should not call method again
        self.mock_engine.model_executor.set_disable_speculative_decoding.assert_not_called()
        self.assertFalse(result)
    
    def test_step(self):
        """Test the step method"""
        # Mock threshold_switcher to return an action
        self.manager.threshold_switcher.check_and_update = MagicMock(return_value=BanditAction.SWITCH_TO_NGRAM)
        
        # Run step
        action = self.manager.step()
        
        # Check that action was executed
        self.assertEqual(action, BanditAction.SWITCH_TO_NGRAM)
        self.mock_engine.switch_to_ngram_draft_model.assert_called_once()
        
        # Next step should update reward
        self.manager.optimizer.update_reward = MagicMock()
        self.manager.step()
        self.manager.optimizer.update_reward.assert_called_once()


if __name__ == '__main__':
    unittest.main() 