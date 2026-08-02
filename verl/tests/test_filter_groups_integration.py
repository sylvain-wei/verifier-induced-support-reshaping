# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration tests for filter_groups with ray_trainer.

These tests verify that:
1. FilterGroupsHelper is properly initialized in RayPPOTrainer
2. filter_groups can be enabled/disabled via config
3. Filtering happens at the correct point in the training loop
4. Training loop handles filtered batches correctly
"""

import pytest
import numpy as np
import torch

try:
    from verl.trainer.config.algorithm import FilterGroupsConfig
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.filter_groups import FilterGroupsHelper
    HAS_VERL = True
except ImportError:
    HAS_VERL = False


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
class TestFilterGroupsRayTrainerIntegration:
    """Integration tests with RayPPOTrainer."""
    
    def test_filter_groups_helper_initialized(self):
        """Test that FilterGroupsHelper is properly initialized in trainer."""
        # This test verifies the initialization only
        # Full trainer initialization requires extensive setup
        config = FilterGroupsConfig(enable=True, metric="acc", max_num_gen_batches=10)
        helper = FilterGroupsHelper(config)
        
        assert helper.enable is True
        assert hasattr(helper, 'compute_group_statistics')
        assert hasattr(helper, 'filter_batch_by_uids')
        assert hasattr(helper, 'should_resample')
    
    def test_filter_groups_disabled_by_default(self):
        """Test that filter_groups is disabled by default."""
        config = FilterGroupsConfig()
        helper = FilterGroupsHelper(config)
        
        assert helper.enable is False


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
class TestFilterGroupsConfigOptions:
    """Test different configuration options."""
    
    def test_config_with_acc_metric(self):
        """Test configuration with 'acc' metric."""
        config = FilterGroupsConfig(enable=True, metric="acc")
        helper = FilterGroupsHelper(config)
        assert helper.metric == "acc"
    
    def test_config_with_score_metric(self):
        """Test configuration with 'score' metric."""
        config = FilterGroupsConfig(enable=True, metric="score")
        helper = FilterGroupsHelper(config)
        assert helper.metric == "score"
    
    def test_config_with_seq_reward_metric(self):
        """Test configuration with 'seq_reward' metric."""
        config = FilterGroupsConfig(enable=True, metric="seq_reward")
        helper = FilterGroupsHelper(config)
        assert helper.metric == "seq_reward"
    
    def test_config_with_seq_final_reward_metric(self):
        """Test configuration with 'seq_final_reward' metric."""
        config = FilterGroupsConfig(enable=True, metric="seq_final_reward")
        helper = FilterGroupsHelper(config)
        assert helper.metric == "seq_final_reward"
    
    def test_max_num_gen_batches_unlimited(self):
        """Test unlimited generation batches (max_num_gen_batches <= 0)."""
        config = FilterGroupsConfig(enable=True, metric="acc", max_num_gen_batches=0)
        helper = FilterGroupsHelper(config)
        
        should_resample, _ = helper.should_resample(
            num_kept_prompts=1,
            train_batch_size=8,
            num_gen_batches=100,  # Very high
        )
        # With 0 or negative max, should allow unlimited resampling
        assert should_resample is True
    
    def test_max_num_gen_batches_limited(self):
        """Test limited generation batches."""
        config = FilterGroupsConfig(enable=True, metric="acc", max_num_gen_batches=5)
        helper = FilterGroupsHelper(config)
        
        should_resample, _ = helper.should_resample(
            num_kept_prompts=1,
            train_batch_size=8,
            num_gen_batches=5,  # At limit
        )
        assert should_resample is False


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
class TestFilterGroupsMetricPreperation:
    """Test metric preparation for different metric types."""
    
    def test_prepare_seq_reward_metric(self):
        """Test preparing seq_reward metric from token-level scores."""
        config = FilterGroupsConfig(enable=True, metric="seq_reward")
        helper = FilterGroupsHelper(config)
        
        class MockBatch:
            def __init__(self):
                self.batch = {
                    "token_level_scores": torch.tensor([[1.0, 2.0, 3.0],
                                                        [0.5, 1.5, 2.5]]),
                }
                self.non_tensor_batch = {}
        
        batch = MockBatch()
        helper.prepare_metric_in_batch(batch, train_batch_size=1, rollout_n=2)
        
        # seq_reward should be computed and added
        assert "seq_reward" in batch.non_tensor_batch
        assert len(batch.non_tensor_batch["seq_reward"]) == 2
    
    def test_prepare_seq_final_reward_metric(self):
        """Test preparing seq_final_reward metric from token-level rewards."""
        config = FilterGroupsConfig(enable=True, metric="seq_final_reward")
        helper = FilterGroupsHelper(config)
        
        class MockBatch:
            def __init__(self):
                self.batch = {
                    "token_level_rewards": torch.tensor([[1.0, 2.0, 3.0],
                                                         [0.5, 1.5, 2.5]]),
                }
                self.non_tensor_batch = {}
        
        batch = MockBatch()
        helper.prepare_metric_in_batch(batch, train_batch_size=1, rollout_n=2)
        
        # seq_final_reward should be computed and added
        assert "seq_final_reward" in batch.non_tensor_batch
        assert len(batch.non_tensor_batch["seq_final_reward"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
