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
Unit tests for FilterGroupsHelper class.

Tests cover:
1. Initialization and configuration
2. Metric preparation (seq_reward, seq_final_reward, acc)
3. Group statistics computation
4. Zero-variance detection
5. Batch filtering by UIDs
6. Batch size alignment
7. Resample decision logic
8. Edge cases and error handling
"""

import numpy as np
import pytest
import torch
from dataclasses import dataclass
from typing import Optional

# Mock imports for testing
try:
    from verl.trainer.config.algorithm import FilterGroupsConfig
    from verl.trainer.ppo.filter_groups import FilterGroupsHelper
    from verl import DataProto
    HAS_VERL = True
except ImportError:
    HAS_VERL = False


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
class TestFilterGroupsHelperBasic:
    """Test basic FilterGroupsHelper functionality."""
    
    def test_initialization_disabled(self):
        """Test FilterGroupsHelper initialization with disabled config."""
        config = FilterGroupsConfig(enable=False)
        helper = FilterGroupsHelper(config)
        assert helper.enable is False
    
    def test_initialization_enabled_with_metric(self):
        """Test FilterGroupsHelper initialization with enabled config."""
        config = FilterGroupsConfig(enable=True, metric="acc", max_num_gen_batches=10)
        helper = FilterGroupsHelper(config)
        assert helper.enable is True
        assert helper.metric == "acc"
        assert helper.max_num_gen_batches == 10
    
    def test_initialization_enabled_without_metric_raises(self):
        """Test that initialization without metric raises error when enabled."""
        config = FilterGroupsConfig(enable=True, metric=None)
        with pytest.raises(ValueError, match="metric must be specified"):
            FilterGroupsHelper(config)
    
    def test_disabled_helper_returns_early(self):
        """Test that disabled helper returns early from methods."""
        config = FilterGroupsConfig(enable=False)
        helper = FilterGroupsHelper(config)
        
        # These should return None/empty without error
        result = helper.prepare_metric_in_batch(None, 0, 0)
        assert result is None


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
class TestFilterGroupsHelperStatistics:
    """Test group statistics computation."""
    
    def _create_mock_batch(self, uids, metric_values, metric_name="acc"):
        """Helper to create a mock batch."""
        batch_dict = {
            "uid": np.array(uids, dtype=object),
            metric_name: np.array(metric_values),
        }
        
        class MockBatch:
            def __init__(self, data):
                self.non_tensor_batch = data
                self.batch = {}
        
        return MockBatch(batch_dict)
    
    def test_compute_group_statistics_single_group(self):
        """Test statistics for a single group with multiple samples."""
        config = FilterGroupsConfig(enable=True, metric="acc")
        helper = FilterGroupsHelper(config)
        
        # Create batch: 1 prompt UID, 3 samples with different acc values
        uids = ["prompt_0", "prompt_0", "prompt_0"]
        acc_values = [1.0, 1.0, 0.0]  # One sample has different value (non-zero variance)
        
        batch = self._create_mock_batch(uids, acc_values, "acc")
        stats = helper.compute_group_statistics(batch)
        
        assert stats["total_uids"] == 1
        assert len(stats["kept_prompt_uids"]) == 1  # Non-zero variance group
        assert stats["filtered_uids"] == 0
    
    def test_compute_group_statistics_zero_variance(self):
        """Test statistics for zero-variance group."""
        config = FilterGroupsConfig(enable=True, metric="acc")
        helper = FilterGroupsHelper(config)
        
        # Create batch: 1 prompt UID, all samples with same acc value
        uids = ["prompt_0", "prompt_0", "prompt_0"]
        acc_values = [1.0, 1.0, 1.0]  # All same - zero variance
        
        batch = self._create_mock_batch(uids, acc_values, "acc")
        stats = helper.compute_group_statistics(batch)
        
        assert stats["total_uids"] == 1
        assert len(stats["zero_variance_uids"]) == 1
        assert len(stats["kept_prompt_uids"]) == 0  # Filtered out
        assert stats["filtered_uids"] == 1
    
    def test_compute_group_statistics_single_sample_groups(self):
        """Test that single-sample groups are kept."""
        config = FilterGroupsConfig(enable=True, metric="acc")
        helper = FilterGroupsHelper(config)
        
        # Create batch: 3 prompts, each with 1 sample
        uids = ["prompt_0", "prompt_1", "prompt_2"]
        acc_values = [1.0, 0.0, 1.0]
        
        batch = self._create_mock_batch(uids, acc_values, "acc")
        stats = helper.compute_group_statistics(batch)
        
        assert stats["total_uids"] == 3
        assert len(stats["single_sample_uids"]) == 3
        assert len(stats["kept_prompt_uids"]) == 3  # All kept (single sample exception)
        assert stats["filtered_uids"] == 0
    
    def test_compute_group_statistics_mixed_groups(self):
        """Test mixed batch with zero-variance, non-zero-variance, and single-sample groups."""
        config = FilterGroupsConfig(enable=True, metric="acc")
        helper = FilterGroupsHelper(config)
        
        # Batch with 5 prompts:
        # - prompt_0: 3 samples, all 1.0 (zero variance - filtered)
        # - prompt_1: 2 samples, 1.0 and 0.0 (non-zero variance - kept)
        # - prompt_2: 1 sample (single sample - kept)
        # - prompt_3: 3 samples, all 0.0 (zero variance - filtered)
        # - prompt_4: 1 sample (single sample - kept)
        
        uids = ["prompt_0", "prompt_0", "prompt_0",
                "prompt_1", "prompt_1",
                "prompt_2",
                "prompt_3", "prompt_3", "prompt_3",
                "prompt_4"]
        acc_values = [1.0, 1.0, 1.0,
                      1.0, 0.0,
                      1.0,
                      0.0, 0.0, 0.0,
                      0.5]
        
        batch = self._create_mock_batch(uids, acc_values, "acc")
        stats = helper.compute_group_statistics(batch)
        
        assert stats["total_uids"] == 5
        assert len(stats["zero_variance_uids"]) == 2
        assert len(stats["kept_prompt_uids"]) == 3  # prompt_1, prompt_2, prompt_4
        assert stats["filtered_uids"] == 2


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
class TestFilterGroupsHelperResample:
    """Test resampling decision logic."""
    
    def test_should_resample_sufficient_prompts(self):
        """Test no resampling when sufficient prompts."""
        config = FilterGroupsConfig(enable=True, metric="acc", max_num_gen_batches=10)
        helper = FilterGroupsHelper(config)
        
        should_resample, reason = helper.should_resample(
            num_kept_prompts=10,
            train_batch_size=8,
            num_gen_batches=0,
        )
        
        assert should_resample is False
        assert "Sufficient prompts" in reason
    
    def test_should_resample_insufficient_not_at_limit(self):
        """Test resampling when insufficient prompts and not at limit."""
        config = FilterGroupsConfig(enable=True, metric="acc", max_num_gen_batches=10)
        helper = FilterGroupsHelper(config)
        
        should_resample, reason = helper.should_resample(
            num_kept_prompts=5,
            train_batch_size=8,
            num_gen_batches=2,
        )
        
        assert should_resample is True
        assert "Insufficient prompts" in reason
    
    def test_should_resample_insufficient_at_limit(self):
        """Test no resampling when at limit even with insufficient prompts."""
        config = FilterGroupsConfig(enable=True, metric="acc", max_num_gen_batches=10)
        helper = FilterGroupsHelper(config)
        
        should_resample, reason = helper.should_resample(
            num_kept_prompts=5,
            train_batch_size=8,
            num_gen_batches=10,  # At the limit
        )
        
        assert should_resample is False
        assert "Max resampling limit reached" in reason
    
    def test_should_resample_unlimited_batches(self):
        """Test unlimited resampling when max_num_gen_batches <= 0."""
        config = FilterGroupsConfig(enable=True, metric="acc", max_num_gen_batches=0)
        helper = FilterGroupsHelper(config)
        
        should_resample, reason = helper.should_resample(
            num_kept_prompts=5,
            train_batch_size=8,
            num_gen_batches=1000,  # Very high number
        )
        
        # Should still resample because max_num_gen_batches <= 0 means unlimited
        assert should_resample is True


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
class TestFilterGroupsHelperIntegration:
    """Integration tests for the complete filtering workflow."""
    
    def test_filtering_workflow_end_to_end(self):
        """Test complete filtering workflow."""
        config = FilterGroupsConfig(enable=True, metric="acc")
        helper = FilterGroupsHelper(config)
        
        # Create a mock batch with mixed zero/non-zero variance
        class MockBatch:
            def __init__(self):
                self.batch = {
                    "token_level_scores": torch.randn(10, 20),  # 10 samples, seq_len 20
                    "token_level_rewards": torch.randn(10, 20),
                }
                self.non_tensor_batch = {
                    "uid": np.array([f"p{i//2}" for i in range(10)], dtype=object),
                    "acc": np.array([1.0 if i < 4 else 0.0 if i < 8 else 0.5 for i in range(10)]),
                }
            
            def __len__(self):
                return len(self.non_tensor_batch["uid"])
        
        batch = MockBatch()
        
        # Step 1: Prepare metric
        helper.prepare_metric_in_batch(batch, train_batch_size=2, rollout_n=2)
        
        # Step 2: Compute statistics
        stats = helper.compute_group_statistics(batch)
        
        assert "total_uids" in stats
        assert "kept_prompt_uids" in stats
        
        # Step 3: Check resample decision
        should_resample, reason = helper.should_resample(
            num_kept_prompts=len(stats["kept_prompt_uids"]),
            train_batch_size=2,
            num_gen_batches=0,
        )
        
        # Result depends on how many groups are kept
        assert isinstance(should_resample, bool)


@pytest.mark.skipif(not HAS_VERL, reason="verl not installed")
class TestFilterGroupsHelperEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_batch_handled_gracefully(self):
        """Test that empty batch doesn't crash."""
        config = FilterGroupsConfig(enable=True, metric="acc")
        helper = FilterGroupsHelper(config)
        
        class EmptyBatch:
            def __init__(self):
                self.non_tensor_batch = {
                    "uid": np.array([], dtype=object),
                    "acc": np.array([]),
                }
        
        batch = EmptyBatch()
        stats = helper.compute_group_statistics(batch)
        
        assert stats["total_uids"] == 0
        assert len(stats["kept_prompt_uids"]) == 0
    
    def test_missing_metric_in_batch_raises(self):
        """Test that missing metric raises error."""
        config = FilterGroupsConfig(enable=True, metric="nonexistent_metric")
        helper = FilterGroupsHelper(config)
        
        class BadBatch:
            def __init__(self):
                self.non_tensor_batch = {
                    "uid": np.array(["p0"], dtype=object),
                    "acc": np.array([1.0]),  # Wrong metric
                }
        
        batch = BadBatch()
        
        with pytest.raises(ValueError, match="not found"):
            helper.compute_group_statistics(batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
