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
Filter Groups Helper for DAPO Dynamic Sampling

This module provides utilities for filtering zero-variance prompt groups during training.
When enabled, the filter_groups mechanism:
1. Groups trajectories by their prompt UID
2. Computes variance of rewards within each group
3. Filters out zero-variance groups (all samples have same reward)
4. Resamples new data if too many groups are filtered out
5. Repeats until sufficient data is accumulated or max iterations reached

This is a core component of DAPO (Dynamic All-Prompt Optimization) to ensure training
stability and avoid degenerate batches where gradients are ineffective.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from verl import DataProto
from verl.trainer.config.algorithm import FilterGroupsConfig


class FilterGroupsHelper:
    """
    Helper class for managing filter_groups during DAPO training.
    
    Responsible for:
    - Detecting zero-variance groups
    - Filtering qualified groups
    - Computing group statistics
    - Determining when to resample
    
    Attributes:
        config (FilterGroupsConfig): Configuration for filter_groups behavior
        enable (bool): Whether filter_groups is enabled
        metric (str): Metric name to use for variance computation ("acc", "score", etc.)
        max_num_gen_batches (int): Maximum number of generation batches allowed
    """
    
    def __init__(self, config: FilterGroupsConfig):
        """
        Initialize the FilterGroupsHelper.
        
        Args:
            config: FilterGroupsConfig with enable, metric, max_num_gen_batches fields
        """
        self.config = config
        self.enable = config.enable
        self.metric = config.metric
        self.max_num_gen_batches = config.max_num_gen_batches
        
        if not self.enable:
            return
            
        if self.metric is None:
            raise ValueError(
                f"FilterGroupsConfig.metric must be specified when enable=True, "
                f"got None. Valid options: 'acc', 'score', 'seq_reward', 'seq_final_reward'"
            )
    
    def prepare_metric_in_batch(
        self, 
        batch: DataProto,
        train_batch_size: int,
        rollout_n: int,
    ) -> None:
        """
        Prepare the metric field in batch for filtering.
        
        Some metrics (like seq_reward, seq_final_reward) need to be computed from
        token-level rewards, while others (like acc) are already in non_tensor_batch.
        
        Args:
            batch: The DataProto batch
            train_batch_size: Training batch size (num unique prompts)
            rollout_n: Number of rollouts per prompt
            
        Raises:
            ValueError: If metric is not found in batch
        """
        if not self.enable:
            return
        
        # Compute derived metrics if needed
        if self.metric == "seq_final_reward":
            if "token_level_rewards" in batch.batch:
                batch.non_tensor_batch["seq_final_reward"] = (
                    batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                )
            else:
                raise ValueError(
                    f"Metric 'seq_final_reward' requires 'token_level_rewards' in batch.batch, "
                    f"but it's not present. Available keys: {list(batch.batch.keys())}"
                )
        elif self.metric == "seq_reward":
            if "token_level_scores" in batch.batch:
                batch.non_tensor_batch["seq_reward"] = (
                    batch.batch["token_level_scores"].sum(dim=-1).numpy()
                )
            else:
                raise ValueError(
                    f"Metric 'seq_reward' requires 'token_level_scores' in batch.batch, "
                    f"but it's not present. Available keys: {list(batch.batch.keys())}"
                )
        
        # Verify metric exists after preparation
        if self.metric not in batch.non_tensor_batch:
            raise ValueError(
                f"Metric '{self.metric}' not found in batch.non_tensor_batch after preparation. "
                f"Available keys: {list(batch.non_tensor_batch.keys())}"
            )
    
    def compute_group_statistics(self, batch: DataProto) -> Dict[str, dict]:
        """
        Compute statistics for each prompt group.
        
        Groups trajectories by prompt UID and computes variance of the metric.
        A group is zero-variance if all metric values are identical.
        
        Args:
            batch: The DataProto batch containing uid and metric fields
            
        Returns:
            Dictionary mapping metric names to their group statistics:
            {
                "prompt_uid2metric_vals": {uid: [val1, val2, ...], ...},
                "prompt_uid2metric_std": {uid: std_value, ...},
                "zero_variance_uids": [uid1, uid2, ...],  # Groups with std == 0 and len > 1
                "single_sample_uids": [uid3, ...],  # Groups with only 1 sample
                "kept_prompt_uids": [uid1, uid2, ...],  # Non-zero-variance groups
                "total_uids": total_count,
                "filtered_uids": filtered_count,
            }
        """
        if not self.enable:
            return {}
        
        # Check if metric exists in batch
        if self.metric not in batch.non_tensor_batch:
            raise ValueError(
                f"Metric '{self.metric}' not found in batch.non_tensor_batch. "
                f"Available keys: {list(batch.non_tensor_batch.keys())}"
            )
        
        # Group metric values by prompt UID
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(
            batch.non_tensor_batch["uid"],
            batch.non_tensor_batch[self.metric],
            strict=True
        ):
            prompt_uid2metric_vals[uid].append(metric_val)
        
        # Compute variance for each group
        prompt_uid2metric_std = {}
        zero_variance_uids = []
        single_sample_uids = []
        
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
            std = np.std(metric_vals)
            prompt_uid2metric_std[prompt_uid] = std
            
            if len(metric_vals) == 1:
                single_sample_uids.append(prompt_uid)
            elif std == 0:
                zero_variance_uids.append(prompt_uid)
        
        # Determine which groups to keep
        # Keep groups with non-zero variance OR groups with only 1 sample
        kept_prompt_uids = [
            uid 
            for uid, std in prompt_uid2metric_std.items() 
            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
        ]
        
        return {
            "prompt_uid2metric_vals": prompt_uid2metric_vals,
            "prompt_uid2metric_std": prompt_uid2metric_std,
            "zero_variance_uids": zero_variance_uids,
            "single_sample_uids": single_sample_uids,
            "kept_prompt_uids": kept_prompt_uids,
            "total_uids": len(prompt_uid2metric_vals),
            "filtered_uids": len(zero_variance_uids),
        }
    
    def filter_batch_by_uids(
        self, 
        batch: DataProto,
        kept_prompt_uids: List[str],
    ) -> Tuple[DataProto, int]:
        """
        Filter batch to only keep trajectories from specified prompt UIDs.
        
        Args:
            batch: The DataProto batch
            kept_prompt_uids: List of prompt UIDs to keep
            
        Returns:
            Tuple of (filtered_batch, num_kept_trajectories)
        """
        if not self.enable:
            return batch, len(batch)
        
        # Find trajectory indices that belong to kept UIDs
        kept_traj_idxs = []
        for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)
        
        # Filter batch using indices
        filtered_batch = batch[kept_traj_idxs]
        return filtered_batch, len(kept_traj_idxs)
    
    def should_resample(
        self,
        num_kept_prompts: int,
        train_batch_size: int,
        num_gen_batches: int,
    ) -> Tuple[bool, str]:
        """
        Determine whether to resample or proceed with training.
        
        Resampling logic:
        - If num_kept_prompts < train_batch_size and num_gen_batches < max_num_gen_batches:
          Resample (continue loop)
        - Otherwise: Proceed with training (break loop)
        
        Args:
            num_kept_prompts: Number of prompts after filtering
            train_batch_size: Minimum required prompts for training
            num_gen_batches: Current number of generation batches
            
        Returns:
            Tuple of (should_resample, reason)
            - should_resample: True if should resample, False if should proceed
            - reason: Explanation of the decision
        """
        if not self.enable:
            return False, "filter_groups disabled"
        
        # Check if we have enough prompts
        if num_kept_prompts >= train_batch_size:
            return False, f"Sufficient prompts: {num_kept_prompts} >= {train_batch_size}"
        
        # Check if we can still resample
        if self.max_num_gen_batches > 0 and num_gen_batches >= self.max_num_gen_batches:
            reason = (
                f"Max resampling limit reached: {num_gen_batches} >= {self.max_num_gen_batches}. "
                f"Only {num_kept_prompts} prompts kept (need {train_batch_size}). "
                f"Consider setting max_num_gen_batches=0 for unlimited resampling or "
                f"checking if data is too difficult."
            )
            return False, reason
        
        # Resample
        return True, (
            f"Insufficient prompts after filtering: {num_kept_prompts} < {train_batch_size}. "
            f"Resampling (batch {num_gen_batches + 1})..."
        )
    
    def align_batch_size(
        self,
        batch: DataProto,
        train_batch_size: int,
        rollout_n: int,
    ) -> DataProto:
        """
        Align batch size to match training requirements.
        
        After accumulating multiple generation batches, we need to ensure:
        - Number of unique prompts = train_batch_size
        - Total trajectories = train_batch_size * rollout_n
        
        This is done by slicing (not padding) to the required size.
        
        Args:
            batch: The accumulated batch
            train_batch_size: Target number of unique prompts
            rollout_n: Rollouts per prompt
            
        Returns:
            Aligned batch
        """
        if not self.enable:
            return batch
        
        target_traj_size = train_batch_size * rollout_n
        if len(batch) > target_traj_size:
            batch = batch[:target_traj_size]
        
        return batch
    
    def log_filtering_info(
        self,
        num_gen_batches: int,
        stats: Dict,
        num_prompts_accumulated: int,
        train_batch_size: int,
        should_resample: bool,
    ) -> None:
        """
        Log diagnostic information about the filtering process.
        
        Args:
            num_gen_batches: Current generation batch number
            stats: Group statistics from compute_group_statistics
            num_prompts_accumulated: Total prompts accumulated so far
            train_batch_size: Target training batch size
            should_resample: Whether we're resampling
        """
        if not self.enable:
            return
        
        if not stats:
            return
        
        print(f"\n[FilterGroups] Generation batch #{num_gen_batches}")
        print(f"  Metric: {self.metric}")
        print(f"  Total groups: {stats['total_uids']}")
        print(f"  Zero-variance groups: {stats['filtered_uids']}")
        print(f"  Single-sample groups: {len(stats['single_sample_uids'])}")
        print(f"  Kept groups: {len(stats['kept_prompt_uids'])}")
        print(f"  Accumulated prompts: {num_prompts_accumulated} / {train_batch_size}")
        
        if should_resample:
            print(f"  Action: Resampling...")
        else:
            print(f"  Action: Proceeding with training")
        
        if stats['filtered_uids'] > 0:
            print(f"  Example zero-variance UIDs: {stats['zero_variance_uids'][:3]}")


__all__ = ["FilterGroupsHelper"]
