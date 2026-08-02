# Filter Groups (Dynamic Sampling) - Usage Guide

## Overview

Filter Groups is a dynamic sampling mechanism in DAPO (Dynamic All-Prompt Optimization) that improves training stability by filtering out low-variance (degenerate) batches where all samples receive the same reward for a given prompt.

**Problem it solves**: When generating responses for a prompt, if all rollouts receive identical reward scores (e.g., all correct or all incorrect), the gradient signal is weak, and training on such batches is ineffective.

**Solution**: Detect these zero-variance groups and filter them out, keeping only groups that have sufficient variance in reward signals.

## Configuration

To enable filter_groups in your training config:

```yaml
algorithm:
  filter_groups:
    enable: true                          # Enable/disable filter_groups
    metric: "acc"                         # Metric to compute variance on: "acc", "score", "seq_reward", "seq_final_reward"
    max_num_gen_batches: 10               # Max resampling iterations (0 = unlimited)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | bool | False | Enable filter_groups feature |
| `metric` | str | None | Metric for variance computation (required if enable=True) |
| `max_num_gen_batches` | int | 0 | Maximum generation batches for resampling (≤0 = unlimited) |

### Metric Options

- **`acc`**: Accuracy metric (0 or 1). Directly available from reward manager.
- **`score`**: Raw reward score. Available in reward_extra_infos_dict.
- **`seq_reward`**: Sequence-level reward = sum of token_level_scores.
- **`seq_final_reward`**: Sequence-level final reward = sum of token_level_rewards.

## How It Works

### Data Flow

```
1. Generate responses for each prompt
   ↓
2. Compute rewards for all responses
   ↓
3. [FILTER_GROUPS] Group responses by prompt UID
   ↓
4. [FILTER_GROUPS] Compute variance of metric within each group
   ↓
5. [FILTER_GROUPS] Remove zero-variance groups (all same reward)
   ↓
6. If too few valid groups remain:
   - Resample new data and go back to step 1
   - Stop after max_num_gen_batches iterations
   ↓
7. Continue with advantage computation and training
```

### Implementation Details

#### 1. Group Creation
```python
# Responses are grouped by prompt UID
# For prompt P with N rollouts:
#   Group[P] = [response_0, response_1, ..., response_N]
```

#### 2. Variance Computation
```python
# For each group, compute metric values:
Group[P].metrics = [metric(response_0), metric(response_1), ..., metric(response_N)]

# Compute standard deviation:
Group[P].std = np.std(Group[P].metrics)

# Zero-variance if:
#   Group[P].std == 0  AND  len(Group[P]) > 1
# (single-sample groups are kept by default)
```

#### 3. Filtering Decision
```python
# Keep groups if:
#   1. Group has non-zero variance, OR
#   2. Group has only 1 sample (can't compute variance)

# Discard groups if:
#   1. All samples in group have identical metric value
```

#### 4. Batch Accumulation
```python
# If filtered data is insufficient for training batch:
#   num_kept_prompts < train_batch_size
#
# AND num_gen_batches < max_num_gen_batches:
#   Resample and go back to step 1
#
# ELSE:
#   Proceed with training using filtered data
```

## Example Configurations

### Conservative Filtering (Recommended for starting)
```yaml
algorithm:
  filter_groups:
    enable: true
    metric: "acc"                    # Simple accuracy metric
    max_num_gen_batches: 5           # Limit resampling to avoid infinite loops
```

### Aggressive Filtering (For clean datasets)
```yaml
algorithm:
  filter_groups:
    enable: true
    metric: "seq_final_reward"       # Use final accumulated reward
    max_num_gen_batches: 0           # Unlimited resampling until clean data
```

### Disabled (Baseline)
```yaml
algorithm:
  filter_groups:
    enable: false                    # No filtering
```

## Integration with Training Loop

### In `ray_trainer.py`:

```python
# 1. Initialize in __init__:
self.filter_groups_helper = FilterGroupsHelper(self.config.algorithm.filter_groups)

# 2. In training loop, after reward computation:
if self.filter_groups_helper.enable:
    # Prepare metrics
    self.filter_groups_helper.prepare_metric_in_batch(batch, ...)
    
    # Compute statistics
    stats = self.filter_groups_helper.compute_group_statistics(batch)
    
    # Filter batch
    batch, _ = self.filter_groups_helper.filter_batch_by_uids(
        batch, stats["kept_prompt_uids"]
    )
    
    # Log statistics
    self.filter_groups_helper.log_filtering_info(...)
```

## Testing

### Running Unit Tests
```bash
# Test FilterGroupsHelper only
pytest tests/test_filter_groups.py -v

# Test specific test class
pytest tests/test_filter_groups.py::TestFilterGroupsHelperBasic -v

# Test with coverage
pytest tests/test_filter_groups.py --cov=verl.trainer.ppo.filter_groups
```

### Running Integration Tests
```bash
# Test integration with ray_trainer
pytest tests/test_filter_groups_integration.py -v
```

### Smoke Test with Training
```bash
# Run minimal training with filter_groups enabled
python -m verl.trainer.main_ppo \
    --config_entrypoint verl.recipe.dapo.dapo_ray_trainer:RayDAPOTrainer \
    --config your_config.yaml \
    --trainer.total_epochs 1 \
    --data.train_batch_size 4 \
    --algorithm.filter_groups.enable true \
    --algorithm.filter_groups.metric acc \
    --algorithm.filter_groups.max_num_gen_batches 3
```

## Monitoring and Debugging

### Console Output
When filter_groups is enabled, you'll see log output like:

```
[FilterGroups] Generation batch #0
  Metric: acc
  Total groups: 8
  Zero-variance groups: 2
  Single-sample groups: 1
  Kept groups: 7
  Accumulated prompts: 7 / 8
  Action: Resampling...

[FilterGroups] Generation batch #1
  Metric: acc
  Total groups: 8
  Zero-variance groups: 1
  Single-sample groups: 0
  Kept groups: 7
  Accumulated prompts: 14 / 8
  Action: Proceeding with training
```

### Key Metrics to Monitor

| Metric | Meaning |
|--------|---------|
| Total groups | Number of unique prompts in this batch |
| Zero-variance groups | Number of prompts where all rollouts got same reward |
| Kept groups | Number of prompts that pass the variance filter |
| Accumulated prompts | Running count of kept prompts (compared to train_batch_size) |

### Common Issues

#### Issue: "Only X prompts kept (need Y)"
- **Cause**: Too many zero-variance groups; dataset might be too easy or too hard
- **Solution**: 
  - Increase `max_num_gen_batches` to allow more resampling
  - Check if dataset is appropriate difficulty
  - Use different metric (e.g., `seq_final_reward` instead of `acc`)

#### Issue: Infinite resampling loop
- **Cause**: `max_num_gen_batches` not set or 0 with impossible data
- **Solution**: Set reasonable `max_num_gen_batches` value (e.g., 10-20)

#### Issue: No effect on training
- **Cause**: Very few zero-variance groups in dataset
- **Solution**: This is expected; filter_groups only activates when needed. Monitor the logs to confirm.

## Performance Considerations

### Training Overhead
- **Enabled**: ~5-10% slower due to grouping and variance computation
- **Disabled**: Baseline speed

### Memory Usage
- **Minimal impact**: Only stores variance statistics for active batch

### When to Use
- ✅ **Use** when: Training on challenging datasets where many prompts have uniform difficulty
- ❌ **Don't use** when: Working with well-balanced datasets or when speed is critical

## Migration from DAPO Reference Implementation

If migrating from the DAPO reference implementation (in `recipe/dapo/dapo_ray_trainer.py`):

### Differences
- Reference implementation uses manual while loop for batch accumulation
- FilterGroupsHelper provides cleaner abstraction for the same logic
- Helper handles edge cases (missing metrics, empty batches, etc.)

### Compatibility
- Configuration format is identical
- Behavior is same when both are enabled
- FilterGroupsHelper is backward-compatible via graceful error handling

## References

- DAPO Paper: See `docs/algo/dapo.md`
- Reference Implementation: `recipe/dapo/dapo_ray_trainer.py` (lines 237-297)
- FilterGroupsHelper Source: `verl/trainer/ppo/filter_groups.py`
- Configuration: `verl/trainer/config/algorithm.py` (lines 43-56)

## Support and Contribution

For issues or improvements:
1. Check existing issues in the repository
2. Create detailed bug reports with:
   - Configuration YAML
   - Console output (with and without filter_groups)
   - Dataset characteristics
3. Contribute improvements via pull requests following CLAUDE.md guidelines
