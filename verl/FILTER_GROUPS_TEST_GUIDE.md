# Filter Groups Testing Guide

## Overview

This document provides comprehensive testing procedures for the filter_groups implementation in VERL.

## Test Structure

```
tests/
├── test_filter_groups.py              # Unit tests for FilterGroupsHelper
├── test_filter_groups_integration.py  # Integration tests with ray_trainer
└── manual_testing/                    # Manual testing procedures
```

## Running Tests

### 1. Unit Tests

#### Test FilterGroupsHelper Initialization
```bash
pytest tests/test_filter_groups.py::TestFilterGroupsHelperBasic -v
```

**What it tests:**
- Initialization with/without metrics
- Error handling for missing required fields
- Disabled helper behavior

**Expected output:**
```
test_initialization_disabled PASSED
test_initialization_enabled_with_metric PASSED
test_initialization_enabled_without_metric_raises PASSED
test_disabled_helper_returns_early PASSED
```

#### Test Statistics Computation
```bash
pytest tests/test_filter_groups.py::TestFilterGroupsHelperStatistics -v
```

**What it tests:**
- Single group variance computation
- Zero-variance group detection
- Single-sample group handling
- Mixed batch with different group types

**Expected output:**
```
test_compute_group_statistics_single_group PASSED
test_compute_group_statistics_zero_variance PASSED
test_compute_group_statistics_single_sample_groups PASSED
test_compute_group_statistics_mixed_groups PASSED
```

#### Test Resampling Decision Logic
```bash
pytest tests/test_filter_groups.py::TestFilterGroupsHelperResample -v
```

**What it tests:**
- Decision with sufficient data
- Decision with insufficient data (not at limit)
- Decision with insufficient data (at limit)
- Unlimited resampling logic

**Expected output:**
```
test_should_resample_sufficient_prompts PASSED
test_should_resample_insufficient_not_at_limit PASSED
test_should_resample_insufficient_at_limit PASSED
test_should_resample_unlimited_batches PASSED
```

#### Run All Unit Tests
```bash
pytest tests/test_filter_groups.py -v
```

**Total expected:** 15+ tests, all PASSED

### 2. Integration Tests

#### Test With RayTrainer
```bash
pytest tests/test_filter_groups_integration.py -v
```

**What it tests:**
- FilterGroupsHelper initialization in trainer
- Default disabled state
- Configuration options (different metrics)
- Metric preparation for different types

**Expected output:**
```
test_filter_groups_helper_initialized PASSED
test_filter_groups_disabled_by_default PASSED
test_config_with_acc_metric PASSED
test_config_with_score_metric PASSED
test_config_with_seq_reward_metric PASSED
test_config_with_seq_final_reward_metric PASSED
test_max_num_gen_batches_unlimited PASSED
test_max_num_gen_batches_limited PASSED
test_prepare_seq_reward_metric PASSED
test_prepare_seq_final_reward_metric PASSED
```

### 3. Smoke Test (Minimal Training)

#### Prerequisites
```bash
# Ensure you have a small dataset or can use existing test data
cd PROJECT_ROOT/verl
```

#### Test Configuration
Create `test_filter_groups_config.yaml`:

```yaml
# Minimal config for testing
data:
  train_batch_size: 4
  gen_batch_size: 4
  train_files: ["path/to/test/data.jsonl"]
  val_files: ["path/to/test/data.jsonl"]

trainer:
  total_epochs: 1
  total_training_steps: 2  # Just 2 steps

algorithm:
  filter_groups:
    enable: true
    metric: "acc"
    max_num_gen_batches: 3

actor_rollout_ref:
  rollout:
    n: 2  # 2 rollouts per prompt
```

#### Run Smoke Test
```bash
python -m verl.trainer.main_ppo \
    --config test_filter_groups_config.yaml \
    2>&1 | tee smoke_test.log
```

**Expected behavior:**
1. Training starts normally
2. After generation and reward computation:
   ```
   [FilterGroups] Generation batch #0
     Metric: acc
     Total groups: 4
     Zero-variance groups: X
     ...
   ```
3. Training loop completes with metrics logged
4. No errors or crashes

**Success criteria:**
- ✅ No exceptions raised
- ✅ Filter groups logs appear
- ✅ Training completes
- ✅ Metrics are logged

### 4. Configuration Testing

#### Test: Enable/Disable Toggle
```bash
# Disable (should run faster without filtering overhead)
python -m verl.trainer.main_ppo --config config.yaml \
    --algorithm.filter_groups.enable false

# Enable (should filter low-variance groups)
python -m verl.trainer.main_ppo --config config.yaml \
    --algorithm.filter_groups.enable true \
    --algorithm.filter_groups.metric acc
```

#### Test: Different Metrics
```bash
# Test with different metric types
for metric in "acc" "score" "seq_reward" "seq_final_reward"; do
    python -m verl.trainer.main_ppo --config config.yaml \
        --algorithm.filter_groups.enable true \
        --algorithm.filter_groups.metric $metric
done
```

#### Test: Resampling Limits
```bash
# Limited resampling
python -m verl.trainer.main_ppo --config config.yaml \
    --algorithm.filter_groups.enable true \
    --algorithm.filter_groups.max_num_gen_batches 5

# Unlimited resampling
python -m verl.trainer.main_ppo --config config.yaml \
    --algorithm.filter_groups.enable true \
    --algorithm.filter_groups.max_num_gen_batches 0
```

## Performance Testing

### Latency Benchmark

```bash
# Run without filter_groups
time python -m verl.trainer.main_ppo --config config.yaml \
    --algorithm.filter_groups.enable false \
    --trainer.total_training_steps 10 > baseline.log 2>&1

# Run with filter_groups
time python -m verl.trainer.main_ppo --config config.yaml \
    --algorithm.filter_groups.enable true \
    --algorithm.filter_groups.metric acc \
    --trainer.total_training_steps 10 > filtered.log 2>&1

# Compare timing in logs
grep "timing" baseline.log | head -5
grep "timing" filtered.log | head -5
```

**Expected result:** <10% overhead

### Memory Usage
```bash
# Monitor memory with nvidia-smi or similar tools
watch -n 1 nvidia-smi

# Run training and observe memory usage stays consistent
python -m verl.trainer.main_ppo --config config.yaml \
    --algorithm.filter_groups.enable true
```

## Edge Case Testing

### 1. Zero-Variance Dominant Dataset
Create synthetic dataset where most prompts have all same reward:

```python
# Test data generation
import json

data = []
for i in range(100):
    data.append({
        "prompt": f"Question {i}",
        "answer": f"Answer {i}",
        "reward": 1.0 if i < 80 else 0.5  # 80% have reward 1.0
    })

with open("zero_var_data.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
```

**Expected behavior:**
- Filter_groups aggressively filters degenerate batches
- Resampling triggers frequently
- Eventually reaches training batch size or max iterations

### 2. All Different Rewards Dataset
Create dataset where all rollouts get different rewards:

```python
data = []
for i in range(100):
    data.append({
        "prompt": f"Question {i}",
        "answers": [f"Answer {i}-{j}" for j in range(10)],
        "rewards": list(np.random.rand(10))  # All different
    })
```

**Expected behavior:**
- Almost no filtering (all groups have variance)
- No resampling needed
- Training proceeds normally

### 3. Single Sample Groups Only
Create dataset with only single samples (no rollouts):

```python
config:
  actor_rollout_ref:
    rollout:
      n: 1  # Only 1 rollout per prompt
```

**Expected behavior:**
- All groups kept (single-sample exception)
- No filtering, no resampling
- Training unchanged

## Regression Testing

### Test Against Reference Implementation

Compare behavior with DAPO reference implementation in `recipe/dapo/dapo_ray_trainer.py`:

```bash
# Run DAPO reference trainer
python recipe/dapo/dapo_ray_trainer.py --config reference_config.yaml \
    --algorithm.filter_groups.enable true > reference_output.log 2>&1

# Run new ray_trainer with same config
python -m verl.trainer.main_ppo --config reference_config.yaml \
    --algorithm.filter_groups.enable true > new_output.log 2>&1

# Compare outputs
diff reference_output.log new_output.log
```

**Expected:** Functionally equivalent results (same filtered groups, same training progression)

## Documentation Tests

### Test Code Examples in Documentation

```bash
# Extract and run examples from FILTER_GROUPS_USAGE.md
# Example 1: Conservative Filtering Config
python -m verl.trainer.main_ppo \
    --config minimal_config.yaml \
    --algorithm.filter_groups.enable true \
    --algorithm.filter_groups.metric acc \
    --algorithm.filter_groups.max_num_gen_batches 5
```

## Continuous Integration

### GitHub Actions Workflow (Example)

```yaml
name: Test Filter Groups

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e .
      - name: Run unit tests
        run: pytest tests/test_filter_groups.py -v
      - name: Run integration tests
        run: pytest tests/test_filter_groups_integration.py -v
      - name: Run smoke test
        run: |
          python -m verl.trainer.main_ppo \
            --config test_filter_groups_config.yaml
```

## Troubleshooting

### Tests Fail with "verl not installed"
```bash
# Install verl in development mode
pip install -e PROJECT_ROOT/verl
```

### Tests Fail with "DataProto" issues
```bash
# Ensure all imports are available
python -c "from verl import DataProto; print('DataProto OK')"
```

### Training Crashes with filter_groups
1. Check config is valid: `--algorithm.filter_groups.enable true`
2. Check metric is available: `--algorithm.filter_groups.metric acc`
3. Check dataset format matches expectations
4. Check logs for specific error messages

### Resampling Never Stops
1. Increase `max_num_gen_batches`: Default 0 (unlimited)
2. Reduce `train_batch_size`: Less required prompts
3. Check dataset difficulty
4. Monitor logs to see how many groups are filtered

## Checklist for Release

- [ ] All unit tests pass: `pytest tests/test_filter_groups.py -v`
- [ ] All integration tests pass: `pytest tests/test_filter_groups_integration.py -v`
- [ ] Smoke test completes: Training runs with filter_groups enabled
- [ ] No regression: Same results as reference implementation
- [ ] Documentation updated: FILTER_GROUPS_USAGE.md is current
- [ ] Code review: Changes reviewed and approved
- [ ] Performance: <10% overhead confirmed
- [ ] Edge cases: Tested with zero-variance and all-variance datasets

## Support

For test failures or issues:
1. Run with `-v` flag for verbose output
2. Check test logs for specific error messages
3. Open issue with:
   - Test name and command
   - Full error traceback
   - System configuration (Python version, CUDA, etc.)
   - Data sample (if available)
