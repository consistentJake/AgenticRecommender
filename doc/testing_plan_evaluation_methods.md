# Testing Plan: Two Evaluation Methods + Basket Prediction

## Overview

This document outlines the testing plan for the new evaluation methods and basket prediction support.

---

## Phase 1: Unit Tests for Basket Metrics

### Test 1.1: Basic Metric Calculations
```python
# Test compute_basket_hit
predictions = ["Italian", "Chinese", "Thai"]
ground_truth = {"Thai", "Indian"}
assert compute_basket_hit(predictions, ground_truth, 3) == 1  # Thai is in top 3
assert compute_basket_hit(predictions, ground_truth, 1) == 0  # Neither in top 1

# Test compute_basket_recall
# Recall@3 = |{Thai} intersection {Thai, Indian}| / 2 = 0.5
assert compute_basket_recall(predictions, ground_truth, 3) == 0.5

# Test compute_basket_precision
# Precision@3 = |{Thai}| / 3 = 0.333
assert abs(compute_basket_precision(predictions, ground_truth, 3) - 0.333) < 0.01
```

### Test 1.2: Edge Cases
```python
# Empty predictions
assert compute_basket_hit([], {"Thai"}, 5) == 0
assert compute_basket_recall([], {"Thai"}, 5) == 0.0

# Empty ground truth
assert compute_basket_hit(["Thai"], set(), 5) == 0

# Single item ground truth
assert compute_basket_recall(["Thai"], {"Thai"}, 1) == 1.0
```

### Test 1.3: NDCG Calculation
```python
# Perfect ranking
predictions = ["Thai", "Indian"]
ground_truth = {"Thai", "Indian"}
# All relevant items at positions 1,2 -> ideal NDCG
assert compute_basket_ndcg(predictions, ground_truth, 5) == 1.0

# Suboptimal ranking
predictions = ["Chinese", "Thai", "Indian"]
ground_truth = {"Thai", "Indian"}
# Hits at positions 2,3 instead of 1,2
ndcg = compute_basket_ndcg(predictions, ground_truth, 5)
assert 0 < ndcg < 1  # Should be less than 1
```

---

## Phase 2: Integration Tests for Data Loading

### Test 2.1: Test Data Loading
```python
from agentic_recommender.data.enriched_loader import EnrichedDataLoader, DataConfig

config = DataConfig(data_dir=Path("/home/zhenkai/Downloads/data_se/data_se/"))
loader = EnrichedDataLoader(config)

# Verify test data loading
test_df = loader.load_test_orders()
assert len(test_df) > 0
assert 'customer_id' in test_df.columns
assert 'order_id' in test_df.columns

# Verify merged test data
merged_test = loader.load_merged_test()
assert 'cuisine' in merged_test.columns
assert len(merged_test) > 0
```

### Test 2.2: User Overlap Verification
```python
train_df = loader.load_merged()
test_df = loader.load_merged_test()

train_users = set(train_df['customer_id'].unique())
test_users = set(test_df['customer_id'].unique())

overlap = train_users & test_users
cold_start = test_users - train_users

print(f"Training users: {len(train_users)}")
print(f"Test users: {len(test_users)}")
print(f"Overlapping users: {len(overlap)}")
print(f"Cold-start users: {len(cold_start)}")

# Expect some overlap
assert len(overlap) > 0
```

---

## Phase 3: Test Sample Builder Tests

### Test 3.1: Method 1 - Leave-Last-Out
```python
from agentic_recommender.evaluation.rerank_eval import build_test_samples

# Build test samples with basket
samples = build_test_samples(
    train_df,
    n_samples=10,
    min_history=5,
    return_basket=True,
)

assert len(samples) == 10
for sample in samples:
    assert 'ground_truth_items' in sample
    assert isinstance(sample['ground_truth_items'], set)
    assert len(sample['ground_truth_items']) >= 1
    assert 'basket_size' in sample
    assert sample['basket_size'] == len(sample['ground_truth_items'])
```

### Test 3.2: Method 2 - Test File Samples
```python
from agentic_recommender.evaluation.rerank_eval import build_test_samples_from_test_file

samples = build_test_samples_from_test_file(
    train_df=train_df,
    test_df=test_df,
    prediction_target="cuisine",
)

print(f"Test samples: {len(samples)}")

for sample in samples:
    # Every sample should have basket ground truth
    assert 'ground_truth_items' in sample
    assert isinstance(sample['ground_truth_items'], set)

    # Customer should be in training data
    customer_id = sample['customer_id']
    assert customer_id in train_df['customer_id'].values

    # Order history should exist
    assert len(sample['order_history']) > 0
```

---

## Phase 4: LightGCN and Swing Caching Tests

### Test 4.1: Method-Specific Cache Creation
```python
from agentic_recommender.similarity.lightGCN import (
    LightGCNEmbeddingManager,
    LightGCNConfig,
    filter_interactions_leave_last_out,
    get_all_interactions,
)
from pathlib import Path

# Get interactions for both methods
interactions_m1 = filter_interactions_leave_last_out(train_df)
interactions_m2 = get_all_interactions(train_df)

print(f"Method 1 interactions: {len(interactions_m1)}")
print(f"Method 2 interactions: {len(interactions_m2)}")

# Method 2 should have more interactions (includes last orders)
assert len(interactions_m2) > len(interactions_m1)

# Train and cache for method1
config = LightGCNConfig(epochs=5, embedding_dim=32)  # Small for testing
manager = LightGCNEmbeddingManager(config)
manager.load_or_train("test_dataset", interactions_m1, method="method1")

# Verify cache exists
cache_path = Path.home() / ".cache" / "agentic_recommender" / "lightgcn" / "test_dataset_method1_lightgcn.pkl"
assert cache_path.exists()
```

### Test 4.2: Swing Caching
```python
from agentic_recommender.similarity.methods import CuisineSwingMethod, CuisineSwingConfig

swing = CuisineSwingMethod(CuisineSwingConfig())
swing.fit(interactions_m1)
swing.save_to_cache("test_dataset", "method1")

# Verify cache exists
cache_path = Path.home() / ".cache" / "agentic_recommender" / "swing" / "test_dataset_method1_swing.pkl"
assert cache_path.exists()

# Test loading
swing2 = CuisineSwingMethod(CuisineSwingConfig())
assert swing2.load_from_cache("test_dataset", "method1") == True
assert swing2._fitted == True
```

---

## Phase 5: End-to-End Workflow Tests

### Test 5.1: Method 1 End-to-End
```bash
# Edit config to use method1
# evaluation_method: "method1"

python -m agentic_recommender.workflow.workflow_runner \
  --stages run_enhanced_rerank_evaluation

# Expected output checks:
# - Log: "EVALUATION METHOD: METHOD1"
# - Log: "Loading/training LightGCN with method1 cache"
# - Log: "Training interactions (N-1 orders): X"
# - Output JSON contains basket metrics
```

### Test 5.2: Method 2 End-to-End
```bash
# Edit config to use method2
# evaluation_method: "method2"

# First ensure test data exists
python -m agentic_recommender.workflow.workflow_runner \
  --stages load_data

# Then run evaluation
python -m agentic_recommender.workflow.workflow_runner \
  --stages run_enhanced_rerank_evaluation

# Expected output checks:
# - Log: "EVALUATION METHOD: METHOD2"
# - Log: "Loading test data"
# - Log: "Skipping X cold-start users" (if any)
# - Log: "Loading/training LightGCN with method2 cache"
```

### Test 5.3: Cache Isolation Verification
```bash
# Run method1 first
# Then run method2
# Verify both cache files exist and are different

ls -la ~/.cache/agentic_recommender/lightgcn/
# Should see: data_se_method1_lightgcn.pkl, data_se_method2_lightgcn.pkl

ls -la ~/.cache/agentic_recommender/swing/
# Should see: data_se_method1_swing.pkl, data_se_method2_swing.pkl
```

---

## Phase 6: Metric Validation

### Test 6.1: Basket Metrics in Output
```python
import json

with open("outputs/stage8_enhanced_rerank_results.json") as f:
    results = json.load(f)

# Verify basket metrics are present
assert 'basket_hit@5' in results
assert 'basket_recall@5' in results
assert 'basket_precision@5' in results
assert 'basket_ndcg@5' in results
assert 'basket_mrr' in results
assert 'avg_basket_size' in results

# Verify reasonable values
assert 0 <= results['basket_hit@5'] <= 1
assert 0 <= results['basket_recall@5'] <= 1
assert results['avg_basket_size'] >= 1  # At least 1 item per basket
```

### Test 6.2: Compare Methods
```python
# Run both methods and compare results

# Method 1 results
with open("outputs/stage8_method1_results.json") as f:
    m1_results = json.load(f)

# Method 2 results
with open("outputs/stage8_method2_results.json") as f:
    m2_results = json.load(f)

print("Method 1 vs Method 2 Comparison:")
print(f"  Method 1 Hit@5: {m1_results['hit@5']:.4f}")
print(f"  Method 2 Hit@5: {m2_results['hit@5']:.4f}")
print(f"  Method 1 Basket Recall@5: {m1_results['basket_recall@5']:.4f}")
print(f"  Method 2 Basket Recall@5: {m2_results['basket_recall@5']:.4f}")
```

---

## Quick Start Test Commands

```bash
# 1. Run basic import test
python -c "from agentic_recommender.evaluation.basket_metrics import compute_basket_hit; print('Imports OK')"

# 2. Run method1 evaluation (quick test with 10 samples)
# First update config: n_samples: 10, evaluation_method: "method1"
python -m agentic_recommender.workflow.workflow_runner --stages run_enhanced_rerank_evaluation

# 3. Check output
cat outputs/stage8_enhanced_rerank_results.json | python -m json.tool | head -30

# 4. Verify cache files
ls -la ~/.cache/agentic_recommender/lightgcn/
ls -la ~/.cache/agentic_recommender/swing/
```

---

## Expected Issues to Watch For

1. **Cold-start users**: Method 2 will skip users only in test file
2. **Empty baskets**: Some orders might have single item (basket_size=1)
3. **Cache conflicts**: Switching methods should use different caches
4. **Memory usage**: Large datasets may need batch processing
5. **Test file not found**: Method 2 requires load_test_data enabled
