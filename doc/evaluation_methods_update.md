# Evaluation Methods Update

## Overview

This update implements two evaluation methods and extends evaluation to support basket prediction (multiple items per order).

**Date**: 2026-01-25

---

## Key Changes

### 1. Two Evaluation Methods

#### Method 1: PureTrainingData (Leave-Last-Out)
- **Training data**: N-1 orders per user (excludes last order)
- **Test data**: Last order per user from training data
- **Use case**: When you only have training data and want to evaluate

#### Method 2: FullHistoryTest (Train-Test Split)
- **Training data**: ALL training data interactions
- **Test data**: Orders from separate test file (`orders_se_test.txt`)
- **Use case**: When you have a proper train/test split

### 2. Basket Prediction Support
- Ground truth is now a **set of items** (cuisines in an order) rather than single item
- New basket-aware metrics: Recall@K, Precision@K, NDCG@K, MRR, Hit@K

### 3. Method-Specific Caching
- LightGCN and Swing models now cache separately per method
- Cache files: `{dataset}_{method}_lightgcn.pkl`, `{dataset}_{method}_swing.pkl`
- Prevents data leakage between evaluation methods

---

## Files Modified

### `agentic_recommender/data/enriched_loader.py`
- Added `orders_test_file` parameter to `DataConfig`
- Added `load_test_orders()` method
- Added `load_merged_test()` method for merging test data with vendors/products

### `agentic_recommender/evaluation/basket_metrics.py` (NEW)
- Created `BasketMetrics` dataclass
- Implemented basket metric functions:
  - `compute_basket_hit()`: At least one item hit in top-K
  - `compute_basket_recall()`: |R_K intersection G| / |G|
  - `compute_basket_precision()`: |R_K intersection G| / K
  - `compute_basket_ndcg()`: Ranking-aware with binary relevance
  - `compute_basket_mrr()`: First hit reciprocal rank
  - `aggregate_basket_metrics()`: Aggregate across samples

### `agentic_recommender/evaluation/rerank_eval.py`
- Extended `EnhancedRerankConfig` with:
  - `evaluation_method`: "method1" or "method2"
  - `prediction_target`: "cuisine", "vendor", or "product"
  - `enable_basket_metrics`: Multi-item evaluation toggle
- Extended `EnhancedRerankMetrics` with basket metrics
- Added `build_test_samples_from_test_file()` for Method 2
- Extended `build_test_samples()` with `return_basket` option
- Updated `_compute_metrics()` to compute basket metrics

### `agentic_recommender/similarity/lightGCN.py`
- Updated `_get_cache_path()` to include method name
- Added `method` parameter to `load_or_train()`
- Added helper functions:
  - `filter_interactions_leave_last_out()`: Get interactions excluding last order per user
  - `get_all_interactions()`: Get all interactions

### `agentic_recommender/similarity/methods.py`
- Added caching to `CuisineSwingMethod`:
  - `save_to_cache()`: Save fitted model with method-specific name
  - `load_from_cache()`: Load from method-specific cache

### `agentic_recommender/workflow/workflow_runner.py`
- Updated `stage_load_data()` to optionally load and save test data
- Rewrote `stage_run_enhanced_rerank_evaluation()` to:
  - Support both evaluation methods
  - Use method-specific caching for LightGCN and Swing
  - Build appropriate test samples for each method

### `agentic_recommender/workflow/workflow_config_linux.yaml`
- Added `load_test_data` and `test_data` output to load_data stage
- Added new settings to run_enhanced_rerank_evaluation:
  - `evaluation_method`
  - `prediction_target`
  - `enable_basket_metrics`
  - `filter_seen_items`

---

## Test Sample Structure

### Single-Item (backward compatible):
```python
{
    'customer_id': str,
    'order_history': List[Dict],
    'ground_truth_cuisine': str,
    'target_hour': int,
    'target_day_of_week': int,
}
```

### Basket (multi-item):
```python
{
    'customer_id': str,
    'order_history': List[Dict],
    'ground_truth_items': Set[str],  # Multiple items
    'ground_truth_primary': str,      # Primary cuisine
    'target_hour': int,
    'target_day_of_week': int,
    'order_id': str,
    'basket_size': int,
}
```

---

## Configuration Examples

### Method 1 (Leave-Last-Out)
```yaml
run_enhanced_rerank_evaluation:
  settings:
    evaluation_method: "method1"
    enable_basket_metrics: true
```

### Method 2 (Train-Test Split)
```yaml
load_data:
  settings:
    load_test_data: true
  output:
    test_data: "outputs/stage1_test_data.parquet"

run_enhanced_rerank_evaluation:
  settings:
    evaluation_method: "method2"
    enable_basket_metrics: true
  input:
    test_data: "outputs/stage1_test_data.parquet"
```

---

## Cache File Locations

```
~/.cache/agentic_recommender/
  lightgcn/
    data_se_method1_lightgcn.pkl
    data_se_method2_lightgcn.pkl
  swing/
    data_se_method1_swing.pkl
    data_se_method2_swing.pkl
```

---

## Verification Steps

### Method 1 Verification
```bash
# Set evaluation_method: "method1" in config
python -m agentic_recommender.workflow.workflow_runner \
  --stages run_enhanced_rerank_evaluation
```

**Expected logs**:
- "EVALUATION METHOD: METHOD1"
- "Loading/training LightGCN with method1 cache"
- "Training interactions (N-1 orders): X"

### Method 2 Verification
```bash
# Set evaluation_method: "method2" in config
python -m agentic_recommender.workflow.workflow_runner \
  --stages load_data run_enhanced_rerank_evaluation
```

**Expected logs**:
- "EVALUATION METHOD: METHOD2"
- "Loading/training LightGCN with method2 cache"
- "Training interactions (all): X"
- "Skipping X cold-start users"

### Basket Metrics Verification
- Check output JSON for `basket_hit@5`, `basket_recall@5`, etc.
- Verify `avg_basket_size` is reasonable (> 1 for multi-item baskets)
