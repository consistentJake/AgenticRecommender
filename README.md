# Agentic Sequential Recommendation System

A cuisine-level recommendation system for food delivery with **two-round LLM reranking**, **LightGCN collaborative filtering**, and **basket prediction support**.

## Current System Architecture

```
Training Data → LightGCN + Swing (cached per method)
    → Candidate Generation (top 20 cuisines)
    → Round 1: LLM Reranking
    → LightGCN User-Cuisine Scores
    → Round 2: LLM Reflection (final reranking)
    → Evaluation Metrics (single-item + basket)
```

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run complete pipeline (Method 1: Leave-Last-Out)
python -m agentic_recommender.workflow.workflow_runner \
    --config agentic_recommender/workflow/workflow_config_linux.yaml \
    --stages load_data run_enhanced_rerank_evaluation

# Check outputs
ls -la outputs/stage8_*
```

---

## Evaluation Methods

### Method 1: PureTrainingData (Leave-Last-Out)

| Aspect | Description |
|--------|-------------|
| **Training** | N-1 orders per user (excludes last order) |
| **Testing** | Last order per user from training data |
| **LightGCN/Swing** | Trained on N-1 orders only |
| **Use Case** | When you only have training data |

### Method 2: FullHistoryTest (Train-Test Split)

| Aspect | Description |
|--------|-------------|
| **Training** | ALL orders from training data |
| **Testing** | Orders from separate test file (`orders_se_test.txt`) |
| **LightGCN/Swing** | Trained on all training data |
| **Use Case** | Proper train/test split evaluation |

### Key Difference

- **Method 1**: Prevents data leakage by excluding test order from training
- **Method 2**: Uses separate test file, cold-start users are skipped

---

## Configuration

Edit `agentic_recommender/workflow/workflow_config_linux.yaml`:

```yaml
stages:
  load_data:
    enabled: true
    settings:
      load_test_data: true  # Enable for Method 2

  run_enhanced_rerank_evaluation:
    enabled: true
    settings:
      # Choose evaluation method
      evaluation_method: "method1"  # or "method2"

      # Basket prediction
      prediction_target: "cuisine"
      enable_basket_metrics: true

      # Candidate generation
      n_candidates: 20
      items_per_seed: 5

      # LightGCN
      dataset_name: "data_se"
      lightgcn_epochs: 50
      lightgcn_embedding_dim: 64

      # Test settings
      n_samples: -1  # -1 = all samples
      min_history: 5
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/stage1_merged_data.parquet` | Merged training data |
| `outputs/stage1_test_data.parquet` | Merged test data (if load_test_data=true) |
| `outputs/stage8_enhanced_rerank_results.json` | Evaluation metrics |
| `outputs/stage8_enhanced_rerank_samples.json` | Test samples with ground truth |
| `outputs/stage8_enhanced_rerank_detailed.json` | Per-sample predictions |

---

## Evaluation Metrics

### Single-Item Metrics (Primary Cuisine)
- **Hit@K** (K=1,3,5,10): Whether ground truth cuisine is in top-K
- **NDCG@K**: Normalized DCG for ranking quality
- **MRR@K**: Mean Reciprocal Rank

### Basket Metrics (Multi-Item Ground Truth)
- **Basket Hit@K**: At least one ground truth item in top-K
- **Basket Recall@K**: |predicted ∩ ground_truth| / |ground_truth|
- **Basket Precision@K**: |predicted ∩ ground_truth| / K
- **Basket NDCG@K**: Ranking-aware with binary relevance
- **Basket MRR**: First correct item's reciprocal rank

---

## Cache Files

Models are cached per evaluation method to prevent data leakage:

```
~/.cache/agentic_recommender/
├── lightgcn/
│   ├── data_se_method1_lightgcn.pkl
│   └── data_se_method2_lightgcn.pkl
└── swing/
    ├── data_se_method1_swing.pkl
    └── data_se_method2_swing.pkl
```

---

## Running Different Methods

### Method 1 (Leave-Last-Out)
```bash
# Edit config: evaluation_method: "method1"
python -m agentic_recommender.workflow.workflow_runner \
    --stages run_enhanced_rerank_evaluation
```

### Method 2 (Train-Test Split)
```bash
# Edit config: evaluation_method: "method2", load_test_data: true
python -m agentic_recommender.workflow.workflow_runner \
    --stages load_data run_enhanced_rerank_evaluation
```

---

## Module Structure

```
agentic_recommender/
├── data/
│   └── enriched_loader.py      # Data loading (train + test)
├── evaluation/
│   ├── basket_metrics.py       # Basket-aware metrics (NEW)
│   └── rerank_eval.py          # Two-round evaluation
├── similarity/
│   ├── lightGCN.py             # LightGCN embeddings + helpers
│   └── methods.py              # Swing similarity with caching
└── workflow/
    ├── workflow_runner.py      # Main entry point
    └── workflow_config_linux.yaml
```

---

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `EnrichedDataLoader` | `data/enriched_loader.py` | Load train/test data |
| `build_test_samples` | `evaluation/rerank_eval.py` | Method 1 test samples |
| `build_test_samples_from_test_file` | `evaluation/rerank_eval.py` | Method 2 test samples |
| `filter_interactions_leave_last_out` | `similarity/lightGCN.py` | Method 1 interactions |
| `get_all_interactions` | `similarity/lightGCN.py` | Method 2 interactions |
| `CuisineSwingMethod` | `similarity/methods.py` | Swing with caching |
| `LightGCNEmbeddingManager` | `similarity/lightGCN.py` | LightGCN with caching |
| `BasketMetrics` | `evaluation/basket_metrics.py` | Basket metric computation |

---

## Test Sample Structure

### Single-Item (backward compatible)
```python
{
    'customer_id': str,
    'order_history': List[Dict],
    'ground_truth_cuisine': str,
    'target_hour': int,
    'target_day_of_week': int,
}
```

### Basket (multi-item)
```python
{
    'customer_id': str,
    'order_history': List[Dict],
    'ground_truth_items': Set[str],  # Multiple cuisines
    'ground_truth_primary': str,
    'basket_size': int,
    'order_id': str,
}
```

---

## Verification

### Quick Test (10 samples)
```bash
# Set n_samples: 10 in config
python -m agentic_recommender.workflow.workflow_runner \
    --stages run_enhanced_rerank_evaluation

# Check results
cat outputs/stage8_enhanced_rerank_results.json | python -m json.tool
```

### Verify Cache Isolation
```bash
# Run method1, then method2
ls -la ~/.cache/agentic_recommender/lightgcn/
ls -la ~/.cache/agentic_recommender/swing/
# Should see separate files for each method
```

---

## Example Output

```json
{
  "ndcg@5": 0.4523,
  "ndcg@10": 0.5012,
  "hit@1": 0.25,
  "hit@5": 0.65,
  "hit@10": 0.78,
  "basket_hit@5": 0.72,
  "basket_recall@5": 0.45,
  "basket_ndcg@5": 0.51,
  "avg_basket_size": 1.8,
  "total_samples": 500,
  "valid_samples": 485
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Test data not found" | Enable `load_test_data: true` and run `load_data` stage |
| "Method 2 requires test_data input" | Add `test_data` to input section |
| Cache not updating | Delete cache files in `~/.cache/agentic_recommender/` |
| Cold-start users skipped | Expected - only users in both train/test are evaluated |

---

## Documentation

- `docs/evaluation_methods_update.md` - Detailed implementation notes
- `docs/testing_plan_evaluation_methods.md` - Testing procedures
- `agentic_recommender/evaluation/basket_evaluation.md` - Basket metric formulas



python -m agentic_recommender.workflow.workflow_runner --config                                       workflow_config_linux.yaml --stages load_data run_enhanced_rerank_evaluation