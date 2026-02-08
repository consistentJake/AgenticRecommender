# Agentic Sequential Recommendation System

A multi-agent recommendation system for **[food delivery dataset (Delivery Hero)](https://github.com/deliveryhero/dh-reco-dataset)** with **two-round LLM evaluation**, **LightGCN collaborative filtering**, **Swing user-user similarity**, and **basket prediction support**.

## Multi-Agent Architecture

The repeat evaluation pipeline (Stage 5) is built around 6 specialized agents coordinated by an orchestrator:

```
┌─────────────────────────────────────────────────────────┐
│                 RecommendationManager                   │
│                    (Orchestrator)                        │
│                                                         │
│  1. SimilarityAgent (LightGCN)                          │
│     └─ Predict top-K cuisines per user                  │
│                                                         │
│  2. CuisinePredictorAgent (Round 1 LLM)                 │
│     └─ LLM ranks cuisines + frequency ensemble          │
│                                                         │
│  3. VendorProfilerAgent (GeohashIndex)                  │
│     └─ Fetch candidate vendors for predicted cuisines   │
│                                                         │
│  4. UserProfilerAgent (Swing + Records)                 │
│     └─ Build user context from similar users' orders    │
│                                                         │
│  5. VendorRankerAgent (Round 2 LLM)                     │
│     └─ LLM ranks vendors → final recommendation        │
└─────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| # | Stage | Purpose |
|---|-------|---------|
| 1 | `load_data` | Load/merge food delivery data |
| 2 | `build_users` | Build enriched user representations |
| 3 | `build_cuisines` | Build cuisine profiles |
| 4 | `run_enhanced_rerank_evaluation` | Two-round LLM + LightGCN reflection (rerank) |
| 5 | `run_repeat_evaluation` | Agent-based two-round LLM (cuisine → vendor) |

## Quick Start

```bash
# 1. Set up environment
source venv/bin/activate

# 2. Set LLM API key (required)
export OPENROUTER_API_KEY="your-openrouter-api-key"

# 3. Run complete pipeline (rerank evaluation)
python -m agentic_recommender.workflow.workflow_runner \
    --config workflow_config_se.yaml \
    --stages load_data run_enhanced_rerank_evaluation

# 4. Run agent-based repeat evaluation
python -m agentic_recommender.workflow.workflow_runner \
    --config workflow_config_se.yaml \
    --stages run_repeat_evaluation

# Check outputs
ls -la outputs/data_se/stage8_*  # rerank results
ls -la outputs/data_se/stage9_*  # repeat results
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key from [OpenRouter](https://openrouter.ai/) for LLM inference |
| `OPENAI_COMPATIBLE_API_KEY` | No | Alternative key when using OpenAI-compatible providers |

The pipeline reads the API key from the config YAML first, then falls back to the environment variable. For security, prefer setting the env var rather than putting keys in config files.

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

Edit `agentic_recommender/workflow/workflow_config_se.yaml` (or `workflow_config_sg.yaml`):

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
| `outputs/data_se/stage1_merged_data.parquet` | Merged training data |
| `outputs/data_se/stage1_test_data.parquet` | Merged test data (if load_test_data=true) |
| `outputs/data_se/stage8_enhanced_rerank_results.json` | Rerank evaluation metrics |
| `outputs/data_se/stage8_enhanced_rerank_detailed.json` | Per-sample rerank predictions |
| `outputs/data_se/stage9_repeat_results.json` | Repeat evaluation metrics (Hit@K, NDCG, MRR) |
| `outputs/data_se/stage9_repeat_detailed.json` | Per-sample repeat predictions |

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

Models and precomputed data are cached to avoid redundant computation:

```
~/.cache/agentic_recommender/
├── lightgcn/
│   ├── data_se_method1_lightgcn.pkl
│   ├── data_se_method2_lightgcn.pkl
│   └── data_se_repeat_lightgcn.pkl
├── swing/
│   └── data_se_repeat_swing_user.pkl
├── swing_user/
│   └── data_se_repeat_swing_user.pkl
├── geohash_index/
│   └── <hash>.pkl
├── user_records/
│   └── <hash>.pkl
└── user_lookups/
    └── <hash>.pkl
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
├── agents/
│   ├── base.py                # RecommendationAgent ABC
│   ├── similarity_agent.py    # LightGCN cuisine similarity
│   ├── user_profiler.py       # Swing + user record building
│   ├── vendor_profiler.py     # Geohash-based vendor lookup
│   ├── cuisine_predictor.py   # Round 1 LLM (cuisine prediction)
│   ├── vendor_ranker.py       # Round 2 LLM (vendor ranking)
│   └── orchestrator.py        # RecommendationManager (coordinates agents)
├── data/
│   ├── enriched_loader.py     # Data loading (train + test)
│   ├── repeat_filter.py       # Repeat order dataset filtering
│   └── geohash_index.py       # Geospatial vendor indexing
├── evaluation/
│   ├── rerank_eval.py         # Two-round rerank evaluation
│   ├── repeat_evaluator.py    # Repeat evaluation (async + agent-based)
│   └── basket_metrics.py      # Basket-aware metrics
├── similarity/
│   ├── lightGCN.py            # LightGCN embeddings + helpers
│   └── methods.py             # Swing similarity with caching
└── workflow/
    ├── workflow_runner.py     # Main entry point
    ├── workflow_config_se.yaml # Sweden dataset config
    └── workflow_config_sg.yaml # Singapore dataset config
```

---

## Key Components

### Agents

| Agent | File | Purpose |
|-------|------|---------|
| `RecommendationAgent` | `agents/base.py` | Abstract base class for all agents |
| `SimilarityAgent` | `agents/similarity_agent.py` | LightGCN collaborative filtering for cuisine similarity |
| `UserProfilerAgent` | `agents/user_profiler.py` | Swing user-user similarity + user record building |
| `VendorProfilerAgent` | `agents/vendor_profiler.py` | Geohash-based candidate vendor lookup |
| `CuisinePredictorAgent` | `agents/cuisine_predictor.py` | Round 1 LLM cuisine prediction with frequency ensemble |
| `VendorRankerAgent` | `agents/vendor_ranker.py` | Round 2 LLM vendor ranking |
| `RecommendationManager` | `agents/orchestrator.py` | Orchestrates all agents for a single recommendation |

### Other Components

| Component | File | Purpose |
|-----------|------|---------|
| `EnrichedDataLoader` | `data/enriched_loader.py` | Load train/test data |
| `RepeatDatasetFilter` | `data/repeat_filter.py` | Filter repeat orders |
| `GeohashVendorIndex` | `data/geohash_index.py` | Geospatial vendor indexing |
| `AgentBasedAsyncEvaluator` | `evaluation/repeat_evaluator.py` | Async eval using agent pipeline |
| `LightGCNEmbeddingManager` | `similarity/lightGCN.py` | LightGCN with caching |
| `CuisineSwingMethod` | `similarity/methods.py` | Swing with caching |
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
| "API key not found" | Set `OPENROUTER_API_KEY` env var or configure in YAML |
| Cache not updating | Delete cache files in `~/.cache/agentic_recommender/` |
| Cold-start users skipped | Expected - only users in both train/test are evaluated |

