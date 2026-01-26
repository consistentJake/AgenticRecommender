# Implementation Plan: Enhanced Two-Round LLM Reranking System

## Overview
Transform the recommendation system from simple user-user collaborative filtering to a two-round LLM reranking system with cuisine-cuisine swing similarity and LightGCN-based reflection.

**Key Decision: All operations at CUISINE level (not vendor level)**
- Candidate generation: Cuisine-to-cuisine swing similarity
- LightGCN: User-cuisine embeddings
- Evaluation: Cuisine prediction

## New Pipeline Flow
```
Data Loading → User Filtering (min/max) → Full History → Cuisine-Cuisine Swing Candidates (top 20)
    → Round 1: LLM Reranking → LightGCN User-Cuisine Similarity
    → Round 2: Reflection (final reranking) → Metrics (NDCG@K, MRR@K, HitRate@K)
```

---

## Phase 1: Data & Filtering Changes

### 1.1 Remove Purchase History Truncation
**File:** `agentic_recommender/data/representations.py`

**Change:** Remove line 128 (`purchase_history = purchase_history[-20:]`)
- Add optional `max_history` parameter to `from_orders()` method
- When `max_history=None`, keep ALL history (n-1 items for prediction)

### 1.2 Add Max Order History Filter
**File:** `agentic_recommender/workflow/workflow_runner.py` (lines 355-368)

**Change:** Add `max_order_history` filter after the existing `min_orders` filter
```python
max_order_history = stage_cfg.settings.get('max_order_history', None)  # None = infinity
if max_order_history is not None:
    valid_customers = customer_order_counts[
        (customer_order_counts >= min_orders) &
        (customer_order_counts <= max_order_history)
    ].index
```

---

## Phase 2: Cuisine-Cuisine Swing Similarity

### 2.1 Add CuisineSwingMethod Class
**File:** `agentic_recommender/similarity/methods.py`

**New class:** `CuisineSwingMethod` and `CuisineSwingConfig`
- Operates at cuisine level (not vendor level)
- `cuisine_users`: Dict[cuisine, Set[user_id]] - which users ordered this cuisine
- `user_cuisines`: Dict[user_id, Set[cuisine]] - which cuisines this user ordered
- Formula: `sim(c1, c2) = Σ(u ∈ common_users) 1 / ((|U(c1)|+α1)^β × (|U(c2)|+α1)^β × (|C(u)|+α2))`

**Key methods:**
- `fit(interactions: List[Tuple[user_id, cuisine]])` - Build indices from user-cuisine pairs
- `compute_similarity(cuisine1, cuisine2)` - Calculate cuisine similarity
- `get_similar_cuisines(cuisine, top_k)` - Get top-k similar cuisines

---

## Phase 3: LightGCN Embedding Manager (User-Cuisine)

### 3.1 Create LightGCNEmbeddingManager
**File:** `agentic_recommender/similarity/lightGCN.py`

**New class:** `LightGCNEmbeddingManager`
- Wraps existing `LightGCN` model with caching
- **Operates on user-cuisine graph** (not user-vendor)
- Cache location: `~/.cache/agentic_recommender/lightgcn/{dataset_name}_embeddings.pkl`
- Cache contents: user_embeddings, cuisine_embeddings, id_maps, cache_key (MD5 hash)

**Key methods:**
- `load_or_train(dataset_name, interactions, force_retrain=False)` - Load from cache or train
  - `interactions`: List[Tuple[user_id, cuisine]] pairs
- `get_user_cuisine_similarity(user_id, cuisine)` - Dot product of embeddings
- `get_user_cuisines_similarities(user_id, cuisines)` - Batch similarity with sorting
- `rerank_by_similarity(user_id, cuisines)` - Return cuisines sorted by similarity

**Caching strategy:**
- Cache key = MD5 hash of sorted (user, cuisine) interactions
- If cache exists and key matches, load embeddings directly
- If no cache or key mismatch, train new model and save

---

## Phase 4: Enhanced Rerank Evaluator

### 4.1 CuisineBasedCandidateGenerator
**File:** `agentic_recommender/evaluation/rerank_eval.py`

**New class:** Generate candidates using cuisine-cuisine swing similarity

**Process:**
1. Get user's n-1 purchase history (excluding last cuisine = ground truth)
2. Deduplicate to unique cuisines
3. For each unique cuisine, get top-k similar cuisines using CuisineSwingMethod
4. Combine all similar cuisines, deduplicate
5. Sort by swing similarity score, take top 20

### 4.2 EnhancedRerankEvaluator
**File:** `agentic_recommender/evaluation/rerank_eval.py`

**New class:** Two-round LLM evaluation

**Round 1 - LLM Reranking:**
- Input: User profile + ALL n-1 history + 20 candidates (shuffled)
- Prompt: "Re-rank all 20 cuisines from most likely to least likely"
- Output: Ordered list [1...20]

**Round 2 - Reflection/Critics:**
- Input:
  - User history summary
  - Round 1 LLM ranking
  - LightGCN similarity scores for 20 candidates
  - LightGCN-based ranking
- Prompt: "Consider both initial ranking and collaborative filtering signals. Produce final ranking."
- Output: Final ordered list [1...20]

### 4.3 Evaluation Metrics
**New dataclass:** `EnhancedRerankMetrics`

Metrics computed:
- **NDCG@K** (K=5,10): `Σ 1/log2(rank+1) / ideal_NDCG`
- **MRR@K**: `1/rank` of first correct item
- **HitRate@K** (K=1,3,5,10): `1 if rank <= K else 0`
- Comparison metrics: Round1 vs Final performance

---

## Phase 5: Workflow Integration

### 5.1 New Stage: run_enhanced_rerank_evaluation
**File:** `agentic_recommender/workflow/workflow_runner.py`

**New method:** `stage_run_enhanced_rerank_evaluation()`

**Flow:**
1. Load merged data
2. Build CuisineSwingMethod from all (user_id, cuisine) interactions
3. Load/train LightGCN embeddings for user-cuisine graph (cached by dataset name)
4. Build test samples (users with min_history orders)
5. For each sample:
   - Generate 20 cuisine candidates via cuisine-cuisine swing
   - Round 1: LLM reranking of cuisines
   - Get LightGCN user-cuisine scores for candidates
   - Round 2: Reflection with all signals
   - Record final cuisine ranking
6. Compute and save metrics (NDCG@K, MRR@K, HitRate@K)

### 5.2 Configuration Updates
**File:** `agentic_recommender/workflow/workflow_config_linux.yaml`

**Updates to build_users:**
```yaml
settings:
  min_orders: 5
  max_order_history: null  # null = no max (infinity)
```

**New stage:**
```yaml
run_enhanced_rerank_evaluation:
  enabled: true
  settings:
    n_candidates: 20
    items_per_seed: 5      # top-k similar items per history item
    dataset_name: "data_se" # for LightGCN cache
    lightgcn_epochs: 50
    lightgcn_embedding_dim: 64
    temperature_round1: 0.3
    temperature_round2: 0.2
    n_samples: 10
    min_history: 5
```

---

## New Prompt Templates

### 5.3 Round 1 Reranking Prompt
**File:** `agentic_recommender/core/templates/rerank/round1.txt`

```
Based on this user's complete order history, RE-RANK all 20 candidate cuisines.

## Complete Order History ({n} orders, oldest to newest):
{history}

## Prediction Context:
Target time: {day_name} at {hour}:00

## Candidates to Rank:
{candidates}

Return JSON: {"ranking": ["cuisine1", ...], "reasoning": "..."}
```

### 5.4 Round 2 Reflection Prompt
**File:** `agentic_recommender/core/templates/rerank/round2.txt`

```
Review the initial ranking and collaborative filtering signals to produce final ranking.

## User's Recent History:
{history_summary}

## Initial LLM Ranking (Round 1):
{round1_ranking}

## Collaborative Filtering Signals (LightGCN):
{lightgcn_scores}

Produce FINAL re-ranking balancing both signals.
Return JSON: {"final_ranking": [...], "reflection": "..."}
```

---

## Files to Modify/Create

| File | Action | Changes |
|------|--------|---------|
| `data/representations.py` | Modify | Remove line 128 truncation, add `max_history` param |
| `workflow/workflow_runner.py` | Modify | Add `max_order_history` filter, add new stage |
| `workflow/workflow_config_linux.yaml` | Modify | Add new settings |
| `similarity/methods.py` | Modify | Add `CuisineSwingMethod`, `CuisineSwingConfig` |
| `similarity/lightGCN.py` | Modify | Add `LightGCNEmbeddingManager` class (user-cuisine) |
| `evaluation/rerank_eval.py` | Modify | Add `CuisineBasedCandidateGenerator`, `EnhancedRerankEvaluator`, `EnhancedRerankMetrics` |
| `core/templates/rerank/round1.txt` | Create | Round 1 prompt template |
| `core/templates/rerank/round2.txt` | Create | Round 2 prompt template |

---

## Implementation Order

1. **Phase 1** - Data changes (representations.py, workflow_runner.py user filtering)
2. **Phase 2** - Cuisine-cuisine swing (methods.py)
3. **Phase 3** - LightGCN manager with caching for user-cuisine (lightGCN.py)
4. **Phase 4** - Enhanced evaluator (rerank_eval.py)
5. **Phase 5** - Workflow integration and prompts

---

## Verification Plan

1. **Unit tests:**
   - CuisineSwingMethod: Test similarity computation on mock (user, cuisine) data
   - LightGCNEmbeddingManager: Test caching (train once, load from cache on second run)
   - Candidate generation: Test dedup and sorting logic with cuisine histories

2. **Integration test:**
   - Run full pipeline with `n_samples=5` on data_se
   - Verify Round 1 and Round 2 produce valid cuisine rankings (20 items each)
   - Check metrics output (NDCG@K, MRR@K, HitRate@K)

3. **Manual verification:**
   - Inspect candidate generation debug info (unique cuisines, similar cuisines per seed)
   - Compare Round 1 vs Final rankings to see reflection impact
   - Check LightGCN cache files exist at `~/.cache/agentic_recommender/lightgcn/`

4. **Run command:**
   ```bash
   python -m agentic_recommender.workflow.workflow_runner
   ```
