# Repeated Dataset Evaluation Pipeline — Design & Implementation Plan

## 1. Overview

Add a new **Stage 9: `run_repeat_evaluation`** that predicts repeated orders using a two-round LLM pipeline:

- **Round 1**: Predict top 3 primary cuisines (using LightGCN on customer-cuisine)
- **Round 2**: Rank candidate vendors filtered by geohash + Round 1 cuisines (using Swing user-user collaborative filtering)

**Item definition**: `vendor_id||primary_cuisine`
**Category definition**: `primary_cuisine`
**Ground truth**: `vendor_id||primary_cuisine` of the test order
**Metrics**: Hit@1/3/5/10, NDCG@1/3/5/10, MRR

---

## 2. Architecture Diagram

```
                    Filtered Training Data
                    (users with >= N orders,
                     test orders with repeat vendors)
                            │
               ┌────────────┼────────────────┐
               │            │                │
        LightGCN       GeohashIndex      Swing (user-user)
    (customer→cuisine)  (geohash→         (customer→
     embeddings         cuisine→           vendor||cuisine)
                        vendors)
               │            │                │
               └────────────┼────────────────┘
                            │
                    ┌───────┴───────┐
                    │   Round 1     │
                    │ User history  │
                    │ + LightGCN    │
                    │ top 10 cuisines│
                    │ → predict top │
                    │   3 cuisines  │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │  Candidate    │
                    │  Selection    │
                    │ geohash match │
                    │ + cuisine     │
                    │ filter        │
                    │ (max 20       │
                    │  vendors)     │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │   Round 2     │
                    │ R1 cuisines   │
                    │ + candidates  │
                    │ + similar     │
                    │   users' CF   │
                    │ → rank vendors│
                    └───────┬───────┘
                            │
                    Hit@K, NDCG@K, MRR
```

---

## 3. Detailed Component Design

### 3.1 Repeated Dataset Filter

**File**: `agentic_recommender/data/repeat_filter.py`

```python
class RepeatDatasetFilter:
    """Filter training/test data for repeated order evaluation."""
    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "repeat_filter"

    def filter(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        min_history_items: int = 5,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Filter dataset for repeated order prediction.

        Steps:
        1. Group training orders by customer_id (each unique order_id = 1 order)
        2. Keep users with >= min_history_items unique orders
        3. For each retained user, collect set of vendor_ids from training
        4. Filter test orders: only keep test orders where vendor_id
           is in user's training vendor set
        5. Log and return stats

        Returns:
            (filtered_train_df, filtered_test_df, stats_dict)

        Stats dict includes:
            - original_train_users, filtered_train_users
            - original_train_orders, filtered_train_orders
            - original_test_orders, filtered_test_orders
            - repeat_rate (% of test orders that are repeats)
            - avg_orders_per_user, min/max_orders_per_user
            - avg_unique_vendors_per_user
        """

    def _compute_cache_key(self, train_df, test_df, min_history_items) -> str:
        """MD5 of (train shape + first 2MB, test shape + first 2MB, min_history_items)."""

    def save_cache(self, key, filtered_train, filtered_test, stats) -> bool: ...
    def load_cache(self, key) -> Optional[Tuple]: ...
```

**Logging output example**:
```
REPEAT DATASET FILTER
  Original: 18,234 users, 95,432 train orders, 12,543 test orders
  After min_history=5: 15,678 users (86.0%)
  After repeat filter: 8,234 test orders (65.6% repeat rate)
  Avg orders/user: 8.3 | Min: 5 | Max: 142
  Avg unique vendors/user: 4.7
```

---

### 3.2 Geohash-Cuisine-Vendor Index

**File**: `agentic_recommender/data/geohash_index.py`

```python
class GeohashVendorIndex:
    """Pre-computed index: vendor_geohash → primary_cuisine → [vendor_ids]."""
    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "geohash_index"

    def build(
        self,
        train_df: pd.DataFrame,
        use_cache: bool = True,
    ) -> Dict:
        """
        Build geohash→cuisine→vendors mapping from training data.

        Extracts unique (vendor_id, vendor_geohash, cuisine) tuples.
        Builds:
            self.geohash_cuisine_vendors: {geohash: {cuisine: [vendor_ids]}}
            self.vendor_metadata: {vendor_id: (vendor_geohash, cuisine)}

        Returns stats dict.
        """

    def get_vendors(
        self,
        vendor_geohash: str,
        cuisines: List[str],
        max_candidates: int = 20,
    ) -> List[str]:
        """
        Look up vendors matching geohash and cuisine list.

        Args:
            vendor_geohash: The test order's vendor_geohash (lookup key)
            cuisines: List of primary cuisines from Round 1 prediction
            max_candidates: Maximum vendors to return

        Returns:
            List of vendor_id||cuisine strings (item format)
        """

    def get_stats(self) -> Dict:
        """
        Return index statistics for logging:
            - total_geohashes: number of unique vendor_geohash values
            - total_vendors: number of unique vendors
            - total_cuisines: number of unique cuisines
            - avg_vendors_per_geohash: average vendors sharing a geohash
            - avg_vendors_per_geohash_cuisine: average vendors per (geohash, cuisine) pair
            - max_vendors_per_geohash: largest cluster
            - geohash_cuisine_pairs: total unique (geohash, cuisine) combinations
        """

    def save_cache(self, ...) -> bool: ...
    def load_cache(self, ...) -> Optional[...]: ...
```

**Logging output example**:
```
GEOHASH-CUISINE-VENDOR INDEX
  Unique geohashes: 142
  Unique vendors: 3,456
  Unique cuisines: 39
  Geohash-cuisine pairs: 1,234
  Avg vendors per geohash: 24.3
  Avg vendors per (geohash, cuisine): 2.8
  Max vendors in single geohash: 89
```

---

### 3.3 LightGCN on Categories

**File**: No changes to `agentic_recommender/similarity/lightGCN.py`

The existing `LightGCNEmbeddingManager` already supports this:
- Train with `prediction_target="cuisine"` and `method="repeat"` for cache isolation
- Interactions = `(customer_id, primary_cuisine)` pairs from filtered training data
- Produces user embeddings and cuisine embeddings
- `get_top_cuisines_for_user(user_id, top_k=10)` returns ranked cuisines with scores

**Usage in stage**:
```python
# Extract cuisine-level interactions from filtered training data
cuisine_interactions = []
for _, row in filtered_train_df.drop_duplicates('order_id').iterrows():
    cuisine_interactions.append((str(row['customer_id']), str(row['cuisine'])))
cuisine_interactions = list(set(cuisine_interactions))  # deduplicate

lightgcn_manager.load_or_train(
    dataset_name=dataset_name,
    interactions=cuisine_interactions,
    method="repeat",
    prediction_target="cuisine",
    force_retrain=not use_lightgcn_cache,
)
```

---

### 3.4 Swing User-User Similarity — Add Caching

**Modify**: `agentic_recommender/similarity/methods.py`

Add disk caching to `SwingMethod` (user-user), matching the pattern from `CuisineSwingMethod`:

```python
class SwingMethod(SimilarityMethod):
    CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "swing_user"

    def save_to_cache(self, dataset_name: str, method: str = "repeat") -> bool:
        """Save fitted model (user_items, item_users) to pickle."""

    def load_from_cache(self, dataset_name: str, method: str = "repeat") -> bool:
        """Load fitted model from pickle."""

    def _get_cache_path(self, dataset_name: str, method: str) -> Path: ...

    def get_top_similar_users(
        self,
        user_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Get top-k most similar users by Swing score."""
        # Compute similarity against all other users
        # Return sorted list of (user_id, score)
```

Swing interactions = `(user_id, vendor_id||primary_cuisine)` from filtered training data. This captures vendor-level co-purchase patterns for user-user similarity.

---

### 3.5 Test Sample Builder for Repeat Evaluation

**Add to**: `agentic_recommender/data/repeat_filter.py`

```python
def build_repeat_test_samples(
    filtered_train_df: pd.DataFrame,
    filtered_test_df: pd.DataFrame,
    n_samples: int = -1,
    deterministic: bool = True,
) -> List[Dict]:
    """
    Build test samples from filtered repeat dataset.

    Each sample represents one test order with full training history.

    Returns list of:
    {
        'customer_id': str,
        'order_history': List[Dict],  # training orders for this user
            # Each: {vendor_id, cuisine, day_of_week, hour,
            #        vendor_geohash, item (vendor_id||cuisine)}
        'ground_truth': str,          # vendor_id||cuisine
        'ground_truth_vendor_id': str,
        'ground_truth_cuisine': str,
        'target_hour': int,
        'target_day_of_week': int,
        'target_vendor_geohash': str,  # for geohash lookup
        'order_id': str,
    }
    """
```

---

### 3.6 Async Repeat Evaluator

**File**: `agentic_recommender/evaluation/repeat_evaluator.py`

```python
@dataclass
class RepeatEvalConfig:
    """Configuration for repeat dataset evaluation."""
    # Filter
    min_history_items: int = 5

    # LightGCN
    lightgcn_top_k_cuisines: int = 10   # Cuisines to show in Round 1
    lightgcn_epochs: int = 50
    lightgcn_embedding_dim: int = 64

    # Round 1
    round1_predict_top_k: int = 3       # Cuisines to predict
    temperature_round1: float = 0.3
    max_tokens_round1: int = 4096

    # Candidate selection
    max_candidate_vendors: int = 20

    # Round 2 / Swing
    top_similar_users: int = 5
    max_records_per_similar_user: int = 5
    temperature_round2: float = 0.2
    max_tokens_round2: int = 4096

    # General
    enable_thinking: bool = True
    prediction_target: str = "vendor_cuisine"
    dataset_name: str = "data_se"
    n_samples: int = 20
    deterministic_sampling: bool = True

    # Async
    enable_async: bool = True
    max_workers: int = 25
    checkpoint_interval: int = 50
    retry_attempts: int = 3


class AsyncRepeatEvaluator:
    """Async evaluator for repeated dataset two-round evaluation."""

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(
        self,
        async_provider: AsyncLLMProvider,
        lightgcn_manager: LightGCNEmbeddingManager,
        swing_model: SwingMethod,
        geohash_index: GeohashVendorIndex,
        train_df: pd.DataFrame,
        config: RepeatEvalConfig,
    ):
        self.provider = async_provider
        self.lightgcn = lightgcn_manager
        self.swing = swing_model
        self.geohash_index = geohash_index
        self.train_df = train_df  # for looking up similar users' records
        self.config = config
        # Build user->records lookup from train_df for fast access
        self._user_records = self._build_user_records_index(train_df)

    def _build_user_records_index(self, train_df) -> Dict[str, List[Dict]]:
        """Pre-build {user_id: [records]} for fast similar user lookup."""

    async def evaluate_async(self, test_samples, output_path, verbose=True) -> Dict:
        """Same worker pool pattern as existing AsyncRerankEvaluator."""

    async def _process_sample(self, idx, sample, verbose) -> Dict:
        """Process one test sample through both rounds."""
        # 1. Get LightGCN top K cuisines for user
        lightgcn_scores = self.lightgcn.get_top_cuisines_for_user(
            sample['customer_id'], top_k=self.config.lightgcn_top_k_cuisines
        )

        # 2. Round 1: LLM predicts top 3 cuisines
        round1_result = await self._async_round1(sample, lightgcn_scores)
        predicted_cuisines = round1_result['predicted_cuisines']

        # 3. Candidate selection via geohash index
        candidate_vendors = self.geohash_index.get_vendors(
            sample['target_vendor_geohash'],
            predicted_cuisines,
            max_candidates=self.config.max_candidate_vendors,
        )

        # 4. Get similar users' collaborative info
        similar_users_info = self._get_similar_users_records(
            sample['customer_id'],
            predicted_cuisines,
        )

        # 5. Round 2: LLM ranks candidate vendors
        round2_result = await self._async_round2(
            sample, predicted_cuisines, candidate_vendors, similar_users_info
        )

        # 6. Compute ground truth rank
        ground_truth = sample['ground_truth']
        final_rank = self._find_rank(round2_result['ranking'], ground_truth)

        return { ... }  # Full detailed result dict

    def _build_round1_prompt(self, sample, lightgcn_scores) -> str:
        """
        Round 1 prompt structure:
        - User's historical items: vendor_id || primary_cuisine || (weekday, hour)
        - Top 10 primary cuisines ranked by LightGCN scores
        - Test order's weekday + hour
        - Task: predict top 3 primary cuisines

        Return JSON: {"predicted_cuisines": ["cuisine1", "cuisine2", "cuisine3"],
                       "reasoning": "..."}
        """

    def _build_round2_prompt(self, sample, round1_cuisines,
                              candidate_vendors, similar_users_info) -> str:
        """
        Round 2 prompt structure:
        a) Round 1 cuisine predictions with order
        b) Candidate vendors (vendor_id || primary_cuisine)
        c) Similar users' records: (vendor_id, cuisine, weekday+hour),
           max 5 records per similar user
        d) Test order weekday + hour
        Task: rank candidate vendors

        Return JSON: {"final_ranking": ["vendor||cuisine", ...],
                       "reflection": "..."}
        """

    def _get_similar_users_records(
        self,
        user_id: str,
        cuisines: List[str],
    ) -> List[Dict]:
        """
        Get collaborative filtering info from similar users.

        1. Use Swing to find top_similar_users most similar users
        2. For each similar user, look up their training records
        3. Filter records to only those with cuisine in the given list
        4. Cap at max_records_per_similar_user per user

        Returns:
        [
            {
                "user_id": "U2",
                "similarity_score": 0.85,
                "records": [
                    {"vendor_id": "V1", "cuisine": "Thai",
                     "day_of_week": 3, "hour": 12},
                    ...
                ]
            },
            ...
        ]
        """

    def _parse_round1_cuisine_response(self, response, all_cuisines) -> List[str]:
        """Extract cuisine predictions. Validate against known cuisines."""

    def _parse_round2_vendor_response(self, response, candidates) -> List[str]:
        """Extract vendor ranking. Validate against candidate list."""

    def _find_rank(self, ranking, target) -> int:
        """1-indexed rank of target, 0 if not found."""
```

---

### 3.7 Metrics Computation

Reuse or adapt the existing metrics pattern. For each test sample, we compute:

| Metric | K values | Description |
|--------|----------|-------------|
| Hit@K | 1, 3, 5, 10 | 1 if ground truth in top K, else 0 |
| NDCG@K | 1, 3, 5, 10 | 1/log2(rank+1) if in top K, else 0 |
| MRR | - | 1/rank of first correct prediction |

Aggregate = mean across all test samples.

**Metrics function**:
```python
def compute_repeat_metrics(
    results: List[Dict],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict:
    """
    Compute aggregate metrics from detailed results.

    Returns:
    {
        "total_samples": N,
        "valid_samples": M,  # samples where ground truth was in candidates
        "gt_in_candidates_rate": float,
        "metrics": {
            "hit@1": float, "hit@3": float, "hit@5": float, "hit@10": float,
            "ndcg@1": float, "ndcg@3": float, "ndcg@5": float, "ndcg@10": float,
            "mrr": float,
        },
        "avg_candidate_count": float,
        "avg_time_ms": float,
    }
    """
```

---

### 3.8 Detailed Result Structure

Each sample's result (saved to JSONL streaming + detailed JSON):

```json
{
    "sample_idx": 0,
    "customer_id": "abc123",
    "ground_truth": "V456||Thai",
    "ground_truth_vendor_id": "V456",
    "ground_truth_cuisine": "Thai",
    "target_hour": 12,
    "target_day_of_week": 3,
    "target_vendor_geohash": "w21z7q",
    "order_history_count": 15,

    "lightgcn_top_cuisines": [["Thai", 0.45], ["Chinese", 0.42], ["Indian", 0.38], ...],

    "round1_prompt": "...",
    "round1_raw_response": "...",
    "round1_predicted_cuisines": ["Thai", "Chinese", "Indian"],
    "round1_reasoning": "...",

    "candidate_vendors": ["V456||Thai", "V789||Thai", "V012||Chinese", ...],
    "candidate_count": 15,
    "gt_in_candidates": true,

    "similar_users": [
        {"user_id": "U2", "similarity_score": 0.85, "records": [...]},
        ...
    ],

    "round2_prompt": "...",
    "round2_raw_response": "...",
    "final_ranking": ["V456||Thai", "V789||Thai", ...],
    "final_reflection": "...",

    "ground_truth_rank": 2,
    "time_ms": 3500.0
}
```

---

## 4. Configuration (YAML)

Added to `workflow_config_qwen32_linux.yaml`:

```yaml
    # Stage 9: Repeated Dataset Evaluation
    run_repeat_evaluation:
      enabled: true
      description: "Two-round LLM evaluation on repeated orders dataset"
      input:
        merged_data: "outputs/stage1_merged_data.parquet"
        test_data: "outputs/stage1_test_data.parquet"
      output:
        results_json: "outputs/stage9_repeat_results.json"
        samples_json: "outputs/stage9_repeat_samples.json"
        detailed_json: "outputs/stage9_repeat_detailed.json"
      settings:
        dataset_name: "data_se"

        # Dataset filtering
        min_history_items: 5              # Min unique orders per user in training

        # Per-component caching
        use_filter_cache: true            # Cache filtered dataset
        use_geohash_cache: true           # Cache geohash-cuisine-vendor index
        use_lightgcn_cache: true          # Cache LightGCN model
        use_swing_cache: true             # Cache Swing user-user model

        # LightGCN (customer → primary_cuisine)
        lightgcn_epochs: 50
        lightgcn_embedding_dim: 64
        lightgcn_top_k_cuisines: 10       # Top cuisines to present in Round 1

        # Round 1: Cuisine prediction
        round1_predict_top_k: 3           # Number of cuisines to predict
        temperature_round1: 0.3

        # Candidate vendor selection
        max_candidate_vendors: 20         # Max vendors for Round 2

        # Swing user-user similarity + collaborative filtering
        top_similar_users: 5              # Top similar users for CF info
        max_records_per_similar_user: 5   # Max records shown per similar user

        # Round 2: Vendor ranking
        temperature_round2: 0.2

        # LLM settings
        enable_thinking: true

        # Evaluation settings
        n_samples: 20                     # Test samples (-1 = all)
        deterministic_sampling: true

        # Async parallel processing
        enable_async: true
        max_workers: 25
        checkpoint_interval: 50
        retry_attempts: 3
```

---

## 5. Files Summary

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `agentic_recommender/data/repeat_filter.py` | Dataset filtering + test sample building |
| CREATE | `agentic_recommender/data/geohash_index.py` | Geohash-cuisine-vendors index |
| CREATE | `agentic_recommender/evaluation/repeat_evaluator.py` | Two-round async evaluator + metrics |
| CREATE | `tests/test_repeat_filter.py` | Tests for repeat filter |
| CREATE | `tests/test_geohash_index.py` | Tests for geohash index |
| CREATE | `tests/test_repeat_evaluator.py` | Tests for evaluator (mock provider) |
| MODIFY | `agentic_recommender/similarity/methods.py` | Add caching to SwingMethod |
| MODIFY | `agentic_recommender/workflow/workflow_runner.py` | Add stage 9 method + registration |
| MODIFY | `agentic_recommender/workflow/workflow_config_qwen32_linux.yaml` | Add stage 9 config block |

No changes to `agentic_recommender/similarity/lightGCN.py`.

---

## 6. Implementation Order & Test Strategy

Each task produces working, tested code before proceeding to the next.
Tests are written as Python files and run via `python -m pytest` without manual approval.

### Task 1: RepeatDatasetFilter + Tests
**Create**: `agentic_recommender/data/repeat_filter.py`, `tests/test_repeat_filter.py`

Test cases:
- Filter with synthetic DataFrame: verify user count, order count after filtering
- Verify all retained test orders have vendors in corresponding user's training set
- Verify users below min_history_items are excluded
- Verify stats dict has correct keys and values
- Verify cache save/load round-trip

```bash
python -m pytest tests/test_repeat_filter.py -v
```

### Task 2: GeohashVendorIndex + Tests
**Create**: `agentic_recommender/data/geohash_index.py`, `tests/test_geohash_index.py`

Test cases:
- Build index from synthetic data, verify structure
- Lookup by geohash + cuisine list returns correct vendors
- Verify max_candidates limit is respected
- Verify vendors are returned as `vendor_id||cuisine` format
- Verify get_stats() returns correct counts
- Verify cache save/load round-trip

```bash
python -m pytest tests/test_geohash_index.py -v
```

### Task 3: SwingMethod caching + Tests
**Modify**: `agentic_recommender/similarity/methods.py`

Test cases (add to `tests/test_repeat_filter.py` or new file):
- Train SwingMethod, save to cache, load from cache, verify identical results
- Verify get_top_similar_users returns sorted users
- Verify unknown user returns empty list

```bash
python -m pytest tests/test_repeat_filter.py -v -k swing
```

### Task 4: Test sample builder + Tests
**Add to**: `agentic_recommender/data/repeat_filter.py`

Test cases:
- Build samples from filtered data, verify structure
- Verify ground_truth vendor exists in order_history vendors
- Verify target_vendor_geohash is populated
- Verify deterministic sampling produces same order
- Verify n_samples limit works

```bash
python -m pytest tests/test_repeat_filter.py -v -k "test_sample"
```

### Task 5: AsyncRepeatEvaluator + Tests
**Create**: `agentic_recommender/evaluation/repeat_evaluator.py`, `tests/test_repeat_evaluator.py`

Test cases:
- Round 1 prompt contains user history in correct format
- Round 1 prompt contains LightGCN top cuisines
- Round 1 response parsing extracts cuisines correctly
- Round 2 prompt contains candidate vendors and similar user info
- Round 2 response parsing extracts vendor ranking
- End-to-end with mock provider: verify result dict structure
- Metrics computation: verify Hit@K, NDCG@K, MRR calculations

```bash
python -m pytest tests/test_repeat_evaluator.py -v
```

### Task 6: Workflow Integration + Config
**Modify**: `workflow_runner.py`, `workflow_config_qwen32_linux.yaml`

Test: Run `--stages run_repeat_evaluation` with mock provider, verify output files created.

```bash
python -m agentic_recommender.workflow.workflow_runner \
  --config workflow_config_qwen32_linux.yaml \
  --stages run_repeat_evaluation
```

---

## 7. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Geohash field | `vendor_geohash` from test order | User confirmed. Simulates "browsing restaurants in an area." |
| LightGCN reuse | Existing LightGCNEmbeddingManager as-is | Already supports different `prediction_target` and `method` cache keys |
| Swing items | `vendor_id\|\|cuisine` for user-user | Captures fine-grained vendor co-purchase patterns |
| Separate stage | Stage 9, independent from Stage 8 | No changes to existing pipeline |
| GT in candidates | Do NOT inject if missing | Honest metrics for pipeline recall |
| Prompts | Inline in evaluator | Matches existing `async_evaluator.py` pattern |
| Test strategy | Test-driven, pytest files, no manual approval | Tests run automatically during development |

---

## 8. Logging Requirements

The stage should print structured stats at each precomputation step:

```
================================================================
  STAGE 9: REPEATED DATASET EVALUATION
================================================================

STEP 1/5: FILTERING REPEAT DATASET
  Original training data: 617,234 rows, 18,234 users
  After min_history=5: 15,678 users (86.0%)
  Test data: 12,543 orders
  After repeat vendor filter: 8,234 test orders (65.6% repeat rate)
  Avg orders/user: 8.3 | Min: 5 | Max: 142
  Avg unique vendors/user: 4.7

STEP 2/5: BUILDING GEOHASH-CUISINE-VENDOR INDEX
  Unique vendor geohashes: 142
  Unique vendors: 3,456
  Unique cuisines: 39
  Geohash-cuisine pairs: 1,234
  Avg vendors per geohash: 24.3
  Avg vendors per (geohash, cuisine) pair: 2.8
  Max vendors in single geohash: 89

STEP 3/5: TRAINING LIGHTGCN (customer → cuisine)
  Users: 15,678 | Cuisines: 39
  Unique interactions: 62,345
  Training on cpu...
  Epoch 0: Loss = 0.6931
  Epoch 49: Loss = 0.2345
  Training complete!

STEP 4/5: TRAINING SWING (user-user, items=vendor||cuisine)
  Users: 15,678 | Items: 3,456
  Total interactions: 95,432

STEP 5/5: RUNNING EVALUATION
  Test samples: 20
  Concurrent workers: 25
  ...

================================================================
  RESULTS
================================================================
  Hit@1:  0.350  |  NDCG@1:  0.350
  Hit@3:  0.550  |  NDCG@3:  0.472
  Hit@5:  0.650  |  NDCG@5:  0.498
  Hit@10: 0.750  |  NDCG@10: 0.523
  MRR:    0.425
  GT in candidates rate: 0.850
  Avg candidates: 14.3
================================================================
```

---

## 9. Verification Checklist

- [ ] `python -m pytest tests/test_repeat_filter.py -v` passes
- [ ] `python -m pytest tests/test_geohash_index.py -v` passes
- [ ] `python -m pytest tests/test_repeat_evaluator.py -v` passes
- [ ] `python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --list` shows Stage 9
- [ ] `python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --stages run_repeat_evaluation` completes
- [ ] Output files created: `stage9_repeat_results.json`, `stage9_repeat_detailed.json`, `stage9_repeat_samples.json`
- [ ] Results JSON contains Hit@1/3/5/10, NDCG@1/3/5/10, MRR
- [ ] Detailed JSON contains LLM prompts, responses, rankings for each sample
- [ ] Cache files created in `~/.cache/agentic_recommender/`
- [ ] Re-run with cache hits (verify "loaded from cache" messages)
