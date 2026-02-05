# AgenticRecommender System Code Review Report

**Date:** 2026-01-26
**Reviewer:** Claude (Opus 4.5)
**Scope:** Full system review covering data preparation, LLM integration, model integration (Swing, LightGCN), and metric calculation correctness
**Entry Point:** `agentic_recommender/workflow/workflow_runner.py`

---

## Executive Summary

The AgenticRecommender system implements a multi-stage recommendation pipeline combining traditional collaborative filtering methods (Swing, LightGCN) with LLM-based reranking for food delivery cuisine prediction. The overall architecture is sound and follows good software engineering practices, but there are several issues that need attention, including one critical security issue.

**Key Findings:**
- 1 Critical security issue (exposed API keys)
- 2 High priority bugs
- 4 Medium priority issues
- 2 Low priority code quality issues

---

## Table of Contents

1. [Critical Issues](#1-critical-issues)
2. [Data Preparation Review](#2-data-preparation-review)
3. [LLM Integration Review](#3-llm-integration-review)
4. [Model Integration Review](#4-model-integration-review)
5. [Metric Calculation Review](#5-metric-calculation-review)
6. [Architectural Issues](#6-architectural-issues)
7. [Performance Considerations](#7-performance-considerations)
8. [Summary of Issues](#8-summary-of-issues-by-severity)
9. [Recommendations](#9-recommendations)

---

## 1. Critical Issues

### 1.1 Exposed API Keys - SECURITY VULNERABILITY

**Severity:** CRITICAL
**Files affected:**
- `agentic_recommender/workflow/workflow_config_linux.yaml:178`
- `agentic_recommender/evaluation/topk.py:637-638`

**Evidence:**

```yaml
# workflow_config_linux.yaml:178
openrouter:
  model_name: "google/gemini-2.0-flash-001"
  api_key: "******"
```

```python
# topk.py:637-638
if __name__ == "__main__":
    API_KEY = "******"
    run_topk_evaluation(api_key=API_KEY, n_samples=20)
```

**Impact:** API keys committed to source control can be scraped by bots, leading to unauthorized usage and potential financial liability.

**Recommendation:**
1. Immediately rotate the exposed API key
2. Remove hardcoded keys from source files
3. Use environment variables: `os.environ.get('OPENROUTER_API_KEY')`
4. Add `*.yaml` with secrets to `.gitignore` or use a template file
5. Consider using a secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault)

---

## 2. Data Preparation Review

### 2.1 EnrichedDataLoader (`data/enriched_loader.py`)

**Status:** ✅ Well Implemented

**Strengths:**
- Clean dataclass configuration (`DataConfig`) for maintainability
- Proper lazy loading with internal caching (`_orders`, `_merged`, etc.)
- Consistent handling of missing values with `fillna()`
- Preserves all important fields for downstream tasks:
  - `customer_id` - User identity for CF
  - `order_id` - Order grouping for co-purchase patterns
  - `geohash` - User and vendor locations
  - `chain_id` - Chain restaurant grouping
  - Temporal features (`hour`, `day_of_week`, `day_num`)

**Code Quality:**
```python
# Good pattern: Lazy loading with caching (lines 55-69)
def load_orders(self) -> pd.DataFrame:
    if self._orders is None:
        path = self.config.data_dir / self.config.orders_file
        self._orders = pd.read_csv(path)
        # ... processing
    return self._orders
```

**Minor Issue:**
- `load_singapore_data()` at line 247 has a hardcoded Mac path that won't work on Linux:
```python
def load_singapore_data(
    data_dir: str = "/Users/zhenkai/Downloads/data_se"  # Mac-specific path
) -> EnrichedDataLoader:
```

### 2.2 Data Leakage Prevention

**Status:** ✅ Correctly Implemented

**Location:** `similarity/lightGCN.py:546-600` (`filter_interactions_leave_last_out`)

The leave-last-out logic correctly excludes the test order from training data:

```python
def filter_interactions_leave_last_out(orders_df, prediction_target="cuisine"):
    for customer_id, group in orders_df.groupby('customer_id'):
        # Sort by time
        if 'day_num' in group.columns:
            sorted_group = group.sort_values(['day_num', 'hour'])

        unique_orders = sorted_group.drop_duplicates('order_id')

        # Need at least 2 orders to have N-1
        if len(unique_orders) < 2:
            continue

        # Correctly excludes last order
        excluded_last = unique_orders.iloc[:-1]
        remaining_order_ids = set(excluded_last['order_id'].unique())
        training_rows = sorted_group[sorted_group['order_id'].isin(remaining_order_ids)]
```

**Verification:** The `iloc[:-1]` correctly selects all rows except the last, ensuring the ground truth order is never included in training.

### 2.3 Method 1 vs Method 2 Cache Isolation

**Status:** ✅ Correctly Separated

Cache keys properly distinguish between evaluation methods:

```python
# lightGCN.py:259-263
def _get_cache_path(self, dataset_name: str, method: str = "full") -> Path:
    self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if method and method != "full":
        return self.CACHE_DIR / f"{dataset_name}_{method}_lightgcn.pkl"
    return self.CACHE_DIR / f"{dataset_name}_embeddings.pkl"
```

**Cache Structure:**
```
~/.cache/agentic_recommender/
├── lightgcn/
│   ├── data_se_method1_lightgcn.pkl  # Leave-last-out
│   └── data_se_method2_lightgcn.pkl  # Train-test split
└── swing/
    ├── data_se_method1_swing.pkl
    └── data_se_method2_swing.pkl
```

---

## 3. LLM Integration Review

### 3.1 Provider Architecture (`models/llm_provider.py`)

**Status:** ✅ Well Structured

**Strengths:**
- Clean abstraction with `LLMProvider` ABC
- Multiple provider support (Gemini SDK, OpenRouter, Mock)
- Good error handling with logging
- Performance tracking (calls, tokens, time)
- Structured logging integration via `get_logger()`

**Class Hierarchy:**
```
LLMProvider (ABC)
├── GeminiProvider (supports both direct Gemini and OpenRouter)
├── OpenRouterProvider (dedicated OpenRouter client)
└── MockLLMProvider (for testing)
```

### 3.2 Issues Found

**Issue 1: Error Swallowing (lines 571-572)**

```python
except Exception as e:
    return f"ERROR: {str(e)}"  # Returns error string instead of raising
```

**Problem:** Downstream code may process error strings as valid LLM responses, leading to subtle bugs.

**Recommendation:** Either raise the exception or return a sentinel object:
```python
from dataclasses import dataclass

@dataclass
class LLMError:
    message: str

# Then check: if isinstance(response, LLMError): handle_error()
```

**Issue 2: Bare `except` Clauses (lines 1299, 1324)**

```python
except:
    pass  # Silently ignores JSON parsing errors
```

**Recommendation:** Catch specific exceptions:
```python
except (json.JSONDecodeError, KeyError, TypeError) as e:
    logger.warning(f"JSON parsing failed: {e}")
```

**Issue 3: Model Name Overwriting (lines 171-175)**

```python
if self.model_name.startswith("gemini"):
    if "2.0" in self.model_name or "flash" in self.model_name:
        self.model_name = "google/gemini-flash-1.5"  # Silently changes model
```

**Problem:** User specifies one model but gets another without warning.

**Recommendation:** Log a warning or raise an error:
```python
original_model = self.model_name
self.model_name = "google/gemini-flash-1.5"
logger.warning(f"Model name translated: {original_model} -> {self.model_name}")
```

---

## 4. Model Integration Review

### 4.1 Swing Algorithm (`similarity/methods.py`)

**Status:** ✅ Correctly Implemented

**Formula Verification (lines 77-84):**

```python
# SwingMethod.compute_similarity()
for item in common:
    item_pop = len(self.item_users.get(item, set()))
    weight = 1.0 / (
        ((len(items1) + cfg.alpha1) ** cfg.beta) *
        ((len(items2) + cfg.alpha1) ** cfg.beta) *
        (item_pop + cfg.alpha2)
    )
    similarity += weight
```

**Mathematical Verification:**

This correctly implements Alibaba's Swing formula:

$$sim(u_1, u_2) = \sum_{i \in I(u_1) \cap I(u_2)} \frac{1}{(|I(u_1)|+\alpha_1)^\beta \times (|I(u_2)|+\alpha_1)^\beta \times (|U(i)|+\alpha_2)}$$

Where:
- $I(u)$ = items purchased by user $u$
- $U(i)$ = users who purchased item $i$
- $\alpha_1, \alpha_2$ = smoothing parameters
- $\beta$ = power weight

**Default Parameters:**
- `alpha1 = 5.0` (user activity smoothing)
- `alpha2 = 1.0` (item popularity smoothing)
- `beta = 0.3` (power weight)

### 4.2 CuisineSwingMethod (`similarity/methods.py:277-533`)

**Status:** ✅ Correctly Adapted

The cuisine-to-cuisine variant correctly adapts the formula:

```python
# CuisineSwingMethod.compute_similarity() (lines 385-421)
def compute_similarity(self, cuisine1: str, cuisine2: str) -> float:
    users1 = self.cuisine_users.get(cuisine1, set())
    users2 = self.cuisine_users.get(cuisine2, set())
    common_users = users1 & users2

    for user in common_users:
        user_activity = len(self.user_cuisines.get(user, set()))
        weight = 1.0 / (
            ((len(users1) + cfg.alpha1) ** cfg.beta) *
            ((len(users2) + cfg.alpha1) ** cfg.beta) *
            (user_activity + cfg.alpha2)
        )
        similarity += weight
```

### 4.3 LightGCN (`similarity/lightGCN.py`)

**Status:** ⚠️ Issues Found

**Issue 1: Unused Variable Assignment (line 350)**

```python
device = self.config.embedding_dim  # Bug: assigns int to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Immediately overwritten
```

**Impact:** Low - the correct value is assigned on the next line, but this is confusing.

**Issue 2: Negative Sampling Not Verified (lines 151-152, 372)**

```python
# During training
batch_neg = np.random.randint(0, num_items, size=len(batch_idx))
```

**Problem:** Random negative sampling may accidentally sample items the user has interacted with (false negatives), which can degrade model quality.

**Recommendation:**
```python
def sample_negative(user_idx, num_items, positive_items_per_user):
    """Sample a true negative item for a user."""
    user_positives = positive_items_per_user.get(user_idx, set())
    neg = np.random.randint(0, num_items)
    while neg in user_positives:
        neg = np.random.randint(0, num_items)
    return neg
```

**Issue 3: Graph Normalization - Correct**

```python
# build_sparse_graph() lines 106-111
rowsum = np.array(adj_mat.sum(1))
d_inv_sqrt = np.power(rowsum, -0.5).flatten()
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # Handle zero-degree nodes
d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
```

The symmetric normalization $D^{-1/2} A D^{-1/2}$ is the correct formulation for LightGCN.

**LightGCN Forward Pass - Correct:**

```python
def forward(self):
    all_embs = [self.embedding.weight]
    emb = self.embedding.weight

    for _ in range(self.n_layers):
        emb = torch.sparse.mm(self.graph, emb)  # Message passing
        all_embs.append(emb)

    # Layer combination via mean (the "Light" in LightGCN)
    final_embs = torch.stack(all_embs, dim=1)
    final_embs = torch.mean(final_embs, dim=1)

    return final_embs
```

---

## 5. Metric Calculation Review

### 5.1 NDCG Calculation

**Single-Item NDCG (`topk.py:385-389`):**

```python
ndcg = sum(
    1.0 / math.log2(r.rank + 1) if r.rank > 0 else 0
    for r in valid_results
) / n_valid
```

**Analysis:** This computes DCG, but for single-item binary relevance where IDCG = 1.0 (when the item is ranked first), this is numerically equivalent to NDCG. The naming is technically correct but could be clearer.

**Enhanced Rerank NDCG (`rerank_eval.py:1404-1414`):**

```python
def dcg(rank, k):
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / (math.log2(rank + 1))

def ndcg(ranks, k):
    dcg_scores = [dcg(r, k) for r in ranks if r > 0]
    ideal_dcg = 1.0  # Correct for binary single-item relevance
    return sum(dcg_scores) / (len(ranks) * ideal_dcg) if ranks else 0.0
```

**Status:** ✅ Correct for single-item evaluation

### 5.2 Basket NDCG (`basket_metrics.py:151-190`)

**Status:** ✅ Correctly Implemented

```python
def compute_basket_ndcg(predictions: List[str], ground_truth: Set[str], k: int) -> float:
    # Compute DCG
    dcg = 0.0
    for i, pred in enumerate(predictions[:k]):
        if pred in ground_truth:
            # Binary relevance: 1 if in ground truth, 0 otherwise
            # Position is 1-indexed, so use i+2 (since i starts at 0)
            dcg += 1.0 / math.log2(i + 2)

    # Compute IDCG (ideal DCG)
    # Best case: all ground truth items ranked first
    n_relevant = min(len(ground_truth), k)
    idcg = 0.0
    for i in range(n_relevant):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg
```

**Verification:**
- DCG formula: $\sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$ ✅
- IDCG computed assuming perfect ranking ✅
- Division by zero handled ✅

### 5.3 MRR Calculation

**Status:** ✅ Correct

**Single-Item MRR (`topk.py:380-383`):**

```python
mrr = sum(
    1.0 / r.rank if r.rank > 0 else 0
    for r in valid_results
) / n_valid
```

**Basket MRR (`basket_metrics.py:193-213`):**

```python
def compute_basket_mrr(predictions: List[str], ground_truth: Set[str]) -> float:
    for i, pred in enumerate(predictions):
        if pred in ground_truth:
            return 1.0 / (i + 1)  # 1-indexed rank
    return 0.0
```

### 5.4 Hit@K

**Status:** ✅ Correct

```python
# topk.py:376-377
def hit_at_k(k: int) -> float:
    return sum(1 for r in valid_results if 0 < r.rank <= k) / n_valid
```

### 5.5 Basket Recall and Precision

**Status:** ✅ Correct

```python
# basket_metrics.py:107-127
def compute_basket_recall(predictions: List[str], ground_truth: Set[str], k: int) -> float:
    top_k = set(predictions[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)  # |R∩G| / |G|

def compute_basket_precision(predictions: List[str], ground_truth: Set[str], k: int) -> float:
    top_k = set(predictions[:k])
    hits = len(top_k & ground_truth)
    return hits / k  # |R∩G| / K
```

**Mathematical Verification:**
- Recall@K = $\frac{|R_k \cap G|}{|G|}$ ✅
- Precision@K = $\frac{|R_k \cap G|}{K}$ ✅

---

## 6. Architectural Issues

### 6.1 Ground Truth Injection in Candidate Generation

**Location:** `rerank_eval.py:211-243`

```python
# 4. IMPORTANT: Add ground truth if not present
if ground_truth:
    if ground_truth in candidates:
        debug_info['ground_truth_rank'] = candidates.index(ground_truth) + 1
    else:
        # Add ground truth to a random position (not first, not last)
        insert_pos = random.randint(1, max(1, len(candidates) - 1))
        candidates.insert(insert_pos, ground_truth)
        debug_info['ground_truth_added'] = True
```

**Analysis:**

This is a documented design decision for evaluation purposes. The code separates two concerns:
1. **Candidate generation quality**: Tracked by `gt_in_candidates` metric
2. **LLM reranking quality**: Measured by Hit@K, NDCG, etc.

The `ground_truth_added` flag allows filtering analysis if needed.

**Recommendation:** Ensure this behavior is clearly documented in evaluation reports.

### 6.2 JSON Parsing Robustness

**Location:** `rerank_eval.py:1287-1304`

```python
try:
    json_match = re.search(r'\{[^{}]*"ranking"[^{}]*\}', response, re.DOTALL)
```

**Problem:** This regex fails on nested JSON objects because `[^{}]*` cannot match nested braces.

**Example Failure Case:**
```json
{"ranking": ["chinese", "indian"], "metadata": {"source": "llm"}}
```

**Recommendation:**
```python
def extract_json_object(text: str, required_key: str) -> Optional[dict]:
    """Extract first valid JSON object containing required_key."""
    import json

    # Find potential JSON starts
    for i, char in enumerate(text):
        if char == '{':
            # Try progressively longer substrings
            for j in range(len(text), i, -1):
                try:
                    obj = json.loads(text[i:j])
                    if required_key in obj:
                        return obj
                except json.JSONDecodeError:
                    continue
    return None
```

### 6.3 Position Bias Mitigation

**Status:** ✅ Good Practice

```python
# rerank_eval.py:1217-1219
shuffled = candidates.copy()
random.shuffle(shuffled)
candidates_str = ", ".join(shuffled)
```

Shuffling candidates before presenting to the LLM reduces position bias.

---

## 7. Performance Considerations

### 7.1 Caching Strategy

**Status:** ✅ Well Implemented

**Multi-Level Caching:**

1. **Stage-level caching** (`workflow/stage_cache.py`):
   - MD5 hash of inputs for cache validation
   - Automatic invalidation on input changes

2. **Model-level caching**:
   - LightGCN embeddings cached to disk
   - Swing similarity cached in memory with symmetric key optimization:
     ```python
     cache_key = (min(user1, user2), max(user1, user2))  # Symmetric
     ```

3. **Method-specific isolation**:
   - Separate caches for `method1` and `method2` evaluations

### 7.2 Potential Memory Issues

**Location:** `similarity/methods.py:63-91` (`get_similar()`)

```python
def get_similar(self, entity_id: str, exclude: Set[str] = None) -> List[Tuple[str, float]]:
    for other_id in self._get_candidate_entities(entity_id):
        sim = self.compute_similarity(entity_id, other_id)
```

**Problem:** Computes similarities against ALL entities, which is O(N) per query.

**Impact:** For large datasets (>100K users/items), this becomes a bottleneck.

**Recommendation:** Consider approximate nearest neighbor search:
```python
# Using FAISS
import faiss

index = faiss.IndexFlatIP(embedding_dim)  # Inner product
index.add(embeddings)
distances, indices = index.search(query_embedding, k=10)
```

### 7.3 Batch Processing

The LLM evaluation processes samples sequentially. Consider parallel processing:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(evaluate_sample, test_samples))
```

---

## 8. Summary of Issues by Severity

### Critical (Fix Immediately)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | Exposed API keys | `workflow_config_linux.yaml:178`, `topk.py:637` | Security breach, financial liability |

### High Priority

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 2 | LightGCN negative sampling may include positives | `lightGCN.py:151-152, 372` | Model quality degradation |
| 3 | Error handling returns strings instead of raising | `llm_provider.py:571-572` | Silent failures, bugs |

### Medium Priority

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 4 | JSON parsing regex fails on nested objects | `rerank_eval.py:1287` | Parsing failures |
| 5 | Unused variable in LightGCN | `lightGCN.py:350` | Code clarity |
| 6 | Hardcoded Mac paths | `enriched_loader.py:247` | Cross-platform issues |
| 7 | Model name silently changed | `llm_provider.py:171-175` | Confusion |

### Low Priority

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 8 | Bare except clauses | `rerank_eval.py:1299, 1324` | Debugging difficulty |
| 9 | Missing type hints in some modules | Various | IDE support, documentation |

---

## 9. Recommendations

### Immediate Actions (This Week)

1. **Rotate exposed API keys** and remove from source control
2. **Add `.env` file support** for secrets management
3. **Fix LightGCN negative sampling** with verified negatives

### Short-Term (This Month)

4. **Improve error handling** - use custom exception classes or result objects
5. **Fix JSON parsing** with proper nested object handling
6. **Add integration tests** for metric calculations with known test cases
7. **Remove hardcoded paths** - use environment variables or config

### Long-Term (This Quarter)

8. **Add comprehensive type hints** throughout codebase
9. **Implement approximate nearest neighbor** for scalability
10. **Add batch/parallel processing** for LLM evaluation
11. **Create API documentation** with examples

---

## Appendix A: Files Reviewed

| File | Lines | Status |
|------|-------|--------|
| `workflow/workflow_runner.py` | ~1500 | Reviewed |
| `workflow/workflow_config_linux.yaml` | 185 | Reviewed |
| `similarity/lightGCN.py` | 634 | Reviewed |
| `similarity/methods.py` | 533 | Reviewed |
| `similarity/base.py` | 119 | Reviewed |
| `evaluation/rerank_eval.py` | 1543 | Reviewed |
| `evaluation/topk.py` | 639 | Reviewed |
| `evaluation/basket_metrics.py` | 365 | Reviewed |
| `data/enriched_loader.py` | 260 | Reviewed |
| `models/llm_provider.py` | 810 | Reviewed |

---

## Appendix B: Metric Formulas Reference

| Metric | Formula | Implementation Status |
|--------|---------|----------------------|
| Hit@K | $\frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[rank_i \leq K]$ | ✅ Correct |
| MRR | $\frac{1}{N}\sum_{i=1}^{N} \frac{1}{rank_i}$ | ✅ Correct |
| NDCG@K | $\frac{DCG@K}{IDCG@K}$ | ✅ Correct |
| Recall@K | $\frac{|R_K \cap G|}{|G|}$ | ✅ Correct |
| Precision@K | $\frac{|R_K \cap G|}{K}$ | ✅ Correct |
| Swing | $\sum_{i \in I_1 \cap I_2} \frac{1}{(|I_1|+\alpha)^\beta (|I_2|+\alpha)^\beta (|U_i|+\alpha)}$ | ✅ Correct |

---

*End of Report*
