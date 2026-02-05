# Implementation Plan for System Improvements

**Date:** 2026-01-26
**Based on:** Code Review Report 2026-01-26
**Priority Order:** Critical → High → Medium → Low

---

## Overview

This document provides detailed implementation steps for each issue identified in the code review. Issues are grouped by priority with estimated effort and specific code changes.

---

## Phase 1: Critical Security Fix (Immediate - Day 1)

### Issue 1.1: Remove Exposed API Keys

**Effort:** 2-3 hours
**Risk:** High if not addressed immediately

#### Step 1: Rotate the Exposed Key

1. Log into OpenRouter dashboard: https://openrouter.ai/keys
2. Revoke the exposed key: `******`
3. Generate a new API key
4. Store securely (password manager, not in code)

#### Step 2: Create Environment Variable Support

**File:** `agentic_recommender/models/llm_provider.py`

Add at the top of the file:
```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()
```

**File:** New file `.env.template`
```bash
# OpenRouter API Configuration
OPENROUTER_API_KEY=your-api-key-here

# Optional: Gemini Direct API
GEMINI_API_KEY=your-gemini-key-here

# Optional: Override default model
# OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

**File:** `.gitignore` (add these lines)
```gitignore
# Environment files with secrets
.env
.env.local
.env.*.local

# Config files that may contain secrets
**/config.local.yaml
**/workflow_config_local*.yaml
```

#### Step 3: Update Config File

**File:** `agentic_recommender/workflow/workflow_config_linux.yaml`

Replace lines 176-178:
```yaml
openrouter:
  model_name: "google/gemini-2.0-flash-001"
  # API key loaded from OPENROUTER_API_KEY environment variable
  # api_key: null  # DO NOT hardcode keys here
```

#### Step 4: Remove Hardcoded Key from topk.py

**File:** `agentic_recommender/evaluation/topk.py`

Replace lines 635-638:
```python
if __name__ == "__main__":
    import os
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    run_topk_evaluation(api_key=api_key, n_samples=20)
```

#### Step 5: Update Provider Factory

**File:** `agentic_recommender/models/llm_provider.py`

In `create_llm_provider()` function, ensure environment variable fallback:
```python
def create_llm_provider(provider_type: Optional[str] = None, **kwargs) -> LLMProvider:
    # ... existing code ...

    if target_type == "openrouter":
        api_key = (
            kwargs.get('api_key') or
            config["openrouter"].get("api_key") or
            os.environ.get('OPENROUTER_API_KEY')
        )
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        # ... rest of function
```

#### Verification Steps

```bash
# 1. Ensure no keys in repo
git grep -n "sk-or-v1" -- '*.py' '*.yaml' '*.yml'

# 2. Test with environment variable
export OPENROUTER_API_KEY="your-new-key"
python -c "from agentic_recommender.models.llm_provider import create_llm_provider; p = create_llm_provider('openrouter'); print(p.get_model_info())"

# 3. Test error when key missing
unset OPENROUTER_API_KEY
python -c "from agentic_recommender.models.llm_provider import create_llm_provider; p = create_llm_provider('openrouter')"
# Should raise ValueError
```

---

## Phase 2: High Priority Fixes (Days 2-3)

### Issue 2.1: Fix LightGCN Negative Sampling

**Effort:** 3-4 hours
**Files:** `agentic_recommender/similarity/lightGCN.py`

#### Current Problem

```python
# Line 151-152, 372 - May sample positive items as negatives
batch_neg = np.random.randint(0, num_items, size=len(batch_idx))
```

#### Implementation

**Step 1:** Add helper function after line 28:

```python
def sample_verified_negatives(
    batch_users: np.ndarray,
    num_items: int,
    user_positive_items: Dict[int, Set[int]],
    max_attempts: int = 10
) -> np.ndarray:
    """
    Sample negative items ensuring they are true negatives.

    Args:
        batch_users: Array of user indices
        num_items: Total number of items
        user_positive_items: Dict mapping user_idx to set of positive item indices
        max_attempts: Max sampling attempts before falling back

    Returns:
        Array of negative item indices
    """
    batch_size = len(batch_users)
    negatives = np.random.randint(0, num_items, size=batch_size)

    for i, user_idx in enumerate(batch_users):
        positives = user_positive_items.get(user_idx, set())
        attempts = 0

        while negatives[i] in positives and attempts < max_attempts:
            negatives[i] = np.random.randint(0, num_items)
            attempts += 1

        # If still positive after max attempts, it's likely a very active user
        # In this case, random sampling is acceptable (low probability of collision)

    return negatives
```

**Step 2:** Build positive items index in `_train()` method (around line 340):

```python
def _train(self, interactions: List[Tuple[str, str]], verbose: bool = True):
    # ... existing ID mapping code ...

    # Build positive items per user for negative sampling
    user_positive_items: Dict[int, Set[int]] = {}
    for uid, cuisine in interactions:
        user_idx = self.user_to_idx[uid]
        cuisine_idx = self.cuisine_to_idx[cuisine]
        if user_idx not in user_positive_items:
            user_positive_items[user_idx] = set()
        user_positive_items[user_idx].add(cuisine_idx)

    # ... continue with existing training code ...
```

**Step 3:** Update training loop (around line 372):

```python
# Replace:
batch_neg = np.random.randint(0, num_cuisines, size=len(batch_idx))

# With:
batch_neg = sample_verified_negatives(
    batch_users,
    num_cuisines,
    user_positive_items
)
```

**Step 4:** Also update the standalone `train()` function (line 151):

```python
# Build positive items index
user_positive_items = {}
for row in train_data:
    user_idx, item_idx = row[0], row[1]
    if user_idx not in user_positive_items:
        user_positive_items[user_idx] = set()
    user_positive_items[user_idx].add(item_idx)

# In training loop, replace line 152:
batch_neg = sample_verified_negatives(
    batch_users,
    num_items,
    user_positive_items
)
```

#### Verification

```python
# Test that negatives don't overlap with positives
def test_negative_sampling():
    user_positives = {0: {1, 2, 3}, 1: {4, 5}}
    batch_users = np.array([0, 0, 1, 1])
    negatives = sample_verified_negatives(batch_users, 10, user_positives)

    for i, user in enumerate(batch_users):
        assert negatives[i] not in user_positives[user], \
            f"Negative {negatives[i]} is in positives for user {user}"
    print("Negative sampling test passed!")

test_negative_sampling()
```

---

### Issue 2.2: Fix Error Handling in LLM Provider

**Effort:** 2-3 hours
**Files:** `agentic_recommender/models/llm_provider.py`

#### Implementation

**Step 1:** Create error classes at the top of the file:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMResponse:
    """Wrapper for LLM responses that can indicate success or failure."""
    text: str
    success: bool = True
    error_message: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return not self.success

    @classmethod
    def error(cls, message: str) -> 'LLMResponse':
        return cls(text="", success=False, error_message=message)

    @classmethod
    def ok(cls, text: str) -> 'LLMResponse':
        return cls(text=text, success=True)


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMAPIError(LLMProviderError):
    """Raised when API call fails."""
    pass


class LLMConfigError(LLMProviderError):
    """Raised when configuration is invalid."""
    pass
```

**Step 2:** Update OpenRouterProvider.generate() method:

```python
def generate(
    self,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    system_prompt: str = None,
    json_mode: bool = False,
    raise_on_error: bool = False,  # New parameter
    **kwargs
) -> str:
    """
    Generate using OpenRouter API.

    Args:
        ...
        raise_on_error: If True, raise exception on API errors instead of returning error string
    """
    # ... existing code up to the try block ...

    try:
        response = self._requests.post(...)
        response.raise_for_status()
        # ... success handling ...
        return text.strip()

    except self._requests.exceptions.Timeout as e:
        error_msg = f"OpenRouter API timeout after 60s: {str(e)}"
        if raise_on_error:
            raise LLMAPIError(error_msg) from e
        return f"ERROR: {error_msg}"

    except self._requests.exceptions.HTTPError as e:
        error_msg = f"OpenRouter API HTTP error: {str(e)}"
        if raise_on_error:
            raise LLMAPIError(error_msg) from e
        return f"ERROR: {error_msg}"

    except Exception as e:
        error_msg = f"OpenRouter API error: {str(e)}"
        if raise_on_error:
            raise LLMAPIError(error_msg) from e
        return f"ERROR: {error_msg}"
```

**Step 3:** Add helper method for safe generation:

```python
def generate_safe(self, prompt: str, **kwargs) -> LLMResponse:
    """
    Generate with structured response that indicates success/failure.

    Returns:
        LLMResponse object with success flag
    """
    try:
        text = self.generate(prompt, raise_on_error=True, **kwargs)
        return LLMResponse.ok(text)
    except LLMProviderError as e:
        return LLMResponse.error(str(e))
```

**Step 4:** Update calling code to check for errors:

```python
# In evaluation code, add error checking:
response = provider.generate(prompt)

# Check for error string (backward compatible)
if response.startswith("ERROR:"):
    logger.warning(f"LLM error: {response}")
    continue  # Skip this sample

# Or use the new safe method:
result = provider.generate_safe(prompt)
if result.is_error:
    logger.warning(f"LLM error: {result.error_message}")
    continue
```

---

## Phase 3: Medium Priority Fixes (Days 4-7)

### Issue 3.1: Fix JSON Parsing for Nested Objects

**Effort:** 2 hours
**Files:** `agentic_recommender/evaluation/rerank_eval.py`

#### Implementation

Add helper function around line 1280:

```python
import json
import re
from typing import Optional, Dict, Any

def extract_json_object(text: str, required_key: str = None) -> Optional[Dict[str, Any]]:
    """
    Extract the first valid JSON object from text, optionally requiring a specific key.

    Handles:
    - JSON embedded in markdown code blocks
    - JSON with surrounding text
    - Nested JSON objects

    Args:
        text: Text potentially containing JSON
        required_key: If specified, JSON must contain this key

    Returns:
        Parsed JSON dict or None if not found
    """
    # Strip markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Try to find JSON objects
    brace_count = 0
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                # Found complete JSON object
                candidate = text[start_idx:i+1]
                try:
                    parsed = json.loads(candidate)
                    if required_key is None or required_key in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
                start_idx = None

    return None


def extract_ranking_from_response(response: str, candidates: List[str]) -> Dict[str, Any]:
    """
    Extract ranking from LLM response with multiple fallback strategies.

    Args:
        response: Raw LLM response
        candidates: Valid candidate items for validation

    Returns:
        Dict with 'ranking' list and optional 'reasoning' string
    """
    # Strategy 1: Try structured JSON extraction
    parsed = extract_json_object(response, required_key='ranking')
    if parsed:
        ranking = parsed.get('ranking', [])
        if ranking:
            return {
                'ranking': validate_ranking(ranking, candidates),
                'reasoning': parsed.get('reasoning', '')
            }

    # Strategy 2: Try final_ranking key (for round 2)
    parsed = extract_json_object(response, required_key='final_ranking')
    if parsed:
        ranking = parsed.get('final_ranking', [])
        if ranking:
            return {
                'ranking': validate_ranking(ranking, candidates),
                'reflection': parsed.get('reflection', '')
            }

    # Strategy 3: Extract cuisines from text
    ranking = extract_cuisines_from_text(response, candidates)
    return {'ranking': ranking, 'reasoning': ''}


def validate_ranking(ranking: List[str], candidates: List[str]) -> List[str]:
    """Validate ranking against candidates and fill missing items."""
    candidate_lower = {c.lower(): c for c in candidates}
    valid = []
    seen = set()

    for item in ranking:
        if isinstance(item, str):
            key = item.lower().strip()
            if key in candidate_lower and key not in seen:
                valid.append(candidate_lower[key])
                seen.add(key)

    # Add missing candidates
    for c in candidates:
        if c.lower() not in seen:
            valid.append(c)

    return valid
```

Update `_parse_round1_response()` and `_parse_round2_response()`:

```python
def _parse_round1_response(self, response: str, candidates: List[str]) -> Dict[str, Any]:
    """Parse Round 1 LLM response."""
    return extract_ranking_from_response(response, candidates)

def _parse_round2_response(self, response: str, candidates: List[str]) -> Dict[str, Any]:
    """Parse Round 2 LLM response."""
    result = extract_ranking_from_response(response, candidates)
    # Rename reasoning to reflection for round 2
    if 'reasoning' in result:
        result['reflection'] = result.pop('reasoning')
    return result
```

---

### Issue 3.2: Fix Unused Variable in LightGCN

**Effort:** 5 minutes
**File:** `agentic_recommender/similarity/lightGCN.py`

#### Implementation

Replace line 350:
```python
# Before:
device = self.config.embedding_dim  # Bug: assigns int to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# After:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

### Issue 3.3: Remove Hardcoded Paths

**Effort:** 30 minutes
**Files:** `agentic_recommender/data/enriched_loader.py`

#### Implementation

Replace lines 246-259:
```python
def load_singapore_data(
    data_dir: str = None
) -> EnrichedDataLoader:
    """
    Convenience function to load Singapore dataset.

    Args:
        data_dir: Path to data directory. If None, uses DATA_DIR environment variable
                  or falls back to a default path.

    Returns:
        EnrichedDataLoader instance
    """
    import os

    if data_dir is None:
        data_dir = os.environ.get('AGENTIC_DATA_DIR')

    if data_dir is None:
        # Platform-specific defaults
        import platform
        if platform.system() == 'Darwin':  # macOS
            data_dir = os.path.expanduser("~/Downloads/data_se")
        else:  # Linux
            data_dir = os.path.expanduser("~/data/agentic_recommender/data_se")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Set AGENTIC_DATA_DIR environment variable or pass data_dir parameter."
        )

    config = DataConfig(data_dir=Path(data_dir))
    return EnrichedDataLoader(config)
```

---

### Issue 3.4: Log Model Name Changes

**Effort:** 15 minutes
**File:** `agentic_recommender/models/llm_provider.py`

#### Implementation

Update lines 171-175:
```python
if self.use_openrouter:
    # ... existing code ...

    # Convert model name for OpenRouter if needed
    original_model = self.model_name
    if self.model_name.startswith("gemini"):
        if "2.0" in self.model_name or "flash" in self.model_name:
            self.model_name = "google/gemini-flash-1.5"
        else:
            self.model_name = "google/gemini-pro-1.5"

        if original_model != self.model_name:
            self._log_event(
                f"[GeminiProvider] Model name translated for OpenRouter: "
                f"{original_model} -> {self.model_name}",
                level=logging.WARNING
            )
```

---

## Phase 4: Low Priority & Code Quality (Days 8-14)

### Issue 4.1: Replace Bare Except Clauses

**Effort:** 1 hour
**Files:** `agentic_recommender/evaluation/rerank_eval.py`

#### Implementation

Find all `except:` and replace with specific exceptions:

```python
# Line ~1299 - JSON parsing
try:
    json_match = re.search(r'\{[^{}]*"ranking"[^{}]*\}', response, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        # ...
except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
    logger.debug(f"JSON parsing failed: {e}")
    # Fall through to fallback
```

```python
# Line ~1324 - Round 2 parsing
try:
    # ...
except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
    logger.debug(f"Round 2 JSON parsing failed: {e}")
```

---

### Issue 4.2: Add Type Hints

**Effort:** 4-6 hours (can be done incrementally)
**Files:** Multiple

#### Priority Order for Type Hints

1. **Public API functions** (most important)
2. **Data classes and configs**
3. **Internal helpers**

#### Example - Adding Type Hints to rerank_eval.py

```python
from typing import List, Dict, Tuple, Set, Optional, Any, Union

def build_test_samples(
    orders_df: pd.DataFrame,
    n_samples: int = 10,
    min_history: int = 5,
    seed: int = 42,
    prediction_target: str = "cuisine",
    return_basket: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build test samples for evaluation (Method 1: Leave-Last-Out).

    Args:
        orders_df: DataFrame with order data
        n_samples: Number of samples to create (-1 for all)
        min_history: Minimum orders per user
        seed: Random seed
        prediction_target: "cuisine", "vendor", or "product"
        return_basket: Include multi-item ground truth

    Returns:
        List of test sample dictionaries containing:
        - customer_id: str
        - order_history: List[Dict[str, Any]]
        - ground_truth_cuisine: str
        - target_hour: int
        - target_day_of_week: int
        - ground_truth_items: Set[str] (if return_basket=True)
    """
    # ... implementation
```

---

## Phase 5: Performance Improvements (Optional, Week 2+)

### Issue 5.1: Implement Approximate Nearest Neighbor for Similarity

**Effort:** 1-2 days
**Files:** New file `agentic_recommender/similarity/ann.py`

#### Implementation Sketch

```python
"""
Approximate Nearest Neighbor similarity using FAISS.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import faiss

class ANNSimilarity:
    """
    Fast approximate nearest neighbor similarity search using FAISS.

    Usage:
        ann = ANNSimilarity(embedding_dim=64)
        ann.build_index(embeddings, ids)
        similar = ann.get_similar("user123", k=10)
    """

    def __init__(self, embedding_dim: int = 64, use_gpu: bool = False):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}

    def build_index(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        index_type: str = "IVFFlat"
    ):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: (N, D) array of embeddings
            ids: List of N string identifiers
            index_type: "Flat" (exact), "IVFFlat" (approximate), "HNSW" (graph-based)
        """
        n_samples = len(embeddings)

        # Build ID mappings
        self.id_to_idx = {id_: i for i, id_ in enumerate(ids)}
        self.idx_to_id = {i: id_ for id_, i in self.id_to_idx.items()}

        # Create index
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "IVFFlat":
            n_clusters = min(int(np.sqrt(n_samples)), 100)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            self.index.train(embeddings.astype(np.float32))
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 neighbors

        # Add embeddings
        self.index.add(embeddings.astype(np.float32))

        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )

    def get_similar(
        self,
        query_id: str,
        k: int = 10,
        exclude: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar items.

        Args:
            query_id: ID of query item
            k: Number of neighbors
            exclude: IDs to exclude from results

        Returns:
            List of (id, similarity) tuples
        """
        exclude = exclude or set()
        exclude.add(query_id)

        query_idx = self.id_to_idx.get(query_id)
        if query_idx is None:
            return []

        # Get extra results to account for exclusions
        fetch_k = k + len(exclude) + 10

        query_vec = self.index.reconstruct(query_idx).reshape(1, -1)
        distances, indices = self.index.search(query_vec, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            item_id = self.idx_to_id[idx]
            if item_id not in exclude:
                results.append((item_id, float(dist)))
                if len(results) >= k:
                    break

        return results
```

---

## Testing Plan

### Unit Tests to Add

```python
# tests/test_security.py
def test_no_api_keys_in_source():
    """Ensure no API keys are committed to source."""
    import subprocess
    result = subprocess.run(
        ['git', 'grep', '-l', 'sk-or-v1'],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0, f"API keys found in: {result.stdout}"

# tests/test_negative_sampling.py
def test_negative_sampling_no_collision():
    """Ensure negative samples don't include positive items."""
    from agentic_recommender.similarity.lightGCN import sample_verified_negatives
    import numpy as np

    user_positives = {0: {0, 1, 2}, 1: {3, 4, 5}}
    batch_users = np.array([0, 0, 0, 1, 1])

    for _ in range(100):  # Run multiple times
        negatives = sample_verified_negatives(batch_users, 10, user_positives)
        for i, user in enumerate(batch_users):
            assert negatives[i] not in user_positives[user]

# tests/test_json_parsing.py
def test_extract_nested_json():
    """Test JSON extraction with nested objects."""
    from agentic_recommender.evaluation.rerank_eval import extract_json_object

    text = 'Here is the result: {"ranking": ["a", "b"], "meta": {"score": 0.9}}'
    result = extract_json_object(text, 'ranking')
    assert result == {"ranking": ["a", "b"], "meta": {"score": 0.9}}

def test_extract_json_from_markdown():
    """Test JSON extraction from markdown code blocks."""
    text = '''```json
    {"ranking": ["chinese", "indian"]}
    ```'''
    result = extract_json_object(text, 'ranking')
    assert result['ranking'] == ["chinese", "indian"]
```

---

## Rollout Checklist

### Phase 1 (Critical)
- [ ] Rotate API key on OpenRouter
- [ ] Create `.env.template` file
- [ ] Update `.gitignore`
- [ ] Remove hardcoded keys from all files
- [ ] Test with environment variables
- [ ] Commit and push changes

### Phase 2 (High)
- [ ] Implement verified negative sampling
- [ ] Add tests for negative sampling
- [ ] Update error handling in LLM providers
- [ ] Test error handling behavior

### Phase 3 (Medium)
- [ ] Improve JSON parsing robustness
- [ ] Fix unused variable
- [ ] Remove hardcoded paths
- [ ] Add model name change logging
- [ ] Run full test suite

### Phase 4 (Low)
- [ ] Replace bare except clauses
- [ ] Add type hints to public APIs
- [ ] Update documentation

### Phase 5 (Optional)
- [ ] Implement ANN for similarity (if scaling needed)
- [ ] Add parallel processing for evaluation

---

## Estimated Total Effort

| Phase | Priority | Estimated Hours |
|-------|----------|-----------------|
| 1 | Critical | 2-3 hours |
| 2 | High | 5-7 hours |
| 3 | Medium | 4-5 hours |
| 4 | Low | 5-7 hours |
| 5 | Optional | 8-16 hours |

**Total (Phases 1-4):** 16-22 hours over ~2 weeks

---

*End of Implementation Plan*
