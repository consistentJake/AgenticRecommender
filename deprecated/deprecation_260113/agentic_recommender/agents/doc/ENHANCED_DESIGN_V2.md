# Enhanced Design V2: Modular Similarity, Top-K Evaluation, and Enriched Representations

## Overview

This document extends the previous design with:
1. **Enhanced Cuisine Representation** - Peaking hours/days patterns
2. **Modular Similarity Calculation** - Pluggable similarity methods
3. **OpenRouter LLM Provider** - Using Gemini 2.5 Flash for testing
4. **Top-K Hit Ratio Evaluation** - Sequential recommendation metrics

---

## 1. Enhanced Cuisine Representation

### 1.1 Cuisine Profile with Temporal Patterns

```python
@dataclass
class CuisineProfile:
    """
    Complete cuisine profile with behavioral patterns.

    Key insight: Peaking patterns help predict when a cuisine
    is most likely to be ordered.
    """

    cuisine_type: str

    # === Temporal Patterns ===

    # Peak hours: {hour: order_frequency}
    # e.g., {"12": 0.25, "13": 0.20, "19": 0.30} for lunch/dinner peaks
    hour_distribution: Dict[int, float]
    peak_hours: List[int]  # Top 3 hours sorted by frequency

    # Peak weekdays: {day: order_frequency}
    # 0=Monday, 6=Sunday
    # e.g., {"5": 0.20, "6": 0.18} for weekend peaks
    weekday_distribution: Dict[int, float]
    peak_weekdays: List[int]  # Top 3 weekdays sorted by frequency

    # Time bucket preferences
    meal_time_distribution: Dict[str, float]  # {"breakfast": 0.1, "lunch": 0.3, "dinner": 0.4, "late_night": 0.2}

    # === Popularity Metrics ===
    total_orders: int
    unique_customers: int
    avg_orders_per_customer: float

    # === Price Characteristics ===
    avg_price: float
    price_std: float
    price_range: Tuple[float, float]

    # === Sequential Patterns ===
    # What cuisines users typically order BEFORE this one
    preceded_by: Dict[str, float]  # {cuisine: transition_probability}
    # What cuisines users typically order AFTER this one
    followed_by: Dict[str, float]  # {cuisine: transition_probability}

    # === Location Patterns ===
    popular_in_areas: List[str]  # Top geohash zones

    # === Co-order Patterns ===
    # Other cuisines ordered in the SAME order (multi-vendor orders)
    co_ordered_with: Dict[str, float]  # {cuisine: co_occurrence_frequency}

    @classmethod
    def from_orders(
        cls,
        cuisine_type: str,
        orders_df: pd.DataFrame,
        vendors_df: pd.DataFrame
    ) -> 'CuisineProfile':
        """
        Build cuisine profile from order data.

        Args:
            cuisine_type: The cuisine to analyze
            orders_df: All orders (with customer_id, order_id, vendor_id, etc.)
            vendors_df: Vendor info (with vendor_id, primary_cuisine)
        """
        # Merge to get cuisine
        merged = orders_df.merge(
            vendors_df[['vendor_id', 'primary_cuisine']],
            on='vendor_id',
            how='left'
        )

        # Filter to this cuisine
        cuisine_orders = merged[merged['primary_cuisine'] == cuisine_type]

        if len(cuisine_orders) == 0:
            return cls._empty_profile(cuisine_type)

        # Extract hour from order_time
        cuisine_orders = cuisine_orders.copy()
        cuisine_orders['hour'] = cuisine_orders['order_time'].str.split(':').str[0].astype(int)

        # Hour distribution
        hour_counts = cuisine_orders['hour'].value_counts(normalize=True)
        hour_distribution = hour_counts.to_dict()
        peak_hours = hour_counts.head(3).index.tolist()

        # Weekday distribution
        weekday_counts = cuisine_orders['day_of_week'].value_counts(normalize=True)
        weekday_distribution = weekday_counts.to_dict()
        peak_weekdays = weekday_counts.head(3).index.tolist()

        # Meal time buckets
        def hour_to_meal(h):
            if 6 <= h < 11:
                return 'breakfast'
            elif 11 <= h < 15:
                return 'lunch'
            elif 15 <= h < 18:
                return 'afternoon'
            elif 18 <= h < 22:
                return 'dinner'
            else:
                return 'late_night'

        cuisine_orders['meal_time'] = cuisine_orders['hour'].apply(hour_to_meal)
        meal_distribution = cuisine_orders['meal_time'].value_counts(normalize=True).to_dict()

        # Popularity metrics
        total_orders = cuisine_orders['order_id'].nunique()
        unique_customers = cuisine_orders['customer_id'].nunique()

        # Build and return profile
        return cls(
            cuisine_type=cuisine_type,
            hour_distribution=hour_distribution,
            peak_hours=peak_hours,
            weekday_distribution=weekday_distribution,
            peak_weekdays=peak_weekdays,
            meal_time_distribution=meal_distribution,
            total_orders=total_orders,
            unique_customers=unique_customers,
            avg_orders_per_customer=total_orders / max(unique_customers, 1),
            avg_price=0.0,  # Computed from products
            price_std=0.0,
            price_range=(0.0, 1.0),
            preceded_by={},  # Computed separately with sequence analysis
            followed_by={},
            popular_in_areas=[],
            co_ordered_with={}
        )
```

### 1.2 Cuisine Profile Registry

```python
class CuisineRegistry:
    """
    Registry of all cuisine profiles for fast lookup.
    Pre-computed once and cached.
    """

    def __init__(self):
        self.profiles: Dict[str, CuisineProfile] = {}
        self._loaded = False

    def build_from_data(
        self,
        orders_df: pd.DataFrame,
        vendors_df: pd.DataFrame
    ):
        """Build profiles for all cuisines."""
        cuisines = vendors_df['primary_cuisine'].dropna().unique()

        for cuisine in cuisines:
            self.profiles[cuisine] = CuisineProfile.from_orders(
                cuisine, orders_df, vendors_df
            )

        self._loaded = True
        print(f"Built profiles for {len(self.profiles)} cuisines")

    def get_profile(self, cuisine: str) -> Optional[CuisineProfile]:
        """Get profile for a cuisine."""
        return self.profiles.get(cuisine)

    def get_peak_cuisines_for_time(
        self,
        hour: int,
        weekday: int,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get cuisines most likely to be ordered at this time.

        Returns: List of (cuisine, probability) tuples
        """
        scores = []
        for cuisine, profile in self.profiles.items():
            hour_score = profile.hour_distribution.get(hour, 0)
            day_score = profile.weekday_distribution.get(weekday, 0)
            combined = hour_score * 0.6 + day_score * 0.4
            scores.append((cuisine, combined))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
```

---

## 2. Modular Similarity Calculation

### 2.1 Similarity Interface (Abstract Base)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class SimilarityConfig:
    """Base configuration for similarity methods."""
    top_k: int = 10
    min_threshold: float = 0.0
    cache_enabled: bool = True


class SimilarityMethod(ABC):
    """
    Abstract base class for similarity calculation methods.

    All similarity methods must implement:
    - fit(): Build index from interaction data
    - compute_similarity(): Calculate similarity between two entities
    - get_similar(): Get top-k similar entities
    """

    def __init__(self, config: SimilarityConfig = None):
        self.config = config or SimilarityConfig()
        self._cache: Dict[Tuple[str, str], float] = {}
        self._fitted = False

    @abstractmethod
    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'SimilarityMethod':
        """
        Build similarity index from interactions.

        Args:
            interactions: List of (entity1_id, entity2_id) tuples
                         For user-item: (user_id, item_id)
                         For item-item: (item_id, co_item_id)

        Returns:
            self (for chaining)
        """
        pass

    @abstractmethod
    def compute_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute similarity between two entities.

        Args:
            entity1: First entity ID
            entity2: Second entity ID

        Returns:
            Similarity score (typically 0.0 to 1.0)
        """
        pass

    def get_similar(
        self,
        entity_id: str,
        exclude: set = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k similar entities.

        Args:
            entity_id: Query entity ID
            exclude: Set of entity IDs to exclude

        Returns:
            List of (entity_id, similarity_score) tuples, sorted desc
        """
        exclude = exclude or set()
        exclude.add(entity_id)

        similarities = []
        for other_id in self._get_candidate_entities(entity_id):
            if other_id in exclude:
                continue

            sim = self.compute_similarity(entity_id, other_id)
            if sim >= self.config.min_threshold:
                similarities.append((other_id, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:self.config.top_k]

    @abstractmethod
    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        """Get candidate entities for similarity computation."""
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of this similarity method."""
        pass

    def clear_cache(self):
        """Clear similarity cache."""
        self._cache.clear()
```

### 2.2 Swing Similarity Implementation

```python
@dataclass
class SwingConfig(SimilarityConfig):
    """Swing-specific configuration."""
    alpha1: float = 5.0   # User activity smoothing
    alpha2: float = 1.0   # Item popularity smoothing
    beta: float = 0.3     # Power weight for user activity


class SwingSimilarity(SimilarityMethod):
    """
    Alibaba's Swing algorithm for user-user similarity.

    Anti-noise property: Penalizes popular items and active users.

    Formula:
    sim(u1, u2) = Σ(i ∈ common_items) 1 / ((|I(u1)|+α1)^β × (|I(u2)|+α1)^β × (|U(i)|+α2))
    """

    def __init__(self, config: SwingConfig = None):
        super().__init__(config or SwingConfig())
        self.user_items: Dict[str, set] = {}
        self.item_users: Dict[str, set] = {}

    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'SwingSimilarity':
        """Build user-item and item-user indices."""
        self.user_items.clear()
        self.item_users.clear()
        self.clear_cache()

        for user_id, item_id in interactions:
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)

            if item_id not in self.item_users:
                self.item_users[item_id] = set()
            self.item_users[item_id].add(user_id)

        self._fitted = True
        return self

    def compute_similarity(self, user1: str, user2: str) -> float:
        """Compute Swing similarity between two users."""
        # Check cache
        cache_key = (min(user1, user2), max(user1, user2))
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        items1 = self.user_items.get(user1, set())
        items2 = self.user_items.get(user2, set())
        common = items1 & items2

        if not common:
            return 0.0

        cfg = self.config
        similarity = 0.0

        for item in common:
            item_pop = len(self.item_users.get(item, set()))
            weight = 1.0 / (
                ((len(items1) + cfg.alpha1) ** cfg.beta) *
                ((len(items2) + cfg.alpha1) ** cfg.beta) *
                (item_pop + cfg.alpha2)
            )
            similarity += weight

        if self.config.cache_enabled:
            self._cache[cache_key] = similarity

        return similarity

    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        return list(self.user_items.keys())

    def get_method_name(self) -> str:
        return "swing"
```

### 2.3 Cosine Similarity Implementation

```python
@dataclass
class CosineConfig(SimilarityConfig):
    """Cosine similarity configuration."""
    normalize: bool = True


class CosineSimilarity(SimilarityMethod):
    """
    Cosine similarity based on user-item vectors.

    sim(u1, u2) = (v1 · v2) / (||v1|| × ||v2||)
    """

    def __init__(self, config: CosineConfig = None):
        super().__init__(config or CosineConfig())
        self.user_vectors: Dict[str, Dict[str, float]] = {}
        self.all_items: set = set()

    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'CosineSimilarity':
        """Build user vectors from interactions."""
        self.user_vectors.clear()
        self.all_items.clear()
        self.clear_cache()

        # Count interactions per user-item pair
        for user_id, item_id in interactions:
            if user_id not in self.user_vectors:
                self.user_vectors[user_id] = {}
            self.user_vectors[user_id][item_id] = \
                self.user_vectors[user_id].get(item_id, 0) + 1
            self.all_items.add(item_id)

        # Normalize if configured
        if self.config.normalize:
            for user_id in self.user_vectors:
                total = sum(self.user_vectors[user_id].values())
                for item_id in self.user_vectors[user_id]:
                    self.user_vectors[user_id][item_id] /= total

        self._fitted = True
        return self

    def compute_similarity(self, user1: str, user2: str) -> float:
        """Compute cosine similarity."""
        cache_key = (min(user1, user2), max(user1, user2))
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        v1 = self.user_vectors.get(user1, {})
        v2 = self.user_vectors.get(user2, {})

        if not v1 or not v2:
            return 0.0

        # Compute dot product
        common_items = set(v1.keys()) & set(v2.keys())
        dot_product = sum(v1[i] * v2[i] for i in common_items)

        # Compute norms
        norm1 = sum(v ** 2 for v in v1.values()) ** 0.5
        norm2 = sum(v ** 2 for v in v2.values()) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        if self.config.cache_enabled:
            self._cache[cache_key] = similarity

        return similarity

    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        return list(self.user_vectors.keys())

    def get_method_name(self) -> str:
        return "cosine"
```

### 2.4 Jaccard Similarity Implementation

```python
class JaccardSimilarity(SimilarityMethod):
    """
    Jaccard similarity based on item set overlap.

    sim(u1, u2) = |I(u1) ∩ I(u2)| / |I(u1) ∪ I(u2)|
    """

    def __init__(self, config: SimilarityConfig = None):
        super().__init__(config or SimilarityConfig())
        self.user_items: Dict[str, set] = {}

    def fit(self, interactions: List[Tuple[str, str]], **kwargs) -> 'JaccardSimilarity':
        """Build user-item sets."""
        self.user_items.clear()
        self.clear_cache()

        for user_id, item_id in interactions:
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)

        self._fitted = True
        return self

    def compute_similarity(self, user1: str, user2: str) -> float:
        """Compute Jaccard similarity."""
        cache_key = (min(user1, user2), max(user1, user2))
        if self.config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        items1 = self.user_items.get(user1, set())
        items2 = self.user_items.get(user2, set())

        if not items1 or not items2:
            return 0.0

        intersection = len(items1 & items2)
        union = len(items1 | items2)

        similarity = intersection / union if union > 0 else 0.0

        if self.config.cache_enabled:
            self._cache[cache_key] = similarity

        return similarity

    def _get_candidate_entities(self, entity_id: str) -> List[str]:
        return list(self.user_items.keys())

    def get_method_name(self) -> str:
        return "jaccard"
```

### 2.5 Similarity Factory

```python
class SimilarityFactory:
    """
    Factory for creating and switching similarity methods.

    Usage:
        factory = SimilarityFactory()

        # Get specific method
        swing = factory.create("swing", SwingConfig(alpha1=5.0))

        # Get default method
        default = factory.get_default()

        # Switch method at runtime
        factory.set_default("cosine")
    """

    METHODS = {
        "swing": (SwingSimilarity, SwingConfig),
        "cosine": (CosineSimilarity, CosineConfig),
        "jaccard": (JaccardSimilarity, SimilarityConfig),
    }

    def __init__(self, default_method: str = "swing"):
        self._default_method = default_method
        self._instances: Dict[str, SimilarityMethod] = {}

    def create(
        self,
        method_name: str,
        config: SimilarityConfig = None
    ) -> SimilarityMethod:
        """Create a similarity method instance."""
        if method_name not in self.METHODS:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.METHODS.keys())}")

        method_class, config_class = self.METHODS[method_name]
        config = config or config_class()

        return method_class(config)

    def get_or_create(
        self,
        method_name: str,
        config: SimilarityConfig = None
    ) -> SimilarityMethod:
        """Get cached instance or create new one."""
        if method_name not in self._instances:
            self._instances[method_name] = self.create(method_name, config)
        return self._instances[method_name]

    def get_default(self) -> SimilarityMethod:
        """Get the default similarity method."""
        return self.get_or_create(self._default_method)

    def set_default(self, method_name: str):
        """Set the default similarity method."""
        if method_name not in self.METHODS:
            raise ValueError(f"Unknown method: {method_name}")
        self._default_method = method_name

    @classmethod
    def available_methods(cls) -> List[str]:
        """List available similarity methods."""
        return list(cls.METHODS.keys())
```

---

## 3. OpenRouter LLM Provider

### 3.1 Configuration

```python
# From /doc/design/APIs.md
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model for testing
DEFAULT_MODEL = "google/gemini-2.0-flash-001"  # Gemini 2.5 Flash
```

### 3.2 OpenRouter Provider Implementation

```python
class OpenRouterProvider(LLMProvider):
    """
    OpenRouter API provider for LLM inference.

    Supports multiple models through OpenRouter's unified API.
    Using Gemini 2.5 Flash for testing (fast and cost-effective).
    """

    DEFAULT_MODEL = "google/gemini-2.0-flash-001"
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name or self.DEFAULT_MODEL

        if not self.api_key:
            raise ValueError("OpenRouter API key required")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AgenticRecommender",
            "X-Title": "Agentic Recommender System",
        }

        # Metrics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str = None,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """Generate using OpenRouter API."""
        import requests
        import time

        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if json_mode:
            messages[-1]["content"] += "\n\nRespond only with valid JSON."

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            text = ""
            if data.get("choices"):
                text = data["choices"][0]["message"]["content"]

            # Track metrics
            duration = time.time() - start_time
            self.total_calls += 1
            self.total_time += duration

            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", len(prompt.split()) + len(text.split()))
            self.total_tokens += tokens

            return text.strip()

        except Exception as e:
            return f"ERROR: {str(e)}"

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, **kwargs) for p in prompts]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "OpenRouter",
            "model_name": self.model_name,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "avg_time_per_call": self.total_time / max(self.total_calls, 1),
        }
```

---

## 4. Top-K Hit Ratio Evaluation

### 4.1 Sequential Recommendation Task Definition

Instead of Yes/No classification, we evaluate:
- **Given**: User's order history (sequence of orders)
- **Task**: Predict the NEXT cuisine/item the user will order
- **Metric**: Is the ground truth in the top-K predictions?

### 4.2 Evaluation Metrics

```python
@dataclass
class TopKMetrics:
    """Metrics for Top-K evaluation."""

    k: int

    # Core metrics
    hit_rate: float        # % of times ground truth is in top-K
    mrr: float             # Mean Reciprocal Rank (1/rank of correct answer)
    ndcg: float            # Normalized Discounted Cumulative Gain

    # Detailed breakdown
    hits_at_1: float       # Hit@1 (exact match)
    hits_at_3: float       # Hit@3
    hits_at_5: float       # Hit@5
    hits_at_10: float      # Hit@10

    # Statistics
    total_samples: int
    avg_prediction_time_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            f"hit@{self.k}": self.hit_rate,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit@1": self.hits_at_1,
            "hit@3": self.hits_at_3,
            "hit@5": self.hits_at_5,
            "hit@10": self.hits_at_10,
            "total_samples": self.total_samples,
            "avg_time_ms": self.avg_prediction_time_ms,
        }
```

### 4.3 Sequential Recommendation Evaluator

```python
class SequentialRecommendationEvaluator:
    """
    Evaluator for sequential recommendation with Top-K Hit Ratio.

    Workflow:
    1. For each test user:
       a. Get user's order history (all but last order)
       b. Ground truth = cuisine of last order
       c. Agent generates top-K cuisine predictions
       d. Check if ground truth is in predictions

    2. Aggregate metrics across all test users
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        similarity_method: SimilarityMethod,
        cuisine_registry: CuisineRegistry,
        k_values: List[int] = [1, 3, 5, 10]
    ):
        self.llm = llm_provider
        self.similarity = similarity_method
        self.cuisine_registry = cuisine_registry
        self.k_values = k_values

    def evaluate(
        self,
        test_samples: List[Dict[str, Any]],
        verbose: bool = False
    ) -> TopKMetrics:
        """
        Evaluate on test samples.

        Args:
            test_samples: List of test cases, each with:
                - customer_id: str
                - order_history: List[Dict] (orders BEFORE target)
                - ground_truth_cuisine: str (cuisine of next order)

        Returns:
            TopKMetrics with evaluation results
        """
        results = []
        total_time = 0.0

        for i, sample in enumerate(test_samples):
            start_time = time.time()

            # Get top-K predictions from agent
            predictions = self._get_predictions(
                customer_id=sample['customer_id'],
                order_history=sample['order_history'],
                k=max(self.k_values)
            )

            elapsed = time.time() - start_time
            total_time += elapsed

            # Record result
            ground_truth = sample['ground_truth_cuisine']
            rank = self._find_rank(predictions, ground_truth)

            results.append({
                'ground_truth': ground_truth,
                'predictions': predictions,
                'rank': rank,  # 0 if not found, 1-indexed if found
                'time_ms': elapsed * 1000
            })

            if verbose and (i + 1) % 10 == 0:
                print(f"Evaluated {i+1}/{len(test_samples)} samples...")

        # Compute metrics
        return self._compute_metrics(results)

    def _get_predictions(
        self,
        customer_id: str,
        order_history: List[Dict],
        k: int
    ) -> List[Tuple[str, float]]:
        """
        Get top-K cuisine predictions for a user.

        Returns: List of (cuisine, confidence_score) tuples
        """
        # Build prompt for LLM
        prompt = self._build_prediction_prompt(customer_id, order_history, k)

        # Get LLM response
        response = self.llm.generate(
            prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.3,
            json_mode=True
        )

        # Parse predictions
        return self._parse_predictions(response, k)

    def _build_prediction_prompt(
        self,
        customer_id: str,
        order_history: List[Dict],
        k: int
    ) -> str:
        """Build prompt for top-K prediction."""

        # Format order history
        history_lines = []
        for i, order in enumerate(order_history[-10:], 1):  # Last 10 orders
            history_lines.append(
                f"{i}. {order['cuisine']} | {order['day_name']} {order['hour']}:00 | ${order['price']:.2f}"
            )
        history_str = "\n".join(history_lines)

        # Get available cuisines
        available_cuisines = list(self.cuisine_registry.profiles.keys())

        return f"""Based on this user's order history, predict the top {k} cuisines they are most likely to order next.

## Order History (most recent last):
{history_str}

## Available Cuisines:
{', '.join(available_cuisines[:30])}... (and more)

## Task:
Predict the TOP {k} cuisines this user is most likely to order next, ranked by probability.

Consider:
1. User's cuisine preferences (frequency patterns)
2. Temporal patterns (current time/day)
3. Price sensitivity
4. Sequential patterns (what typically follows recent orders)

Return JSON format:
{{
    "predictions": [
        {{"cuisine": "cuisine_name", "confidence": 0.95, "reason": "brief reason"}},
        {{"cuisine": "cuisine_name", "confidence": 0.82, "reason": "brief reason"}},
        ...
    ]
}}"""

    def _get_system_prompt(self) -> str:
        return (
            "You are a sequential recommendation agent. "
            "Given a user's order history, predict what cuisine they will order next. "
            "Base predictions on patterns in their history, temporal preferences, and cuisine relationships."
        )

    def _parse_predictions(
        self,
        response: str,
        k: int
    ) -> List[Tuple[str, float]]:
        """Parse LLM response into predictions."""
        try:
            import json
            data = json.loads(response)
            predictions = data.get('predictions', [])

            result = []
            for pred in predictions[:k]:
                cuisine = pred.get('cuisine', '')
                confidence = pred.get('confidence', 0.5)
                result.append((cuisine.lower(), confidence))

            return result
        except:
            return []

    def _find_rank(
        self,
        predictions: List[Tuple[str, float]],
        ground_truth: str
    ) -> int:
        """Find rank of ground truth in predictions (1-indexed, 0 if not found)."""
        ground_truth_lower = ground_truth.lower()
        for i, (cuisine, _) in enumerate(predictions):
            if cuisine == ground_truth_lower:
                return i + 1
        return 0

    def _compute_metrics(self, results: List[Dict]) -> TopKMetrics:
        """Compute evaluation metrics from results."""
        n = len(results)
        if n == 0:
            return TopKMetrics(k=max(self.k_values), hit_rate=0, mrr=0, ndcg=0,
                              hits_at_1=0, hits_at_3=0, hits_at_5=0, hits_at_10=0,
                              total_samples=0, avg_prediction_time_ms=0)

        # Calculate hits at various K
        def hit_at_k(k):
            return sum(1 for r in results if 0 < r['rank'] <= k) / n

        # Calculate MRR
        mrr = sum(1.0 / r['rank'] if r['rank'] > 0 else 0 for r in results) / n

        # Calculate NDCG (simplified)
        import math
        ndcg = sum(
            1.0 / math.log2(r['rank'] + 1) if r['rank'] > 0 else 0
            for r in results
        ) / n

        # Average time
        avg_time = sum(r['time_ms'] for r in results) / n

        return TopKMetrics(
            k=max(self.k_values),
            hit_rate=hit_at_k(max(self.k_values)),
            mrr=mrr,
            ndcg=ndcg,
            hits_at_1=hit_at_k(1),
            hits_at_3=hit_at_k(3),
            hits_at_5=hit_at_k(5),
            hits_at_10=hit_at_k(10),
            total_samples=n,
            avg_prediction_time_ms=avg_time
        )
```

### 4.4 Test Data Preparation for Top-K Evaluation

```python
class TopKTestDataBuilder:
    """
    Build test data for Top-K evaluation from Singapore dataset.

    Strategy:
    1. For each user with 5+ orders
    2. Use first N-1 orders as history
    3. Use Nth order's cuisine as ground truth
    """

    def __init__(
        self,
        orders_df: pd.DataFrame,
        vendors_df: pd.DataFrame,
        min_history: int = 5
    ):
        self.orders = orders_df
        self.vendors = vendors_df
        self.min_history = min_history

    def build_test_samples(
        self,
        n_samples: int = 100,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Build test samples for evaluation.

        Returns:
            List of test samples with:
            - customer_id: str
            - order_history: List[Dict]
            - ground_truth_cuisine: str
        """
        import random
        random.seed(seed)

        # Merge to get cuisines
        merged = self.orders.merge(
            self.vendors[['vendor_id', 'primary_cuisine']],
            on='vendor_id',
            how='left'
        )

        # Group by customer
        customer_orders = merged.groupby('customer_id')

        # Filter customers with enough history
        eligible_customers = [
            cid for cid, group in customer_orders
            if group['order_id'].nunique() >= self.min_history
        ]

        # Sample customers
        sampled = random.sample(eligible_customers, min(n_samples, len(eligible_customers)))

        samples = []
        for customer_id in sampled:
            customer_data = merged[merged['customer_id'] == customer_id]

            # Sort by order_day and time
            customer_data = customer_data.sort_values(['order_day', 'order_time'])

            # Get unique orders
            orders_list = []
            for order_id in customer_data['order_id'].unique():
                order_rows = customer_data[customer_data['order_id'] == order_id]
                first_row = order_rows.iloc[0]

                orders_list.append({
                    'order_id': order_id,
                    'cuisine': first_row['primary_cuisine'],
                    'day_of_week': first_row['day_of_week'],
                    'day_name': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][first_row['day_of_week']],
                    'hour': int(first_row['order_time'].split(':')[0]),
                    'price': order_rows['unit_price'].sum() if 'unit_price' in order_rows else 0,
                })

            if len(orders_list) >= self.min_history:
                samples.append({
                    'customer_id': customer_id,
                    'order_history': orders_list[:-1],  # All but last
                    'ground_truth_cuisine': orders_list[-1]['cuisine'],
                })

        return samples
```

### 4.5 Running Evaluation

```python
def run_topk_evaluation():
    """
    Full evaluation pipeline with Top-K Hit Ratio.

    Usage:
        python -m agentic_recommender.evaluation.run_topk
    """
    import pandas as pd
    from pathlib import Path

    # Load data
    data_dir = Path("/Users/zhenkai/Downloads/data_sg")
    orders = pd.read_csv(data_dir / "orders_sg_train.txt")
    vendors = pd.read_csv(data_dir / "vendors_sg.txt")

    print("=" * 60)
    print("TOP-K HIT RATIO EVALUATION")
    print("=" * 60)

    # Build test samples
    print("\n1. Building test samples...")
    test_builder = TopKTestDataBuilder(orders, vendors, min_history=5)
    test_samples = test_builder.build_test_samples(n_samples=100)
    print(f"   Created {len(test_samples)} test samples")

    # Build cuisine registry
    print("\n2. Building cuisine profiles...")
    cuisine_registry = CuisineRegistry()
    cuisine_registry.build_from_data(orders, vendors)

    # Build similarity index
    print("\n3. Building similarity index...")
    similarity = SimilarityFactory().create("swing")
    interactions = [
        (row.customer_id, row.vendor_id)
        for row in orders.itertuples()
    ]
    similarity.fit(interactions)

    # Initialize LLM
    print("\n4. Initializing LLM (OpenRouter + Gemini 2.5 Flash)...")
    llm = OpenRouterProvider(
        api_key="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",
        model_name="google/gemini-2.0-flash-001"
    )

    # Run evaluation
    print("\n5. Running evaluation...")
    evaluator = SequentialRecommendationEvaluator(
        llm_provider=llm,
        similarity_method=similarity,
        cuisine_registry=cuisine_registry,
        k_values=[1, 3, 5, 10]
    )

    metrics = evaluator.evaluate(test_samples, verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples: {metrics.total_samples}")
    print(f"Avg prediction time: {metrics.avg_prediction_time_ms:.2f}ms")
    print()
    print("Hit Rates:")
    print(f"  Hit@1:  {metrics.hits_at_1:.2%}")
    print(f"  Hit@3:  {metrics.hits_at_3:.2%}")
    print(f"  Hit@5:  {metrics.hits_at_5:.2%}")
    print(f"  Hit@10: {metrics.hits_at_10:.2%}")
    print()
    print(f"MRR:  {metrics.mrr:.4f}")
    print(f"NDCG: {metrics.ndcg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_topk_evaluation()
```

---

## 5. Test Suite for Top-K Evaluation

```python
# tests/test_topk_evaluation.py

"""
Test suite for Top-K Hit Ratio evaluation.
"""

import pytest
from agentic_recommender.evaluation.topk import (
    TopKMetrics,
    SequentialRecommendationEvaluator,
    TopKTestDataBuilder
)


class TestTopKMetrics:
    """Test metrics computation."""

    def test_hit_at_1_with_exact_match(self):
        """If prediction[0] == ground_truth, hit@1 = 1.0"""
        results = [{'rank': 1, 'time_ms': 100}]
        # Simplified metric computation
        hit_at_1 = sum(1 for r in results if r['rank'] == 1) / len(results)
        assert hit_at_1 == 1.0

    def test_hit_at_5_with_rank_3(self):
        """If ground truth is at rank 3, hit@5 = 1.0, hit@1 = 0.0"""
        results = [{'rank': 3, 'time_ms': 100}]
        hit_at_1 = sum(1 for r in results if 0 < r['rank'] <= 1) / len(results)
        hit_at_5 = sum(1 for r in results if 0 < r['rank'] <= 5) / len(results)
        assert hit_at_1 == 0.0
        assert hit_at_5 == 1.0

    def test_mrr_calculation(self):
        """MRR should be mean of 1/rank."""
        results = [
            {'rank': 1, 'time_ms': 100},  # 1/1 = 1.0
            {'rank': 2, 'time_ms': 100},  # 1/2 = 0.5
            {'rank': 5, 'time_ms': 100},  # 1/5 = 0.2
        ]
        expected_mrr = (1.0 + 0.5 + 0.2) / 3
        mrr = sum(1.0 / r['rank'] for r in results) / len(results)
        assert abs(mrr - expected_mrr) < 0.01


class TestTestDataBuilder:
    """Test data preparation."""

    def test_builds_correct_sample_structure(self, sample_orders_df, all_vendors):
        """Test samples have required fields."""
        builder = TopKTestDataBuilder(sample_orders_df, all_vendors, min_history=3)
        samples = builder.build_test_samples(n_samples=5)

        for sample in samples:
            assert 'customer_id' in sample
            assert 'order_history' in sample
            assert 'ground_truth_cuisine' in sample
            assert isinstance(sample['order_history'], list)

    def test_history_excludes_ground_truth(self, sample_orders_df, all_vendors):
        """Order history should not include the ground truth order."""
        builder = TopKTestDataBuilder(sample_orders_df, all_vendors, min_history=3)
        samples = builder.build_test_samples(n_samples=5)

        for sample in samples:
            history_order_ids = [o['order_id'] for o in sample['order_history']]
            # Ground truth order should not be in history
            # (We use last order as ground truth)
```

---

## 6. Implementation Summary

| Component | Priority | Status |
|-----------|----------|--------|
| Cuisine profiles with peaking patterns | High | Designed |
| Modular similarity (Swing/Cosine/Jaccard) | High | Designed |
| OpenRouter LLM provider | High | Designed |
| Top-K Hit Ratio evaluator | High | Designed |
| Test suite | High | Designed |

### Next Steps

1. **Implement data loader** with enriched representations
2. **Implement similarity module** with factory pattern
3. **Implement OpenRouter provider**
4. **Implement Top-K evaluator**
5. **Run evaluation** on Singapore dataset

Ready to start implementation?
