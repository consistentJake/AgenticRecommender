# Data Enrichment & Test-First Design

## 1. Information Loss Analysis

### 1.1 Original Data Schema

```
orders_sg_train.txt (3.4M rows):
├── customer_id      # User identity (e.g., f374c8c54c)
├── geohash          # User location (e.g., w21zt)
├── order_id         # Order grouping (1 order = multiple products)
├── vendor_id        # Restaurant ID
├── product_id       # Product ID
├── day_of_week      # 0-6
├── order_time       # HH:MM:SS
└── order_day        # "61 days" (relative timestamp)

vendors_sg.txt (7,411 rows):
├── vendor_id        # Restaurant ID
├── chain_id         # Chain restaurant grouping (can be empty)
├── geohash          # Restaurant location
└── primary_cuisine  # Cuisine type (chinese, pizza, etc.)

products_sg.txt (1M rows):
├── vendor_id        # Restaurant ID
├── product_id       # Product ID
├── name             # Product name
└── unit_price       # Normalized price
```

### 1.2 Information Currently LOST

| Field | Impact | Use Case |
|-------|--------|----------|
| `customer_id` | **CRITICAL** | User identity for CF similarity |
| `order_id` | **CRITICAL** | Co-purchase patterns (basket analysis) |
| `user geohash` | HIGH | Location-based preferences |
| `vendor_id` | HIGH | Restaurant loyalty patterns |
| `vendor geohash` | MEDIUM | Proximity-based recommendations |
| `chain_id` | MEDIUM | Brand preference patterns |
| `product_id` | LOW | Individual product tracking |

### 1.3 Order Composition Statistics

```
Items per order:
- 1 item:  799,896 orders (42%)
- 2 items: 501,887 orders (26%)
- 3 items: 222,583 orders (12%)
- 4+ items: 380,000+ orders (20%)
```

**Insight**: 58% of orders contain multiple items - losing order_id means losing co-purchase patterns!

---

## 2. Enriched Data Representations

### 2.1 User Representation

```python
@dataclass
class EnrichedUser:
    """Complete user representation for recommendation."""

    # Identity
    customer_id: str                    # Original customer ID
    primary_geohash: str                # Most frequent order location

    # Cuisine Preferences (CF signal)
    cuisine_distribution: Dict[str, float]  # {cuisine: frequency}
    cuisine_sequence: List[str]             # Recent cuisines in order

    # Temporal Patterns
    day_distribution: Dict[int, float]      # {0-6: frequency}
    hour_distribution: Dict[int, float]     # {0-23: frequency}
    preferred_meal_times: List[str]         # ["lunch", "dinner"]

    # Price Behavior
    avg_order_value: float
    price_range: Tuple[float, float]        # (min, max)
    price_sensitivity: float                # Variance indicator

    # Vendor Loyalty
    top_vendors: List[Tuple[str, int]]      # [(vendor_id, order_count)]
    vendor_diversity: float                 # 0-1 (0=loyal, 1=explorer)
    chain_preferences: Dict[str, float]     # {chain_id: frequency}

    # Co-purchase Patterns
    typical_basket_size: float
    frequently_paired_cuisines: List[Tuple[str, str, float]]  # (c1, c2, co-occurrence)

    # Location Patterns
    location_distribution: Dict[str, float]  # {geohash: frequency}
    max_distance_traveled: float             # Geohash distance metric

    # Activity Metrics
    total_orders: int
    order_frequency: float                   # Orders per day
    recency: int                             # Days since last order

    @classmethod
    def from_order_history(cls, customer_id: str, orders: pd.DataFrame) -> 'EnrichedUser':
        """Build enriched user from order history."""
        # Implementation in test section below
        pass
```

### 2.2 Item (Cuisine/Vendor) Representation

```python
@dataclass
class EnrichedVendor:
    """Complete vendor representation."""

    # Identity
    vendor_id: str
    chain_id: Optional[str]
    geohash: str
    primary_cuisine: str

    # Popularity Metrics
    total_orders: int
    unique_customers: int
    repeat_customer_rate: float

    # Product Portfolio
    num_products: int
    price_range: Tuple[float, float]
    avg_price: float

    # Customer Segments
    customer_geohash_distribution: Dict[str, float]  # Where customers come from

    # Temporal Patterns
    peak_hours: List[int]
    peak_days: List[int]


@dataclass
class EnrichedCuisine:
    """Cuisine-level representation for broader patterns."""

    cuisine_type: str

    # Cross-cuisine patterns
    frequently_followed_by: Dict[str, float]   # What cuisines users order next
    frequently_preceded_by: Dict[str, float]   # What cuisines users ordered before

    # Customer demographics
    typical_customer_locations: List[str]
    typical_order_times: Dict[int, float]

    # Price characteristics
    avg_price: float
    price_variance: float

    # Co-purchase within orders
    common_companions: Dict[str, float]  # Other cuisines in same order
```

### 2.3 Order Representation (Preserving Grouping)

```python
@dataclass
class EnrichedOrder:
    """Single order with all items grouped."""

    order_id: str
    customer_id: str
    vendor_id: str

    # Temporal
    day_of_week: int
    hour: int
    order_day: int  # Relative day number

    # Location
    customer_geohash: str
    vendor_geohash: str

    # Items in this order
    items: List[OrderItem]

    # Computed
    total_value: float
    num_items: int
    cuisine: str


@dataclass
class OrderItem:
    """Single item within an order."""
    product_id: str
    product_name: str
    unit_price: float
```

---

## 3. Test-First Validation Framework

### 3.1 Module Testing Strategy

Each module will have:
1. **Unit Tests**: Isolated function tests with mock data
2. **Integration Tests**: Tests against real Singapore dataset samples
3. **Validation Tests**: Compare outputs against expected patterns

### 3.2 Test Data Setup

```python
# tests/fixtures/sample_data.py

"""
Test fixtures using REAL data samples from Singapore dataset.
Each fixture is a validated snapshot that tests can rely on.
"""

import pytest
from pathlib import Path

ORIGINAL_DATA_DIR = Path("/Users/zhenkai/Downloads/data_sg")

@pytest.fixture
def sample_orders_df():
    """Load first 1000 orders for fast testing."""
    import pandas as pd
    orders = pd.read_csv(ORIGINAL_DATA_DIR / "orders_sg_train.txt", nrows=1000)
    return orders

@pytest.fixture
def sample_customer_history():
    """Load complete history for a specific test customer."""
    import pandas as pd
    orders = pd.read_csv(ORIGINAL_DATA_DIR / "orders_sg_train.txt")
    # Pick customer with moderate history (10-50 orders)
    customer_counts = orders['customer_id'].value_counts()
    test_customer = customer_counts[(customer_counts >= 10) & (customer_counts <= 50)].index[0]
    return orders[orders['customer_id'] == test_customer]

@pytest.fixture
def all_vendors():
    """Load all vendors."""
    import pandas as pd
    return pd.read_csv(ORIGINAL_DATA_DIR / "vendors_sg.txt")

@pytest.fixture
def all_products():
    """Load all products."""
    import pandas as pd
    return pd.read_csv(ORIGINAL_DATA_DIR / "products_sg.txt")
```

### 3.3 Stage 1: Data Loader Tests

```python
# tests/test_data_loader.py

"""
Test Stage 1: Data loading and basic parsing.
Run with: pytest tests/test_data_loader.py -v
"""

import pytest
from agentic_recommender.data.enriched_loader import (
    load_orders,
    load_vendors,
    load_products,
    merge_order_data
)


class TestDataLoading:
    """Test basic data loading from original files."""

    def test_load_orders_returns_expected_columns(self, sample_orders_df):
        """Verify all expected columns are present."""
        expected_cols = [
            'customer_id', 'geohash', 'order_id', 'vendor_id',
            'product_id', 'day_of_week', 'order_time', 'order_day'
        ]
        for col in expected_cols:
            assert col in sample_orders_df.columns, f"Missing column: {col}"

    def test_customer_id_not_null(self, sample_orders_df):
        """Customer ID should never be null."""
        assert sample_orders_df['customer_id'].notna().all()

    def test_order_id_groups_products(self, sample_orders_df):
        """Same order_id should have same customer_id and vendor_id."""
        grouped = sample_orders_df.groupby('order_id').agg({
            'customer_id': 'nunique',
            'vendor_id': 'nunique'
        })
        assert (grouped['customer_id'] == 1).all(), "Order has multiple customers"
        assert (grouped['vendor_id'] == 1).all(), "Order has multiple vendors"

    def test_geohash_format(self, sample_orders_df):
        """Geohash should be 4-5 character string starting with 'w'."""
        geohashes = sample_orders_df['geohash'].dropna()
        assert all(g.startswith('w') for g in geohashes)
        assert all(4 <= len(g) <= 6 for g in geohashes)


class TestMergedData:
    """Test merged order + vendor + product data."""

    def test_merge_preserves_all_orders(self, sample_orders_df, all_vendors, all_products):
        """Merge should not lose orders."""
        merged = merge_order_data(sample_orders_df, all_vendors, all_products)
        # Left join should preserve all orders
        assert len(merged) == len(sample_orders_df)

    def test_merge_adds_cuisine(self, sample_orders_df, all_vendors, all_products):
        """Merged data should include cuisine from vendors."""
        merged = merge_order_data(sample_orders_df, all_vendors, all_products)
        assert 'cuisine' in merged.columns or 'primary_cuisine' in merged.columns

    def test_merge_adds_product_name(self, sample_orders_df, all_vendors, all_products):
        """Merged data should include product name."""
        merged = merge_order_data(sample_orders_df, all_vendors, all_products)
        assert 'product_name' in merged.columns or 'name' in merged.columns
```

### 3.4 Stage 2: User Representation Tests

```python
# tests/test_user_representation.py

"""
Test Stage 2: User vector representation.
Run with: pytest tests/test_user_representation.py -v
"""

import pytest
import numpy as np
from agentic_recommender.data.representations import EnrichedUser


class TestEnrichedUser:
    """Test user representation building."""

    def test_cuisine_distribution_sums_to_one(self, sample_customer_history):
        """Cuisine distribution should be normalized."""
        user = EnrichedUser.from_order_history("test", sample_customer_history)
        total = sum(user.cuisine_distribution.values())
        assert abs(total - 1.0) < 0.01, f"Distribution sums to {total}, not 1.0"

    def test_day_distribution_valid_keys(self, sample_customer_history):
        """Day distribution keys should be 0-6."""
        user = EnrichedUser.from_order_history("test", sample_customer_history)
        for day in user.day_distribution.keys():
            assert 0 <= day <= 6, f"Invalid day: {day}"

    def test_hour_distribution_valid_keys(self, sample_customer_history):
        """Hour distribution keys should be 0-23."""
        user = EnrichedUser.from_order_history("test", sample_customer_history)
        for hour in user.hour_distribution.keys():
            assert 0 <= hour <= 23, f"Invalid hour: {hour}"

    def test_basket_size_matches_data(self, sample_customer_history):
        """Average basket size should match computed from orders."""
        user = EnrichedUser.from_order_history("test", sample_customer_history)

        # Compute expected basket size
        items_per_order = sample_customer_history.groupby('order_id').size()
        expected = items_per_order.mean()

        assert abs(user.typical_basket_size - expected) < 0.1

    def test_total_orders_count(self, sample_customer_history):
        """Total orders should match unique order_ids."""
        user = EnrichedUser.from_order_history("test", sample_customer_history)
        expected = sample_customer_history['order_id'].nunique()
        assert user.total_orders == expected


class TestUserSimilarity:
    """Test user-to-user similarity computation."""

    def test_same_user_similarity_is_one(self, sample_customer_history):
        """User should have similarity 1.0 with themselves."""
        user = EnrichedUser.from_order_history("test", sample_customer_history)
        similarity = user.compute_similarity(user)
        assert abs(similarity - 1.0) < 0.01

    def test_similarity_is_symmetric(self, sample_orders_df):
        """similarity(A, B) should equal similarity(B, A)."""
        customers = sample_orders_df['customer_id'].unique()[:2]

        user1_data = sample_orders_df[sample_orders_df['customer_id'] == customers[0]]
        user2_data = sample_orders_df[sample_orders_df['customer_id'] == customers[1]]

        user1 = EnrichedUser.from_order_history(customers[0], user1_data)
        user2 = EnrichedUser.from_order_history(customers[1], user2_data)

        sim_12 = user1.compute_similarity(user2)
        sim_21 = user2.compute_similarity(user1)

        assert abs(sim_12 - sim_21) < 0.001
```

### 3.5 Stage 3: CF Score Tests

```python
# tests/test_cf_score.py

"""
Test Stage 3: Collaborative filtering score computation.
Run with: pytest tests/test_cf_score.py -v
"""

import pytest
from agentic_recommender.similarity.enhanced_cf import (
    compute_cf_score,
    SwingSimilarity,
    build_interaction_index
)


class TestInteractionIndex:
    """Test building interaction index for Swing."""

    def test_index_has_all_users(self, sample_orders_df):
        """Index should contain all users from orders."""
        interactions = build_interaction_index(sample_orders_df, granularity='cuisine')
        user_ids = set(i[0] for i in interactions)
        expected_users = set(sample_orders_df['customer_id'].unique())
        assert user_ids == expected_users

    def test_cuisine_granularity_reduces_items(self, sample_orders_df, all_vendors):
        """Cuisine granularity should have fewer unique items than product."""
        cuisine_interactions = build_interaction_index(
            sample_orders_df, granularity='cuisine', vendors=all_vendors
        )
        product_interactions = build_interaction_index(
            sample_orders_df, granularity='product'
        )

        cuisine_items = len(set(i[1] for i in cuisine_interactions))
        product_items = len(set(i[1] for i in product_interactions))

        assert cuisine_items < product_items


class TestSwingSimilarity:
    """Test Swing similarity computation."""

    def test_swing_fit_creates_index(self, sample_orders_df):
        """Fitting should create user and item indices."""
        swing = SwingSimilarity()
        interactions = [(row.customer_id, row.vendor_id)
                       for row in sample_orders_df.itertuples()]
        swing.fit(interactions)

        assert len(swing.user_items) > 0
        assert len(swing.item_users) > 0

    def test_swing_self_similarity_high(self, sample_orders_df):
        """User should have highest similarity with themselves (trivial)."""
        swing = SwingSimilarity()
        interactions = [(row.customer_id, row.vendor_id)
                       for row in sample_orders_df.itertuples()]
        swing.fit(interactions)

        # Pick a user
        test_user = sample_orders_df['customer_id'].iloc[0]
        similar = swing.get_similar_users(test_user)

        # All similar users should have lower similarity than 1.0
        for user, score in similar:
            assert score <= 1.0


class TestCFScore:
    """Test CF score computation for user-item pairs."""

    def test_cf_score_in_range(self, sample_customer_history, all_vendors):
        """CF score should be between 0 and 1."""
        from agentic_recommender.data.representations import EnrichedUser

        user = EnrichedUser.from_order_history("test", sample_customer_history)

        # Pick a random cuisine
        test_cuisine = all_vendors['primary_cuisine'].dropna().iloc[0]

        score, context = compute_cf_score(user, test_cuisine)

        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_cf_score_higher_for_preferred_cuisine(self, sample_customer_history, all_vendors):
        """CF score should be higher for user's preferred cuisines."""
        from agentic_recommender.data.representations import EnrichedUser

        user = EnrichedUser.from_order_history("test", sample_customer_history)

        # Find user's top cuisine
        top_cuisine = max(user.cuisine_distribution, key=user.cuisine_distribution.get)

        # Find a cuisine user never ordered
        ordered_cuisines = set(user.cuisine_distribution.keys())
        all_cuisines = set(all_vendors['primary_cuisine'].dropna().unique())
        never_ordered = all_cuisines - ordered_cuisines

        if never_ordered:
            random_never = list(never_ordered)[0]

            score_top, _ = compute_cf_score(user, top_cuisine)
            score_never, _ = compute_cf_score(user, random_never)

            assert score_top > score_never, \
                f"Top cuisine score {score_top} should be > never ordered {score_never}"
```

### 3.6 Stage 4: LLM Provider Tests

```python
# tests/test_llm_providers.py

"""
Test Stage 4: LLM providers.
Run with: pytest tests/test_llm_providers.py -v

Note: Some tests require API keys and will be skipped if not available.
"""

import pytest
import os
from agentic_recommender.models.llm_provider import (
    create_llm_provider,
    ClaudeProvider,
    MockLLMProvider
)


class TestMockProvider:
    """Test mock provider (no API needed)."""

    def test_mock_returns_response(self):
        """Mock provider should return a response."""
        provider = MockLLMProvider()
        response = provider.generate("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_mock_tracks_calls(self):
        """Mock provider should track call count."""
        provider = MockLLMProvider()
        provider.generate("Test 1")
        provider.generate("Test 2")
        assert provider.call_count == 2


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
class TestClaudeProvider:
    """Test Claude provider (requires API key)."""

    def test_claude_simple_response(self):
        """Claude should return a valid response."""
        provider = ClaudeProvider()
        response = provider.generate(
            "Say 'hello' and nothing else.",
            max_tokens=10
        )
        assert "hello" in response.lower()

    def test_claude_json_mode(self):
        """Claude should return valid JSON in json_mode."""
        import json
        provider = ClaudeProvider()
        response = provider.generate(
            "Return a JSON object with key 'test' and value 'success'",
            json_mode=True,
            max_tokens=50
        )
        parsed = json.loads(response)
        assert 'test' in parsed


class TestProviderFactory:
    """Test provider factory function."""

    def test_create_mock_provider(self):
        """Factory should create mock provider."""
        provider = create_llm_provider(provider_type="mock")
        assert isinstance(provider, MockLLMProvider)

    def test_invalid_provider_raises(self):
        """Factory should raise for invalid provider type."""
        with pytest.raises(ValueError):
            create_llm_provider(provider_type="invalid_provider")
```

### 3.7 Stage 5: Enhanced Reflector Tests

```python
# tests/test_enhanced_reflector.py

"""
Test Stage 5: Enhanced Reflector with CF integration.
Run with: pytest tests/test_enhanced_reflector.py -v
"""

import pytest
from agentic_recommender.agents.reflector import EnhancedReflector
from agentic_recommender.models.llm_provider import MockLLMProvider


class TestEnhancedReflector:
    """Test enhanced reflector with CF signals."""

    @pytest.fixture
    def mock_reflector(self):
        """Create reflector with mock LLM."""
        from agentic_recommender.similarity.swing import SwingSimilarity

        provider = MockLLMProvider(responses={
            "final_decision": '{"final_decision": "Yes", "confidence": 0.8, "reasoning": "Test"}'
        })
        swing = SwingSimilarity()

        return EnhancedReflector(
            llm_provider=provider,
            swing_similarity=swing
        )

    def test_reflector_returns_decision(self, mock_reflector, sample_customer_history):
        """Reflector should return a decision dict."""
        from agentic_recommender.data.food_delivery_adapter import CandidateProduct

        first_judgment = {"decision": "Yes", "reasoning": "Test reasoning"}
        candidate = CandidateProduct(name="Test", cuisine="chinese", price=0.5)

        # Convert to list of OrderRecord
        orders = []  # Simplified for test

        result = mock_reflector.forward(
            first_judgment=first_judgment,
            user_orders=orders,
            candidate=candidate
        )

        assert 'final_decision' in result
        assert result['final_decision'] in ['Yes', 'No']

    def test_cf_score_included_in_context(self, mock_reflector, sample_customer_history):
        """CF score should be computed and included."""
        from agentic_recommender.data.food_delivery_adapter import CandidateProduct

        first_judgment = {"decision": "Yes", "reasoning": "Test"}
        candidate = CandidateProduct(name="Test", cuisine="chinese", price=0.5)

        result = mock_reflector.forward(
            first_judgment=first_judgment,
            user_orders=[],
            candidate=candidate
        )

        # Result should include CF contribution
        assert 'cf_score' in result or 'cf_contribution' in result
```

### 3.8 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific stage
pytest tests/test_data_loader.py -v        # Stage 1
pytest tests/test_user_representation.py -v # Stage 2
pytest tests/test_cf_score.py -v            # Stage 3
pytest tests/test_llm_providers.py -v       # Stage 4
pytest tests/test_enhanced_reflector.py -v  # Stage 5

# Run with coverage
pytest tests/ --cov=agentic_recommender --cov-report=html

# Run only fast tests (no API calls)
pytest tests/ -v -m "not slow"
```

---

## 4. Implementation Order

### Phase 1: Data Infrastructure (This Sprint)

1. **Create enriched data loader** (`data/enriched_loader.py`)
   - Test: `test_data_loader.py`
   - Preserves all original fields

2. **Create representations module** (`data/representations.py`)
   - Test: `test_user_representation.py`
   - EnrichedUser, EnrichedVendor, EnrichedCuisine

### Phase 2: CF Enhancement

3. **Enhance Swing with order grouping** (`similarity/enhanced_cf.py`)
   - Test: `test_cf_score.py`
   - Co-purchase pattern integration

### Phase 3: LLM Module

4. **Add Claude provider** (`models/llm_provider.py`)
   - Test: `test_llm_providers.py`

### Phase 4: Reflector Integration

5. **Create enhanced reflector** (`agents/reflector.py`)
   - Test: `test_enhanced_reflector.py`

---

## 5. Validation with Real Data

Each module will include a `validate_with_real_data()` function:

```python
def validate_with_real_data():
    """
    Run validation against Singapore dataset.
    Prints human-readable output for manual inspection.
    """
    print("=" * 60)
    print("VALIDATION: EnrichedUser Representation")
    print("=" * 60)

    # Load real data
    orders = pd.read_csv("/Users/zhenkai/Downloads/data_sg/orders_sg_train.txt")

    # Pick a customer with good history
    customer_id = "f374c8c54c"  # From the data sample we saw
    customer_orders = orders[orders['customer_id'] == customer_id]

    print(f"\nCustomer: {customer_id}")
    print(f"Total order rows: {len(customer_orders)}")
    print(f"Unique orders: {customer_orders['order_id'].nunique()}")

    # Build representation
    user = EnrichedUser.from_order_history(customer_id, customer_orders)

    print(f"\n--- Cuisine Distribution ---")
    for cuisine, freq in sorted(user.cuisine_distribution.items(),
                                 key=lambda x: -x[1])[:5]:
        print(f"  {cuisine}: {freq:.2%}")

    print(f"\n--- Top Vendors ---")
    for vendor, count in user.top_vendors[:5]:
        print(f"  {vendor}: {count} orders")

    print(f"\n--- Temporal Patterns ---")
    print(f"  Avg basket size: {user.typical_basket_size:.2f}")
    print(f"  Most active hour: {max(user.hour_distribution, key=user.hour_distribution.get)}")

    print("=" * 60)


if __name__ == "__main__":
    validate_with_real_data()
```

---

## 6. Next Steps

1. **Immediate**: Create `tests/` directory structure
2. **Then**: Implement `data/enriched_loader.py` with tests
3. **Validate**: Run against real Singapore data sample
4. **Iterate**: Move to next module only when current passes

Do you want me to start implementing Stage 1 (Data Loader) with its tests?
