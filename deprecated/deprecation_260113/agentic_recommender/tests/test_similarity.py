"""
Test Stage 2: Similarity module.

Run with: pytest agentic_recommender/tests/test_similarity.py -v
"""

import pytest
from typing import List, Tuple

from agentic_recommender.similarity import (
    SimilarityMethod,
    SimilarityConfig,
    SwingMethod,
    SwingConfig,
    CosineMethod,
    CosineConfig,
    JaccardMethod,
    JaccardConfig,
    SimilarityFactory,
)


# Test data
SAMPLE_INTERACTIONS: List[Tuple[str, str]] = [
    ("user1", "pizza"),
    ("user1", "burger"),
    ("user1", "chinese"),
    ("user2", "pizza"),
    ("user2", "burger"),
    ("user2", "sushi"),
    ("user3", "chinese"),
    ("user3", "indian"),
    ("user3", "thai"),
    ("user4", "pizza"),
    ("user4", "chinese"),
    ("user4", "indian"),
]


class TestSwingMethod:
    """Test Swing similarity method."""

    def test_fit_creates_indices(self):
        """Fit should create user-item and item-user indices."""
        swing = SwingMethod()
        swing.fit(SAMPLE_INTERACTIONS)

        assert len(swing.user_items) == 4
        assert len(swing.item_users) > 0
        assert swing._fitted

    def test_compute_similarity_same_user_high(self):
        """Users with same items should have higher similarity."""
        swing = SwingMethod()
        swing.fit(SAMPLE_INTERACTIONS)

        # user1 and user2 share pizza and burger
        sim_12 = swing.compute_similarity("user1", "user2")
        # user1 and user3 share only chinese
        sim_13 = swing.compute_similarity("user1", "user3")

        assert sim_12 > sim_13

    def test_compute_similarity_no_overlap_zero(self):
        """Users with no common items should have 0 similarity."""
        interactions = [
            ("user1", "pizza"),
            ("user2", "sushi"),
        ]
        swing = SwingMethod()
        swing.fit(interactions)

        sim = swing.compute_similarity("user1", "user2")
        assert sim == 0.0

    def test_compute_similarity_is_symmetric(self):
        """similarity(A, B) should equal similarity(B, A)."""
        swing = SwingMethod()
        swing.fit(SAMPLE_INTERACTIONS)

        sim_12 = swing.compute_similarity("user1", "user2")
        sim_21 = swing.compute_similarity("user2", "user1")

        assert abs(sim_12 - sim_21) < 0.0001

    def test_get_similar_returns_sorted(self):
        """get_similar should return results sorted by similarity."""
        swing = SwingMethod(SwingConfig(top_k=10, min_threshold=0.0))
        swing.fit(SAMPLE_INTERACTIONS)

        similar = swing.get_similar("user1")

        # Should be sorted descending
        if len(similar) > 1:
            for i in range(len(similar) - 1):
                assert similar[i][1] >= similar[i + 1][1]

    def test_get_similar_excludes_self(self):
        """get_similar should not include the query user."""
        swing = SwingMethod()
        swing.fit(SAMPLE_INTERACTIONS)

        similar = swing.get_similar("user1")
        user_ids = [u for u, _ in similar]

        assert "user1" not in user_ids

    def test_cache_works(self):
        """Cache should store computed similarities."""
        swing = SwingMethod(SwingConfig(cache_enabled=True))
        swing.fit(SAMPLE_INTERACTIONS)

        # First call
        sim1 = swing.compute_similarity("user1", "user2")
        cache_size_1 = len(swing._cache)

        # Second call (should hit cache)
        sim2 = swing.compute_similarity("user1", "user2")
        cache_size_2 = len(swing._cache)

        assert sim1 == sim2
        assert cache_size_1 == cache_size_2  # No new entries

    def test_get_user_items(self):
        """get_user_items should return user's items."""
        swing = SwingMethod()
        swing.fit(SAMPLE_INTERACTIONS)

        items = swing.get_user_items("user1")
        assert "pizza" in items
        assert "burger" in items


class TestCosineMethod:
    """Test Cosine similarity method."""

    def test_fit_creates_vectors(self):
        """Fit should create user vectors."""
        cosine = CosineMethod()
        cosine.fit(SAMPLE_INTERACTIONS)

        assert len(cosine.user_vectors) == 4
        assert cosine._fitted

    def test_compute_similarity_in_range(self):
        """Cosine similarity should be in [0, 1]."""
        cosine = CosineMethod()
        cosine.fit(SAMPLE_INTERACTIONS)

        for u1 in ["user1", "user2", "user3"]:
            for u2 in ["user1", "user2", "user3"]:
                sim = cosine.compute_similarity(u1, u2)
                assert 0.0 <= sim <= 1.0

    def test_compute_similarity_self_is_one(self):
        """User should have similarity 1.0 with themselves."""
        cosine = CosineMethod()
        cosine.fit(SAMPLE_INTERACTIONS)

        sim = cosine.compute_similarity("user1", "user1")
        assert abs(sim - 1.0) < 0.01

    def test_compute_similarity_is_symmetric(self):
        """Cosine similarity should be symmetric."""
        cosine = CosineMethod()
        cosine.fit(SAMPLE_INTERACTIONS)

        sim_12 = cosine.compute_similarity("user1", "user2")
        sim_21 = cosine.compute_similarity("user2", "user1")

        assert abs(sim_12 - sim_21) < 0.0001


class TestJaccardMethod:
    """Test Jaccard similarity method."""

    def test_fit_creates_sets(self):
        """Fit should create user-item sets."""
        jaccard = JaccardMethod()
        jaccard.fit(SAMPLE_INTERACTIONS)

        assert len(jaccard.user_items) == 4
        assert jaccard._fitted

    def test_compute_similarity_in_range(self):
        """Jaccard similarity should be in [0, 1]."""
        jaccard = JaccardMethod()
        jaccard.fit(SAMPLE_INTERACTIONS)

        for u1 in ["user1", "user2", "user3"]:
            for u2 in ["user1", "user2", "user3"]:
                sim = jaccard.compute_similarity(u1, u2)
                assert 0.0 <= sim <= 1.0

    def test_compute_similarity_self_is_one(self):
        """User should have similarity 1.0 with themselves."""
        jaccard = JaccardMethod()
        jaccard.fit(SAMPLE_INTERACTIONS)

        sim = jaccard.compute_similarity("user1", "user1")
        assert abs(sim - 1.0) < 0.01

    def test_compute_similarity_exact_formula(self):
        """Test Jaccard formula: |A∩B| / |A∪B|."""
        interactions = [
            ("user1", "a"),
            ("user1", "b"),
            ("user1", "c"),
            ("user2", "b"),
            ("user2", "c"),
            ("user2", "d"),
        ]
        jaccard = JaccardMethod()
        jaccard.fit(interactions)

        # user1: {a, b, c}, user2: {b, c, d}
        # intersection: {b, c} = 2
        # union: {a, b, c, d} = 4
        # Jaccard = 2/4 = 0.5
        sim = jaccard.compute_similarity("user1", "user2")
        assert abs(sim - 0.5) < 0.01


class TestSimilarityFactory:
    """Test similarity factory."""

    def test_create_swing(self):
        """Factory should create Swing method."""
        factory = SimilarityFactory()
        method = factory.create("swing")

        assert isinstance(method, SwingMethod)

    def test_create_cosine(self):
        """Factory should create Cosine method."""
        factory = SimilarityFactory()
        method = factory.create("cosine")

        assert isinstance(method, CosineMethod)

    def test_create_jaccard(self):
        """Factory should create Jaccard method."""
        factory = SimilarityFactory()
        method = factory.create("jaccard")

        assert isinstance(method, JaccardMethod)

    def test_create_with_config(self):
        """Factory should respect custom config."""
        factory = SimilarityFactory()
        config = SwingConfig(alpha1=10.0, top_k=20)
        method = factory.create("swing", config)

        assert method.config.alpha1 == 10.0
        assert method.config.top_k == 20

    def test_create_invalid_raises(self):
        """Factory should raise for unknown method."""
        factory = SimilarityFactory()

        with pytest.raises(ValueError):
            factory.create("unknown_method")

    def test_get_or_create_caches(self):
        """get_or_create should cache instances."""
        factory = SimilarityFactory()

        method1 = factory.get_or_create("swing")
        method2 = factory.get_or_create("swing")

        assert method1 is method2

    def test_get_default(self):
        """get_default should return configured default."""
        factory = SimilarityFactory(default_method="cosine")
        method = factory.get_default()

        assert isinstance(method, CosineMethod)

    def test_set_default(self):
        """set_default should change default method."""
        factory = SimilarityFactory(default_method="swing")
        factory.set_default("jaccard")

        assert factory.get_default_name() == "jaccard"

    def test_available_methods(self):
        """available_methods should list all methods."""
        methods = SimilarityFactory.available_methods()

        assert "swing" in methods
        assert "cosine" in methods
        assert "jaccard" in methods

    def test_get_method_info(self):
        """get_method_info should return method details."""
        info = SimilarityFactory.get_method_info("swing")

        assert info["name"] == "swing"
        assert "SwingMethod" in info["class"]


class TestSimilarityWithRealData:
    """Test similarity methods with real Singapore data."""

    def test_swing_with_real_interactions(self, all_orders, all_vendors):
        """Test Swing with real order data."""
        # Build interactions (customer -> cuisine)
        merged = all_orders.head(50000).merge(
            all_vendors[['vendor_id', 'primary_cuisine']],
            on='vendor_id',
            how='left'
        )

        interactions = [
            (str(row.customer_id), str(row.primary_cuisine))
            for row in merged.dropna(subset=['primary_cuisine']).itertuples()
        ]

        swing = SwingMethod(SwingConfig(top_k=5, min_threshold=0.0))
        swing.fit(interactions)

        stats = swing.get_stats()
        assert stats['num_users'] > 0
        assert stats['num_items'] > 0

        # Get similar users for first user
        first_user = list(swing.user_items.keys())[0]
        similar = swing.get_similar(first_user)

        # Should return some similar users
        assert isinstance(similar, list)

    def test_all_methods_same_data(self, all_orders, all_vendors):
        """All methods should work with same data."""
        # Build interactions
        merged = all_orders.head(10000).merge(
            all_vendors[['vendor_id', 'primary_cuisine']],
            on='vendor_id',
            how='left'
        )

        interactions = [
            (str(row.customer_id), str(row.primary_cuisine))
            for row in merged.dropna(subset=['primary_cuisine']).itertuples()
        ]

        factory = SimilarityFactory()

        for method_name in factory.available_methods():
            method = factory.create(method_name)
            method.fit(interactions)

            stats = method.get_stats()
            assert stats['fitted'] is True
            assert stats['method'] == method_name


# Validation function
def validate_similarity_methods():
    """Validate similarity methods with real data."""
    import pandas as pd
    from pathlib import Path

    print("=" * 60)
    print("VALIDATION: Similarity Methods")
    print("=" * 60)

    # Load data
    data_dir = Path("/Users/zhenkai/Downloads/data_sg")
    orders = pd.read_csv(data_dir / "orders_sg_train.txt", nrows=100000)
    vendors = pd.read_csv(data_dir / "vendors_sg.txt")

    # Build interactions
    merged = orders.merge(
        vendors[['vendor_id', 'primary_cuisine']],
        on='vendor_id',
        how='left'
    )

    interactions = [
        (str(row.customer_id), str(row.primary_cuisine))
        for row in merged.dropna(subset=['primary_cuisine']).itertuples()
    ]

    print(f"\nTotal interactions: {len(interactions):,}")

    # Test all methods
    factory = SimilarityFactory()

    for method_name in factory.available_methods():
        print(f"\n--- {method_name.upper()} ---")

        method = factory.create(method_name, SimilarityConfig(top_k=5))
        method.fit(interactions)

        stats = method.get_stats()
        print(f"  Users: {stats.get('num_users', 'N/A')}")
        print(f"  Items: {stats.get('num_items', 'N/A')}")

        # Get similar users for a test user
        test_user = list(method._get_candidate_entities("x"))[0]
        similar = method.get_similar(test_user)

        print(f"  Similar to {test_user[:10]}...:")
        for user, score in similar[:3]:
            print(f"    {user[:10]}...: {score:.4f}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    validate_similarity_methods()
