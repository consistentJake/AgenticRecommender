"""
Test Stage 1: Data loading and representations.

Run with: pytest agentic_recommender/tests/test_data_loader.py -v
"""

import pytest
import pandas as pd
from pathlib import Path

from agentic_recommender.data.enriched_loader import (
    EnrichedDataLoader,
    DataConfig,
    load_singapore_data,
)
from agentic_recommender.data.representations import (
    EnrichedUser,
    CuisineProfile,
    CuisineRegistry,
)


class TestEnrichedDataLoader:
    """Test enriched data loading from original files."""

    def test_load_orders_has_required_columns(self, data_dir):
        """Verify all expected columns are present in orders."""
        config = DataConfig(data_dir=data_dir)
        loader = EnrichedDataLoader(config)
        orders = loader.load_orders()

        required_cols = [
            'customer_id', 'geohash', 'order_id', 'vendor_id',
            'product_id', 'day_of_week', 'order_time', 'order_day'
        ]
        for col in required_cols:
            assert col in orders.columns, f"Missing column: {col}"

    def test_load_orders_parses_hour(self, data_dir):
        """Hour should be extracted from order_time."""
        config = DataConfig(data_dir=data_dir, parse_time=True)
        loader = EnrichedDataLoader(config)
        orders = loader.load_orders()

        assert 'hour' in orders.columns
        assert orders['hour'].min() >= 0
        assert orders['hour'].max() <= 23

    def test_load_orders_computes_day_name(self, data_dir):
        """Day name should be computed from day_of_week."""
        config = DataConfig(data_dir=data_dir, compute_day_name=True)
        loader = EnrichedDataLoader(config)
        orders = loader.load_orders()

        assert 'day_name' in orders.columns
        valid_days = {'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Unknown'}
        assert set(orders['day_name'].unique()).issubset(valid_days)

    def test_load_vendors_has_required_columns(self, data_dir):
        """Verify vendors have required columns."""
        config = DataConfig(data_dir=data_dir)
        loader = EnrichedDataLoader(config)
        vendors = loader.load_vendors()

        required_cols = ['vendor_id', 'primary_cuisine', 'geohash', 'chain_id']
        for col in required_cols:
            assert col in vendors.columns, f"Missing column: {col}"

    def test_load_products_has_required_columns(self, data_dir):
        """Verify products have required columns."""
        config = DataConfig(data_dir=data_dir)
        loader = EnrichedDataLoader(config)
        products = loader.load_products()

        required_cols = ['vendor_id', 'product_id', 'name', 'unit_price']
        for col in required_cols:
            assert col in products.columns, f"Missing column: {col}"

    def test_load_merged_preserves_orders(self, data_dir):
        """Merged data should preserve all orders."""
        config = DataConfig(data_dir=data_dir)
        loader = EnrichedDataLoader(config)

        orders = loader.load_orders()
        merged = loader.load_merged()

        # Left join should preserve all order rows
        assert len(merged) == len(orders)

    def test_load_merged_has_cuisine(self, data_dir):
        """Merged data should include cuisine from vendors."""
        config = DataConfig(data_dir=data_dir)
        loader = EnrichedDataLoader(config)
        merged = loader.load_merged()

        assert 'cuisine' in merged.columns

    def test_load_merged_has_both_geohashes(self, data_dir):
        """Merged data should have user and vendor geohashes."""
        config = DataConfig(data_dir=data_dir)
        loader = EnrichedDataLoader(config)
        merged = loader.load_merged()

        assert 'user_geohash' in merged.columns
        assert 'vendor_geohash' in merged.columns

    def test_order_id_groups_same_customer(self, sample_orders):
        """Same order_id should have same customer_id."""
        grouped = sample_orders.groupby('order_id').agg({
            'customer_id': 'nunique',
        })
        assert (grouped['customer_id'] == 1).all(), "Order has multiple customers"

    def test_order_id_groups_same_vendor(self, sample_orders):
        """Same order_id should have same vendor_id."""
        grouped = sample_orders.groupby('order_id').agg({
            'vendor_id': 'nunique',
        })
        assert (grouped['vendor_id'] == 1).all(), "Order has multiple vendors"

    def test_geohash_format(self, sample_orders):
        """Geohash should start with 'w' (Singapore)."""
        geohashes = sample_orders['geohash'].dropna()
        assert all(str(g).startswith('w') for g in geohashes)

    def test_get_stats_returns_valid_counts(self, data_dir):
        """Stats should return positive counts."""
        loader = load_singapore_data(str(data_dir))
        stats = loader.get_stats()

        assert stats['unique_customers'] > 0
        assert stats['unique_orders'] > 0
        assert stats['unique_vendors'] > 0
        assert stats['avg_items_per_order'] >= 1.0


class TestEnrichedUser:
    """Test user representation building."""

    def test_from_orders_creates_valid_user(self, sample_customer_orders, sample_customer_id):
        """Should create user with valid fields."""
        user = EnrichedUser.from_orders(sample_customer_id, sample_customer_orders)

        assert user.customer_id == sample_customer_id
        assert user.total_orders > 0
        assert user.total_items > 0

    def test_cuisine_distribution_sums_to_one(self, sample_customer_orders, sample_customer_id):
        """Cuisine distribution should be normalized."""
        user = EnrichedUser.from_orders(sample_customer_id, sample_customer_orders)

        if user.cuisine_distribution:
            total = sum(user.cuisine_distribution.values())
            assert abs(total - 1.0) < 0.01, f"Distribution sums to {total}, not 1.0"

    def test_day_distribution_has_valid_keys(self, sample_customer_orders, sample_customer_id):
        """Day distribution keys should be 0-6."""
        user = EnrichedUser.from_orders(sample_customer_id, sample_customer_orders)

        for day in user.day_distribution.keys():
            assert 0 <= day <= 6, f"Invalid day: {day}"

    def test_hour_distribution_has_valid_keys(self, sample_customer_orders, sample_customer_id):
        """Hour distribution keys should be 0-23."""
        user = EnrichedUser.from_orders(sample_customer_id, sample_customer_orders)

        for hour in user.hour_distribution.keys():
            assert 0 <= hour <= 23, f"Invalid hour: {hour}"

    def test_basket_size_is_positive(self, sample_customer_orders, sample_customer_id):
        """Average basket size should be >= 1."""
        user = EnrichedUser.from_orders(sample_customer_id, sample_customer_orders)
        assert user.typical_basket_size >= 1.0

    def test_vendor_diversity_in_range(self, sample_customer_orders, sample_customer_id):
        """Vendor diversity should be between 0 and 1."""
        user = EnrichedUser.from_orders(sample_customer_id, sample_customer_orders)
        assert 0.0 <= user.vendor_diversity <= 1.0

    def test_to_dict_is_serializable(self, sample_customer_orders, sample_customer_id):
        """to_dict should return JSON-serializable dict."""
        import json
        user = EnrichedUser.from_orders(sample_customer_id, sample_customer_orders)
        user_dict = user.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(user_dict)
        assert len(json_str) > 0


class TestCuisineProfile:
    """Test cuisine profile building."""

    def test_from_orders_creates_valid_profile(self, merged_orders):
        """Should create profile with valid fields."""
        # Get a cuisine that exists
        cuisines = merged_orders['cuisine'].dropna().unique()
        if len(cuisines) == 0:
            pytest.skip("No cuisines in sample data")

        cuisine = cuisines[0]
        profile = CuisineProfile.from_orders(cuisine, merged_orders)

        assert profile.cuisine_type == cuisine
        assert profile.total_orders > 0

    def test_hour_distribution_has_valid_keys(self, merged_orders):
        """Hour distribution keys should be 0-23."""
        cuisines = merged_orders['cuisine'].dropna().unique()
        if len(cuisines) == 0:
            pytest.skip("No cuisines in sample data")

        profile = CuisineProfile.from_orders(cuisines[0], merged_orders)

        for hour in profile.hour_distribution.keys():
            assert 0 <= hour <= 23, f"Invalid hour: {hour}"

    def test_weekday_distribution_has_valid_keys(self, merged_orders):
        """Weekday distribution keys should be 0-6."""
        cuisines = merged_orders['cuisine'].dropna().unique()
        if len(cuisines) == 0:
            pytest.skip("No cuisines in sample data")

        profile = CuisineProfile.from_orders(cuisines[0], merged_orders)

        for day in profile.weekday_distribution.keys():
            assert 0 <= day <= 6, f"Invalid day: {day}"

    def test_peak_hours_are_sorted_by_frequency(self, merged_orders):
        """Peak hours should be most frequent hours."""
        cuisines = merged_orders['cuisine'].dropna().unique()
        if len(cuisines) == 0:
            pytest.skip("No cuisines in sample data")

        profile = CuisineProfile.from_orders(cuisines[0], merged_orders)

        if len(profile.peak_hours) > 1:
            # First peak hour should have highest frequency
            freq_0 = profile.hour_distribution.get(profile.peak_hours[0], 0)
            freq_1 = profile.hour_distribution.get(profile.peak_hours[1], 0)
            assert freq_0 >= freq_1

    def test_meal_time_distribution_has_valid_keys(self, merged_orders):
        """Meal time distribution should have valid meal names."""
        cuisines = merged_orders['cuisine'].dropna().unique()
        if len(cuisines) == 0:
            pytest.skip("No cuisines in sample data")

        profile = CuisineProfile.from_orders(cuisines[0], merged_orders)

        valid_meals = {'breakfast', 'lunch', 'afternoon', 'dinner', 'late_night'}
        for meal in profile.meal_time_distribution.keys():
            assert meal in valid_meals, f"Invalid meal time: {meal}"


class TestCuisineRegistry:
    """Test cuisine registry."""

    def test_build_creates_profiles(self, merged_orders):
        """Build should create profiles for all cuisines."""
        registry = CuisineRegistry()
        registry.build_from_data(merged_orders)

        assert len(registry.profiles) > 0
        assert registry._loaded

    def test_get_profile_returns_valid_profile(self, merged_orders):
        """get_profile should return CuisineProfile."""
        registry = CuisineRegistry()
        registry.build_from_data(merged_orders)

        cuisines = list(registry.profiles.keys())
        if cuisines:
            profile = registry.get_profile(cuisines[0])
            assert isinstance(profile, CuisineProfile)

    def test_get_profile_returns_none_for_unknown(self, merged_orders):
        """get_profile should return None for unknown cuisine."""
        registry = CuisineRegistry()
        registry.build_from_data(merged_orders)

        profile = registry.get_profile("definitely_not_a_cuisine_xyz123")
        assert profile is None

    def test_get_peak_cuisines_returns_sorted(self, merged_orders):
        """get_peak_cuisines_for_time should return sorted list."""
        registry = CuisineRegistry()
        registry.build_from_data(merged_orders)

        peaks = registry.get_peak_cuisines_for_time(hour=12, weekday=3, top_k=5)

        assert len(peaks) <= 5
        # Should be sorted by score descending
        if len(peaks) > 1:
            assert peaks[0][1] >= peaks[1][1]

    def test_get_all_cuisines(self, merged_orders):
        """get_all_cuisines should return list of cuisine names."""
        registry = CuisineRegistry()
        registry.build_from_data(merged_orders)

        cuisines = registry.get_all_cuisines()
        assert isinstance(cuisines, list)
        assert len(cuisines) > 0


# Validation function for manual inspection
def validate_with_real_data():
    """
    Run validation against Singapore dataset.
    Prints human-readable output for manual inspection.
    """
    from agentic_recommender.data.enriched_loader import load_singapore_data

    print("=" * 60)
    print("VALIDATION: Data Loader and Representations")
    print("=" * 60)

    # Load data
    loader = load_singapore_data()
    merged = loader.load_merged()
    stats = loader.get_stats()

    print(f"\n--- Dataset Statistics ---")
    for key, value in stats.items():
        print(f"  {key}: {value:,.2f}" if isinstance(value, float) else f"  {key}: {value:,}")

    # Pick a customer with good history
    customer_counts = merged['customer_id'].value_counts()
    test_customer = customer_counts[(customer_counts >= 10) & (customer_counts <= 50)].index[0]
    customer_orders = merged[merged['customer_id'] == test_customer]

    print(f"\n--- Sample User: {test_customer} ---")
    user = EnrichedUser.from_orders(test_customer, customer_orders)

    print(f"  Total orders: {user.total_orders}")
    print(f"  Total items: {user.total_items}")
    print(f"  Avg basket size: {user.typical_basket_size:.2f}")
    print(f"  Vendor diversity: {user.vendor_diversity:.2f}")

    print(f"\n  Top cuisines:")
    for cuisine, freq in sorted(user.cuisine_distribution.items(), key=lambda x: -x[1])[:5]:
        print(f"    {cuisine}: {freq:.2%}")

    print(f"\n  Peak hours: {user.peak_hours}")
    print(f"  Peak weekdays: {user.peak_weekdays}")

    # Build cuisine registry
    print(f"\n--- Cuisine Registry ---")
    registry = CuisineRegistry()
    registry.build_from_data(merged)

    # Show top 3 cuisines by order count
    cuisine_orders = [(c, p.total_orders) for c, p in registry.profiles.items()]
    cuisine_orders.sort(key=lambda x: -x[1])

    for cuisine, count in cuisine_orders[:3]:
        profile = registry.get_profile(cuisine)
        print(f"\n  {cuisine}:")
        print(f"    Orders: {profile.total_orders:,}")
        print(f"    Peak hours: {profile.peak_hours}")
        print(f"    Peak weekdays: {profile.peak_weekdays}")
        print(f"    Meal distribution: {profile.meal_time_distribution}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    validate_with_real_data()
