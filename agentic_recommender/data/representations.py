"""
Enriched representations for users, cuisines, and vendors.

These representations capture behavioral patterns for recommendation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter


@dataclass
class EnrichedUser:
    """
    Complete user representation for recommendation.

    Captures:
    - Cuisine preferences
    - Temporal patterns (hours, weekdays)
    - Price behavior
    - Vendor loyalty
    - Co-purchase patterns
    - Location patterns
    """

    # Identity
    customer_id: str
    primary_geohash: str = ""

    # Cuisine preferences
    cuisine_distribution: Dict[str, float] = field(default_factory=dict)
    # Detailed purchase history: each entry has vendor_id, cuisine, hour, day_of_week
    purchase_history: List[Dict[str, Any]] = field(default_factory=list)

    # Temporal patterns
    day_distribution: Dict[int, float] = field(default_factory=dict)
    hour_distribution: Dict[int, float] = field(default_factory=dict)
    peak_hours: List[int] = field(default_factory=list)
    peak_weekdays: List[int] = field(default_factory=list)

    # Price behavior
    avg_price: float = 0.0
    price_std: float = 0.0
    price_range: Tuple[float, float] = (0.0, 1.0)

    # Vendor loyalty
    top_vendors: List[Tuple[str, int]] = field(default_factory=list)
    vendor_diversity: float = 0.0
    chain_preferences: Dict[str, float] = field(default_factory=dict)

    # Co-purchase patterns
    typical_basket_size: float = 1.0

    # Activity metrics
    total_orders: int = 0
    total_items: int = 0

    @classmethod
    def from_orders(
        cls,
        customer_id: str,
        orders_df: pd.DataFrame,
        already_filtered: bool = True,
        max_history: int = None
    ) -> 'EnrichedUser':
        """
        Build enriched user from order data.

        Args:
            customer_id: Customer identifier
            orders_df: DataFrame with customer's orders (from merged data)
                      Required columns: order_id, cuisine, day_of_week, hour,
                                       vendor_id, chain_id, user_geohash, unit_price
            already_filtered: If True, skip filtering by customer_id (for performance)
            max_history: Optional maximum number of orders to keep in purchase_history.
                        If None, keep ALL history (useful for evaluation).
                        If specified, keeps last N orders.
        """
        if len(orders_df) == 0:
            return cls(customer_id=customer_id)

        # Only filter if not already filtered (for backwards compatibility)
        if not already_filtered:
            orders_df = orders_df[orders_df['customer_id'] == customer_id].copy()
            if len(orders_df) == 0:
                return cls(customer_id=customer_id)

        # Primary geohash (most frequent) - handle different column names
        geohash_col = 'user_geohash' if 'user_geohash' in orders_df.columns else 'geohash'
        if geohash_col in orders_df.columns:
            geohash_counts = orders_df[geohash_col].value_counts()
            primary_geohash = geohash_counts.index[0] if len(geohash_counts) > 0 else ""
        else:
            primary_geohash = ""

        # Cuisine distribution (normalized) - handle missing cuisine column
        cuisine_col = 'cuisine' if 'cuisine' in orders_df.columns else 'primary_cuisine'
        if cuisine_col not in orders_df.columns:
            cuisine_distribution = {}
            cuisine_sequence = []
        else:
            cuisine_counts = orders_df.groupby('order_id')[cuisine_col].first().value_counts()
            total_cuisine = cuisine_counts.sum()
            cuisine_distribution = {k: v / total_cuisine for k, v in cuisine_counts.items()} if total_cuisine > 0 else {}

        # Detailed purchase history (chronological)
        # Each entry: {vendor_id, cuisine, hour, day_of_week}
        purchase_history = []
        if cuisine_col in orders_df.columns:
            # Sort orders chronologically
            sort_cols = ['day_num', 'hour'] if 'day_num' in orders_df.columns else ['order_id']
            sorted_orders = orders_df.sort_values(sort_cols)

            # Group by order_id to get one entry per order
            seen_orders = set()
            for _, row in sorted_orders.iterrows():
                order_id = row.get('order_id')
                if order_id in seen_orders:
                    continue
                seen_orders.add(order_id)

                purchase_history.append({
                    'vendor_id': str(row.get('vendor_id', '')),
                    'cuisine': row.get(cuisine_col, 'unknown'),
                    'hour': int(row.get('hour', 12)),
                    'day_of_week': int(row.get('day_of_week', 0)),
                })

            # Optionally truncate history (if max_history is specified)
            if max_history is not None:
                purchase_history = purchase_history[-max_history:]

        # Day distribution
        day_counts = orders_df.groupby('order_id')['day_of_week'].first().value_counts(normalize=True)
        day_distribution = day_counts.to_dict()
        peak_weekdays = day_counts.head(3).index.tolist()

        # Hour distribution
        if 'hour' in orders_df.columns:
            hour_counts = orders_df.groupby('order_id')['hour'].first().value_counts(normalize=True)
            hour_distribution = hour_counts.to_dict()
            peak_hours = hour_counts.head(3).index.tolist()
        else:
            hour_distribution = {}
            peak_hours = []

        # Price behavior
        order_totals = orders_df.groupby('order_id')['unit_price'].sum()
        avg_price = order_totals.mean() if len(order_totals) > 0 else 0.0
        price_std = order_totals.std() if len(order_totals) > 1 else 0.0
        price_range = (order_totals.min(), order_totals.max()) if len(order_totals) > 0 else (0.0, 0.0)

        # Vendor loyalty
        vendor_counts = orders_df.groupby('order_id')['vendor_id'].first().value_counts()
        top_vendors = [(str(v), int(c)) for v, c in vendor_counts.head(5).items()]

        # Vendor diversity (0 = always same vendor, 1 = all different)
        unique_vendors = orders_df['vendor_id'].nunique()
        total_orders = orders_df['order_id'].nunique()
        vendor_diversity = unique_vendors / total_orders if total_orders > 0 else 0.0

        # Chain preferences
        if 'chain_id' in orders_df.columns:
            chain_df = orders_df[orders_df['chain_id'] != '']
            if len(chain_df) > 0:
                chain_counts = chain_df.groupby('order_id')['chain_id'].first().value_counts(normalize=True)
                chain_preferences = chain_counts.to_dict()
            else:
                chain_preferences = {}
        else:
            chain_preferences = {}

        # Basket size
        items_per_order = orders_df.groupby('order_id').size()
        typical_basket_size = items_per_order.mean() if len(items_per_order) > 0 else 1.0

        return cls(
            customer_id=customer_id,
            primary_geohash=primary_geohash,
            cuisine_distribution=cuisine_distribution,
            purchase_history=purchase_history,
            day_distribution=day_distribution,
            hour_distribution=hour_distribution,
            peak_hours=peak_hours,
            peak_weekdays=peak_weekdays,
            avg_price=float(avg_price),
            price_std=float(price_std) if not np.isnan(price_std) else 0.0,
            price_range=price_range,
            top_vendors=top_vendors,
            vendor_diversity=float(vendor_diversity),
            chain_preferences=chain_preferences,
            typical_basket_size=float(typical_basket_size),
            total_orders=int(total_orders),
            total_items=int(len(orders_df)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'customer_id': self.customer_id,
            'primary_geohash': self.primary_geohash,
            'cuisine_distribution': self.cuisine_distribution,
            'purchase_history': self.purchase_history,
            'day_distribution': self.day_distribution,
            'hour_distribution': self.hour_distribution,
            'peak_hours': self.peak_hours,
            'peak_weekdays': self.peak_weekdays,
            'avg_price': self.avg_price,
            'price_std': self.price_std,
            'price_range': list(self.price_range),
            'top_vendors': self.top_vendors,
            'vendor_diversity': self.vendor_diversity,
            'chain_preferences': self.chain_preferences,
            'typical_basket_size': self.typical_basket_size,
            'total_orders': self.total_orders,
            'total_items': self.total_items,
        }


@dataclass
class CuisineProfile:
    """
    Cuisine profile with temporal patterns.

    Captures when this cuisine is typically ordered.
    """

    cuisine_type: str

    # Temporal patterns
    hour_distribution: Dict[int, float] = field(default_factory=dict)
    peak_hours: List[int] = field(default_factory=list)
    weekday_distribution: Dict[int, float] = field(default_factory=dict)
    peak_weekdays: List[int] = field(default_factory=list)
    meal_time_distribution: Dict[str, float] = field(default_factory=dict)

    # Popularity metrics
    total_orders: int = 0
    unique_customers: int = 0
    avg_orders_per_customer: float = 0.0

    # Price characteristics
    avg_price: float = 0.0
    price_std: float = 0.0
    price_range: Tuple[float, float] = (0.0, 1.0)

    # Sequential patterns
    preceded_by: Dict[str, float] = field(default_factory=dict)
    followed_by: Dict[str, float] = field(default_factory=dict)

    # Location patterns
    popular_in_areas: List[str] = field(default_factory=list)

    @classmethod
    def from_orders(
        cls,
        cuisine_type: str,
        orders_df: pd.DataFrame,
        vendors_df: pd.DataFrame = None
    ) -> 'CuisineProfile':
        """
        Build cuisine profile from order data.

        Args:
            cuisine_type: The cuisine to analyze
            orders_df: All orders (merged with vendor info)
            vendors_df: Optional vendors dataframe for additional info
        """
        # Filter to this cuisine
        if 'cuisine' in orders_df.columns:
            cuisine_orders = orders_df[orders_df['cuisine'] == cuisine_type].copy()
        elif 'primary_cuisine' in orders_df.columns:
            cuisine_orders = orders_df[orders_df['primary_cuisine'] == cuisine_type].copy()
        else:
            return cls(cuisine_type=cuisine_type)

        if len(cuisine_orders) == 0:
            return cls(cuisine_type=cuisine_type)

        # Hour distribution
        if 'hour' in cuisine_orders.columns:
            hour_counts = cuisine_orders.groupby('order_id')['hour'].first().value_counts(normalize=True)
            hour_distribution = hour_counts.to_dict()
            peak_hours = hour_counts.head(3).index.tolist()
        else:
            hour_distribution = {}
            peak_hours = []

        # Weekday distribution
        if 'day_of_week' in cuisine_orders.columns:
            weekday_counts = cuisine_orders.groupby('order_id')['day_of_week'].first().value_counts(normalize=True)
            weekday_distribution = weekday_counts.to_dict()
            peak_weekdays = weekday_counts.head(3).index.tolist()
        else:
            weekday_distribution = {}
            peak_weekdays = []

        # Meal time distribution
        meal_distribution = {}
        if 'hour' in cuisine_orders.columns:
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
            meal_counts = cuisine_orders.groupby('order_id')['meal_time'].first().value_counts(normalize=True)
            meal_distribution = meal_counts.to_dict()

        # Popularity metrics
        total_orders = cuisine_orders['order_id'].nunique()
        unique_customers = cuisine_orders['customer_id'].nunique()
        avg_orders = total_orders / max(unique_customers, 1)

        # Price characteristics
        if 'unit_price' in cuisine_orders.columns:
            order_totals = cuisine_orders.groupby('order_id')['unit_price'].sum()
            avg_price = order_totals.mean() if len(order_totals) > 0 else 0.0
            price_std = order_totals.std() if len(order_totals) > 1 else 0.0
            price_range = (order_totals.min(), order_totals.max()) if len(order_totals) > 0 else (0.0, 0.0)
        else:
            avg_price = 0.0
            price_std = 0.0
            price_range = (0.0, 1.0)

        # Popular areas
        if 'user_geohash' in cuisine_orders.columns:
            area_counts = cuisine_orders['user_geohash'].value_counts()
            popular_areas = area_counts.head(5).index.tolist()
        else:
            popular_areas = []

        return cls(
            cuisine_type=cuisine_type,
            hour_distribution=hour_distribution,
            peak_hours=peak_hours,
            weekday_distribution=weekday_distribution,
            peak_weekdays=peak_weekdays,
            meal_time_distribution=meal_distribution,
            total_orders=int(total_orders),
            unique_customers=int(unique_customers),
            avg_orders_per_customer=float(avg_orders),
            avg_price=float(avg_price),
            price_std=float(price_std) if not np.isnan(price_std) else 0.0,
            price_range=price_range,
            popular_in_areas=popular_areas,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cuisine_type': self.cuisine_type,
            'hour_distribution': self.hour_distribution,
            'peak_hours': self.peak_hours,
            'weekday_distribution': self.weekday_distribution,
            'peak_weekdays': self.peak_weekdays,
            'meal_time_distribution': self.meal_time_distribution,
            'total_orders': self.total_orders,
            'unique_customers': self.unique_customers,
            'avg_orders_per_customer': self.avg_orders_per_customer,
            'avg_price': self.avg_price,
            'price_std': self.price_std,
            'price_range': list(self.price_range),
            'popular_in_areas': self.popular_in_areas,
        }


class CuisineRegistry:
    """
    Registry of all cuisine profiles for fast lookup.
    """

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self):
        self.profiles: Dict[str, CuisineProfile] = {}
        self._loaded = False

    def build_from_data(
        self,
        orders_df: pd.DataFrame,
        vendors_df: pd.DataFrame = None
    ):
        """Build profiles for all cuisines."""
        # Get cuisine column
        if 'cuisine' in orders_df.columns:
            cuisine_col = 'cuisine'
        elif 'primary_cuisine' in orders_df.columns:
            cuisine_col = 'primary_cuisine'
        else:
            # Need to merge with vendors
            if vendors_df is not None:
                orders_df = orders_df.merge(
                    vendors_df[['vendor_id', 'primary_cuisine']],
                    on='vendor_id',
                    how='left'
                )
                cuisine_col = 'primary_cuisine'
            else:
                raise ValueError("No cuisine column found and no vendors_df provided")

        cuisines = orders_df[cuisine_col].dropna().unique()

        for cuisine in cuisines:
            self.profiles[cuisine] = CuisineProfile.from_orders(
                cuisine, orders_df, vendors_df
            )

        self._loaded = True
        print(f"Built profiles for {len(self.profiles)} cuisines")

    def get_profile(self, cuisine: str) -> Optional[CuisineProfile]:
        """Get profile for a cuisine."""
        return self.profiles.get(cuisine)

    def get_all_cuisines(self) -> List[str]:
        """Get list of all cuisines."""
        return list(self.profiles.keys())

    def get_peak_cuisines_for_time(
        self,
        hour: int,
        weekday: int,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get cuisines most likely to be ordered at this time.

        Args:
            hour: Hour of day (0-23)
            weekday: Day of week (0=Mon, 6=Sun)
            top_k: Number of cuisines to return

        Returns:
            List of (cuisine, score) tuples sorted by score descending
        """
        scores = []
        for cuisine, profile in self.profiles.items():
            hour_score = profile.hour_distribution.get(hour, 0)
            day_score = profile.weekday_distribution.get(weekday, 0)
            # Weighted combination
            combined = hour_score * 0.6 + day_score * 0.4
            scores.append((cuisine, combined))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def format_cuisine_info(self, cuisine: str) -> str:
        """Format cuisine info as readable string."""
        profile = self.get_profile(cuisine)
        if not profile:
            return f"No profile for cuisine: {cuisine}"

        lines = [
            f"Cuisine: {cuisine}",
            f"  Total orders: {profile.total_orders}",
            f"  Unique customers: {profile.unique_customers}",
            f"  Peak hours: {profile.peak_hours}",
            f"  Peak weekdays: {[self.DAY_NAMES[d] for d in profile.peak_weekdays if 0 <= d <= 6]}",
            f"  Avg price: ${profile.avg_price:.2f}",
        ]
        return "\n".join(lines)
