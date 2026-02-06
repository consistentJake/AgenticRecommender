"""UserProfilerAgent — wraps Swing user-user CF + user record building."""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from .base import RecommendationAgent

USER_RECORDS_CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "repeat_user_records"
USER_LOOKUPS_CACHE_DIR = Path.home() / ".cache" / "agentic_recommender" / "repeat_user_lookups"


def _hash_df(df) -> str:
    """Compute a hash key from DataFrame shape and content summary."""
    key_parts = [
        str(df.shape),
        str(df['customer_id'].nunique()),
        str(df['order_id'].nunique()),
    ]
    if 'vendor_id' in df.columns:
        key_parts.append(str(df['vendor_id'].nunique()))
    return hashlib.md5("_".join(key_parts).encode()).hexdigest()


class UserProfilerAgent(RecommendationAgent):
    """Wraps Swing model + user record building + precomputed lookups."""

    def __init__(self, name: str = "user_profiler"):
        super().__init__(name)
        self.swing = None
        self._user_records: Dict[str, List[Dict]] = {}
        self._user_lookups: Optional[Dict[str, Dict]] = None

    def initialize(
        self,
        train_df,
        swing_config,
        dataset_name: str,
        no_cache: bool = False,
    ) -> None:
        """Train or load Swing user-user model and build user records.

        Extracts from workflow_runner.py Stage 9 Step 4 and
        repeat_evaluator.py _build_user_records.
        """
        from ..similarity.methods import SwingMethod

        # Build item-level interactions: (customer_id, vendor_id||cuisine)
        swing_interactions = []
        for _, row in train_df.iterrows():
            cid = str(row['customer_id'])
            vid = str(row.get('vendor_id', ''))
            cuisine = str(row.get('cuisine', 'unknown'))
            item = f"{vid}||{cuisine}"
            swing_interactions.append((cid, item))

        self._logger.info("Swing interactions: %d", len(swing_interactions))

        swing_model = SwingMethod(swing_config)
        loaded = False
        if not no_cache:
            loaded = swing_model.load_from_cache(dataset_name, "repeat")
        if not loaded:
            self._logger.info("Training new Swing user-user model...")
            swing_model.fit(swing_interactions)
            swing_model.save_to_cache(dataset_name, "repeat")

        self.swing = swing_model
        self._logger.info("Swing stats: %s", swing_model.get_stats())

        # Build user records (with caching)
        self._build_user_records(train_df, dataset_name, use_cache=not no_cache)
        self._initialized = True

    def _build_user_records(self, train_df, dataset_name: str, use_cache: bool = True):
        """Pre-build training records per user for similar user lookups.

        Same logic as AsyncRepeatEvaluator._build_user_records().
        """
        if use_cache and dataset_name:
            df_key = _hash_df(train_df)
            cache_path = USER_RECORDS_CACHE_DIR / f"{dataset_name}_user_records.pkl"
            try:
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    if data.get('cache_key') == df_key:
                        self._user_records = data['user_records']
                        self._logger.info(
                            "Loaded user records from cache: %d users (key=%s...)",
                            len(self._user_records), df_key[:8],
                        )
                        print(
                            f"[UserProfilerAgent] Loaded user records from cache: "
                            f"{len(self._user_records)} users (key={df_key[:8]}...)"
                        )
                        return
            except Exception as e:
                self._logger.warning("Failed to load user records cache: %s", e)

        # Build from scratch
        seen_orders: Dict[str, set] = {}
        if 'day_num' in train_df.columns:
            df_sorted = train_df.sort_values(['customer_id', 'day_num', 'hour'])
        else:
            df_sorted = train_df

        for _, row in df_sorted.iterrows():
            cid = str(row['customer_id'])
            oid = row['order_id']
            if cid not in seen_orders:
                seen_orders[cid] = set()
                self._user_records[cid] = []
            if oid in seen_orders[cid]:
                continue
            seen_orders[cid].add(oid)
            self._user_records[cid].append({
                'vendor_id': str(row.get('vendor_id', '')),
                'cuisine': row.get('cuisine', 'unknown'),
                'day_of_week': int(row.get('day_of_week', 0)),
                'hour': int(row.get('hour', 12)),
            })

        # Save to cache
        if use_cache and dataset_name:
            try:
                USER_RECORDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path = USER_RECORDS_CACHE_DIR / f"{dataset_name}_user_records.pkl"
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'cache_key': _hash_df(train_df),
                        'user_records': self._user_records,
                    }, f)
                self._logger.info("Saved user records cache: %d users", len(self._user_records))
                print(f"[UserProfilerAgent] Saved user records cache: {len(self._user_records)} users")
            except Exception as e:
                self._logger.warning("Failed to save user records cache: %s", e)

    def precompute_lookups(
        self,
        test_samples: List[Dict[str, Any]],
        similarity_agent,
        config,
    ) -> None:
        """Pre-compute LightGCN + Swing lookups for all test users.

        Same logic as AsyncRepeatEvaluator.precompute_user_lookups().
        """
        dataset_name = config.dataset_name
        user_ids = sorted(set(str(s['customer_id']) for s in test_samples))

        # Try loading from cache
        if dataset_name:
            cache_key_parts = [
                "_".join(user_ids),
                str(config.lightgcn_top_k_cuisines),
                str(config.top_similar_users),
                str(similarity_agent.cache_key or ""),
            ]
            cache_key = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest()
            cache_path = USER_LOOKUPS_CACHE_DIR / f"{dataset_name}_user_lookups.pkl"

            try:
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    if data.get('cache_key') == cache_key:
                        self._user_lookups = data['lookups']
                        self._logger.info(
                            "Loaded user lookups from cache: %d users (key=%s...)",
                            len(self._user_lookups), cache_key[:8],
                        )
                        print(
                            f"[UserProfilerAgent] Loaded user lookups from cache: "
                            f"{len(self._user_lookups)} users (key={cache_key[:8]}...)"
                        )
                        return
            except Exception as e:
                self._logger.warning("Failed to load user lookups cache: %s", e)

        # Build from scratch
        print(f"[UserProfilerAgent] Pre-computing lookups for {len(user_ids)} users...")
        lookups = {}
        for i, uid in enumerate(user_ids):
            top_cuisines = similarity_agent.process(uid, top_k=config.lightgcn_top_k_cuisines)
            similar_users = self.swing.get_top_similar_users(uid, top_k=config.top_similar_users)
            sim_records = {
                sim_uid: self._user_records.get(sim_uid, [])
                for sim_uid, _ in similar_users
            }
            lookups[uid] = {
                'lightgcn_top_cuisines': top_cuisines,
                'similar_users': similar_users,
                'similar_users_full_records': sim_records,
            }
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(user_ids)} users")

        self._user_lookups = lookups
        print(f"[UserProfilerAgent] Pre-computed lookups for {len(lookups)} users")

        # Save to cache
        if dataset_name:
            try:
                USER_LOOKUPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'cache_key': cache_key,
                        'lookups': lookups,
                    }, f)
                self._logger.info("Saved user lookups cache: %d users", len(lookups))
                print(f"[UserProfilerAgent] Saved user lookups cache: {len(lookups)} users")
            except Exception as e:
                self._logger.warning("Failed to save user lookups cache: %s", e)

    def get_lookup(self, customer_id: str) -> Optional[Dict]:
        """Get pre-computed lookup for a user."""
        if self._user_lookups:
            return self._user_lookups.get(customer_id)
        return None

    def process(
        self,
        similar_users: List[Tuple[str, float]],
        similar_users_full_records: Dict[str, List[Dict]],
        top_cuisines: List[str],
        config,
    ) -> List[Dict]:
        """Filter pre-fetched similar users' records by predicted cuisines.

        Same logic as AsyncRepeatEvaluator._filter_similar_users_by_cuisines().
        """
        cuisine_set = set(c.lower() for c in top_cuisines)
        result = []
        total_records = 0
        max_total = config.max_total_similar_records

        for sim_uid, sim_score in similar_users:
            records = similar_users_full_records.get(sim_uid, [])
            filtered = [
                r for r in records
                if r.get('cuisine', '').lower() in cuisine_set
            ]
            filtered = filtered[-config.max_records_per_similar_user:]
            remaining = max_total - total_records
            if remaining <= 0:
                break
            if len(filtered) > remaining:
                filtered = filtered[-remaining:]
            total_records += len(filtered)
            result.append({
                'user_id': sim_uid,
                'similarity': sim_score,
                'records': filtered,
            })

        return result

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['user_records_count'] = len(self._user_records)
        stats['user_lookups_count'] = len(self._user_lookups) if self._user_lookups else 0
        if self.swing:
            stats['swing'] = self.swing.get_stats()
        return stats
