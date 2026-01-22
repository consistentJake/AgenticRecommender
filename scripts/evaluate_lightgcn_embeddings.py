#!/usr/bin/env python3
"""
LightGCN Embedding Evaluation Script

This script validates LightGCN embeddings by checking if a user's top-ranked
cuisines (by embedding similarity) match their actual purchase history.

Expected behavior:
- Top-ranked cuisines should mostly be cuisines the user has purchased
- Some new cuisines in top-K are fine (these are recommendations)
- Low overlap would indicate poor embeddings
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


def load_embeddings(cache_path: str = None) -> dict:
    """Load cached LightGCN embeddings."""
    if cache_path is None:
        cache_path = Path.home() / ".cache/agentic_recommender/lightgcn/data_se_embeddings.pkl"

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"Loaded embeddings from: {cache_path}")
    print(f"  Users: {cache['user_embeddings'].shape}")
    print(f"  Cuisines: {cache['cuisine_embeddings'].shape}")

    return cache


def load_user_history(parquet_path: str = "outputs/stage1_merged_data.parquet") -> pd.DataFrame:
    """Load user purchase history."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    return df


def get_user_cuisine_history(df: pd.DataFrame, customer_id: str) -> dict:
    """Get a user's cuisine purchase history with counts."""
    user_df = df[df['customer_id'] == customer_id]

    # Count orders per cuisine (not items, but unique orders)
    cuisine_counts = user_df.groupby('cuisine')['order_id'].nunique().to_dict()

    return {
        'customer_id': customer_id,
        'total_orders': user_df['order_id'].nunique(),
        'total_items': len(user_df),
        'cuisine_counts': cuisine_counts,
        'cuisines': list(cuisine_counts.keys())
    }


def calculate_user_cuisine_scores(cache: dict, customer_id: str) -> dict:
    """Calculate LightGCN similarity scores between a user and all cuisines."""
    user_to_idx = cache['user_to_idx']
    idx_to_cuisine = cache['idx_to_cuisine']
    user_embeddings = cache['user_embeddings']
    cuisine_embeddings = cache['cuisine_embeddings']

    if customer_id not in user_to_idx:
        return None

    user_idx = user_to_idx[customer_id]
    user_emb = user_embeddings[user_idx]  # (64,)

    # Calculate dot product similarity with all cuisines
    scores = np.dot(cuisine_embeddings, user_emb)  # (39,)

    # Create cuisine -> score mapping
    cuisine_scores = {}
    for idx, score in enumerate(scores):
        cuisine = idx_to_cuisine[idx]
        cuisine_scores[cuisine] = float(score)

    # Sort by score descending
    sorted_cuisines = sorted(cuisine_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        'user_idx': user_idx,
        'scores': cuisine_scores,
        'ranking': [c[0] for c in sorted_cuisines],
        'sorted_scores': sorted_cuisines
    }


def evaluate_user(cache: dict, df: pd.DataFrame, customer_id: str, top_k: int = 10):
    """Evaluate LightGCN embeddings for a single user."""
    print(f"\n{'='*70}")
    print(f"EVALUATING USER: {customer_id}")
    print(f"{'='*70}")

    # Get user history
    history = get_user_cuisine_history(df, customer_id)
    print(f"\n📋 PURCHASE HISTORY:")
    print(f"   Total orders: {history['total_orders']}")
    print(f"   Total items: {history['total_items']}")
    print(f"   Unique cuisines: {len(history['cuisines'])}")
    print(f"\n   Cuisine breakdown (by order count):")
    for cuisine, count in sorted(history['cuisine_counts'].items(), key=lambda x: x[1], reverse=True):
        pct = count / history['total_orders'] * 100
        print(f"      {cuisine}: {count} orders ({pct:.1f}%)")

    # Calculate LightGCN scores
    lgcn_result = calculate_user_cuisine_scores(cache, customer_id)
    if lgcn_result is None:
        print(f"\n❌ User {customer_id} not found in LightGCN embeddings")
        return None

    print(f"\n🔮 LIGHTGCN TOP-{top_k} PREDICTIONS:")
    for i, (cuisine, score) in enumerate(lgcn_result['sorted_scores'][:top_k], 1):
        in_history = "✓ (purchased)" if cuisine in history['cuisines'] else "  (new)"
        print(f"   {i:2d}. {cuisine}: {score:.4f} {in_history}")

    # Calculate metrics
    top_k_cuisines = set(lgcn_result['ranking'][:top_k])
    purchased_cuisines = set(history['cuisines'])

    # Recall@K: What fraction of purchased cuisines appear in top-K?
    overlap = top_k_cuisines & purchased_cuisines
    recall_at_k = len(overlap) / len(purchased_cuisines) if purchased_cuisines else 0

    # Precision@K: What fraction of top-K are actually purchased?
    precision_at_k = len(overlap) / top_k if top_k > 0 else 0

    # Check where each purchased cuisine ranks
    cuisine_ranks = {}
    for cuisine in purchased_cuisines:
        if cuisine in lgcn_result['ranking']:
            cuisine_ranks[cuisine] = lgcn_result['ranking'].index(cuisine) + 1
        else:
            cuisine_ranks[cuisine] = -1

    print(f"\n📊 EVALUATION METRICS (Top-{top_k}):")
    print(f"   Recall@{top_k}: {recall_at_k:.2%} ({len(overlap)}/{len(purchased_cuisines)} purchased cuisines in top-{top_k})")
    print(f"   Precision@{top_k}: {precision_at_k:.2%} ({len(overlap)}/{top_k} of top-{top_k} are purchased)")

    print(f"\n   Rank of each purchased cuisine:")
    for cuisine, rank in sorted(cuisine_ranks.items(), key=lambda x: x[1] if x[1] > 0 else 999):
        count = history['cuisine_counts'][cuisine]
        if rank > 0:
            status = f"rank {rank}" + (" ✓" if rank <= top_k else "")
        else:
            status = "not found ❌"
        print(f"      {cuisine} ({count} orders): {status}")

    return {
        'customer_id': customer_id,
        'history': history,
        'lgcn_ranking': lgcn_result['ranking'],
        'recall_at_k': recall_at_k,
        'precision_at_k': precision_at_k,
        'cuisine_ranks': cuisine_ranks
    }


def evaluate_multiple_users(cache: dict, df: pd.DataFrame, n_users: int = 5, top_k: int = 10, min_orders: int = 5):
    """Evaluate LightGCN embeddings for multiple random users."""
    # Get users with enough orders
    user_order_counts = df.groupby('customer_id')['order_id'].nunique()
    eligible_users = user_order_counts[user_order_counts >= min_orders].index.tolist()

    # Filter to users in embeddings
    users_in_cache = set(cache['user_to_idx'].keys())
    eligible_users = [u for u in eligible_users if u in users_in_cache]

    print(f"\n{'#'*70}")
    print(f"# LIGHTGCN EMBEDDING EVALUATION")
    print(f"# Eligible users: {len(eligible_users)} (min {min_orders} orders)")
    print(f"# Evaluating: {n_users} random users")
    print(f"# Top-K: {top_k}")
    print(f"{'#'*70}")

    # Sample random users
    np.random.seed(42)
    sample_users = np.random.choice(eligible_users, size=min(n_users, len(eligible_users)), replace=False)

    results = []
    for user_id in sample_users:
        result = evaluate_user(cache, df, user_id, top_k=top_k)
        if result:
            results.append(result)

    # Aggregate metrics
    if results:
        avg_recall = np.mean([r['recall_at_k'] for r in results])
        avg_precision = np.mean([r['precision_at_k'] for r in results])

        print(f"\n{'='*70}")
        print(f"AGGREGATE RESULTS ({len(results)} users)")
        print(f"{'='*70}")
        print(f"   Average Recall@{top_k}: {avg_recall:.2%}")
        print(f"   Average Precision@{top_k}: {avg_precision:.2%}")

        # Show distribution of ranks for purchased cuisines
        all_ranks = []
        for r in results:
            all_ranks.extend([rank for rank in r['cuisine_ranks'].values() if rank > 0])

        if all_ranks:
            print(f"\n   Rank distribution for purchased cuisines:")
            print(f"      Mean rank: {np.mean(all_ranks):.1f}")
            print(f"      Median rank: {np.median(all_ranks):.1f}")
            print(f"      Min rank: {min(all_ranks)}")
            print(f"      Max rank: {max(all_ranks)}")

            # Count how many are in top-K
            in_top_k = sum(1 for r in all_ranks if r <= top_k)
            print(f"      In top-{top_k}: {in_top_k}/{len(all_ranks)} ({in_top_k/len(all_ranks):.1%})")

    return results


def main():
    """Main evaluation function."""
    # Load data
    cache = load_embeddings()
    df = load_user_history()

    # Run evaluation
    results = evaluate_multiple_users(
        cache=cache,
        df=df,
        n_users=5,
        top_k=10,
        min_orders=5
    )

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
    Good LightGCN embeddings should show:
    ✓ High Recall@K: Most purchased cuisines appear in top-K predictions
    ✓ Reasonable Precision@K: Top-K contains purchased cuisines (not all random)
    ✓ Low mean/median rank for purchased cuisines

    What's acceptable:
    • Some new cuisines in top-K is GOOD (these are recommendations!)
    • Not all purchased cuisines need to be in top-K
    • Very popular cuisines might rank high even if not purchased (popularity bias)

    Warning signs:
    ✗ Recall@10 < 30%: Model not learning user preferences
    ✗ All users get same top cuisines: Model collapsed to popularity
    """)


if __name__ == "__main__":
    main()
