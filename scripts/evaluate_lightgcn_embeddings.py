#!/usr/bin/env python3
"""
LightGCN Embedding Evaluation Script

This script validates LightGCN embeddings by checking if a user's top-ranked
cuisines (by embedding similarity) match their actual purchase history.

Supports two evaluation methods:
- Method 1 (Leave-Last-Out): LightGCN trained on N-1 orders, evaluates against last order
- Method 2 (Train-Test Split): LightGCN trained on all training data, evaluates against test file

Expected behavior:
- Top-ranked cuisines should mostly be cuisines the user has purchased
- Some new cuisines in top-K are fine (these are recommendations)
- Low overlap would indicate poor embeddings
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def get_cache_path(method: str, dataset_name: str = "data_se") -> Path:
    """Get the cache path for the specified evaluation method."""
    cache_dir = Path.home() / ".cache/agentic_recommender/lightgcn"
    if method == "method1":
        return cache_dir / f"{dataset_name}_method1_lightgcn.pkl"
    elif method == "method2":
        return cache_dir / f"{dataset_name}_method2_lightgcn.pkl"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'method1' or 'method2'")


def load_embeddings(method: str, dataset_name: str = "data_se") -> dict:
    """Load cached LightGCN embeddings for the specified method."""
    cache_path = get_cache_path(method, dataset_name)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Run the evaluation pipeline with {method} first to generate embeddings."
        )

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"Loaded embeddings from: {cache_path}")
    print(f"  Method: {method}")
    print(f"  Users: {cache['user_embeddings'].shape}")
    print(f"  Cuisines: {cache['cuisine_embeddings'].shape}")

    return cache


def load_user_history(parquet_path: str = "outputs/stage1_merged_data.parquet") -> pd.DataFrame:
    """Load user purchase history."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    return df


def load_test_data(parquet_path: str = "outputs/stage1_test_data.parquet") -> pd.DataFrame:
    """Load test data for Method 2."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Test data not found: {parquet_path}\n"
            "Run 'load_data' stage with 'load_test_data: true' first."
        )
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} test rows from {parquet_path}")
    return df


def get_user_cuisine_history(df: pd.DataFrame, customer_id: str,
                              exclude_last_order: bool = False) -> dict:
    """
    Get a user's cuisine purchase history with counts.

    Args:
        df: DataFrame with order history
        customer_id: Customer ID to look up
        exclude_last_order: If True, exclude the last order (for Method 1)
    """
    user_df = df[df['customer_id'] == customer_id].copy()

    if len(user_df) == 0:
        return None

    if exclude_last_order:
        # Get the last order ID (by order_time or order_id)
        if 'order_time' in user_df.columns:
            user_df['order_time'] = pd.to_datetime(user_df['order_time'])
            last_order_id = user_df.groupby('order_id')['order_time'].max().idxmax()
        else:
            # Fallback: use order_id (assuming they're sortable)
            last_order_id = user_df['order_id'].unique()[-1]

        # Exclude last order
        user_df = user_df[user_df['order_id'] != last_order_id]

        if len(user_df) == 0:
            return None

    # Count orders per cuisine (not items, but unique orders)
    cuisine_counts = user_df.groupby('cuisine')['order_id'].nunique().to_dict()

    return {
        'customer_id': customer_id,
        'total_orders': user_df['order_id'].nunique(),
        'total_items': len(user_df),
        'cuisine_counts': cuisine_counts,
        'cuisines': list(cuisine_counts.keys())
    }


def get_user_last_order_cuisines(df: pd.DataFrame, customer_id: str) -> set:
    """Get the cuisines from a user's last order (for Method 1 evaluation)."""
    user_df = df[df['customer_id'] == customer_id].copy()

    if len(user_df) == 0:
        return set()

    # Get the last order ID
    if 'order_time' in user_df.columns:
        user_df['order_time'] = pd.to_datetime(user_df['order_time'])
        last_order_id = user_df.groupby('order_id')['order_time'].max().idxmax()
    else:
        last_order_id = user_df['order_id'].unique()[-1]

    # Get cuisines from last order
    last_order_df = user_df[user_df['order_id'] == last_order_id]
    return set(last_order_df['cuisine'].unique())


def get_user_test_cuisines(test_df: pd.DataFrame, customer_id: str) -> set:
    """Get cuisines from test data for a user (for Method 2 evaluation)."""
    user_test = test_df[test_df['customer_id'] == customer_id]
    return set(user_test['cuisine'].unique())


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


def evaluate_user(cache: dict, df: pd.DataFrame, customer_id: str,
                  method: str, test_df: pd.DataFrame = None, top_k: int = 10):
    """
    Evaluate LightGCN embeddings for a single user.

    Args:
        cache: LightGCN cache with embeddings
        df: Training data DataFrame
        customer_id: Customer to evaluate
        method: 'method1' or 'method2'
        test_df: Test data DataFrame (required for method2)
        top_k: Number of top predictions to evaluate
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING USER: {customer_id} (Method: {method})")
    print(f"{'='*70}")

    # Get user training history (what LightGCN was trained on)
    exclude_last = (method == "method1")
    history = get_user_cuisine_history(df, customer_id, exclude_last_order=exclude_last)

    if history is None:
        print(f"User {customer_id} has no training history")
        return None

    print(f"\n[Training History] (what LightGCN learned from):")
    print(f"   Total orders: {history['total_orders']}")
    print(f"   Total items: {history['total_items']}")
    print(f"   Unique cuisines: {len(history['cuisines'])}")
    print(f"\n   Cuisine breakdown (by order count):")
    for cuisine, count in sorted(history['cuisine_counts'].items(), key=lambda x: x[1], reverse=True):
        pct = count / history['total_orders'] * 100
        print(f"      {cuisine}: {count} orders ({pct:.1f}%)")

    # Get ground truth (what we're evaluating against)
    if method == "method1":
        ground_truth = get_user_last_order_cuisines(df, customer_id)
        print(f"\n[Ground Truth] Last order cuisines: {ground_truth}")
    else:  # method2
        if test_df is None:
            print("Error: test_df required for method2")
            return None
        ground_truth = get_user_test_cuisines(test_df, customer_id)
        if not ground_truth:
            print(f"User {customer_id} not in test data (cold-start in test)")
            return None
        print(f"\n[Ground Truth] Test data cuisines: {ground_truth}")

    # Calculate LightGCN scores
    lgcn_result = calculate_user_cuisine_scores(cache, customer_id)
    if lgcn_result is None:
        print(f"\nUser {customer_id} not found in LightGCN embeddings")
        return None

    print(f"\n[LightGCN TOP-{top_k} PREDICTIONS]:")
    for i, (cuisine, score) in enumerate(lgcn_result['sorted_scores'][:top_k], 1):
        in_training = "(trained)" if cuisine in history['cuisines'] else ""
        in_ground_truth = "* GT" if cuisine in ground_truth else ""
        print(f"   {i:2d}. {cuisine}: {score:.4f} {in_training} {in_ground_truth}")

    # Calculate metrics
    top_k_cuisines = set(lgcn_result['ranking'][:top_k])

    # Metrics against training history (sanity check)
    training_cuisines = set(history['cuisines'])
    training_overlap = top_k_cuisines & training_cuisines
    training_recall = len(training_overlap) / len(training_cuisines) if training_cuisines else 0
    training_precision = len(training_overlap) / top_k if top_k > 0 else 0

    # Metrics against ground truth (actual evaluation)
    gt_overlap = top_k_cuisines & ground_truth
    gt_recall = len(gt_overlap) / len(ground_truth) if ground_truth else 0
    gt_precision = len(gt_overlap) / top_k if top_k > 0 else 0
    hit_at_k = len(gt_overlap) > 0

    # Check where each ground truth cuisine ranks
    gt_ranks = {}
    for cuisine in ground_truth:
        if cuisine in lgcn_result['ranking']:
            gt_ranks[cuisine] = lgcn_result['ranking'].index(cuisine) + 1
        else:
            gt_ranks[cuisine] = -1

    print(f"\n[EVALUATION METRICS] (Top-{top_k}):")
    print(f"\n   Against Training History (sanity check):")
    print(f"      Recall@{top_k}: {training_recall:.2%} ({len(training_overlap)}/{len(training_cuisines)})")
    print(f"      Precision@{top_k}: {training_precision:.2%}")

    print(f"\n   Against Ground Truth (actual evaluation):")
    print(f"      Hit@{top_k}: {'Yes' if hit_at_k else 'No'}")
    print(f"      Recall@{top_k}: {gt_recall:.2%} ({len(gt_overlap)}/{len(ground_truth)})")
    print(f"      Precision@{top_k}: {gt_precision:.2%}")

    print(f"\n   Rank of each ground truth cuisine:")
    for cuisine, rank in sorted(gt_ranks.items(), key=lambda x: x[1] if x[1] > 0 else 999):
        in_training = "(trained)" if cuisine in history['cuisines'] else "(new)"
        if rank > 0:
            status = f"rank {rank}" + (" *" if rank <= top_k else "")
        else:
            status = "not found"
        print(f"      {cuisine} {in_training}: {status}")

    return {
        'customer_id': customer_id,
        'method': method,
        'history': history,
        'ground_truth': ground_truth,
        'lgcn_ranking': lgcn_result['ranking'],
        'training_recall_at_k': training_recall,
        'training_precision_at_k': training_precision,
        'gt_recall_at_k': gt_recall,
        'gt_precision_at_k': gt_precision,
        'hit_at_k': hit_at_k,
        'gt_ranks': gt_ranks
    }


def evaluate_multiple_users(cache: dict, df: pd.DataFrame, method: str,
                            test_df: pd.DataFrame = None,
                            n_users: int = 5, top_k: int = 10, min_orders: int = 5):
    """Evaluate LightGCN embeddings for multiple random users."""
    # Get users with enough orders in training data
    user_order_counts = df.groupby('customer_id')['order_id'].nunique()
    eligible_users = user_order_counts[user_order_counts >= min_orders].index.tolist()

    # Filter to users in embeddings
    users_in_cache = set(cache['user_to_idx'].keys())
    eligible_users = [u for u in eligible_users if u in users_in_cache]

    # For method2, also filter to users in test data
    if method == "method2" and test_df is not None:
        test_users = set(test_df['customer_id'].unique())
        eligible_users = [u for u in eligible_users if u in test_users]

    print(f"\n{'#'*70}")
    print(f"# LIGHTGCN EMBEDDING EVALUATION")
    print(f"# Method: {method}")
    print(f"# Eligible users: {len(eligible_users)} (min {min_orders} orders)")
    print(f"# Evaluating: {n_users} random users")
    print(f"# Top-K: {top_k}")
    print(f"{'#'*70}")

    if len(eligible_users) == 0:
        print("\nNo eligible users found!")
        return []

    # Sample random users
    np.random.seed(42)
    sample_users = np.random.choice(eligible_users, size=min(n_users, len(eligible_users)), replace=False)

    results = []
    for user_id in sample_users:
        result = evaluate_user(cache, df, user_id, method=method, test_df=test_df, top_k=top_k)
        if result:
            results.append(result)

    # Aggregate metrics
    if results:
        avg_training_recall = np.mean([r['training_recall_at_k'] for r in results])
        avg_training_precision = np.mean([r['training_precision_at_k'] for r in results])
        avg_gt_recall = np.mean([r['gt_recall_at_k'] for r in results])
        avg_gt_precision = np.mean([r['gt_precision_at_k'] for r in results])
        hit_rate = np.mean([1 if r['hit_at_k'] else 0 for r in results])

        print(f"\n{'='*70}")
        print(f"AGGREGATE RESULTS ({len(results)} users)")
        print(f"{'='*70}")

        print(f"\n   Against Training History (sanity check):")
        print(f"      Average Recall@{top_k}: {avg_training_recall:.2%}")
        print(f"      Average Precision@{top_k}: {avg_training_precision:.2%}")

        print(f"\n   Against Ground Truth (actual evaluation):")
        print(f"      Hit Rate@{top_k}: {hit_rate:.2%}")
        print(f"      Average Recall@{top_k}: {avg_gt_recall:.2%}")
        print(f"      Average Precision@{top_k}: {avg_gt_precision:.2%}")

        # Show distribution of ranks for ground truth cuisines
        all_ranks = []
        for r in results:
            all_ranks.extend([rank for rank in r['gt_ranks'].values() if rank > 0])

        if all_ranks:
            print(f"\n   Rank distribution for ground truth cuisines:")
            print(f"      Mean rank: {np.mean(all_ranks):.1f}")
            print(f"      Median rank: {np.median(all_ranks):.1f}")
            print(f"      Min rank: {min(all_ranks)}")
            print(f"      Max rank: {max(all_ranks)}")

            # Count how many are in top-K
            in_top_k = sum(1 for r in all_ranks if r <= top_k)
            print(f"      In top-{top_k}: {in_top_k}/{len(all_ranks)} ({in_top_k/len(all_ranks):.1%})")

    return results


def main():
    """Main evaluation function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate LightGCN embeddings for Method 1 or Method 2"
    )
    parser.add_argument(
        "--method",
        choices=["method1", "method2", "both"],
        default="both",
        help="Evaluation method: method1 (leave-last-out), method2 (train-test split), or both"
    )
    parser.add_argument(
        "--dataset",
        default="data_se",
        help="Dataset name (default: data_se)"
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=5,
        help="Number of users to evaluate (default: 5)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K for evaluation (default: 10)"
    )
    parser.add_argument(
        "--min-orders",
        type=int,
        default=5,
        help="Minimum orders per user (default: 5)"
    )
    parser.add_argument(
        "--train-data",
        default="outputs/stage1_merged_data.parquet",
        help="Path to training data parquet"
    )
    parser.add_argument(
        "--test-data",
        default="outputs/stage1_test_data.parquet",
        help="Path to test data parquet (for method2)"
    )

    args = parser.parse_args()

    # Load training data
    df = load_user_history(args.train_data)

    # Load test data for method2
    test_df = None
    if args.method in ["method2", "both"]:
        try:
            test_df = load_test_data(args.test_data)
        except FileNotFoundError as e:
            if args.method == "method2":
                print(f"Error: {e}")
                return
            else:
                print(f"Warning: {e}")
                print("Will only run method1.\n")

    methods_to_run = []
    if args.method == "both":
        methods_to_run = ["method1", "method2"] if test_df is not None else ["method1"]
    else:
        methods_to_run = [args.method]

    all_results = {}

    for method in methods_to_run:
        print(f"\n{'*'*70}")
        print(f"* RUNNING {method.upper()}")
        print(f"{'*'*70}")

        try:
            cache = load_embeddings(method, args.dataset)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        results = evaluate_multiple_users(
            cache=cache,
            df=df,
            method=method,
            test_df=test_df if method == "method2" else None,
            n_users=args.n_users,
            top_k=args.top_k,
            min_orders=args.min_orders
        )
        all_results[method] = results

    # Print comparison if both methods were run
    if len(all_results) == 2 and all(all_results.values()):
        print(f"\n{'='*70}")
        print("METHOD COMPARISON SUMMARY")
        print(f"{'='*70}")

        for method, results in all_results.items():
            if results:
                avg_gt_recall = np.mean([r['gt_recall_at_k'] for r in results])
                hit_rate = np.mean([1 if r['hit_at_k'] else 0 for r in results])
                print(f"\n   {method}:")
                print(f"      Hit Rate@{args.top_k}: {hit_rate:.2%}")
                print(f"      GT Recall@{args.top_k}: {avg_gt_recall:.2%}")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
    Method 1 (Leave-Last-Out):
    - LightGCN trained on N-1 orders, evaluated against last order
    - Tests if model can predict held-out order from same user

    Method 2 (Train-Test Split):
    - LightGCN trained on ALL training data, evaluated against test file
    - Tests if model generalizes to future orders

    Good LightGCN embeddings should show:
    * High Recall@K against training history (sanity check)
    * Reasonable Hit Rate and Recall@K against ground truth
    * Low mean/median rank for ground truth cuisines

    What's acceptable:
    - Some new cuisines in top-K is GOOD (these are recommendations!)
    - Ground truth cuisines not in training are harder to predict
    - Method 2 may show lower metrics (harder task)

    Warning signs:
    * Recall@10 < 30% against training: Model not learning preferences
    * All users get same top cuisines: Model collapsed to popularity
    """)


if __name__ == "__main__":
    main()
