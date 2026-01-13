"""
Test Stage 4: Top-K Hit Ratio evaluation.

Run with: pytest agentic_recommender/tests/test_topk_evaluation.py -v
"""

import pytest
from typing import List, Dict, Any

from agentic_recommender.evaluation.topk import (
    TopKMetrics,
    SequentialRecommendationEvaluator,
    TopKTestDataBuilder,
    EvaluationResult,
)
from agentic_recommender.models.llm_provider import MockLLMProvider


class TestTopKMetrics:
    """Test TopKMetrics dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        metrics = TopKMetrics(k=10)

        assert metrics.k == 10
        assert metrics.hit_rate == 0.0
        assert metrics.mrr == 0.0
        assert metrics.total_samples == 0

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = TopKMetrics(
            k=10,
            hit_rate=0.5,
            mrr=0.75,
            hits_at_1=0.3,
            total_samples=100
        )

        d = metrics.to_dict()

        assert 'hit@10' in d
        assert d['mrr'] == 0.75
        assert d['hit@1'] == 0.3
        assert d['total_samples'] == 100

    def test_str_format(self):
        """Should format as readable string."""
        metrics = TopKMetrics(
            k=10,
            hits_at_1=0.3,
            hits_at_5=0.7,
            mrr=0.5,
            total_samples=100,
            valid_samples=95
        )

        s = str(metrics)

        assert "Hit@1:" in s
        assert "Hit@5:" in s
        assert "MRR:" in s
        assert "95/100" in s


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_create_result(self):
        """Should create valid result."""
        result = EvaluationResult(
            sample_idx=0,
            customer_id="user123",
            ground_truth="pizza",
            predictions=[("pizza", 0.9), ("burger", 0.5)],
            rank=1,
            time_ms=100.0
        )

        assert result.customer_id == "user123"
        assert result.rank == 1
        assert len(result.predictions) == 2


class TestTopKTestDataBuilder:
    """Test test data builder."""

    def test_build_samples_returns_list(self, all_orders, all_vendors):
        """Should return list of samples."""
        builder = TopKTestDataBuilder(all_orders.head(50000), all_vendors, min_history=3)
        samples = builder.build_test_samples(n_samples=10)

        assert isinstance(samples, list)
        assert len(samples) <= 10

    def test_sample_has_required_fields(self, all_orders, all_vendors):
        """Each sample should have required fields."""
        builder = TopKTestDataBuilder(all_orders.head(50000), all_vendors, min_history=3)
        samples = builder.build_test_samples(n_samples=5)

        for sample in samples:
            assert 'customer_id' in sample
            assert 'order_history' in sample
            assert 'ground_truth_cuisine' in sample
            assert isinstance(sample['order_history'], list)

    def test_history_excludes_ground_truth_order(self, all_orders, all_vendors):
        """Order history should not include the ground truth order."""
        builder = TopKTestDataBuilder(all_orders.head(50000), all_vendors, min_history=5)
        samples = builder.build_test_samples(n_samples=5)

        for sample in samples:
            # Ground truth is from last order, so history should have fewer orders
            assert len(sample['order_history']) >= builder.min_history - 1

    def test_get_unique_cuisines(self, all_orders, all_vendors):
        """Should return list of unique cuisines."""
        builder = TopKTestDataBuilder(all_orders.head(10000), all_vendors)
        cuisines = builder.get_unique_cuisines()

        assert isinstance(cuisines, list)
        assert len(cuisines) > 0

    def test_reproducible_with_seed(self, all_orders, all_vendors):
        """Same seed should produce same samples."""
        builder = TopKTestDataBuilder(all_orders.head(50000), all_vendors, min_history=3)

        samples1 = builder.build_test_samples(n_samples=10, seed=42)
        samples2 = builder.build_test_samples(n_samples=10, seed=42)

        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            assert s1['customer_id'] == s2['customer_id']


class TestSequentialRecommendationEvaluator:
    """Test the evaluator."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM that returns predictable responses."""
        responses = {
            "predictions": '{"predictions": [{"cuisine": "chinese", "confidence": 0.9}, {"cuisine": "pizza", "confidence": 0.8}]}'
        }
        return MockLLMProvider(responses=responses)

    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data."""
        return [
            {
                'customer_id': 'user1',
                'order_history': [
                    {'cuisine': 'chinese', 'day_of_week': 0, 'hour': 12, 'price': 10.0},
                    {'cuisine': 'chinese', 'day_of_week': 2, 'hour': 19, 'price': 15.0},
                    {'cuisine': 'pizza', 'day_of_week': 4, 'hour': 20, 'price': 20.0},
                ],
                'ground_truth_cuisine': 'chinese',
            },
            {
                'customer_id': 'user2',
                'order_history': [
                    {'cuisine': 'burger', 'day_of_week': 1, 'hour': 13, 'price': 8.0},
                    {'cuisine': 'pizza', 'day_of_week': 3, 'hour': 18, 'price': 12.0},
                ],
                'ground_truth_cuisine': 'burger',
            },
        ]

    def test_evaluate_returns_metrics(self, mock_llm, sample_test_data):
        """Evaluate should return TopKMetrics."""
        evaluator = SequentialRecommendationEvaluator(
            llm_provider=mock_llm,
            cuisine_list=['chinese', 'pizza', 'burger', 'indian'],
        )

        metrics = evaluator.evaluate(sample_test_data)

        assert isinstance(metrics, TopKMetrics)
        assert metrics.total_samples == 2

    def test_find_rank_found(self):
        """Should find correct rank when cuisine is in predictions."""
        evaluator = SequentialRecommendationEvaluator(
            llm_provider=MockLLMProvider(),
            cuisine_list=['chinese', 'pizza']
        )

        predictions = [('pizza', 0.9), ('chinese', 0.8), ('burger', 0.5)]
        rank = evaluator._find_rank(predictions, 'chinese')

        assert rank == 2  # 1-indexed

    def test_find_rank_not_found(self):
        """Should return 0 when cuisine not in predictions."""
        evaluator = SequentialRecommendationEvaluator(
            llm_provider=MockLLMProvider(),
            cuisine_list=['chinese', 'pizza']
        )

        predictions = [('pizza', 0.9), ('burger', 0.8)]
        rank = evaluator._find_rank(predictions, 'indian')

        assert rank == 0

    def test_find_rank_case_insensitive(self):
        """Should match case-insensitively."""
        evaluator = SequentialRecommendationEvaluator(
            llm_provider=MockLLMProvider(),
            cuisine_list=['chinese', 'pizza']
        )

        predictions = [('PIZZA', 0.9), ('Chinese', 0.8)]
        rank = evaluator._find_rank(predictions, 'chinese')

        assert rank == 2

    def test_parse_predictions_valid_json(self):
        """Should parse valid JSON response."""
        evaluator = SequentialRecommendationEvaluator(
            llm_provider=MockLLMProvider(),
            cuisine_list=['chinese', 'pizza']
        )

        response = '{"predictions": [{"cuisine": "pizza", "confidence": 0.9}, {"cuisine": "chinese", "confidence": 0.7}]}'
        predictions = evaluator._parse_predictions(response, k=5)

        assert len(predictions) == 2
        assert predictions[0][0] == 'pizza'
        assert predictions[0][1] == 0.9

    def test_parse_predictions_with_code_block(self):
        """Should handle markdown code blocks."""
        evaluator = SequentialRecommendationEvaluator(
            llm_provider=MockLLMProvider(),
            cuisine_list=['chinese', 'pizza']
        )

        response = '```json\n{"predictions": [{"cuisine": "pizza", "confidence": 0.9}]}\n```'
        predictions = evaluator._parse_predictions(response, k=5)

        assert len(predictions) == 1
        assert predictions[0][0] == 'pizza'

    def test_compute_metrics_hit_at_1(self):
        """Should compute hit@1 correctly."""
        evaluator = SequentialRecommendationEvaluator(
            llm_provider=MockLLMProvider(),
            cuisine_list=['chinese', 'pizza']
        )

        results = [
            EvaluationResult(0, 'u1', 'pizza', [('pizza', 0.9)], rank=1, time_ms=100),
            EvaluationResult(1, 'u2', 'chinese', [('pizza', 0.9)], rank=0, time_ms=100),
        ]

        metrics = evaluator._compute_metrics(results)

        # 1 hit out of 2
        assert metrics.hits_at_1 == 0.5

    def test_compute_metrics_mrr(self):
        """Should compute MRR correctly."""
        evaluator = SequentialRecommendationEvaluator(
            llm_provider=MockLLMProvider(),
            cuisine_list=['chinese', 'pizza']
        )

        results = [
            EvaluationResult(0, 'u1', 'pizza', [('pizza', 0.9)], rank=1, time_ms=100),  # 1/1 = 1.0
            EvaluationResult(1, 'u2', 'chinese', [('a', 0.9), ('chinese', 0.8)], rank=2, time_ms=100),  # 1/2 = 0.5
        ]

        metrics = evaluator._compute_metrics(results)

        expected_mrr = (1.0 + 0.5) / 2
        assert abs(metrics.mrr - expected_mrr) < 0.01


class TestMetricsComputation:
    """Test metrics computation formulas."""

    def test_hit_rate_all_hits(self):
        """Hit rate should be 1.0 when all predictions are correct."""
        results = [
            EvaluationResult(0, 'u1', 'a', [('a', 0.9)], rank=1, time_ms=100),
            EvaluationResult(1, 'u2', 'b', [('b', 0.9)], rank=1, time_ms=100),
        ]

        evaluator = SequentialRecommendationEvaluator(MockLLMProvider(), [])
        metrics = evaluator._compute_metrics(results)

        assert metrics.hits_at_1 == 1.0

    def test_hit_rate_no_hits(self):
        """Hit rate should be 0.0 when no predictions are correct."""
        results = [
            EvaluationResult(0, 'u1', 'a', [('b', 0.9)], rank=0, time_ms=100),
            EvaluationResult(1, 'u2', 'c', [('d', 0.9)], rank=0, time_ms=100),
        ]

        evaluator = SequentialRecommendationEvaluator(MockLLMProvider(), [])
        metrics = evaluator._compute_metrics(results)

        assert metrics.hits_at_1 == 0.0
        assert metrics.mrr == 0.0

    def test_hit_at_k_increasing(self):
        """Hit@K should increase or stay same as K increases."""
        results = [
            EvaluationResult(0, 'u1', 'a', [('x', 0.9), ('a', 0.8)], rank=2, time_ms=100),
            EvaluationResult(1, 'u2', 'b', [('x', 0.9), ('y', 0.8), ('z', 0.7), ('w', 0.6), ('b', 0.5)], rank=5, time_ms=100),
        ]

        evaluator = SequentialRecommendationEvaluator(MockLLMProvider(), [])
        metrics = evaluator._compute_metrics(results)

        assert metrics.hits_at_1 <= metrics.hits_at_3
        assert metrics.hits_at_3 <= metrics.hits_at_5
        assert metrics.hits_at_5 <= metrics.hits_at_10


# Validation function
def validate_topk_evaluation():
    """Validate Top-K evaluation with real data."""
    import pandas as pd
    from pathlib import Path
    from agentic_recommender.models.llm_provider import OpenRouterProvider

    print("=" * 60)
    print("VALIDATION: Top-K Evaluation")
    print("=" * 60)

    # Load data
    data_dir = Path("/Users/zhenkai/Downloads/data_sg")
    orders = pd.read_csv(data_dir / "orders_sg_train.txt", nrows=100000)
    vendors = pd.read_csv(data_dir / "vendors_sg.txt")

    print(f"\nLoaded {len(orders):,} orders")

    # Build test samples
    builder = TopKTestDataBuilder(orders, vendors, min_history=5)
    test_samples = builder.build_test_samples(n_samples=5)
    cuisines = builder.get_unique_cuisines()

    print(f"Built {len(test_samples)} test samples")
    print(f"Unique cuisines: {len(cuisines)}")

    # Show sample
    if test_samples:
        sample = test_samples[0]
        print(f"\n--- Sample Test Case ---")
        print(f"Customer: {sample['customer_id']}")
        print(f"History ({len(sample['order_history'])} orders):")
        for order in sample['order_history'][-3:]:
            print(f"  - {order['cuisine']} (day={order['day_of_week']}, hour={order['hour']})")
        print(f"Ground truth: {sample['ground_truth_cuisine']}")

    # Run mini evaluation with real LLM
    print("\n--- Running Mini Evaluation (3 samples) ---")

    API_KEY = "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"

    try:
        llm = OpenRouterProvider(api_key=API_KEY)

        evaluator = SequentialRecommendationEvaluator(
            llm_provider=llm,
            cuisine_list=cuisines[:30],
        )

        metrics = evaluator.evaluate(test_samples[:3], verbose=True)

        print("\n--- Results ---")
        print(metrics)

    except Exception as e:
        print(f"Error during evaluation: {e}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    validate_topk_evaluation()
