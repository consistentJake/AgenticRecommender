"""
Unit tests for dataset processing.
Ensures data integrity and correctness.
"""

import pytest
import random
import numpy as np
from agentic_recommender.datasets import BeautyDataset, DeliveryHeroDataset
from agentic_recommender.utils.metrics import hit_rate_at_k, ndcg_at_k, mrr, evaluate_recommendations


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation"""
        # Perfect prediction
        predictions = [1, 2, 3, 4, 5]
        ground_truth = 1
        assert hit_rate_at_k(predictions, ground_truth, k=5) == 1.0
        assert hit_rate_at_k(predictions, ground_truth, k=1) == 1.0
        
        # Miss
        predictions = [2, 3, 4, 5, 6]
        ground_truth = 1
        assert hit_rate_at_k(predictions, ground_truth, k=5) == 0.0
        
        # Multiple ground truths
        predictions = [1, 2, 3, 4, 5]
        ground_truth = [1, 10]
        assert hit_rate_at_k(predictions, ground_truth, k=5) == 1.0
        
        print("✅ Hit Rate@K tests passed")
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation"""
        # Perfect prediction (rank 1)
        predictions = [1, 2, 3, 4, 5]
        ground_truth = 1
        expected_ndcg = 1.0 / np.log2(2)  # 1/log2(1+1)
        assert abs(ndcg_at_k(predictions, ground_truth, k=5) - expected_ndcg) < 1e-6
        
        # Second position
        predictions = [2, 1, 3, 4, 5]
        ground_truth = 1
        expected_ndcg = 1.0 / np.log2(3)  # 1/log2(2+1)
        assert abs(ndcg_at_k(predictions, ground_truth, k=5) - expected_ndcg) < 1e-6
        
        # Miss
        predictions = [2, 3, 4, 5, 6]
        ground_truth = 1
        assert ndcg_at_k(predictions, ground_truth, k=5) == 0.0
        
        print("✅ NDCG@K tests passed")
    
    def test_mrr(self):
        """Test MRR calculation"""
        # First position
        predictions = [1, 2, 3, 4, 5]
        ground_truth = 1
        assert mrr(predictions, ground_truth) == 1.0
        
        # Second position
        predictions = [2, 1, 3, 4, 5]
        ground_truth = 1
        assert mrr(predictions, ground_truth) == 0.5
        
        # Miss
        predictions = [2, 3, 4, 5, 6]
        ground_truth = 1
        assert mrr(predictions, ground_truth) == 0.0
        
        print("✅ MRR tests passed")
    
    def test_batch_evaluation(self):
        """Test batch evaluation function"""
        # Multiple samples
        predictions_list = [
            [1, 2, 3, 4, 5],  # Hit at rank 1
            [2, 1, 3, 4, 5],  # Hit at rank 2
            [2, 3, 4, 5, 6]   # Miss
        ]
        ground_truths = [1, 1, 1]
        
        metrics = evaluate_recommendations(predictions_list, ground_truths, k_values=[1, 5])
        
        # Check expected values
        assert metrics['hr@1'] == 1/3  # Only first prediction hits@1
        assert metrics['hr@5'] == 2/3  # First two hit@5
        assert 0 < metrics['mrr'] < 1   # Should be between 0 and 1
        
        print("✅ Batch evaluation tests passed")


class TestDatasets:
    """Test dataset processing"""
    
    def test_beauty_dataset_creation(self):
        """Test Beauty dataset creation and processing"""
        # Create dataset - will use real data if available, synthetic otherwise
        dataset = BeautyDataset()  # Uses default paths to real data
        dataset.process_data()
        
        # Check basic properties
        assert len(dataset.sessions) > 0
        assert len(dataset.all_items) > 0
        assert len(dataset.item_to_name) > 0
        
        # Check statistics
        stats = dataset.get_statistics()
        assert stats['num_sessions'] > 0
        assert stats['num_items'] > 0
        assert stats['avg_session_length'] > 0
        
        print(f"✅ Beauty dataset created: {stats['num_sessions']} sessions, {stats['num_items']} items")
    
    def test_delivery_hero_dataset_creation(self):
        """Test Delivery Hero dataset creation"""
        dataset = DeliveryHeroDataset(data_path="dummy_path.csv")
        dataset.min_interactions_per_user = 1  # Relax filter for testing
        dataset.min_interactions_per_item = 1
        dataset.process_data()
        
        # Check basic properties
        assert len(dataset.sessions) > 0
        assert len(dataset.all_items) > 0
        
        # DH characteristics: shorter sessions
        stats = dataset.get_statistics()
        assert stats['avg_session_length'] < 10  # Should be shorter than beauty
        
        print(f"✅ DH dataset created: {stats['num_sessions']} sessions, {stats['num_items']} items")
    
    def test_negative_sampling(self):
        """Test negative sampling correctness"""
        dataset = BeautyDataset(data_path="dummy_path.json")
        dataset.min_interactions_per_user = 1
        dataset.min_interactions_per_item = 1
        dataset.process_data()
        
        # Pick a test session
        test_session = random.choice(dataset.sessions)
        user_id = test_session['user_id']
        user_items = test_session['items']
        
        # Generate negatives
        negatives = dataset.negative_sample(user_id, user_items)
        
        # Check properties
        assert len(negatives) == dataset.n_neg_items
        assert len(set(negatives)) == len(negatives)  # No duplicates
        assert not (set(negatives) & set(user_items))  # No overlap with user items
        
        print("✅ Negative sampling tests passed")
    
    def test_candidate_pool_generation(self):
        """Test candidate pool generation"""
        dataset = BeautyDataset(data_path="dummy_path.json")
        dataset.min_interactions_per_user = 1
        dataset.min_interactions_per_item = 1
        dataset.process_data()
        
        test_session = random.choice(dataset.sessions)
        candidates, target_idx = dataset.create_candidate_pool(test_session)
        
        # Check properties
        assert len(candidates) == dataset.n_neg_items + 1  # Target + negatives
        assert 0 <= target_idx < len(candidates)
        
        # Verify target is correct
        expected_target = dataset.extract_ground_truth(test_session)
        assert candidates[target_idx] == expected_target
        
        print("✅ Candidate pool generation tests passed")
    
    def test_data_integrity(self):
        """Test complete data integrity"""
        # Test Beauty dataset
        beauty_dataset = BeautyDataset(data_path="dummy_path.json")
        beauty_dataset.min_interactions_per_user = 1
        beauty_dataset.min_interactions_per_item = 1
        beauty_dataset.process_data()
        assert beauty_dataset.test_data_integrity()
        
        # Test DH dataset  
        dh_dataset = DeliveryHeroDataset(data_path="dummy_path.csv")
        dh_dataset.min_interactions_per_user = 1
        dh_dataset.min_interactions_per_item = 1
        dh_dataset.process_data()
        assert dh_dataset.test_data_integrity()
        
        print("✅ Data integrity tests passed for both datasets")
    
    def test_evaluation_splits(self):
        """Test train/val/test split creation"""
        dataset = BeautyDataset(data_path="dummy_path.json")
        dataset.min_interactions_per_user = 1
        dataset.min_interactions_per_item = 1
        dataset.process_data()
        
        splits = dataset.create_evaluation_splits()
        
        # Check split properties
        assert 'train' in splits and 'val' in splits and 'test' in splits
        assert len(splits['train']) > len(splits['val'])
        assert len(splits['train']) > len(splits['test'])
        
        # Check no overlap
        train_ids = {s['session_id'] for s in splits['train']}
        val_ids = {s['session_id'] for s in splits['val']}
        test_ids = {s['session_id'] for s in splits['test']}
        
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)
        
        print("✅ Evaluation splits tests passed")


def run_all_tests():
    """Run all dataset and metrics tests"""
    print("🧪 Running all dataset and metrics tests...\n")
    
    # Test metrics
    metrics_test = TestMetrics()
    metrics_test.test_hit_rate_at_k()
    metrics_test.test_ndcg_at_k()
    metrics_test.test_mrr()
    metrics_test.test_batch_evaluation()
    
    # Test datasets
    dataset_test = TestDatasets()
    dataset_test.test_beauty_dataset_creation()
    dataset_test.test_delivery_hero_dataset_creation()
    dataset_test.test_negative_sampling()
    dataset_test.test_candidate_pool_generation()
    dataset_test.test_data_integrity()
    dataset_test.test_evaluation_splits()
    
    print("\n🎉 All tests passed! Dataset processing is ready.")


if __name__ == "__main__":
    run_all_tests()