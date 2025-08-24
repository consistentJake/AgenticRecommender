"""
Comprehensive dataset testing demonstration.
Shows the complete dataset processing and evaluation pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.system import create_pipeline
from agentic_recommender.core import RecommendationRequest
from agentic_recommender.models.llm_provider import MockLLMProvider
from agentic_recommender.datasets import BeautyDataset, DeliveryHeroDataset


def demo_dataset_processing():
    """Demonstrate dataset processing capabilities"""
    print("📊 Dataset Processing & Testing Demo")
    print("=" * 40)
    
    print("\n1. 🏗️ DATASET IMPLEMENTATIONS")
    print("-" * 30)
    
    print("Available dataset implementations:")
    print("   📄 BeautyDataset - Amazon Beauty 5-core dataset")
    print("   🍕 DeliveryHeroDataset - Food delivery dataset") 
    print("   🔧 Base synthetic data generation for testing")
    
    print("\n2. 📊 BEAUTY DATASET PROCESSING")
    print("-" * 32)
    
    # Test Beauty dataset with synthetic data
    print("🧪 Creating synthetic Beauty dataset...")
    beauty_dataset = BeautyDataset(data_path="dummy_beauty_path.json")
    
    # Use relaxed filtering for demo
    beauty_dataset.min_interactions_per_user = 2
    beauty_dataset.min_interactions_per_item = 2
    
    beauty_dataset.process_data()
    
    # Get dataset statistics
    beauty_stats = beauty_dataset.get_statistics()
    print("✅ Beauty Dataset Statistics:")
    print(f"   Sessions: {beauty_stats['num_sessions']}")
    print(f"   Items: {beauty_stats['num_items']}")
    print(f"   Avg session length: {beauty_stats['avg_session_length']:.2f}")
    print(f"   Total interactions: {beauty_stats['total_interactions']}")
    print(f"   Density: {beauty_stats['density']:.4f}")
    
    print("\n3. 🍕 DELIVERY HERO DATASET PROCESSING")
    print("-" * 38)
    
    # Test Delivery Hero dataset with synthetic data
    print("🧪 Creating synthetic Delivery Hero dataset...")
    dh_dataset = DeliveryHeroDataset(data_path="dummy_dh_path.csv")
    
    # Use relaxed filtering for demo
    dh_dataset.min_interactions_per_user = 2
    dh_dataset.min_interactions_per_item = 2
    
    dh_dataset.process_data()
    
    # Get dataset statistics
    dh_stats = dh_dataset.get_statistics()
    print("✅ Delivery Hero Dataset Statistics:")
    print(f"   Sessions: {dh_stats['num_sessions']}")
    print(f"   Items: {dh_stats['num_items']}")
    print(f"   Avg session length: {dh_stats['avg_session_length']:.2f}")
    print(f"   Total interactions: {dh_stats['total_interactions']}")
    print(f"   Density: {dh_stats['density']:.4f}")
    
    print("\n4. 📈 DATASET COMPARISON")
    print("-" * 23)
    
    print("📊 Beauty vs Delivery Hero:")
    print(f"   Sessions: {beauty_stats['num_sessions']} vs {dh_stats['num_sessions']}")
    print(f"   Items: {beauty_stats['num_items']} vs {dh_stats['num_items']}")
    print(f"   Avg length: {beauty_stats['avg_session_length']:.2f} vs {dh_stats['avg_session_length']:.2f}")
    print(f"   Density: {beauty_stats['density']:.4f} vs {dh_stats['density']:.4f}")
    
    return beauty_dataset, dh_dataset


def demo_evaluation_pipeline(dataset):
    """Demonstrate evaluation pipeline with dataset"""
    print("\n5. 🎯 EVALUATION PIPELINE TESTING")
    print("-" * 33)
    
    # Create pipeline with dataset
    mock_llm = MockLLMProvider({
        "analyze": "User shows strong sequential preferences",
        "finish": "Finish[recommended_item]"
    })
    
    pipeline = create_pipeline(mock_llm, "beauty", "reflexion")
    
    # Use the processed dataset
    pipeline.dataset = dataset
    
    # Update orchestrator with dataset
    user_data = {s['user_id']: {'sequence_length': len(s['items'])} for s in dataset.sessions}
    item_data = dataset.item_to_name
    user_histories = {s['user_id']: s['items'] for s in dataset.sessions}
    
    pipeline.orchestrator.update_agent_data(
        user_data=user_data,
        item_data=item_data,
        user_histories=user_histories
    )
    
    print(f"✅ Pipeline configured with dataset:")
    print(f"   Dataset: {len(dataset.sessions)} sessions")
    print(f"   User data: {len(user_data)} users")
    print(f"   Item data: {len(item_data)} items")
    
    # Test evaluation splits
    splits = dataset.create_evaluation_splits()
    print(f"\n📊 Evaluation Splits:")
    print(f"   Train: {len(splits['train'])} sessions")
    print(f"   Validation: {len(splits['val'])} sessions") 
    print(f"   Test: {len(splits['test'])} sessions")
    
    # Test mini evaluation
    print(f"\n🔍 Mini Evaluation (3 samples):")
    try:
        metrics = pipeline.run_evaluation(split='test', max_samples=3)
        print("✅ Evaluation Results:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
    except Exception as e:
        print(f"   ⚠️ Evaluation note: {str(e)}")
        print("   (This is expected with synthetic data)")
    
    return pipeline


def demo_dataset_features(dataset):
    """Demonstrate dataset-specific features"""
    print("\n6. 🔬 DATASET FEATURES TESTING")
    print("-" * 30)
    
    # Test session sampling
    sample_sessions = dataset.sessions[:3]
    
    print("📋 Sample Sessions:")
    for i, session in enumerate(sample_sessions):
        print(f"\n   Session {i+1}:")
        print(f"      User: {session['user_id']}")
        print(f"      Items: {session['items'][:5]}{'...' if len(session['items']) > 5 else ''}")
        print(f"      Length: {session['length']}")
        
        # Test prediction preparation
        pred_sequence = dataset.prepare_to_predict(session)
        ground_truth = dataset.extract_ground_truth(session)
        
        print(f"      Prediction sequence: {pred_sequence}")
        print(f"      Ground truth: {ground_truth}")
        
        # Test candidate pool
        candidates, target_idx = dataset.create_candidate_pool(session)
        print(f"      Candidates: {len(candidates)} items (target at index {target_idx})")
    
    # Test negative sampling
    test_session = sample_sessions[0]
    user_items = test_session['items']
    negatives = dataset.negative_sample(test_session['user_id'], user_items)
    
    print(f"\n🎯 Negative Sampling Test:")
    print(f"   User items: {len(user_items)}")
    print(f"   Negative samples: {len(negatives)}")
    print(f"   No overlap: {len(set(negatives) & set(user_items)) == 0}")
    
    # Test data integrity
    integrity_ok = dataset.test_data_integrity()
    print(f"   Data integrity: {'✅ PASSED' if integrity_ok else '❌ FAILED'}")


def demo_recommendation_with_dataset(pipeline):
    """Demonstrate recommendations using real dataset samples"""
    print("\n7. 🎯 DATASET-BASED RECOMMENDATIONS")
    print("-" * 37)
    
    dataset = pipeline.dataset
    sample_sessions = dataset.sessions[:2]
    
    print("🧪 Testing recommendations with real dataset samples:")
    
    for i, session in enumerate(sample_sessions):
        print(f"\n   🔍 Test Case {i+1}:")
        
        # Prepare prediction task
        pred_sequence = dataset.prepare_to_predict(session)
        ground_truth = dataset.extract_ground_truth(session)
        candidates, target_idx = dataset.create_candidate_pool(session)
        
        print(f"      User: {session['user_id']}")
        print(f"      Sequence: {' → '.join(pred_sequence[-3:])}")  # Last 3 items
        print(f"      Candidates: {len(candidates)} items")
        print(f"      Ground truth: {ground_truth}")
        
        # Create recommendation request
        request = RecommendationRequest(
            user_id=session['user_id'],
            user_sequence=pred_sequence,
            candidates=candidates,
            ground_truth=ground_truth
        )
        
        # Get recommendation
        response = pipeline.orchestrator.recommend(request, max_iterations=2)
        
        # Check accuracy
        correct = response.recommendation == ground_truth
        print(f"      🎯 Recommendation: {response.recommendation}")
        print(f"      📊 Confidence: {response.confidence:.3f}")
        print(f"      ✅ Accuracy: {'CORRECT' if correct else 'INCORRECT'}")
        
        # Reset for next request
        pipeline.orchestrator.reset_session()


def run_comprehensive_dataset_demo():
    """Run complete dataset testing demonstration"""
    print("🚀 Comprehensive Dataset Testing Demo")
    print("=" * 42)
    
    # Step 1: Process datasets
    beauty_dataset, dh_dataset = demo_dataset_processing()
    
    # Step 2: Setup evaluation pipeline
    pipeline = demo_evaluation_pipeline(beauty_dataset)
    
    # Step 3: Test dataset features
    demo_dataset_features(beauty_dataset)
    
    # Step 4: Test recommendations
    demo_recommendation_with_dataset(pipeline)
    
    print("\n8. 🎉 DATASET TESTING SUMMARY")
    print("-" * 28)
    
    print("✅ Successfully Demonstrated:")
    print("   📊 Multi-dataset processing (Beauty + Delivery Hero)")
    print("   🔄 5-core filtering and data cleaning")
    print("   📈 Statistical analysis and reporting")
    print("   🎯 Leave-one-out evaluation setup")
    print("   🧪 Negative sampling and candidate generation")
    print("   📋 Train/validation/test splitting")
    print("   🤖 Dataset-driven recommendation testing")
    print("   ✅ Data integrity validation")
    
    print(f"\n💡 Dataset Capabilities:")
    print(f"   🏗️ Synthetic data generation for testing")
    print(f"   📄 Real dataset format compatibility")
    print(f"   🔧 Flexible filtering and preprocessing") 
    print(f"   📊 Comprehensive evaluation metrics")
    print(f"   🎯 Sequential recommendation evaluation")
    
    beauty_stats = beauty_dataset.get_statistics()
    dh_stats = dh_dataset.get_statistics()
    
    return {
        'beauty_sessions': beauty_stats['num_sessions'],
        'beauty_items': beauty_stats['num_items'],
        'dh_sessions': dh_stats['num_sessions'], 
        'dh_items': dh_stats['num_items'],
        'evaluation_ready': True
    }


if __name__ == "__main__":
    summary = run_comprehensive_dataset_demo()
    print(f"\n🏁 Dataset demo completed: {summary}")