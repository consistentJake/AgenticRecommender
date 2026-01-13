#!/usr/bin/env python3
"""
Process Beauty dataset and generate outputs for agentic workflow.
"""

import sys
import os
import json
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.datasets import BeautyDataset
from agentic_recommender.utils.logging import get_component_logger


logger = get_component_logger("data.process_beauty_dataset")


def process_and_save_beauty_dataset():
    """Process Beauty dataset and save outputs for agentic workflow"""
    
    logger.info("🚀 Starting Beauty dataset processing...")
    
    # Create dataset instance
    dataset = BeautyDataset()
    
    # Process the data
    dataset.process_data()
    
    # Get statistics
    stats = dataset.get_statistics()
    logger.info("📊 Dataset Statistics:")
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)
    
    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("💾 Saving processed data to %s", output_dir)
    
    # Save processed dataset
    with open(output_dir / "beauty_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    
    # Save dataset statistics
    with open(output_dir / "beauty_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save evaluation splits
    logger.info("📑 Creating evaluation splits...")
    splits = dataset.create_evaluation_splits()
    
    for split_name, split_data in splits.items():
        with open(output_dir / f"beauty_{split_name}.json", "w") as f:
            json.dump(split_data, f, indent=2)
    
    # Save item mappings
    with open(output_dir / "beauty_item_to_name.json", "w") as f:
        json.dump(dataset.item_to_name, f, indent=2)
    
    with open(output_dir / "beauty_name_to_item.json", "w") as f:
        json.dump(dataset.name_to_item, f, indent=2)
    
    # Create a sample for demo
    logger.info("🎯 Creating evaluation samples...")
    sample_sessions = splits['test'][:10]  # First 10 test sessions
    
    evaluation_samples = []
    for session in sample_sessions:
        # Create candidate pool for each session
        candidates, target_idx = dataset.create_candidate_pool(session)
        prompt_items, target = dataset.prepare_to_predict(session)
        
        sample = {
            'session_id': session['session_id'],
            'user_id': session['user_id'],
            'prompt_items': prompt_items,
            'target_item': target,
            'candidates': candidates,
            'target_index': target_idx,
            'item_names': {
                item_id: dataset.get_item_name(item_id) 
                for item_id in prompt_items + [target]
            }
        }
        evaluation_samples.append(sample)
    
    with open(output_dir / "beauty_evaluation_samples.json", "w") as f:
        json.dump(evaluation_samples, f, indent=2)
    
    logger.info("✅ Beauty dataset processing completed!")
    logger.info("📁 Files saved in: %s", output_dir)
    logger.info("   - beauty_dataset.pkl: Complete dataset object")
    logger.info("   - beauty_stats.json: Dataset statistics")
    logger.info("   - beauty_train.json: Training sessions")
    logger.info("   - beauty_val.json: Validation sessions")
    logger.info("   - beauty_test.json: Test sessions")
    logger.info("   - beauty_item_to_name.json: Item ID to name mapping")
    logger.info("   - beauty_name_to_item.json: Item name to ID mapping")
    logger.info("   - beauty_evaluation_samples.json: Sample evaluation tasks")
    
    # Test data integrity
    logger.info("🔍 Testing data integrity...")
    if dataset.test_data_integrity():
        logger.info("✅ All data integrity tests passed!")
    else:
        logger.error("❌ Data integrity tests failed!")
    
    return dataset


if __name__ == "__main__":
    dataset = process_and_save_beauty_dataset()
