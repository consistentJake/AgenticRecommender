#!/usr/bin/env python3
"""
Process Delivery Hero dataset and generate outputs for agentic workflow.
"""

import sys
import os
import json
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.datasets import DeliveryHeroDataset
from agentic_recommender.utils.logging import get_component_logger


logger = get_component_logger("data.process_delivery_hero_dataset")


def process_and_save_delivery_hero_dataset(city: str = "sg"):
    """Process Delivery Hero dataset and save outputs for agentic workflow"""
    
    logger.info("🚀 Starting Delivery Hero %s dataset processing...", city.upper())
    
    # Create dataset instance
    dataset = DeliveryHeroDataset(city=city)
    
    # Process the data (this may take a while for large datasets)
    dataset.process_data()
    
    # Get statistics
    stats = dataset.get_statistics()
    logger.info("📊 Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info("  %s: %.4f", key, value)
        else:
            logger.info("  %s: %s", key, f"{value:,}")
    
    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("💾 Saving processed data to %s", output_dir)
    
    # Save processed dataset
    with open(output_dir / f"delivery_hero_{city}_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    
    # Save dataset statistics
    with open(output_dir / f"delivery_hero_{city}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save evaluation splits
    logger.info("📑 Creating evaluation splits...")
    splits = dataset.create_evaluation_splits()
    
    for split_name, split_data in splits.items():
        # Convert numpy types to native Python types for JSON serialization
        split_data_clean = []
        for session in split_data:
            session_clean = {}
            for key, value in session.items():
                if key == 'items' or key == 'timestamps':
                    # Convert arrays to native Python lists
                    session_clean[key] = [int(x) if hasattr(x, 'item') else x for x in value]
                else:
                    # Convert individual values
                    if hasattr(value, 'item'):  # numpy scalar
                        session_clean[key] = value.item()
                    else:
                        session_clean[key] = value
            split_data_clean.append(session_clean)
        
        with open(output_dir / f"delivery_hero_{city}_{split_name}.json", "w") as f:
            json.dump(split_data_clean, f, indent=2)
    
    # Save item mappings
    with open(output_dir / f"delivery_hero_{city}_item_to_name.json", "w") as f:
        json.dump(dataset.item_to_name, f, indent=2)
    
    with open(output_dir / f"delivery_hero_{city}_name_to_item.json", "w") as f:
        json.dump(dataset.name_to_item, f, indent=2)
    
    # Create a sample for demo (fewer samples since DH sessions are typically shorter)
    logger.info("🎯 Creating evaluation samples...")
    sample_sessions = splits['test'][:5]  # First 5 test sessions
    
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
    
    with open(output_dir / f"delivery_hero_{city}_evaluation_samples.json", "w") as f:
        json.dump(evaluation_samples, f, indent=2)
    
    logger.info("✅ Delivery Hero %s dataset processing completed!", city.upper())
    logger.info("📁 Files saved in: %s", output_dir)
    logger.info("   - delivery_hero_%s_dataset.pkl: Complete dataset object", city)
    logger.info("   - delivery_hero_%s_stats.json: Dataset statistics", city)
    logger.info("   - delivery_hero_%s_train.json: Training sessions", city)
    logger.info("   - delivery_hero_%s_val.json: Validation sessions", city)
    logger.info("   - delivery_hero_%s_test.json: Test sessions", city)
    logger.info("   - delivery_hero_%s_item_to_name.json: Item ID to name mapping", city)
    logger.info("   - delivery_hero_%s_name_to_item.json: Item name to ID mapping", city)
    logger.info("   - delivery_hero_%s_evaluation_samples.json: Sample evaluation tasks", city)
    
    # Test data integrity
    logger.info("🔍 Testing data integrity...")
    if dataset.test_data_integrity():
        logger.info("✅ All data integrity tests passed!")
    else:
        logger.error("❌ Data integrity tests failed!")
    
    return dataset


def process_all_cities():
    """Process all available cities"""
    cities = ["sg", "se", "tw"]  # Singapore, Stockholm, Taiwan
    
    for city in cities:
        try:
            separator = '=' * 60
            logger.info(separator)
            logger.info("Processing %s dataset...", city.upper())
            logger.info(separator)
            
            dataset = process_and_save_delivery_hero_dataset(city)
            logger.info("✅ %s completed successfully!", city.upper())
            
        except Exception as e:
            logger.error("❌ Error processing %s: %s", city.upper(), e)
            continue
    
    logger.info("🎉 All Delivery Hero datasets processed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Delivery Hero dataset')
    parser.add_argument('--city', type=str, default='sg', 
                       choices=['sg', 'se', 'tw'],
                       help='City to process (sg=Singapore, se=Stockholm, tw=Taiwan)')
    parser.add_argument('--all', action='store_true',
                       help='Process all cities')
    
    args = parser.parse_args()
    
    if args.all:
        process_all_cities()
    else:
        process_and_save_delivery_hero_dataset(args.city)
