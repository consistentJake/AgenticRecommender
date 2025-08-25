#!/usr/bin/env python3
"""
Test improved agentic system with real dataset samples.
"""

import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from improved_agentic_system import ImprovedAgenticSystem


def load_real_samples():
    """Load real samples from processed datasets"""
    outputs_dir = Path("agentic_recommender/data/outputs")
    samples = []
    
    # Load Beauty samples
    beauty_path = outputs_dir / "beauty_evaluation_samples.json"
    if beauty_path.exists():
        with open(beauty_path, 'r') as f:
            beauty_samples = json.load(f)[:2]  # First 2 samples
            for sample in beauty_samples:
                sample['dataset'] = 'beauty'
            samples.extend(beauty_samples)
    
    # Load Delivery Hero samples
    dh_path = outputs_dir / "delivery_hero_se_evaluation_samples.json"
    if dh_path.exists():
        with open(dh_path, 'r') as f:
            dh_samples = json.load(f)[:2]  # First 2 samples  
            for sample in dh_samples:
                sample['dataset'] = 'delivery_hero'
            samples.extend(dh_samples)
    
    return samples


def main():
    print("🧪 Testing Improved System with Real Dataset Samples")
    print("="*60)
    
    # Load real samples
    samples = load_real_samples()
    print(f"📊 Loaded {len(samples)} real samples")
    
    # Initialize improved system
    system = ImprovedAgenticSystem(use_real_gemini=True)
    
    # Test with real samples (limit candidates for efficiency)
    for sample in samples:
        # Limit candidates to first 10 for efficiency
        if 'candidates' in sample and len(sample['candidates']) > 10:
            sample['candidates'] = sample['candidates'][:10]
        
        # Limit prompt items to first 5 for efficiency  
        if 'prompt_items' in sample and len(sample['prompt_items']) > 5:
            sample['prompt_items'] = sample['prompt_items'][:5]
    
    print(f"\n🚀 Running improved recommendations on real data...")
    results = system.batch_recommend(samples)
    
    # Evaluate performance
    print(f"\n📊 Real Data Performance:")
    performance = system.evaluate_performance(results)
    
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Dataset-specific analysis
    beauty_results = [r for r, s in zip(results, samples) if s.get('dataset') == 'beauty']
    dh_results = [r for r, s in zip(results, samples) if s.get('dataset') == 'delivery_hero']
    
    if beauty_results:
        beauty_accuracy = sum(1 for r in beauty_results if r.get('correct_prediction', False)) / len(beauty_results)
        print(f"   Beauty dataset accuracy: {beauty_accuracy:.3f}")
    
    if dh_results:
        dh_accuracy = sum(1 for r in dh_results if r.get('correct_prediction', False)) / len(dh_results)
        print(f"   Delivery Hero accuracy: {dh_accuracy:.3f}")
    
    # Show sample results
    print(f"\n📋 Sample Results:")
    for i, (sample, result) in enumerate(zip(samples[:4], results[:4])):
        dataset = sample.get('dataset', 'unknown')
        session_id = sample.get('session_id', 'unknown')
        
        # Get item names for display
        prompt_names = []
        for item_id in sample.get('prompt_items', [])[:3]:  # First 3 only
            name = sample.get('item_names', {}).get(item_id, item_id)
            prompt_names.append(name[:30] + "..." if len(name) > 30 else name)
        
        target_name = sample.get('item_names', {}).get(sample.get('target_item', ''), sample.get('target_item', 'unknown'))
        rec_name = sample.get('item_names', {}).get(result.get('recommendation', ''), result.get('recommendation', 'unknown'))
        
        print(f"\nSample {i+1} ({dataset}): {session_id}")
        print(f"   Sequence: {' → '.join(prompt_names)}")
        print(f"   Target: {target_name[:50]}...")
        print(f"   Predicted: {rec_name[:50]}...")
        print(f"   Correct: {'✅' if result.get('correct_prediction', False) else '❌'}")
        print(f"   Steps: {result.get('steps_taken', 0)}, Confidence: {result.get('final_confidence', 'unknown')}")
    
    print(f"\n✅ Real data testing completed!")


if __name__ == "__main__":
    main()