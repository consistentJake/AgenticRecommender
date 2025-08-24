"""
Demo script showing training data preparation for the agentic recommendation system.
Demonstrates how to generate training data for agent fine-tuning.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.system import create_pipeline
from agentic_recommender.training import prepare_training_pipeline
from agentic_recommender.models.llm_provider import MockLLMProvider


def demo_training_preparation():
    """Demonstrate training data generation"""
    print("🎓 Training Data Preparation Demo")
    print("=" * 40)
    
    # Create mock LLM for data generation
    mock_llm = MockLLMProvider({
        "analyze": "User shows sequential purchasing patterns",
        "action": "Finish[recommended_item]",
        "reflection": '{"correctness": false, "reason": "Need better analysis"}'
    })
    
    print("\n1. 🏗️ SETUP PIPELINE")
    print("-" * 20)
    
    # Create pipeline and load data
    pipeline = create_pipeline(mock_llm, "beauty", "reflexion")
    pipeline.load_dataset("dummy_path.json", use_synthetic=True)
    
    dataset_stats = pipeline.dataset.get_statistics()
    print(f"   Dataset loaded: {dataset_stats['num_sessions']} sessions, {dataset_stats['num_items']} items")
    
    print("\n2. 📚 GENERATE TRAINING DATA")
    print("-" * 25)
    
    # Generate basic training data (skip reflection for speed)
    output_dir = "training_output_demo"
    from agentic_recommender.training.data_preparation import TrainingDataGenerator
    
    data_generator = TrainingDataGenerator(pipeline.dataset)
    training_examples = data_generator.generate_all_training_data(output_dir, 20)
    
    summary = {
        'training_examples': len(training_examples),
        'reflection_examples': 0,  # Skipped for demo speed
        'output_directory': output_dir,
        'ready_for_training': True
    }
    
    print(f"\n3. 📊 TRAINING DATA SUMMARY")
    print("-" * 25)
    
    print(f"   📚 Agent training examples: {summary['training_examples']}")
    print(f"   🪞 Reflection examples: {summary['reflection_examples']}")
    print(f"   📁 Output directory: {summary['output_directory']}")
    print(f"   ✅ Ready for training: {summary['ready_for_training']}")
    
    print(f"\n4. 📄 SAMPLE TRAINING DATA")
    print("-" * 22)
    
    # Show sample training examples
    training_file = Path(output_dir) / "training_data.jsonl"
    if training_file.exists():
        print("   Sample training examples:")
        with open(training_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 examples
                    break
                import json
                example = json.loads(line)
                print(f"\n   Example {i+1} ({example['metadata']['type']}):")
                print(f"      Input: {example['input'][:80]}...")
                print(f"      Output: {example['output'][:80]}...")
    
    # Show reflection examples
    reflection_file = Path(output_dir) / "reflection_data.jsonl"  
    if reflection_file.exists():
        print("\n   Sample reflection examples:")
        with open(reflection_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 2:  # Show first 2 examples
                    break
                import json
                example = json.loads(line)
                print(f"\n   Reflection {i+1}:")
                print(f"      Task: {example['task']}")
                print(f"      Correct: {example['reflection']['correctness']}")
                print(f"      Reason: {example['reflection']['reason'][:60]}...")
    
    print(f"\n5. 🎯 NEXT STEPS")
    print("-" * 15)
    
    print("   The generated training data can be used for:")
    print("   📖 Agent fine-tuning (Manager think/act patterns)")
    print("   🔍 Analyst training (Sequential pattern recognition)")  
    print("   🪞 Reflection learning (Improvement strategies)")
    print("   📊 Evaluation benchmarks")
    
    print(f"\n   Training files ready in: {output_dir}/")
    print("   - training_data.jsonl (agent examples)")
    print("   - reflection_data.jsonl (reflection examples)")
    print("   - training_summary.json (metadata)")
    
    print("\n🎉 Training preparation demo completed!")
    
    return summary


if __name__ == "__main__":
    summary = demo_training_preparation()