"""
Simple training data generation demo.
Shows the concept of generating training examples for agents.
"""

import sys
from pathlib import Path
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.training.data_preparation import TrainingExample


def demo_training_concepts():
    """Demonstrate training data generation concepts"""
    print("🎓 Training Data Generation Concepts")
    print("=" * 40)
    
    print("\n1. 📚 AGENT TRAINING EXAMPLES")
    print("-" * 25)
    
    # Example Manager training data
    manager_examples = [
        TrainingExample(
            input_prompt="""You are a Manager agent. Analyze this recommendation task:

User sequence: laptop → mouse → keyboard
Candidates: monitor, headphones, webcam

Think step by step about what information you need.""",
            target_output="I need to analyze the user's sequential behavior. The sequence shows tech accessories pattern. I should request user analysis to understand their preferences better.",
            metadata={'type': 'manager_think', 'user_id': 'user_123'}
        ),
        
        TrainingExample(
            input_prompt="""Based on your analysis, choose the best action:

Available actions:
- Analyse[user, user_id]
- Finish[item]

Context: User shows tech accessories pattern""",
            target_output="Analyse[user, user_123]",
            metadata={'type': 'manager_act', 'action': 'Analyse'}
        )
    ]
    
    # Example Analyst training data
    analyst_examples = [
        TrainingExample(
            input_prompt="""Analyze this user for sequential recommendation:

User sequence: foundation → concealer → lipstick
Recent items: foundation → concealer

Provide insights about patterns and next item prediction.""",
            target_output="""User analysis:
1. Preference patterns: Shows preference for beauty/makeup items
2. Sequential behavior: Follows logical makeup application order
3. Recommendation: Based on sequence, user likely needs mascara or eyeshadow to complete the look""",
            metadata={'type': 'analyst_user', 'user_id': 'beauty_user'}
        )
    ]
    
    all_examples = manager_examples + analyst_examples
    
    print(f"   Generated {len(all_examples)} training examples:")
    for i, example in enumerate(all_examples):
        print(f"\n   Example {i+1} ({example.metadata['type']}):")
        print(f"      Input: {example.input_prompt[:60]}...")
        print(f"      Output: {example.target_output[:60]}...")
    
    print("\n2. 🪞 REFLECTION EXAMPLES")  
    print("-" * 20)
    
    # Example reflection training data
    reflection_examples = [
        {
            'task': 'Recommend next item for tech user',
            'attempt': 'User sequence: laptop → mouse → keyboard\nRecommendation: headphones',
            'ground_truth': 'monitor',
            'reflection': {
                'correctness': False,
                'reason': 'Recommended headphones but user chose monitor - missed the display completion pattern',
                'improvement': 'Better analyze sequential patterns: laptop setup usually needs monitor for completion'
            }
        }
    ]
    
    print(f"   Generated {len(reflection_examples)} reflection examples:")
    for i, example in enumerate(reflection_examples):
        print(f"\n   Reflection {i+1}:")
        print(f"      Task: {example['task']}")
        print(f"      Correct: {example['reflection']['correctness']}")
        print(f"      Reason: {example['reflection']['reason'][:80]}...")
    
    print("\n3. 💾 TRAINING DATA FORMAT")
    print("-" * 22)
    
    # Show JSONL format
    sample_jsonl = {
        'input': manager_examples[0].input_prompt,
        'output': manager_examples[0].target_output,
        'metadata': manager_examples[0].metadata
    }
    
    print("   JSONL format for agent training:")
    print(f"   {json.dumps(sample_jsonl, indent=2)[:200]}...")
    
    print("\n4. 🎯 TRAINING APPLICATIONS")
    print("-" * 22)
    
    print("   📖 Manager Training:")
    print("      - Think phase: Situation analysis and planning")
    print("      - Act phase: Action selection and formatting")
    print("")
    print("   🔍 Analyst Training:")  
    print("      - User analysis: Pattern recognition in sequences")
    print("      - Item analysis: Characteristics and relationships")
    print("")
    print("   🪞 Reflection Training:")
    print("      - Quality assessment: Correctness evaluation")
    print("      - Improvement suggestions: Better strategies")
    
    print("\n5. 📊 TRAINING PIPELINE")
    print("-" * 18)
    
    print("   The complete training pipeline would:")
    print("   1. 📊 Load processed dataset (Beauty/Delivery Hero)")
    print("   2. 🎯 Generate agent-specific training examples")
    print("   3. 🪞 Create reflection learning data")
    print("   4. 💾 Save in standard formats (JSONL)")
    print("   5. 🚀 Ready for fine-tuning workflows")
    
    print("\n🎉 Training concepts demonstration completed!")
    
    return {
        'manager_examples': len(manager_examples),
        'analyst_examples': len(analyst_examples), 
        'reflection_examples': len(reflection_examples),
        'total_examples': len(all_examples) + len(reflection_examples)
    }


if __name__ == "__main__":
    summary = demo_training_concepts()
    print(f"\nSummary: {summary}")