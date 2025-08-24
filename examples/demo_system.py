"""
Demo script showing the agentic recommendation system in action.
Demonstrates the complete workflow with mock data.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.agents import Manager, Analyst, Reflector
from agentic_recommender.agents.base import ReflectionStrategy
from agentic_recommender.models.llm_provider import MockLLMProvider, GeminiProvider
from agentic_recommender.utils.metrics import evaluate_recommendations


def demo_agentic_recommendation():
    """Demonstrate the complete agentic recommendation workflow"""
    print("🚀 Agentic Sequential Recommendation System Demo\n")
    
    # Check if we have real API key
    api_key = os.getenv('GEMINI_API_KEY')
    use_real_api = api_key is not None
    
    if use_real_api:
        print("🤖 Using real Gemini API")
        llm_provider = GeminiProvider(api_key)
    else:
        print("🎭 Using mock LLM for demo")
        # Create realistic mock responses
        mock_responses = {
            "analyze": "I should analyze the user's sequential behavior first",
            "action": "Analyse[user, user_123]",
            "user_analysis": "User shows preference for tech accessories, buys complementary items",
            "final_rec": "Finish[wireless_mouse]"
        }
        llm_provider = MockLLMProvider(mock_responses)
    
    # Create agents
    print("\n🧠 Initializing agents...")
    manager = Manager(llm_provider, llm_provider, config={'max_steps': 5})
    analyst = Analyst(llm_provider)
    reflector = Reflector(llm_provider, ReflectionStrategy.REFLEXION)
    
    # Sample user data
    sample_user_data = {
        "user_123": {
            "age": 28,
            "location": "San Francisco", 
            "categories": ["Electronics", "Books", "Gaming"]
        }
    }
    
    sample_item_data = {
        "wireless_mouse": {"name": "Wireless Gaming Mouse", "category": "Electronics"},
        "bluetooth_headphones": {"name": "Bluetooth Headphones", "category": "Electronics"},
        "programming_book": {"name": "Python Programming Guide", "category": "Books"}
    }
    
    user_histories = {
        "user_123": ["laptop", "mechanical_keyboard", "monitor", "mouse_pad"]
    }
    
    # Update analyst with data
    analyst.update_data_sources(
        user_data=sample_user_data,
        item_data=sample_item_data,
        user_histories=user_histories
    )
    
    # Demo task: Recommend next item for user
    print("\n📋 Task: Recommend next item for user who bought: laptop → keyboard → monitor → mouse_pad")
    
    task_context = {
        "user_id": "user_123",
        "user_sequence": ["laptop", "mechanical_keyboard", "monitor", "mouse_pad"],
        "candidates": ["wireless_mouse", "bluetooth_headphones", "programming_book", "webcam", "usb_hub"],
        "task": "sequential_recommendation"
    }
    
    print(f"👤 User sequence: {' → '.join(task_context['user_sequence'])}")
    print(f"🎯 Candidates: {', '.join(task_context['candidates'])}")
    
    # Step 1: Manager thinks about the situation
    print("\n💭 Manager thinking phase...")
    thought = manager.think(task_context)
    print(f"Manager thought: {thought}")
    
    # Step 2: Manager decides on action
    print("\n⚡ Manager action phase...")
    action_type, argument = manager.act(task_context)
    print(f"Manager action: {action_type}[{argument}]")
    
    # Step 3: Execute action if it's analysis
    analysis_result = ""
    if action_type == "Analyse" and len(argument) >= 2:
        print(f"\n🔍 Analyst analyzing {argument[0]} {argument[1]}...")
        analysis_result = analyst.forward(argument)
        print(f"Analysis result: {analysis_result}")
    
    # Step 4: Manager makes final recommendation
    print("\n🎯 Manager making final recommendation...")
    final_thought = manager.think({**task_context, "analysis": analysis_result})
    final_action_type, final_argument = manager.act({**task_context, "analysis": analysis_result})
    print(f"Final action: {final_action_type}[{final_argument}]")
    
    # Step 5: Reflect on the recommendation
    print("\n🪞 Reflector evaluating recommendation...")
    scratchpad = f"""Thought: {thought}
Action: {action_type}[{argument}]
Analysis: {analysis_result}
Final thought: {final_thought}
Final action: {final_action_type}[{final_argument}]"""
    
    reflection = reflector.forward(
        input_task="Recommend next item for tech user",
        scratchpad=scratchpad,
        first_attempt=final_argument,
        ground_truth="wireless_mouse"  # Ground truth for evaluation
    )
    print(f"Reflection: {reflection}")
    
    # Demo evaluation
    print("\n📊 Evaluation Demo:")
    predictions = ["wireless_mouse", "bluetooth_headphones", "usb_hub", "webcam", "programming_book"]
    ground_truth = "wireless_mouse"
    
    metrics = evaluate_recommendations([predictions], [ground_truth], k_values=[1, 3, 5])
    print(f"HR@1: {metrics['hr@1']:.3f}")
    print(f"HR@3: {metrics['hr@3']:.3f}")
    print(f"NDCG@5: {metrics['ndcg@5']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    
    print("\n🎉 Demo completed successfully!")
    
    # Show agent performance
    print("\n📈 Agent Performance:")
    for agent_name, agent in [("Manager", manager), ("Analyst", analyst), ("Reflector", reflector)]:
        stats = agent.get_performance_stats()
        print(f"{agent_name}: {stats['total_calls']} calls, {stats['avg_time_per_call']:.3f}s avg")


if __name__ == "__main__":
    demo_agentic_recommendation()