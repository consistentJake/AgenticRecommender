"""
Complete pipeline demo showing the full agentic recommendation system.
Demonstrates dataset integration, agent coordination, and evaluation.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_recommender.system import create_pipeline
from agentic_recommender.core import RecommendationRequest
from agentic_recommender.models import llm_provider as llm_provider_module
from agentic_recommender.models.llm_provider import (
    MockLLMProvider,
    GeminiProvider,
    create_llm_provider,
)


def demo_complete_pipeline():
    """Demonstrate the complete agentic recommendation pipeline"""
    print("🚀 Complete Agentic Recommendation Pipeline Demo")
    print("=" * 55)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    use_real_api = api_key is not None

    if use_real_api:
        print("🤖 Using real Gemini API (environment key)")
        llm_provider = GeminiProvider(api_key)
    else:
        provider_from_config = None
        try:
            provider_from_config = create_llm_provider('config')
        except Exception as exc:
            print(f"⚠️  Could not create provider from config: {exc}")

        if isinstance(provider_from_config, GeminiProvider):
            llm_provider = provider_from_config
            use_real_api = True
            mode_label = "OpenRouter" if llm_provider.use_openrouter else "Gemini"
            print(f"🤖 Using real Gemini API via config ({mode_label})")
        else:
            print("🎭 Using mock LLM responses")
            # Create comprehensive mock responses
            mock_responses = {
                "analyze": "I should analyze the user's preferences first",
                "user": "User shows strong preference for tech accessories, sequential pattern detected",
                "action": "Finish[wireless_headphones]",
                "reflection": '{"correctness": true, "reason": "Good recommendation based on user sequence"}'
            }
            llm_provider = MockLLMProvider(mock_responses)
    
    print("\n1. 🏗️ PIPELINE INITIALIZATION")
    print("-" * 30)
    
    # Create pipeline with different reflection strategies
    pipelines = {}
    strategies = ['none', 'reflexion', 'last_trial']
    
    for strategy in strategies:
        print(f"   Creating pipeline with {strategy} reflection...")
        pipeline = create_pipeline(
            llm_provider=llm_provider,
            dataset_type="beauty",
            reflection_strategy=strategy
        )
        pipelines[strategy] = pipeline
    
    print("\n2. 📊 DATASET INTEGRATION")
    print("-" * 30)
    
    # Load synthetic dataset
    main_pipeline = pipelines['reflexion']
    print("   Loading synthetic beauty dataset...")
    main_pipeline.load_dataset("dummy_path.json", use_synthetic=True)
    
    print("\n3. 🎯 DEMO PREDICTIONS")
    print("-" * 30)
    
    # Demo scenarios
    demo_scenarios = [
        {
            "name": "Tech Enthusiast",
            "sequence": ["gaming_laptop", "mechanical_keyboard", "gaming_mouse"],
            "candidates": ["wireless_headphones", "monitor", "webcam", "usb_hub", "mouse_pad"]
        },
        {
            "name": "Beauty Lover",
            "sequence": ["foundation", "concealer", "lipstick"],
            "candidates": ["mascara", "eyeshadow", "blush", "nail_polish", "perfume"]
        },
        {
            "name": "Book Reader",
            "sequence": ["fiction_book", "bookmark", "reading_light"],
            "candidates": ["notebook", "pen_set", "book_stand", "reading_chair", "coffee_mug"]
        }
    ]
    
    for scenario in demo_scenarios:
        print(f"\n   📋 Scenario: {scenario['name']}")
        print(f"      Sequence: {' → '.join(scenario['sequence'])}")
        print(f"      Candidates: {', '.join(scenario['candidates'])}")
        
        # Get recommendation
        response = main_pipeline.run_demo_prediction(
            user_sequence=scenario['sequence'],
            candidates=scenario['candidates']
        )
        
        print(f"      🏆 Recommendation: {response.recommendation}")
        print(f"      📊 Confidence: {response.confidence:.3f}")
        
        if response.metadata.get('reflection'):
            ref_data = response.metadata['reflection']
            if isinstance(ref_data, dict) and 'reason' in ref_data:
                print(f"      💭 Reflection: {ref_data['reason']}")
    
    print("\n4. 🔄 STRATEGY COMPARISON")
    print("-" * 30)
    
    # Compare different reflection strategies
    test_request = RecommendationRequest(
        user_id="comparison_user",
        user_sequence=["laptop", "mouse"],
        candidates=["keyboard", "monitor", "headphones"],
        ground_truth="keyboard"
    )
    
    strategy_results = {}
    
    for strategy, pipeline in pipelines.items():
        print(f"   Testing {strategy} strategy...")
        response = pipeline.orchestrator.recommend(test_request, max_iterations=2)
        strategy_results[strategy] = {
            'recommendation': response.recommendation,
            'confidence': response.confidence,
            'iterations': response.metadata.get('iterations', 0)
        }
        pipeline.orchestrator.reset_session()
    
    # Display comparison
    print("\n   📊 Strategy Comparison:")
    for strategy, result in strategy_results.items():
        print(f"      {strategy:12}: {result['recommendation']:15} "
              f"(conf: {result['confidence']:.3f}, iter: {result['iterations']})")
    
    print("\n5. 📈 SYSTEM PERFORMANCE")
    print("-" * 30)
    
    # Show system statistics
    stats = main_pipeline.orchestrator.get_system_stats()
    
    print("   📊 Orchestrator Stats:")
    orch_stats = stats['orchestrator']
    print(f"      Total requests: {orch_stats['total_requests']}")
    print(f"      Success rate: {orch_stats['success_rate']:.3f}")
    print(f"      Current session: {orch_stats['current_session']}")
    
    print("\n   🤖 Agent Performance:")
    for agent_name, agent_stats in stats['agents'].items():
        print(f"      {agent_name:10}: {agent_stats['total_calls']} calls, "
              f"{agent_stats['avg_time_per_call']:.3f}s avg")
    
    print("\n6. 🔍 MINI EVALUATION")
    print("-" * 30)
    
    # Run mini evaluation if we have a dataset
    if hasattr(main_pipeline, 'dataset') and main_pipeline.dataset:
        try:
            print("   Running mini evaluation (3 samples)...")
            metrics = main_pipeline.run_evaluation(split='test', max_samples=3)
            
            print("   📊 Results:")
            for metric_name, value in metrics.items():
                print(f"      {metric_name}: {value:.4f}")
                
        except Exception as e:
            print(f"   ⚠️  Evaluation skipped: {str(e)}")
    else:
        print("   ℹ️  No dataset available for evaluation")
    
    print("\n7. 💾 PIPELINE SUMMARY")  
    print("-" * 30)
    
    summary = main_pipeline.get_pipeline_summary()
    
    print("   🏗️ Configuration:")
    print(f"      Dataset: {summary['dataset']['type']}")
    print(f"      LLM Provider: {'Gemini' if use_real_api else 'Mock'}")
    print(f"      Reflection: reflexion")
    
    if summary['dataset'].get('statistics'):
        ds_stats = summary['dataset']['statistics']
        print(f"      Sessions: {ds_stats.get('num_sessions', 'N/A')}")
        print(f"      Items: {ds_stats.get('num_items', 'N/A')}")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 55)
    
    print("\n📚 Key Features Demonstrated:")
    print("   ✅ Multi-agent coordination (Manager, Analyst, Reflector)")
    print("   ✅ Agent communication protocol") 
    print("   ✅ Dataset integration and processing")
    print("   ✅ Multiple reflection strategies")
    print("   ✅ Performance monitoring and logging")
    print("   ✅ End-to-end recommendation pipeline")
    print("   ✅ Evaluation metrics and analysis")

    log_path = llm_provider_module.LOG_FILE
    if log_path.exists():
        print(f"\n📝 Gemini provider logs saved to: {log_path}")
    else:
        print(f"\n📝 Gemini provider log path: {log_path} (file will appear after next provider run)")

    return main_pipeline


if __name__ == "__main__":
    pipeline = demo_complete_pipeline()
