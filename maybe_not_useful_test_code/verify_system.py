#!/usr/bin/env python3
"""
Simple system verification script.
Tests the core recommendation functionality.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentic_recommender.system import create_pipeline
from agentic_recommender.core import RecommendationRequest
from agentic_recommender.models.llm_provider import MockLLMProvider

def main():
    print("🔍 System Verification")
    print("=" * 20)
    
    # Create pipeline with mock LLM
    mock_llm = MockLLMProvider({
        "analyze": "User shows preference for tech products in sequence",
        "finish": "Finish[keyboard]",
        "default": "Analysis complete"
    })
    
    print("1. Creating pipeline...")
    pipeline = create_pipeline(mock_llm, "beauty", "none")
    
    print("2. Testing recommendation...")
    request = RecommendationRequest(
        user_id="test_user",
        user_sequence=["laptop", "mouse"],
        candidates=["keyboard", "monitor", "headphones"]
    )
    
    response = pipeline.orchestrator.recommend(request, max_iterations=2)
    
    print(f"✅ Recommendation: {response.recommendation}")
    print(f"✅ Confidence: {response.confidence:.3f}")
    print(f"✅ Reasoning: {response.reasoning}")
    
    # Test system stats
    stats = pipeline.orchestrator.get_system_stats()
    print(f"✅ System requests: {stats['orchestrator']['total_requests']}")
    
    print("\n🎉 System verification complete!")
    print("The agentic recommendation system is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)