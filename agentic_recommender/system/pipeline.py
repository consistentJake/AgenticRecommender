"""
End-to-end recommendation pipeline combining datasets, agents, and evaluation.
Complete system integration following MACRec architecture.
"""

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..datasets import BeautyDataset, DeliveryHeroDataset
from ..core import AgentOrchestrator, RecommendationRequest, RecommendationResponse
from ..agents.base import ReflectionStrategy
from ..models.llm_provider import LLMProvider
from ..utils.metrics import evaluate_recommendations
from ..utils.logging import get_logger


class RecommendationPipeline:
    """
    Complete recommendation pipeline integrating all components.
    
    Workflow:
    1. Load and process dataset
    2. Initialize agent orchestrator
    3. Execute recommendation workflow
    4. Evaluate results and collect metrics
    5. Generate reflection-based improvements
    
    Reference: MACRec_Analysis.md:199-259
    """
    
    def __init__(self, llm_provider: LLMProvider,
                 dataset_type: str = "beauty",
                 reflection_strategy: ReflectionStrategy = ReflectionStrategy.REFLEXION,
                 config: Dict[str, Any] = None):
        
        self.llm_provider = llm_provider
        self.dataset_type = dataset_type
        self.config = config or {}
        self.logger = get_logger()
        
        # Initialize components
        self.dataset = None
        self.orchestrator = AgentOrchestrator(
            llm_provider, 
            reflection_strategy, 
            config
        )
        
        # Evaluation results
        self.evaluation_results = []
        self.performance_logs = []
        
        print(f"🚀 Pipeline initialized: {dataset_type} dataset, {reflection_strategy.value} reflection")
    
    def load_dataset(self, data_path: str, metadata_path: str = None,
                    use_synthetic: bool = False):
        """Load and process dataset"""
        print(f"📊 Loading {self.dataset_type} dataset...")
        
        if self.dataset_type == "beauty":
            self.dataset = BeautyDataset(data_path, metadata_path)
        elif self.dataset_type == "delivery_hero":
            self.dataset = DeliveryHeroDataset(data_path)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        # For testing, relax filters
        if use_synthetic:
            self.dataset.min_interactions_per_user = 1
            self.dataset.min_interactions_per_item = 1
        
        # Process dataset
        self.dataset.process_data()
        
        # Update orchestrator with dataset info
        user_data = {s['user_id']: {'sequence_length': len(s['items'])} for s in self.dataset.sessions}
        item_data = self.dataset.item_to_name
        user_histories = {s['user_id']: s['items'] for s in self.dataset.sessions}
        
        self.orchestrator.update_agent_data(
            user_data=user_data,
            item_data=item_data,
            user_histories=user_histories
        )
        
        stats = self.dataset.get_statistics()
        print(f"✅ Dataset loaded: {stats['num_sessions']} sessions, {stats['num_items']} items")
    
    def run_evaluation(self, split: str = "test", max_samples: int = 10) -> Dict[str, float]:
        """
        Run evaluation on dataset split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum samples to evaluate (for speed)
            
        Returns:
            Evaluation metrics
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"🔍 Running evaluation on {split} split (max {max_samples} samples)...")
        
        # Get evaluation data
        splits = self.dataset.create_evaluation_splits()
        eval_sessions = splits[split][:max_samples]
        
        predictions_list = []
        ground_truths = []
        
        for i, session in enumerate(eval_sessions):
            print(f"  Processing sample {i+1}/{len(eval_sessions)}", end="\r")
            
            # Prepare prediction task
            pred_sequence, target = self.dataset.prepare_to_predict(session)
            candidates, target_idx = self.dataset.create_candidate_pool(session)
            
            # Create recommendation request
            request = RecommendationRequest(
                user_id=session['user_id'],
                user_sequence=pred_sequence,
                candidates=candidates,
                ground_truth=target
            )
            
            # Get recommendation
            try:
                response = self.orchestrator.recommend(request, max_iterations=3)
                
                # Find recommendation in candidates
                try:
                    rec_idx = candidates.index(response.recommendation)
                    # Create ranking with recommendation first
                    ranking = [response.recommendation] + [c for c in candidates if c != response.recommendation]
                except ValueError:
                    # Recommendation not in candidates, use original order
                    ranking = candidates
                
                predictions_list.append(ranking)
                ground_truths.append(target)
                
            except Exception as e:
                # Use fallback ranking
                predictions_list.append(candidates)
                ground_truths.append(target)
            
            # Reset for next sample
            self.orchestrator.reset_session()
        
        print(f"\n📊 Evaluating {len(predictions_list)} predictions...")
        
        # Calculate metrics
        metrics = evaluate_recommendations(
            predictions_list, 
            ground_truths, 
            k_values=[1, 3, 5, 10]
        )
        
        # Store results
        self.evaluation_results.append({
            'split': split,
            'timestamp': time.time(),
            'num_samples': len(eval_sessions),
            'metrics': metrics,
            'config': self.config
        })
        
        print("✅ Evaluation completed:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def run_demo_prediction(self, user_sequence: List[str], 
                           candidates: List[str]) -> RecommendationResponse:
        """
        Run a single demo prediction with detailed output.
        
        Args:
            user_sequence: User's interaction history
            candidates: Candidate items to rank
            
        Returns:
            Recommendation response with full reasoning
        """
        print(f"🎯 Demo prediction:")
        print(f"   User sequence: {' → '.join(user_sequence)}")
        print(f"   Candidates: {', '.join(candidates)}")
        
        # Create request
        request = RecommendationRequest(
            user_id="demo_user",
            user_sequence=user_sequence,
            candidates=candidates,
            context={"demo_mode": True}
        )
        
        # Get recommendation
        response = self.orchestrator.recommend(request)
        
        print(f"\n📋 Results:")
        print(f"   Recommendation: {response.recommendation}")
        print(f"   Confidence: {response.confidence:.3f}")
        print(f"   Reasoning: {response.reasoning}")
        
        if response.metadata.get('reflection'):
            ref = response.metadata['reflection']
            print(f"   Reflection: {ref.get('reason', 'No specific feedback')}")
        
        return response
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        dataset_stats = self.dataset.get_statistics() if self.dataset else {}
        orchestrator_stats = self.orchestrator.get_system_stats()
        
        return {
            'dataset': {
                'type': self.dataset_type,
                'statistics': dataset_stats
            },
            'orchestrator': orchestrator_stats,
            'evaluation_history': self.evaluation_results,
            'config': self.config
        }
    
    def save_results(self, output_path: str):
        """Save evaluation results and system logs"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results
        results_file = output_path / f"evaluation_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'pipeline_summary': self.get_pipeline_summary(),
                'evaluation_results': self.evaluation_results
            }, f, indent=2)
        
        print(f"💾 Results saved to {results_file}")


def create_pipeline(llm_provider: LLMProvider, 
                   dataset_type: str = "beauty",
                   reflection_strategy: str = "reflexion") -> RecommendationPipeline:
    """
    Factory function to create recommendation pipeline.
    
    Args:
        llm_provider: LLM provider for agents
        dataset_type: 'beauty' or 'delivery_hero'
        reflection_strategy: 'none', 'last_trial', 'reflexion', 'last_trial_and_reflexion'
        
    Returns:
        Configured pipeline instance
    """
    # Convert string to enum
    strategy_map = {
        'none': ReflectionStrategy.NONE,
        'last_trial': ReflectionStrategy.LAST_ATTEMPT,
        'reflexion': ReflectionStrategy.REFLEXION,
        'last_trial_and_reflexion': ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION
    }
    
    strategy = strategy_map.get(reflection_strategy, ReflectionStrategy.REFLEXION)
    
    return RecommendationPipeline(
        llm_provider=llm_provider,
        dataset_type=dataset_type,
        reflection_strategy=strategy
    )