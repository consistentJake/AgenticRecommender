"""
Training data preparation for agentic recommendation system.
Generates training data for agent fine-tuning and reflection learning.

Based on ThinkRec's dual-objective training approach.
Reference: MACRec_Analysis.md:260-300
"""

import json
import random
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from ..datasets.base_dataset import SequentialDataset
from ..core import AgentOrchestrator, RecommendationRequest
from ..agents.base import ReflectionStrategy
from ..models.llm_provider import LLMProvider
from ..utils.logging import get_component_logger


logger = get_component_logger("training.data_preparation")


@dataclass
class TrainingExample:
    """Single training example for agent training"""
    input_prompt: str
    target_output: str
    metadata: Dict[str, Any]


@dataclass  
class ReflectionExample:
    """Training example for reflection learning"""
    task_description: str
    attempt_scratchpad: str
    ground_truth: str
    reflection_target: Dict[str, Any]


class TrainingDataGenerator:
    """
    Generate training data for agent fine-tuning.
    
    Creates examples for:
    1. Manager decision-making (think/act)
    2. Analyst sequential pattern analysis  
    3. Reflector improvement suggestions
    
    Reference: ThinkRec dual-objective training
    """
    
    def __init__(self, dataset: SequentialDataset, config: Dict[str, Any] = None):
        self.dataset = dataset
        self.config = config or {}
        self.examples = []
        
        logger.info(
            "📚 Training data generator initialized with %s sessions",
            len(dataset.sessions),
        )
    
    def generate_manager_examples(self, num_examples: int = 1000) -> List[TrainingExample]:
        """Generate training examples for Manager agent"""
        
        examples = []
        sessions = self.dataset.sessions[:num_examples]
        
        for session in sessions:
            # Create think/act pairs
            pred_sequence = self.dataset.prepare_to_predict(session)
            target = self.dataset.extract_ground_truth(session)
            candidates, target_idx = self.dataset.create_candidate_pool(session)
            
            # Think phase example  
            # Template reference: previousWorks/MACRec/config/prompts/manager_prompt/analyse.json
            think_prompt = f"""You are a Manager agent. Analyze this recommendation task:

User ID: {session['user_id']}
User sequence: {' → '.join(pred_sequence)}
Candidates: {', '.join(candidates)}

Think step by step about what information you need to make a good recommendation."""
            
            think_target = f"I need to analyze the user's sequential behavior and preferences. The sequence shows {self._describe_sequence_pattern(pred_sequence)}. I should request user analysis to understand their patterns better."
            
            examples.append(TrainingExample(
                input_prompt=think_prompt,
                target_output=think_target,
                metadata={
                    'type': 'manager_think',
                    'user_id': session['user_id'],
                    'sequence_length': len(pred_sequence)
                }
            ))
            
            # Act phase example
            # Template reference: previousWorks/MACRec/config/prompts/manager_prompt/analyse.json  
            act_prompt = f"""Based on your analysis, choose the best action:

Available actions:
- Analyse[user, user_id] - analyze user preferences
- Analyse[item, item_id] - analyze specific item
- Finish[item] - return final recommendation

Context: {think_target}"""
            
            act_target = f"Analyse[user, {session['user_id']}]"
            
            examples.append(TrainingExample(
                input_prompt=act_prompt,
                target_output=act_target,
                metadata={
                    'type': 'manager_act',
                    'user_id': session['user_id'],
                    'action': 'Analyse'
                }
            ))
        
        logger.info("✅ Generated %s Manager training examples", len(examples))
        return examples
    
    def generate_analyst_examples(self, num_examples: int = 1000) -> List[TrainingExample]:
        """Generate training examples for Analyst agent"""
        
        examples = []
        sessions = self.dataset.sessions[:num_examples]
        
        for session in sessions:
            pred_sequence = self.dataset.prepare_to_predict(session)
            target = self.dataset.extract_ground_truth(session)
            
            # User analysis example
            # Template reference: previousWorks/MACRec/config/prompts/agent_prompt/analyst.json
            analysis_prompt = f"""Analyze this user's preferences for sequential recommendation:

User ID: {session['user_id']}
Interaction sequence: {' → '.join(session['items'])}
Recent items: {' → '.join(pred_sequence)}

Provide insights about:
1. User preference patterns
2. Sequential behavior  
3. Next item prediction"""
            
            # Generate realistic analysis
            pattern = self._identify_pattern(session['items'])
            analysis_target = f"""User analysis for {session['user_id']}:
1. Preference patterns: Shows preference for {pattern['category']} items
2. Sequential behavior: {pattern['sequential_pattern']}
3. Recommendation: Based on the sequence, user likely needs {pattern['next_item_type']}"""
            
            examples.append(TrainingExample(
                input_prompt=analysis_prompt,
                target_output=analysis_target,
                metadata={
                    'type': 'analyst_user',
                    'user_id': session['user_id'],
                    'ground_truth': target
                }
            ))
        
        logger.info("✅ Generated %s Analyst training examples", len(examples))
        return examples
    
    def generate_all_training_data(self, output_dir: str, num_examples: int = 500):
        """Generate complete training dataset"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "📚 Generating training data (max %s examples each)...",
            num_examples,
        )
        
        # Generate examples for each agent type
        manager_examples = self.generate_manager_examples(num_examples)
        analyst_examples = self.generate_analyst_examples(num_examples)
        
        # Combine all examples
        all_examples = manager_examples + analyst_examples
        
        # Save training data
        training_file = output_path / "training_data.jsonl"
        with open(training_file, 'w') as f:
            for example in all_examples:
                json.dump({
                    'input': example.input_prompt,
                    'output': example.target_output,
                    'metadata': example.metadata
                }, f)
                f.write('\n')
        
        # Save summary
        summary = {
            'total_examples': len(all_examples),
            'manager_examples': len(manager_examples),
            'analyst_examples': len(analyst_examples),
            'dataset_info': self.dataset.get_statistics(),
            'config': self.config
        }
        
        summary_file = output_path / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("💾 Training data saved:")
        logger.info(
            "   📄 %s (%s examples)",
            training_file,
            len(all_examples),
        )
        logger.info("   📊 %s (summary)", summary_file)
        
        return all_examples
    
    def _describe_sequence_pattern(self, sequence: List[str]) -> str:
        """Describe patterns in a sequence"""
        if len(sequence) < 2:
            return "limited interaction history"
        
        patterns = [
            "progressive engagement with related items",
            "exploration of complementary products", 
            "focused interest in specific category",
            "diverse browsing behavior"
        ]
        
        return random.choice(patterns)
    
    def _identify_pattern(self, items: List[str]) -> Dict[str, str]:
        """Identify patterns in item sequence"""
        
        # Simple pattern identification (would be more sophisticated in practice)
        categories = ["electronics", "beauty", "books", "clothing", "home"]
        behaviors = [
            "buys complementary items in sequence",
            "explores different options before deciding", 
            "has consistent preferences",
            "alternates between different categories"
        ]
        
        return {
            'category': random.choice(categories),
            'sequential_pattern': random.choice(behaviors),
            'next_item_type': 'related accessories or complementary items'
        }


class ReflectionDataGenerator:
    """
    Generate training data for reflection learning.
    
    Creates examples showing how to reflect on recommendation attempts
    and provide improvement suggestions.
    
    Reference: MACRec_Analysis.md:114-176
    """
    
    def __init__(self, orchestrator: AgentOrchestrator, dataset: SequentialDataset):
        self.orchestrator = orchestrator
        self.dataset = dataset
        self.reflection_examples = []
        
        logger.info("🪞 Reflection data generator initialized")
    
    def generate_reflection_examples(self, num_examples: int = 200) -> List[ReflectionExample]:
        """Generate reflection training examples"""
        
        examples = []
        sessions = self.dataset.sessions[:num_examples]
        
        for session in sessions:
            pred_sequence = self.dataset.prepare_to_predict(session)
            target = self.dataset.extract_ground_truth(session)
            candidates, target_idx = self.dataset.create_candidate_pool(session)
            
            # Create request
            request = RecommendationRequest(
                user_id=session['user_id'],
                user_sequence=pred_sequence,
                candidates=candidates,
                ground_truth=target
            )
            
            # Get recommendation attempt
            try:
                response = self.orchestrator.recommend(request, max_iterations=1)
                
                # Build reflection example
                task_desc = f"Recommend next item for user {session['user_id']}"
                scratchpad = f"User sequence: {' → '.join(pred_sequence)}\nRecommendation: {response.recommendation}\nCandidates: {', '.join(candidates)}"
                
                # Determine correctness
                correct = response.recommendation == target
                
                # Generate reflection target
                if correct:
                    reflection_target = {
                        "correctness": True,
                        "reason": "Good recommendation based on user's sequential patterns",
                        "improvement": "Continue analyzing sequential behavior"
                    }
                else:
                    reflection_target = {
                        "correctness": False, 
                        "reason": f"Recommended {response.recommendation} but user actually chose {target}",
                        "improvement": "Better analyze user preferences and sequential patterns"
                    }
                
                examples.append(ReflectionExample(
                    task_description=task_desc,
                    attempt_scratchpad=scratchpad,
                    ground_truth=target,
                    reflection_target=reflection_target
                ))
                
            except Exception as e:
                # Skip failed attempts
                continue
            
            # Reset for next example
            self.orchestrator.reset_session()
        
        logger.info(
            "✅ Generated %s reflection training examples",
            len(examples),
        )
        return examples
    
    def save_reflection_data(self, examples: List[ReflectionExample], output_file: str):
        """Save reflection training data"""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                json.dump({
                    'task': example.task_description,
                    'attempt': example.attempt_scratchpad,
                    'ground_truth': example.ground_truth,
                    'reflection': example.reflection_target
                }, f)
                f.write('\n')
        
        logger.info("💾 Reflection data saved to %s", output_path)


def prepare_training_pipeline(dataset: SequentialDataset,
                             llm_provider: LLMProvider,
                             output_dir: str = "training_data",
                             num_examples: int = 500) -> Dict[str, Any]:
    """
    Complete training pipeline preparation.
    
    Args:
        dataset: Processed dataset
        llm_provider: LLM provider for orchestrator
        output_dir: Directory to save training data
        num_examples: Number of examples to generate
        
    Returns:
        Summary of generated training data
    """
    
    logger.info("🎓 Preparing complete training pipeline...")
    
    # Generate basic training data
    data_generator = TrainingDataGenerator(dataset)
    training_examples = data_generator.generate_all_training_data(output_dir, num_examples)
    
    # Generate reflection data
    from ..core import AgentOrchestrator
    orchestrator = AgentOrchestrator(llm_provider, ReflectionStrategy.REFLEXION)
    orchestrator.update_agent_data(
        user_data={s['user_id']: {'items': len(s['items'])} for s in dataset.sessions},
        item_data=dataset.item_to_name,
        user_histories={s['user_id']: s['items'] for s in dataset.sessions}
    )
    
    reflection_generator = ReflectionDataGenerator(orchestrator, dataset) 
    reflection_examples = reflection_generator.generate_reflection_examples(min(100, num_examples//5))
    reflection_generator.save_reflection_data(
        reflection_examples, 
        f"{output_dir}/reflection_data.jsonl"
    )
    
    # Create training summary
    summary = {
        'training_examples': len(training_examples),
        'reflection_examples': len(reflection_examples),
        'dataset_stats': dataset.get_statistics(),
        'output_directory': output_dir,
        'ready_for_training': True
    }
    
    logger.info("🎉 Training pipeline preparation completed!")
    logger.info("   📚 %s agent training examples", len(training_examples))
    logger.info("   🪞 %s reflection examples", len(reflection_examples))
    logger.info("   📁 Data saved to %s/", output_dir)
    
    return summary
