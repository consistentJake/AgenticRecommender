"""
Agent orchestrator for coordinating multi-agent recommendation workflow.
Based on MACRec's agent communication protocol.
"""

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..agents import Manager, Analyst, Reflector
from ..agents.base import AgentType, ReflectionStrategy
from ..models.llm_provider import LLMProvider
from ..utils.logging import get_component_logger, get_logger


@dataclass
class RecommendationRequest:
    """Request structure for recommendation"""
    user_id: str
    user_sequence: List[str]
    candidates: List[str]
    ground_truth: Optional[str] = None
    context: Dict[str, Any] = None


@dataclass
class RecommendationResponse:
    """Response structure from recommendation"""
    recommendation: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class AgentOrchestrator:
    """
    Central orchestrator coordinating all agents in the recommendation workflow.
    
    Responsibilities:
    - Agent lifecycle management
    - Inter-agent communication protocol
    - Workflow orchestration
    - Performance monitoring
    
    Reference: MACRec_Analysis.md:28-52, 199-259
    """
    
    def __init__(self, llm_provider: LLMProvider, 
                 reflection_strategy: ReflectionStrategy = ReflectionStrategy.REFLEXION,
                 config: Dict[str, Any] = None):
        
        self.config = config or {}
        self.logger = get_logger()
        self.component_logger = get_component_logger("core.orchestrator")
        
        # Initialize agents
        self.manager = Manager(llm_provider, llm_provider, config)
        self.analyst = Analyst(llm_provider, config=config)
        self.reflector = Reflector(llm_provider, reflection_strategy, config)
        
        # Communication state
        self.conversation_history = []
        self.current_session_id = f"session_{int(time.time())}"
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        
        self.component_logger.info(
            "🎭 Orchestrator initialized with %s reflection",
            reflection_strategy.value,
        )
    
    def recommend(self, request: RecommendationRequest, 
                 max_iterations: int = 5) -> RecommendationResponse:
        """
        Main recommendation workflow with agent coordination.
        
        Args:
            request: Recommendation request
            max_iterations: Maximum agent interaction cycles
            
        Returns:
            Final recommendation response
        """
        start_time = time.time()
        self.total_requests += 1
        
        # Log request
        self.logger.log_agent_action(
            agent_name="Orchestrator",
            action_type="request",
            message=f"Processing recommendation for user {request.user_id}",
            context={
                'user_id': request.user_id,
                'sequence_length': len(request.user_sequence),
                'num_candidates': len(request.candidates),
                'session_id': self.current_session_id
            }
        )
        
        # Initialize task context
        task_context = {
            'user_id': request.user_id,
            'user_sequence': request.user_sequence,
            'candidates': request.candidates,
            'ground_truth': request.ground_truth,
            'session_id': self.current_session_id
        }
        
        if request.context:
            task_context.update(request.context)
        
        # Execute workflow
        try:
            result = self._execute_workflow(task_context, max_iterations)
            self.successful_requests += 1
            
            # Log success
            duration = time.time() - start_time
            self.logger.log_agent_action(
                agent_name="Orchestrator",
                action_type="success",
                message=f"Recommendation completed: {result.recommendation}",
                context={
                    'recommendation': result.recommendation,
                    'confidence': result.confidence,
                    'iterations_used': len(self.conversation_history)
                },
                duration_ms=duration * 1000
            )
            
            return result
            
        except Exception as e:
            # Log error
            self.logger.log_agent_action(
                agent_name="Orchestrator",
                action_type="error",
                message=f"Recommendation failed: {str(e)}",
                context={'error_type': type(e).__name__, 'user_id': request.user_id}
            )
            
            # Return fallback response
            return RecommendationResponse(
                recommendation=request.candidates[0] if request.candidates else "no_recommendation",
                confidence=0.1,
                reasoning=f"Fallback due to error: {str(e)}",
                metadata={'error': True, 'error_message': str(e)}
            )
    
    def _execute_workflow(self, task_context: Dict[str, Any], 
                         max_iterations: int) -> RecommendationResponse:
        """
        Execute the multi-agent recommendation workflow.
        
        Workflow:
        1. Manager analyzes situation and plans actions
        2. Agents execute specialized tasks (analysis, search)
        3. Manager synthesizes information and makes recommendation
        4. Reflector evaluates and provides feedback (if enabled)
        
        Reference: MACRec_Analysis.md:28-52
        """
        iteration = 0
        final_recommendation = None
        accumulated_context = task_context.copy()
        
        while iteration < max_iterations and final_recommendation is None:
            iteration += 1
            
            # Manager thinking phase
            thought = self.manager.think(accumulated_context)
            self._log_communication("Manager", "thought", thought, iteration)
            
            # Manager action phase  
            action_type, argument = self.manager.act(accumulated_context)
            self._log_communication("Manager", "action", f"{action_type}[{argument}]", iteration)
            
            # Execute action based on type
            if action_type == "Analyse":
                result = self._handle_analysis_request(argument, accumulated_context)
                accumulated_context['last_analysis'] = result
                
            elif action_type == "Search":
                result = self._handle_search_request(argument, accumulated_context)
                accumulated_context['last_search'] = result
                
            elif action_type == "Finish":
                final_recommendation = argument
                accumulated_context['final_recommendation'] = final_recommendation
                break
            
            else:
                # Unknown action, let manager decide next step
                accumulated_context['error'] = f"Unknown action: {action_type}"
        
        # If no recommendation made, use fallback
        if final_recommendation is None:
            final_recommendation = task_context['candidates'][0] if task_context.get('candidates') else "no_recommendation"
            accumulated_context['final_recommendation'] = final_recommendation
        
        # Generate reflection if enabled
        reflection_insights = self._generate_reflection(task_context, accumulated_context)
        
        # Create response
        response = RecommendationResponse(
            recommendation=final_recommendation,
            confidence=self._calculate_confidence(accumulated_context),
            reasoning=self._extract_reasoning(accumulated_context),
            metadata={
                'iterations': iteration,
                'workflow_trace': self.conversation_history,
                'reflection': reflection_insights,
                'session_id': self.current_session_id
            }
        )
        
        return response
    
    def _handle_analysis_request(self, argument: Any, context: Dict[str, Any]) -> str:
        """Handle analysis request from Manager"""
        self._log_communication("Orchestrator", "routing", f"Routing analysis request: {argument}")
        
        # Route to analyst
        analysis_result = self.analyst.forward(
            argument,
            json_mode=True,
            request_context=self._extract_request_context(context),
            manager_notes=self._collect_manager_notes(),
        )
        
        self._log_communication("Analyst", "analysis", analysis_result)
        return analysis_result
    
    def _handle_search_request(self, argument: Any, context: Dict[str, Any]) -> str:
        """Handle search request from Manager"""
        self._log_communication("Orchestrator", "routing", f"Routing search request: {argument}")
        
        # For now, return mock search result
        # In full implementation, this would route to Searcher agent
        search_result = f"Search results for: {argument}"
        
        self._log_communication("Searcher", "search", search_result)
        return search_result
    
    def _generate_reflection(self, original_context: Dict[str, Any], 
                           final_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reflection on the recommendation process"""
        if self.reflector.reflection_strategy == ReflectionStrategy.NONE:
            return {"reflection_enabled": False}
        
        # Build scratchpad from conversation history
        scratchpad = "\n".join([
            f"{entry['agent']}: {entry['content']}" 
            for entry in self.conversation_history
        ])
        
        # Generate reflection
        reflection = self.reflector.forward(
            input_task=f"Recommend item for user {original_context['user_id']}",
            scratchpad=scratchpad,
            first_attempt=final_context.get('final_recommendation'),
            ground_truth=original_context.get('ground_truth')
        )
        
        self._log_communication("Reflector", "reflection", reflection)
        
        # Parse reflection insights
        try:
            reflection_data = json.loads(self.reflector.reflection_output)
        except:
            reflection_data = {"reason": reflection, "correctness": "unknown"}
        
        return reflection_data

    def _extract_request_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'user_id': context.get('user_id'),
            'user_sequence': context.get('user_sequence'),
            'candidates': context.get('candidates'),
            'ground_truth': context.get('ground_truth'),
        }

    def _collect_manager_notes(self) -> str:
        if not self.conversation_history:
            return ""

        notes = []
        for entry in self.conversation_history[-5:]:
            notes.append(f"{entry['agent']}({entry.get('type')}): {entry.get('content')}")
        return "\n".join(notes)
    
    def _calculate_confidence(self, context: Dict[str, Any]) -> float:
        """Calculate confidence score for recommendation"""
        base_confidence = 0.5
        
        # Increase confidence if we have analysis
        if context.get('last_analysis'):
            base_confidence += 0.2
        
        # Increase confidence if we have search results
        if context.get('last_search'):
            base_confidence += 0.1
        
        # Increase confidence based on user sequence length
        sequence_length = len(context.get('user_sequence', []))
        base_confidence += min(0.2, sequence_length * 0.05)
        
        return min(1.0, base_confidence)
    
    def _extract_reasoning(self, context: Dict[str, Any]) -> str:
        """Extract reasoning from workflow context"""
        reasoning_parts = []
        
        if context.get('last_analysis'):
            reasoning_parts.append(f"User analysis: {context['last_analysis'][:100]}...")
        
        if context.get('last_search'):
            reasoning_parts.append(f"Search insights: {context['last_search'][:100]}...")
        
        reasoning_parts.append(f"Final recommendation: {context.get('final_recommendation', 'none')}")
        
        return " | ".join(reasoning_parts)
    
    def _log_communication(self, agent_name: str, message_type: str, 
                          content: str, iteration: int = None):
        """Log inter-agent communication"""
        comm_entry = {
            'agent': agent_name,
            'type': message_type,
            'content': content,
            'iteration': iteration,
            'timestamp': time.time()
        }
        
        self.conversation_history.append(comm_entry)
        
        # Also log through main logger
        self.logger.log_agent_action(
            agent_name=agent_name,
            action_type=f"communication_{message_type}",
            message=content[:100] + "..." if len(content) > 100 else content,
            context={
                'iteration': iteration,
                'session_id': self.current_session_id,
                'full_content': content,
            }
        )
    
    def update_agent_data(self, user_data: Dict[str, Any] = None,
                         item_data: Dict[str, Any] = None,
                         user_histories: Dict[str, List] = None,
                         item_histories: Dict[str, List] = None):
        """Update data sources for all agents"""
        self.analyst.update_data_sources(
            user_data=user_data,
            item_data=item_data, 
            user_histories=user_histories,
            item_histories=item_histories
        )
        
        self.component_logger.info("📊 Orchestrator data updated")
    
    def reset_session(self):
        """Reset for new recommendation session"""
        self.conversation_history = []
        self.current_session_id = f"session_{int(time.time())}"
        
        # Reset individual agents
        self.manager.reset()
        self.analyst.reset()
        self.reflector.clear_reflections()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'orchestrator': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
                'current_session': self.current_session_id
            },
            'agents': {
                'manager': self.manager.get_performance_stats(),
                'analyst': self.analyst.get_performance_stats(), 
                'reflector': self.reflector.get_performance_stats()
            },
            'communication': {
                'messages_in_session': len(self.conversation_history),
                'last_session_id': self.current_session_id
            }
        }
