"""
Base agent classes for agentic recommendation system.
Based on MACRec's agent architecture.
"""

import time
import json
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from enum import Enum

from ..models.llm_provider import LLMProvider
from ..utils.logging import get_logger


class AgentType(Enum):
    """
    Enumeration of agent types in the system.
    
    Each agent has a specialized role:
    - MANAGER: Central coordination and decision making
    - ANALYST: User and item analysis 
    - REFLECTOR: Self-reflection and improvement
    - SEARCHER: External information retrieval
    - INTERPRETER: Result interpretation and formatting
    """
    MANAGER = "Manager"
    ANALYST = "Analyst"
    REFLECTOR = "Reflector"
    SEARCHER = "Searcher"
    INTERPRETER = "Interpreter"


class Agent(ABC):
    """
    Base agent class following MACRec architecture.
    
    Reference: previousWorks/MACRec/macrec/agents/base.py
    """
    
    def __init__(self, agent_type: AgentType, llm_provider: LLMProvider, 
                 config: Dict[str, Any] = None):
        self.agent_type = agent_type
        self.llm_provider = llm_provider
        self.config = config or {}
        self.logger = get_logger()
        
        # State tracking
        self.is_finished = False
        self.scratchpad = ""
        self.step_count = 0
        
        # Performance tracking
        self.total_calls = 0
        self.total_time = 0.0
    
    @abstractmethod
    def forward(self, **kwargs) -> str:
        """Main forward method. Must be implemented by subclasses."""
        pass
    
    def invoke(self, argument: Any, json_mode: bool = False, **kwargs) -> str:
        """
        Invoke agent with given argument.
        
        Args:
            argument: Agent-specific argument
            json_mode: Whether to expect JSON response
            **kwargs: Additional arguments
            
        Returns:
            Agent response
        """
        start_time = time.time()
        
        try:
            # Log invocation
            self.logger.log_agent_action(
                agent_name=self.agent_type.value,
                action_type="invocation",
                message=f"Invoked with argument: {argument}",
                context={'argument': argument, 'json_mode': json_mode},
                step_number=self.step_count
            )
            
            # Execute forward method
            result = self.forward(argument=argument, json_mode=json_mode, **kwargs)
            
            # Update metrics
            duration = time.time() - start_time
            self.total_calls += 1
            self.total_time += duration
            
            # Log result
            self.logger.log_agent_action(
                agent_name=self.agent_type.value,
                action_type="response",
                message=f"Response: {result[:100]}..." if len(result) > 100 else f"Response: {result}",
                context={'response_length': len(result)},
                step_number=self.step_count,
                duration_ms=duration * 1000
            )
            
            self.step_count += 1
            return result
            
        except Exception as e:
            # Log error
            self.logger.log_agent_action(
                agent_name=self.agent_type.value,
                action_type="error",
                message=f"Error during invocation: {str(e)}",
                context={'argument': argument, 'error_type': type(e).__name__}
            )
            raise
    
    def reset(self):
        """Reset agent state"""
        self.is_finished = False
        self.scratchpad = ""
        self.step_count = 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0
        
        return {
            'agent_type': self.agent_type.value,
            'total_calls': self.total_calls,
            'total_time': self.total_time,
            'avg_time_per_call': avg_time,
            'current_step': self.step_count
        }


class ToolAgent(Agent):
    """
    Agent with tool access capabilities.
    Used for Analyst and Searcher agents.
    
    Reference: MACRec_Analysis.md:81-198
    """
    
    def __init__(self, agent_type: AgentType, llm_provider: LLMProvider, 
                 tools: Dict[str, Any] = None, **kwargs):
        super().__init__(agent_type, llm_provider, **kwargs)
        self.tools = tools or {}
        self.command_history = []
    
    def execute_command(self, command: str, **kwargs) -> str:
        """
        Execute a tool command.
        
        Args:
            command: Command string (e.g., "UserInfo[524]")
            **kwargs: Additional arguments
            
        Returns:
            Command execution result
        """
        start_time = time.time()
        
        # Parse command
        cmd_type, cmd_args = self._parse_command(command)
        
        # Log command execution
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="tool_usage",
            message=f"Executing: {command}",
            context={
                'command_type': cmd_type,
                'arguments': cmd_args,
                'raw_command': command,
            }
        )

        # Execute command
        try:
            if cmd_type in self.tools:
                result = self.tools[cmd_type](cmd_args, **kwargs)
            else:
                result = f"Unknown command: {cmd_type}"
            
            # Track command
            duration = time.time() - start_time
            self.command_history.append({
                'command': command,
                'result': result,
                'duration': duration
            })
            
            # Log result
            self.logger.log_agent_action(
                agent_name=self.agent_type.value,
                action_type="tool_result",
                message=f"Tool result: {result[:100]}..." if len(result) > 100 else f"Tool result: {result}",
                context={'command': command, 'result': result},
                duration_ms=duration * 1000
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            self.logger.log_agent_action(
                agent_name=self.agent_type.value,
                action_type="error",
                message=error_msg,
                context={'command': command, 'error': str(e)}
            )
            return error_msg
    
    def _parse_command(self, command: str) -> tuple[str, List[str]]:
        """
        Parse command string into type and arguments.
        
        Examples:
        - "UserInfo[524]" -> ("UserInfo", ["524"])
        - "UserHistory[524, 10]" -> ("UserHistory", ["524", "10"])
        """
        if '[' not in command or ']' not in command:
            return command, []
        
        cmd_type = command.split('[')[0]
        args_str = command.split('[')[1].rstrip(']')
        
        if not args_str:
            args = []
        else:
            # Split by comma and clean
            args = [arg.strip() for arg in args_str.split(',')]
        
        return cmd_type, args


class ReflectionStrategy(Enum):
    """
    Reflection strategies for Reflector agent.
    
    Reference: MACRec_Analysis.md:118-125
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial'
    REFLEXION = 'reflection'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflection'


def create_agent_prompt(agent_type: AgentType, task_context: Dict[str, Any], 
                       **kwargs) -> str:
    """
    Create prompt for agent based on type and context.
    
    Template references:
    - Manager: previousWorks/MACRec/config/prompts/manager_prompt/analyse.json
    - Analyst: previousWorks/MACRec/config/prompts/agent_prompt/analyst.json  
    - Reflector: previousWorks/MACRec/config/prompts/agent_prompt/reflector.json
    
    Args:
        agent_type: Type of agent
        task_context: Current task context
        **kwargs: Additional prompt parameters
        
    Returns:
        Formatted prompt string
    """
    base_prompt = f"You are a {agent_type.value} agent in a recommendation system."
    
    if agent_type == AgentType.MANAGER:
        return f"""{base_prompt}
        
Your role is to orchestrate the recommendation process through thinking and acting.

Current task: {task_context.get('task', 'sequential recommendation')}
Context: {task_context.get('context', '')}

Available actions:
- Analyse[user/item, id] - request analysis from Analyst
- Search[query] - request external information
- Finish[result] - return final recommendation

Think step by step about what information you need, then choose an action.
"""
    
    elif agent_type == AgentType.ANALYST:
        return f"""{base_prompt}
        
Your role is to analyze user preferences and item characteristics.

Available commands:
- UserInfo[id] - get user profile
- ItemInfo[id] - get item details
- UserHistory[id, k] - get user's recent interactions
- ItemHistory[id, k] - get item's interaction history
- Finish[analysis] - return analysis result

Analyze the given data and provide insights.
"""
    
    elif agent_type == AgentType.REFLECTOR:
        return f"""{base_prompt}
        
Your role is to reflect on recommendation attempts and provide improvement guidance.

You will be given a previous reasoning trial. Determine if the answer is correct,
provide reasons for your judgment, and suggest improvements.

Return JSON format: {{"correctness": boolean, "reason": "explanation"}}
"""
    
    else:
        return base_prompt
