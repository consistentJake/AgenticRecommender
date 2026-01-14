"""
Core module for agentic recommendation system.

Provides prompt templates and management.
"""

from .prompts import PromptManager, PromptTemplate, PromptType, get_prompt_manager

__all__ = ['PromptManager', 'PromptTemplate', 'PromptType', 'get_prompt_manager']