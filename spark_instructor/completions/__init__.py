"""A package for defining completion object models.

This helps Spark understand the schema of our completions, regardless of the model provider.
"""

from .anthropic_completions import AnthropicCompletion, AnthropicContent, AnthropicUsage
from .base import BaseCompletion
from .databricks_completions import DatabricksCompletion
from .ollama_completions import OllamaCompletion
from .openai_completions import OpenAICompletion

__all__ = [
    "AnthropicCompletion",
    "AnthropicContent",
    "AnthropicUsage",
    "OpenAICompletion",
    "DatabricksCompletion",
    "OllamaCompletion",
    "BaseCompletion",
]
