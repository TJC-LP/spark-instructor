"""A package for defining completion object models.

This helps Spark understand the schema of our completions, regardless of the model provider.
"""

from .anthropic import AnthropicCompletion, AnthropicContent, AnthropicUsage
from .base import BaseCompletion
from .databricks import DatabricksCompletion
from .openai import OpenAICompletion

# You can also define __all__ to control what gets imported with "from completions import *"
__all__ = [
    "AnthropicCompletion",
    "AnthropicContent",
    "AnthropicUsage",
    "OpenAICompletion",
    "DatabricksCompletion",
    "BaseCompletion",
]
