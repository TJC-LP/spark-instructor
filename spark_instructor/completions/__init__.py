"""A package for defining completion object models.

This helps Spark understand the schema of our completions, regardless of the model provider.
"""

from .anthropic import AnthropicCompletion, AnthropicContent, AnthropicUsage
from .base import BaseChoice, BaseCompletion, BaseMessage, SparkBase, Usage
from .databricks import DatabricksChoice, DatabricksCompletion, DatabricksMessage
from .openai import OpenAIChoice, OpenAICompletion, OpenAIMessage

# You can also define __all__ to control what gets imported with "from completions import *"
__all__ = [
    "AnthropicCompletion",
    "AnthropicContent",
    "AnthropicUsage",
    "OpenAICompletion",
    "OpenAIMessage",
    "OpenAIChoice",
    "DatabricksCompletion",
    "DatabricksMessage",
    "DatabricksChoice",
    "BaseCompletion",
    "BaseMessage",
    "BaseChoice",
    "Usage",
    "SparkBase",
]
