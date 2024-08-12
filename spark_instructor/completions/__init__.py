"""A package for defining completion object models.

This helps Spark understand the schema of our completions, regardless of the model provider.
"""

from .base import BaseCompletion
from .databricks_completions import DatabricksCompletion
from .ollama_completions import OllamaCompletion
from .openai_completions import OpenAICompletion

__all__ = [
    "OpenAICompletion",
    "DatabricksCompletion",
    "OllamaCompletion",
    "BaseCompletion",
]

try:
    from .anthropic_completions import (  # noqa: F401
        AnthropicCompletion,
        AnthropicContent,
        AnthropicUsage,
        transform_message_to_chat_completion,
    )

    __all__.extend(
        [
            "AnthropicCompletion",
            "AnthropicContent",
            "AnthropicUsage",
            "transform_message_to_chat_completion",
        ]
    )
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def is_anthropic_available():
    """Check if Anthropic-related modules are available."""
    return ANTHROPIC_AVAILABLE


__all__.append("is_anthropic_available")
