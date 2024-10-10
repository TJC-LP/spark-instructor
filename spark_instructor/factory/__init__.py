"""Package for routing factories."""

#
from spark_instructor import is_anthropic_available

from .base import ClientFactory
from .databricks_factory import DatabricksFactory
from .ollama_factory import OllamaFactory
from .openai_factory import O1Factory, OpenAIFactory

__all__ = ["DatabricksFactory", "OllamaFactory", "OpenAIFactory", "ClientFactory", "O1Factory"]

if is_anthropic_available():
    from .anthropic_factory import AnthropicFactory  # noqa: F401

    __all__.append("AnthropicFactory")
