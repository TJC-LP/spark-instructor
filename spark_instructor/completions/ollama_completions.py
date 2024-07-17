"""This module defines Pydantic models for Ollama chat completion API responses.

Ollama uses OpenAI schema.
"""

from openai.types.chat import ChatCompletion as OllamaCompletion

__all__ = ["OllamaCompletion"]
