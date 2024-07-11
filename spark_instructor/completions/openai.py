"""This module defines Pydantic models for OpenAI's chat completion API responses.

OpenAI already provides a well-defined schema which is spark-serializable.
"""

from openai.types.chat import ChatCompletion as OpenAICompletion

__all__ = ["OpenAICompletion"]
