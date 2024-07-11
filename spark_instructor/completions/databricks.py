"""This module defines Pydantic models for Databricks' chat completion API responses.

Databricks uses OpenAI schema.
"""

from openai.types.chat import ChatCompletion as DatabricksCompletion

__all__ = ["DatabricksCompletion"]
