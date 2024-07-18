"""A package for defining instructor clients."""

from typing import TypeVar

from pydantic import BaseModel

from .client import (
    MODEL_CLASS_ROUTE,
    ModelClass,
    get_anthropic_client,
    get_databricks_client,
    get_instructor,
    get_ollama_client,
    get_openai_client,
    infer_model_class,
)

CompletionModel = TypeVar("CompletionModel", bound=BaseModel)
__all__ = [
    "CompletionModel",
    "get_instructor",
    "get_openai_client",
    "get_ollama_client",
    "get_databricks_client",
    "get_anthropic_client",
    "ModelClass",
    "infer_model_class",
    "MODEL_CLASS_ROUTE",
]