"""Module for defining ``spark-instructor`` user-defined functions in Spark."""

from typing import Any, Optional, Tuple, Type, Union

from pydantic import BaseModel

from spark_instructor.client import MODEL_CLASS_ROUTE, ModelClass, infer_model_class
from spark_instructor.completions.anthropic import AnthropicCompletion
from spark_instructor.completions.databricks import DatabricksCompletion
from spark_instructor.completions.openai import OpenAICompletion


def create_object_from_messages(
    model: str, response_model: Type[BaseModel], messages: list[Any], model_class: Optional[ModelClass] = None, **kwargs
) -> BaseModel:
    """Create a completion using the ``instructor`` client."""
    if model_class is None:
        model_class = infer_model_class(model)
    client = MODEL_CLASS_ROUTE[model_class]()
    return client.chat.completions.create(model=model, response_model=response_model, messages=messages, **kwargs)


def create_object_and_completion_from_messages(
    model: str, response_model: Type[BaseModel], messages: list[Any], model_class: Optional[ModelClass] = None, **kwargs
) -> Tuple[BaseModel, Union[AnthropicCompletion, DatabricksCompletion, OpenAICompletion]]:
    """Create a completion using the ``instructor`` client."""
    if model_class is None:
        model_class = infer_model_class(model)
    client = MODEL_CLASS_ROUTE[model_class]()
    object, completion = client.chat.completions.create_with_completion(
        model=model, response_model=response_model, messages=messages, **kwargs
    )
    return object, completion


def create_serialized_completion(
    model: str, response_model: Type[BaseModel], messages: list[Any], model_class: Optional[ModelClass] = None, **kwargs
) -> dict[Any, Any]:
    """Process messages and return the result as a dictionary."""
    completion = create_object_from_messages(model, response_model, messages, model_class, **kwargs)
    return completion.model_dump()
