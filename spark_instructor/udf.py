"""Module for defining ``spark-instructor`` user-defined functions in Spark."""

import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type, Union

from instructor import Mode
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from pyspark.sql.types import StructField, StructType
from sparkdantic.model import create_spark_schema

from spark_instructor.client import ModelClass, get_instructor, infer_model_class
from spark_instructor.completions.anthropic import AnthropicCompletion
from spark_instructor.completions.databricks import DatabricksCompletion
from spark_instructor.completions.openai import OpenAICompletion


@dataclass
class ModelSerializer:
    """
    A class for serializing Pydantic models to Spark schemas.

    This class provides functionality to convert Pydantic models to Spark StructType schemas,
    with fields named according to the snake case version of the model class names.

    Attributes:
        response_model_type (Type[BaseModel]): The Pydantic model type for the main data.
        completion_model_type (Type[BaseModel]): The Pydantic model type for the completion data.
    """

    response_model_type: Type[BaseModel]
    completion_model_type: Type[BaseModel]

    @staticmethod
    def to_snake_case(name: str) -> str:
        """
        Convert a string from camel case to snake case.

        This method takes a camel case string and converts it to snake case.
        For example, 'CamelCase' becomes 'camel_case'.

        Args:
            name (str): The camel case string to convert.

        Returns:
            str: The snake case version of the input string.
        """
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    @property
    def response_model_name(self) -> str:
        """Pydantic model field name in snake-case."""
        return self.to_snake_case(self.response_model_type.__name__)

    @property
    def response_model_spark_schema(self) -> StructType:
        """Response model spark schema."""
        return create_spark_schema(self.response_model_type)

    @property
    def completion_model_name(self) -> str:
        """Pydantic model field name in snake-case."""
        return self.to_snake_case(self.completion_model_type.__name__)

    @property
    def completion_model_spark_schema(self) -> StructType:
        """Response model spark schema."""
        return create_spark_schema(self.completion_model_type)

    @property
    def spark_schema(self) -> StructType:
        """
        Generate a Spark StructType schema for the Pydantic models.

        This property creates a Spark schema that includes two fields:
        one for the main Pydantic model and one for the completion model.
        The field names are derived from the snake case versions of the model class names.

        Returns:
            StructType: A Spark StructType containing two StructFields, one for each model.
                        Each field is named after the snake case version of its model class name
                        and contains the corresponding Spark schema.
        """
        return StructType(
            [
                StructField(self.response_model_name, self.response_model_spark_schema, nullable=True),
                StructField(self.completion_model_name, self.completion_model_spark_schema, nullable=True),
            ]
        )


@dataclass
class MessageRouter:
    """Wrapper for serializing ``instructor`` calls."""

    model: str
    pydantic_model_type: Type[BaseModel]
    model_class: Optional[ModelClass] = None
    mode: Optional[Mode] = None

    def __post_init__(self):
        """Get model class from string."""
        if self.model_class is None:
            self.model_class = infer_model_class(self.model)

    @property
    def completion_type(self) -> Union[Type[AnthropicCompletion], Type[DatabricksCompletion], Type[OpenAICompletion]]:
        """Get completion type."""
        if self.model_class == ModelClass.ANTHROPIC:
            return AnthropicCompletion
        if self.model_class == ModelClass.DATABRICKS:
            return DatabricksCompletion
        return OpenAICompletion

    @property
    def spark_schema(self) -> StructType:
        """Get spark schema."""
        model_serializer = ModelSerializer(self.pydantic_model_type, self.completion_type)
        return model_serializer.spark_schema

    def create_object_from_messages(self, messages: list[ChatCompletionMessageParam], **kwargs) -> BaseModel:
        """Create a pydantic object response from messages."""
        client = get_instructor(self.model_class, self.mode)
        return client.chat.completions.create(
            model=self.model, response_model=self.pydantic_model_type, messages=messages, **kwargs
        )


def create_object_from_messages(
    model: str,
    response_model: Type[BaseModel],
    messages: list[Any],
    model_class: Optional[ModelClass] = None,
    mode: Optional[Mode] = None,
    **kwargs
) -> BaseModel:
    """Create an object using the ``instructor`` client."""
    if model_class is None:
        model_class = infer_model_class(model)
    client = get_instructor(model_class, mode)
    return client.chat.completions.create(model=model, response_model=response_model, messages=messages, **kwargs)


def create_object_and_completion_from_messages(
    model: str,
    response_model: Type[BaseModel],
    messages: list[Any],
    model_class: Optional[ModelClass] = None,
    mode: Optional[Mode] = None,
    **kwargs
) -> Tuple[BaseModel, Union[AnthropicCompletion, DatabricksCompletion, OpenAICompletion]]:
    """Create an object and completion using the ``instructor`` client."""
    if model_class is None:
        model_class = infer_model_class(model)
    client = get_instructor(model_class, mode)
    pydantic_object, completion = client.chat.completions.create_with_completion(
        model=model, response_model=response_model, messages=messages, **kwargs
    )
    return pydantic_object, completion


def create_serialized_completion(
    model: str, response_model: Type[BaseModel], messages: list[Any], model_class: Optional[ModelClass] = None, **kwargs
) -> dict[Any, Any]:
    """Process messages and return the result as a dictionary."""
    completion = create_object_from_messages(model, response_model, messages, model_class, **kwargs)
    return completion.model_dump()
