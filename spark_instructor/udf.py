"""Module for defining ``spark-instructor`` user-defined functions in Spark."""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from instructor import Mode
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from pyspark.sql.functions import udf
from pyspark.sql.types import StructField, StructType
from sparkdantic.model import create_spark_schema

from spark_instructor.client import ModelClass, get_instructor, infer_model_class
from spark_instructor.completions.anthropic import AnthropicCompletion
from spark_instructor.completions.databricks import DatabricksCompletion
from spark_instructor.completions.openai import OpenAICompletion


@dataclass
class ModelSerializer:
    """A class for serializing Pydantic models to Spark schemas.

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
        """Convert a string from camel case to snake case.

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
        """Generate a Spark StructType schema for the Pydantic models.

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
    response_model_type: Type[BaseModel]
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
    def model_serializer(self) -> ModelSerializer:
        """Get model serializer."""
        return ModelSerializer(self.response_model_type, self.completion_type)

    @property
    def spark_schema(self) -> StructType:
        """Get spark schema."""
        return self.model_serializer.spark_schema

    def create_object_from_messages(self, messages: list[ChatCompletionMessageParam], **kwargs) -> BaseModel:
        """Create a pydantic object response from messages."""
        client = get_instructor(self.model_class, self.mode)
        return client.chat.completions.create(
            model=self.model, response_model=self.response_model_type, messages=messages, **kwargs
        )

    def create_object_from_messages_udf(self, **kwargs) -> Callable:
        """Create a Spark UDF which returns a Spark-serializable object response."""

        def _func(messages: list[ChatCompletionMessageParam]) -> BaseModel:
            return self.create_object_from_messages(messages, **kwargs)

        schema = self.model_serializer.response_model_spark_schema

        @udf(returnType=schema)
        def func(messages: list[ChatCompletionMessageParam]) -> Dict[str, Any]:
            return _func(messages).model_dump()

        return func

    def create_object_and_completion_from_messages(
        self, messages: list[ChatCompletionMessageParam], **kwargs
    ) -> Tuple[BaseModel, Union[AnthropicCompletion, DatabricksCompletion, OpenAICompletion]]:
        """Create an object and completion using the ``instructor`` client."""
        client = get_instructor(self.model_class, self.mode)
        pydantic_object, completion = client.chat.completions.create_with_completion(
            model=self.model, response_model=self.response_model_type, messages=messages, **kwargs
        )
        return pydantic_object, completion

    def create_object_and_completion_from_messages_udf(self, **kwargs) -> Callable:
        """Create s Spark UDF which returns a Spark-serializable object response."""

        def _func(
            messages: list[ChatCompletionMessageParam],
        ) -> Tuple[BaseModel, Union[AnthropicCompletion, DatabricksCompletion, OpenAICompletion]]:
            return self.create_object_and_completion_from_messages(messages, **kwargs)

        schema = self.model_serializer.spark_schema
        response_model_name = self.model_serializer.response_model_name
        completion_model_name = self.model_serializer.completion_model_name

        @udf(returnType=schema)
        def func(messages: list[ChatCompletionMessageParam]) -> Dict[str, Any]:
            response_model, completion = _func(messages)
            return {response_model_name: response_model.model_dump(), completion_model_name: completion.model_dump()}

        return func
