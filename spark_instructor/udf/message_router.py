"""Module for ``MessageRouter``."""

import re
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import instructor
from instructor import Mode
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from pyspark.sql.functions import udf
from pyspark.sql.types import StructField, StructType
from sparkdantic.model import create_spark_schema

from spark_instructor import is_anthropic_available
from spark_instructor.client import ModelClass, get_instructor, infer_model_class
from spark_instructor.completions.databricks_completions import DatabricksCompletion
from spark_instructor.completions.openai_completions import OpenAICompletion

if TYPE_CHECKING:
    from spark_instructor.completions.anthropic_completions import AnthropicCompletion

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


@dataclass
class ModelSerializer:
    """A class for serializing Pydantic models to Spark schemas.

    This class provides functionality to convert Pydantic models to Spark StructType schemas,
    with fields named according to the snake case version of the model class names.

    Attributes:
        response_model_type (Type[BaseModel]): The Pydantic model type for the main data.
        completion_model_type (Type[BaseModel]): The Pydantic model type for the completion data.
    """

    response_model_type: Type[BaseModel] | None
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
    def response_model_name(self) -> str | None:
        """Pydantic model field name in snake-case."""
        if self.response_model_type is not None:
            return self.to_snake_case(self.response_model_type.__name__)
        return None

    @property
    def response_model_spark_schema(self) -> StructType | None:
        """Response model spark schema."""
        if self.response_model_type is not None:
            return create_spark_schema(self.response_model_type)
        return None

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
        return (
            StructType(
                [
                    StructField(self.response_model_name, self.response_model_spark_schema, nullable=True),
                    StructField(self.completion_model_name, self.completion_model_spark_schema, nullable=True),
                ]
            )
            if self.response_model_spark_schema and self.response_model_name
            else StructType(
                [
                    StructField(self.completion_model_name, self.completion_model_spark_schema, nullable=True),
                ]
            )
        )


@dataclass
class MessageRouter(Generic[ResponseModel]):
    """A wrapper for serializing ``instructor`` calls and managing model interactions.

    This class provides methods to create Pydantic objects and completions from chat messages,
    and to generate Spark UDFs for these operations.

    Attributes:
        model (str): The name of the model to use.
        response_model_type (Type[ResponseModel]): The Pydantic model type for the response.
        model_class (Optional[ModelClass]): The class of the model (e.g., ``ModelClass.OPENAI``).
            If not provided, it will be inferred based on the ``model``.
        mode (Optional[Mode]): The mode for the instructor client.
        base_url (Optional[str]): The base URL for API calls.
        api_key (Optional[str]): The API key for authentication.

    Notes:
        **WARNING:** ``MessageRouter`` is now deprecated. Use ``instruct`` instead.
    """

    model: str
    response_model_type: Type[ResponseModel]
    model_class: Optional[ModelClass] = None
    mode: Optional[Mode] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        """Initialize the ``model_class`` if not provided.

        The ``model_class`` will be inferred based on the ``model`` attribute.
        """
        warnings.warn(
            "`MessageRouter` is deprecated and may be removed in future versions. " "Please use `instruct` instead.",
            DeprecationWarning,
        )
        if self.model_class is None:
            self.model_class = infer_model_class(self.model)

    @property
    def completion_type(self) -> Union[Type["AnthropicCompletion"], Type[DatabricksCompletion], Type[OpenAICompletion]]:
        """Get the appropriate completion type based on the ``model_class`` attribute.

        Returns:
            Union[Type[AnthropicCompletion], Type[DatabricksCompletion], Type[OpenAICompletion]]:
                The completion type corresponding to the model class.
        """
        if self.model_class == ModelClass.ANTHROPIC:
            if is_anthropic_available():
                from spark_instructor.completions.anthropic_completions import (
                    AnthropicCompletion,
                )

                return AnthropicCompletion
            else:
                raise ImportError(
                    "Please install ``anthropic`` by running ``pip install anthropic`` "
                    "or ``pip install spark-instructor[anthropic]``"
                )
        if self.model_class == ModelClass.DATABRICKS:
            return DatabricksCompletion
        return OpenAICompletion

    @property
    def model_serializer(self) -> ModelSerializer:
        """Get the model serializer for the response model type and completion type.

        Returns:
            ModelSerializer: An instance of ModelSerializer.
        """
        return ModelSerializer(self.response_model_type, self.completion_type)

    @property
    def spark_schema(self) -> StructType:
        """Get the Spark schema for the model.

        Returns:
            StructType: The Spark schema corresponding to the model.
        """
        return self.model_serializer.spark_schema

    def get_instructor(self) -> instructor.Instructor:
        """Get an instance of the instructor client.

        Returns:
            instructor.Instructor: An initialized instructor client.
        """
        return get_instructor(
            model_class=self.model_class, mode=self.mode, api_key=self.api_key, base_url=self.base_url
        )

    def create_object_from_messages(self, messages: list[ChatCompletionMessageParam], **kwargs: Any) -> ResponseModel:
        """Create a Pydantic object response from chat messages.

        Args:
            messages (list[ChatCompletionMessageParam]): The list of chat messages.
            **kwargs (Any): Additional keyword arguments for the chat completion.

        Returns:
            ResponseModel: A Pydantic object representing the response.
        """
        client = self.get_instructor()
        return client.chat.completions.create(
            model=self.model, response_model=self.response_model_type, messages=messages, **kwargs
        )

    def create_object_from_messages_udf(self, **kwargs: Any) -> Callable:
        """Create a Spark UDF that returns a ``StructType`` response based on the ``response_model_type`` attribute.

        Args:
            **kwargs (Any): Additional keyword arguments for the chat completion.

        Returns:
            Callable: A Spark UDF that takes messages and returns a serialized object.
        """

        def _func(messages: list[ChatCompletionMessageParam]) -> ResponseModel:
            return self.create_object_from_messages(messages, **kwargs)

        schema = self.model_serializer.response_model_spark_schema
        assert schema, "Null response models are not supported by `MessageRouter`"

        @udf(returnType=schema)
        def func(messages: list[ChatCompletionMessageParam]) -> Dict[str, Any]:
            return _func(messages).model_dump()

        return func

    def create_object_and_completion_from_messages(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> Tuple[ResponseModel, Union["AnthropicCompletion", DatabricksCompletion, OpenAICompletion]]:
        """Create a Pydantic object response and completion using the ``instructor`` client.

        The completion will be of the type corresponding to the ``model_class`` attribute.

        Args:
            messages (list[ChatCompletionMessageParam]): The list of chat messages.
            **kwargs (Any): Additional keyword arguments for the chat completion.

        Returns:
            Tuple[ResponseModel, Union[AnthropicCompletion, DatabricksCompletion, OpenAICompletion]]:
                A tuple containing the Pydantic object and the completion.
        """
        client = self.get_instructor()
        pydantic_object, completion = client.chat.completions.create_with_completion(
            model=self.model, response_model=self.response_model_type, messages=messages, **kwargs
        )
        return pydantic_object, completion

    def create_object_and_completion_from_messages_udf(self, **kwargs: Any) -> Callable:
        """Create a Spark UDF that returns a ``StructType``.

        The response will be based on the ``response_model_type`` and ``model_class`` attributes.

        Args:
            **kwargs (Any): Additional keyword arguments for the chat completion.

        Returns:
            Callable: A Spark UDF that takes messages and returns a dictionary with
                      serialized object and completion.
        """

        def _func(
            messages: list[ChatCompletionMessageParam],
        ) -> Tuple[ResponseModel, Union["AnthropicCompletion", DatabricksCompletion, OpenAICompletion]]:
            return self.create_object_and_completion_from_messages(messages, **kwargs)

        schema = self.model_serializer.spark_schema
        response_model_name = self.model_serializer.response_model_name
        assert response_model_name, "Null response models are not supported by `MessageRouter`"
        completion_model_name = self.model_serializer.completion_model_name

        @udf(returnType=schema)
        def func(messages: list[ChatCompletionMessageParam]) -> Dict[str, Any]:
            response_model, completion = _func(messages)
            return {response_model_name: response_model.model_dump(), completion_model_name: completion.model_dump()}

        return func
