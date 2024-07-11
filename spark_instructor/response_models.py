"""Sample response models for parsing."""

from typing import Optional

from pydantic_spark.base import SparkBase  # type: ignore

from spark_instructor.completions import (
    AnthropicCompletion,
    DatabricksCompletion,
    OpenAICompletion,
)


class ResponseModelWithCompletion(SparkBase):
    """Sample response model for parsing objects with completion."""

    anthropic_completion: Optional[AnthropicCompletion] = None
    databricks_completion: Optional[DatabricksCompletion] = None
    openai_completion: Optional[OpenAICompletion] = None


class User(SparkBase):
    """A user."""

    name: str
    age: int
