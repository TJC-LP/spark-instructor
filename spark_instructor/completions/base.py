"""A module for defining base classes for each completion type."""

from typing import Any, Optional

from pydantic_spark.base import SparkBase  # type: ignore


class Usage(SparkBase):
    """Usage model defining number of tokens."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class BaseCompletion(SparkBase):
    """Completion model for defining what model was used and number of tokens."""

    id: str
    model: str
    usage: Usage


class FunctionCall(SparkBase):
    """Function call model for defining AI calls to functions."""

    arguments: str
    name: str


class ToolCall(SparkBase):
    """Tool call model for defining AI tool usage."""

    id: str
    function: FunctionCall
    type: str


class BaseMessage(SparkBase):
    """Message model for defining LLM messages."""

    role: str
    content: Optional[str] = None


class BaseChoice(SparkBase):
    """Choice model for defining LLM choices."""

    finish_reason: str
    index: int
    logprobs: Optional[Any] = None
