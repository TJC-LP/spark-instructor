"""This module defines Pydantic models for Anthropic's chat completion API responses.

It extends the base models from spark_instructor.completions.base to provide
specific implementations for Anthropic's API structure. These models can be used
to parse and validate Anthropic API responses, ensuring type safety and
providing easy access to response data.

Anthropic's completion models are not serializable out of the box, primarily due to Tools schema.
We may add support for Anthropic Tools later on.
"""

from typing import List, Optional

from spark_instructor.completions.base import BaseCompletion, BaseModel


class AnthropicContent(BaseModel):
    """Represents a content item in an Anthropic chat completion response.

    Attributes:
        text (str): The text content of the message.
        type (str): The type of the content, typically "text" for Anthropic responses.
    """

    text: str
    type: str


class AnthropicUsage(BaseModel):
    """Represents the token usage information for an Anthropic chat completion.

    Attributes:
        input_tokens (int): The number of tokens in the input prompt.
        output_tokens (int): The number of tokens in the generated completion.
    """

    input_tokens: int
    output_tokens: int


class AnthropicCompletion(BaseCompletion):
    """Represents a complete Anthropic chat completion response.

    This class extends BaseCompletion to include all fields specific to
    Anthropic's API response structure.

    Attributes:
        content (List[AnthropicContent]): A list of content items in the completion.
        role (str): The role of the message (e.g., "assistant").
        stop_reason (str): The reason why the completion stopped.
        stop_sequence (Optional[str]): The stop sequence used, if any. Defaults to None.
        type (str): The type of the completion, typically "message" for Anthropic.
        usage (AnthropicUsage): An object containing token usage information.
    """

    content: List[AnthropicContent]
    role: str
    stop_reason: str
    stop_sequence: Optional[str] = None
    type: str
    usage: AnthropicUsage
