"""This module defines Pydantic models for Anthropic's chat completion API responses.

It extends the base models from spark_instructor.completions.base to provide
specific implementations for Anthropic's API structure. These models can be used
to parse and validate Anthropic API responses, ensuring type safety and
providing easy access to response data.

Anthropic's completion models are not serializable out of the box, primarily due to Tools schema.
We may add support for Anthropic Tools later on.
"""

import json
import time
from typing import List, Literal, Optional, cast

from anthropic.types import Message
from anthropic.types.beta.prompt_caching import PromptCachingBetaMessage
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_use_block import ToolUseBlock
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from spark_instructor.completions.base import BaseCompletion, BaseModel

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


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


def anthropic_tool_call_to_openai_tool_call(block: ToolUseBlock) -> ChatCompletionMessageToolCall:
    """Convert an Anthropic tool call to an OpenAI tool call."""
    return ChatCompletionMessageToolCall(
        id=block.id,
        function=Function(name=block.name, arguments=json.dumps(block.input)),
        type="function",
    )


def transform_message_to_chat_completion(
    message: Message | PromptCachingBetaMessage, enable_created_at: bool = False
) -> ChatCompletion:
    """Transform a Message object to a ChatCompletion object.

    This function converts the structure of a Message (from the original schema)
    to a ChatCompletion (matching the target schema). It combines text blocks,
    converts tool use blocks to tool calls, maps stop reasons to finish reasons,
    and restructures usage information.

    Args:
        message (Message | PromptCachingBetaMessage): The original Message object to be transformed.
        enable_created_at (bool): Whether to include a unix timestamp.

    Returns:
        ChatCompletion: A new ChatCompletion object structured according to the target schema.

    Note:
        - Text blocks are combined into a single content string.
        - Tool use blocks are converted to ChatCompletionMessageToolCall objects.
        - The stop_reason is mapped to a corresponding finish_reason.
        - Usage information is restructured to fit the CompletionUsage model.
        - The timestamp is 0 by default but is updated when ``enable_created_at`` is True
    """
    # Convert content to a single string
    content = " ".join([block.text for block in message.content if isinstance(block, TextBlock)]) or None

    # Create tool calls from ToolUseBlock instances
    tool_calls = [
        anthropic_tool_call_to_openai_tool_call(block) for block in message.content if isinstance(block, ToolUseBlock)
    ]

    # Create the ChatCompletionMessage
    chat_message = ChatCompletionMessage(
        content=content, tool_calls=tool_calls if tool_calls else None, role="assistant"
    )

    # Map stop_reason to finish_reason
    finish_reason_map = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop", "tool_use": "tool_calls"}
    finish_reason: FinishReason = cast(
        FinishReason,
        finish_reason_map.get(message.stop_reason, "stop") if message.stop_reason else "stop",
    )

    # Create the Choice
    choice = Choice(finish_reason=finish_reason, index=0, message=chat_message)  # Assuming single choice

    # Create CompletionUsage
    prompt_tokens_details = (
        PromptTokensDetails(cached_tokens=message.usage.cache_read_input_tokens)
        if isinstance(message, PromptCachingBetaMessage)
        else None
    )
    usage = CompletionUsage(
        completion_tokens=message.usage.output_tokens,
        prompt_tokens=message.usage.input_tokens,
        total_tokens=message.usage.input_tokens + message.usage.output_tokens,
        prompt_tokens_details=prompt_tokens_details,
    )

    # Create the ChatCompletion
    return ChatCompletion(
        id=message.id,
        choices=[choice],
        created=int(time.time()) if enable_created_at else 0,
        model=message.model,
        object="chat.completion",
        usage=usage,
    )
