"""This module defines Pydantic models for Anthropic's chat completion API responses.

It extends the base models from spark_instructor.completions.base to provide
specific implementations for Anthropic's API structure. These models can be used
to parse and validate Anthropic API responses, ensuring type safety and
providing easy access to response data.

Anthropic's completion models are not serializable out of the box, primarily due to Tools schema.
We may add support for Anthropic Tools later on.
"""

import json
from typing import List, Literal, Optional, cast

from anthropic.types import Message
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_use_block import ToolUseBlock
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionUsage

from spark_instructor.completions.base import BaseCompletion, BaseModel

# print("Schema to transform:")
# print(Message.schema_json(indent=2))
# print("Schema to match:")
# print(ChatCompletion.schema_json(indent=2))

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


def transform_message_to_chat_completion(message: Message) -> ChatCompletion:
    """Transform a Message object to a ChatCompletion object.

    This function converts the structure of a Message (from the original schema)
    to a ChatCompletion (matching the target schema). It combines text blocks,
    converts tool use blocks to tool calls, maps stop reasons to finish reasons,
    and restructures usage information.

    Args:
        message (Message): The original Message object to be transformed.

    Returns:
        ChatCompletion: A new ChatCompletion object structured according to the target schema.

    Note:
        - Text blocks are combined into a single content string.
        - Tool use blocks are converted to ChatCompletionMessageToolCall objects.
        - The stop_reason is mapped to a corresponding finish_reason.
        - Usage information is restructured to fit the CompletionUsage model.
        - The created timestamp is set to 0 and may need to be updated.
    """
    # Convert content to a single string
    content = " ".join(
        [block.text if isinstance(block, TextBlock) else f"[Tool Use: {block.name}]" for block in message.content]
    )

    # Create tool calls from ToolUseBlock instances
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=block.id, function=Function(name=block.name, arguments=json.dumps(block.input)), type="function"
        )
        for block in message.content
        if isinstance(block, ToolUseBlock)
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
    usage = CompletionUsage(
        completion_tokens=message.usage.output_tokens,
        prompt_tokens=message.usage.input_tokens,
        total_tokens=message.usage.input_tokens + message.usage.output_tokens,
    )

    # Create the ChatCompletion
    return ChatCompletion(
        id=message.id,
        choices=[choice],
        created=0,  # You might want to add a timestamp here
        model=message.model,
        object="chat.completion",
        usage=usage,
    )


# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     from instructor import Mode
#
#     from spark_instructor.client import get_anthropic_client
#     from spark_instructor.response_models import User
#
#     load_dotenv()
#
#     client = get_anthropic_client(mode=Mode.ANTHROPIC_TOOLS)
#     _, tool_completion = client.chat.completions.create_with_completion(
#         response_model=User,
#         model="claude-3-5-sonnet-20240620",
#         messages=[{"role": "user", "content": "Create a user"}],
#         max_tokens=400,
#     )
#     print(tool_completion.model_dump_json())
#     raw_message = Message(
#         **{
#             "id": "msg_01PPz1sb8XEJfT3NBLxn6C7y",
#             "content": [{"text": '{\n  "name": "John Doe",\n  "age": 30\n}', "type": "text"}],
#             "model": "claude-3-5-sonnet-20240620",
#             "role": "assistant",
#             "stop_reason": "end_turn",
#             "stop_sequence": None,
#             "type": "message",
#             "usage": {"input_tokens": 156, "output_tokens": 23},
#         }
#     )
#     print(transform_message_to_chat_completion(raw_message))
#     print(transform_message_to_chat_completion(tool_completion))
