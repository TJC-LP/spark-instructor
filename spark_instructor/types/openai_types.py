"""A module for defining OpenAI classes for each completion type."""

from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)

from spark_instructor.utils.types import typeddict_to_pydantic

__all__ = ["ImageURLPD", "ChatCompletionMessageToolCallParamPD", "ImageURL", "ChatCompletionMessageToolCallParam"]

ImageURLPD = typeddict_to_pydantic(ImageURL)
ChatCompletionMessageToolCallParamPD = typeddict_to_pydantic(ChatCompletionMessageToolCallParam)
