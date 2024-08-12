"""Module for creating an Anthropic factory."""

from typing import List, Optional, Type, TypeVar, cast

import instructor
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from spark_instructor.client.base import get_anthropic_aclient
from spark_instructor.completions.anthropic_completions import (
    Message,
    transform_message_to_chat_completion,
)
from spark_instructor.factory.base import ClientFactory
from spark_instructor.types.base import SparkChatCompletionMessages
from spark_instructor.utils.image import convert_image_url_to_image_block_param

T = TypeVar("T", bound=BaseModel)


class AnthropicFactory(ClientFactory):
    """An Anthropic factory."""

    @classmethod
    def from_config(
        cls,
        mode: Optional[instructor.Mode] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "AnthropicFactory":
        """Get an anthropic client."""
        return cls(get_anthropic_aclient(mode or instructor.Mode.ANTHROPIC_JSON, base_url, api_key))

    def format_completion(self, completion: Message) -> ChatCompletion:
        """Format an Anthropic ``Message`` as an OpenAI ``ChatCompletion``."""
        return transform_message_to_chat_completion(completion, enable_created_at=True)

    def format_messages(
        self, messages: SparkChatCompletionMessages, ignore_system: bool = False
    ) -> List[ChatCompletionMessageParam]:
        """Format messages so that images serialize properly."""
        messages_unpacked = (
            [message() for message in messages.root if message.role != "system"]
            if ignore_system
            else [message() for message in messages.root]
        )
        for message in messages_unpacked:
            if message["role"] == "user":
                casted_message = cast(ChatCompletionUserMessageParam, message)
                if isinstance(casted_message["content"], list):
                    for i, content in enumerate(casted_message["content"]):
                        casted_content = cast(ChatCompletionContentPartParam, content)
                        if casted_content["type"] == "image_url":
                            casted_content_image = cast(ChatCompletionContentPartImageParam, casted_content)
                            casted_message["content"][i] = convert_image_url_to_image_block_param(
                                casted_content_image["image_url"]
                            )
        return messages_unpacked

    async def create(
        self,
        response_model: Type[T] | None,
        messages: SparkChatCompletionMessages,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int,
        **kwargs,
    ) -> ChatCompletion:
        """Create a completion."""
        system_message = [m for m in messages.root if m.role == "system"].pop()
        completion = await self.async_client.chat.completions.create(
            system=system_message.as_system()["content"],
            response_model=response_model,  # type: ignore
            messages=self.format_messages(messages, ignore_system=True),
            model=model,
            max_retries=max_retries,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return self.format_completion(cast(Message, completion))
