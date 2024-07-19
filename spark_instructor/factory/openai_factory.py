"""Module for creating an OpenAI factory."""

from typing import List, Optional

import instructor
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from spark_instructor.client.base import get_openai_aclient
from spark_instructor.factory.base import ClientFactory
from spark_instructor.types import SparkChatCompletionMessages


class OpenAIFactory(ClientFactory):
    """An OpenAI factory.

    Used as default factory for most providers.
    """

    @classmethod
    def from_config(
        cls,
        mode: Optional[instructor.Mode] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> "OpenAIFactory":
        """Create an OpenAI factory from custom inputs."""
        return cls(get_openai_aclient(mode or instructor.Mode.TOOLS, base_url, api_key))

    def format_messages(self, messages: SparkChatCompletionMessages) -> List[ChatCompletionMessageParam]:
        """Format messages by using default callable."""
        return [message() for message in messages.root]

    def format_completion(self, completion: ChatCompletion) -> ChatCompletion:
        """Return standard OpenAI completion message."""
        return completion
