"""Module for creating an OpenAI factory."""

from typing import List, Optional, Tuple, Type, TypeVar, cast

import instructor
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from spark_instructor.client.base import get_openai_aclient
from spark_instructor.factory.base import ClientFactory
from spark_instructor.types import SparkChatCompletionMessages

T = TypeVar("T", bound=BaseModel)


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
        **kwargs,
    ) -> "OpenAIFactory":
        """Create an OpenAI factory from custom inputs."""
        return cls(get_openai_aclient(mode or instructor.Mode.TOOLS, base_url, api_key))

    def format_messages(self, messages: SparkChatCompletionMessages) -> List[ChatCompletionMessageParam]:
        """Format messages by using default callable."""
        return [message() for message in messages.root]

    def format_completion(self, completion: ChatCompletion) -> ChatCompletion:
        """Return standard OpenAI completion message."""
        return completion


class O1Factory(OpenAIFactory):
    """An OpenAI o1 factory.

    Used as the factory for o1 models which have unique functionality.
    """

    @classmethod
    def from_config(
        cls,
        mode: Optional[instructor.Mode] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "O1Factory":
        """Create an OpenAI factory from custom inputs."""
        return cls(get_openai_aclient(mode or instructor.Mode.JSON_O1, base_url, api_key))

    def format_messages(self, messages: SparkChatCompletionMessages) -> List[ChatCompletionMessageParam]:
        """Format messages by using default callable."""
        # Ignore system messages and images as they are not yet supported
        return [message(string_only=True) for message in messages.root if message.role != "system"]

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
        """Create a completion response.

        This method sends a request to the API and returns the completion response.
        If a response_model is provided, it structures the response accordingly.
        We handle o1 ``max_completion_tokens`` independently.

        Args:
            response_model (Type[T] | None): Optional Pydantic model class for structuring the response.
            messages (SparkChatCompletionMessages): The input messages for the completion.
            model (str): The name or identifier of the AI model to use.
            max_tokens (int): The maximum number of tokens in the completion response.
            temperature (float): The sampling temperature for the model's output.
                Always set to 1 regardless of input (not supported for o1 yet).
            max_retries (int): The maximum number of retry attempts for failed requests.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            ChatCompletion: The completion response, formatted according to the OpenAI standard.
        """
        completion = await self.async_client.chat.completions.create(
            response_model=response_model,  # type: ignore
            messages=self.format_messages(messages),
            model=model,
            max_retries=max_retries,
            max_completion_tokens=max_tokens,
            temperature=1,
            **kwargs,
        )
        return self.format_completion(cast(ChatCompletion, completion))

    async def create_with_completion(
        self,
        response_model: Type[T],
        messages: SparkChatCompletionMessages,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int,
        **kwargs,
    ) -> Tuple[T, ChatCompletion]:
        """Create a Pydantic model instance along with the full completion response.

        This method sends a request to the API, formats the response into a Pydantic model,
        and returns both the model instance and the full completion details.
        We handle o1 ``max_completion_tokens`` independently.

        Args:
            response_model (Type[T]): The Pydantic model class for structuring the response.
            messages (SparkChatCompletionMessages): The input messages for the completion.
            model (str): The name or identifier of the AI model to use.
            max_tokens (int): The maximum number of tokens in the completion response.
            temperature (float): The sampling temperature for the model's output.
                Always set to 1 regardless of input (not supported by o1 yet).
            max_retries (int): The maximum number of retry attempts for failed requests.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            Tuple[T, ChatCompletion]: A tuple containing the Pydantic model instance and the full completion.
        """
        pydantic_object, completion = await self.async_client.chat.completions.create_with_completion(
            response_model=response_model,
            messages=self.format_messages(messages),
            model=model,
            max_retries=max_retries,
            max_completion_tokens=max_tokens,
            temperature=1,
            **kwargs,
        )
        return pydantic_object, self.format_completion(completion)
