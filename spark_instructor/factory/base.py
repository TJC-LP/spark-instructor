"""Module for defining abstract classes for instructor client retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type, TypeVar

import instructor
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from spark_instructor.completions import OpenAICompletion
from spark_instructor.types import SparkChatCompletionMessages

T = TypeVar("T", bound=BaseModel)


@dataclass
class ClientFactory(ABC):
    """An abstract base class for defining client factories to route API traffic to different providers.

    This class serves as a template for creating specific client factories for various AI model providers
    (e.g., OpenAI, Anthropic, etc.). It provides a standardized interface for creating clients,
    formatting messages, and handling completions across different API providers.

    Attributes:
        async_client (instructor.AsyncInstructor): An asynchronous instructor client for making API calls.

    The ClientFactory class defines several abstract methods that must be implemented by subclasses:
    - from_config: For creating a factory instance from configuration parameters.
    - format_messages: For converting Spark-specific message formats to provider-specific formats.
    - format_completion: For standardizing completion responses from different providers.

    It also provides concrete implementations for creating completions with or without Pydantic models.

    Usage:
        Subclass ClientFactory for each API provider, implementing the abstract methods according to
        the provider's specific requirements. Then use these subclasses to create provider-specific
        clients and handle API interactions in a standardized way across your application.

    Example:
        ```python
        class OpenAIFactory(ClientFactory):
            @classmethod
            def from_config(cls, mode=None, base_url=None, api_key=None, **kwargs):
                # Implementation for creating an OpenAI client
                ...

            def format_messages(self, messages):
                # Implementation for formatting messages for OpenAI
                ...

            def format_completion(self, completion):
                # Implementation for formatting OpenAI completion
                ...

        # Using the factory
        openai_factory = OpenAIFactory.from_config(api_key="your-api-key")
        completion = await openai_factory.create(...)
        ```
    """

    async_client: instructor.AsyncInstructor

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        mode: Optional[instructor.Mode] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "ClientFactory":
        """Create a client factory instance from configuration parameters.

        This method should be implemented to initialize the factory with provider-specific settings.

        Args:
            mode (Optional[instructor.Mode]): The mode of operation for the instructor client.
            base_url (Optional[str]): The base URL for the API endpoint.
            api_key (Optional[str]): The API key for authentication.
            **kwargs: Additional keyword arguments for provider-specific configuration.

        Returns:
            ClientFactory: An instance of the concrete ClientFactory subclass.
        """
        pass

    @abstractmethod
    def format_messages(self, messages: SparkChatCompletionMessages) -> List[ChatCompletionMessageParam]:
        """Format Spark completion messages to provider-specific chat completion messages.

        This method should be implemented to convert the standardized Spark message format
        to the format expected by the specific API provider.

        Args:
            messages (SparkChatCompletionMessages): The messages in Spark format.

        Returns:
            List[ChatCompletionMessageParam]: The formatted messages ready for the API provider.
        """
        pass

    @abstractmethod
    def format_completion(self, completion: Any) -> OpenAICompletion:
        """Format a provider-specific completion to a standardized OpenAI-style completion.

        This method should be implemented to convert the completion response from the
        provider's format to a standardized OpenAICompletion format.

        Args:
            completion (Any): The completion response from the API provider.

        Returns:
            OpenAICompletion: The formatted completion in OpenAI-compatible format.
        """
        pass

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

        Args:
            response_model (Type[T]): The Pydantic model class for structuring the response.
            messages (SparkChatCompletionMessages): The input messages for the completion.
            model (str): The name or identifier of the AI model to use.
            max_tokens (int): The maximum number of tokens in the completion response.
            temperature (float): The sampling temperature for the model's output.
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
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return pydantic_object, self.format_completion(completion)

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

        Args:
            response_model (Type[T] | None): Optional Pydantic model class for structuring the response.
            messages (SparkChatCompletionMessages): The input messages for the completion.
            model (str): The name or identifier of the AI model to use.
            max_tokens (int): The maximum number of tokens in the completion response.
            temperature (float): The sampling temperature for the model's output.
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
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return self.format_completion(completion)
