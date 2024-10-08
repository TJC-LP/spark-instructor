"""A module for defining base classes for each completion type."""

from typing import List, Literal, Optional, Union

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field, RootModel

from spark_instructor.types.openai_types import (
    ChatCompletionMessageToolCallParamPD,
    ImageURL,
    ImageURLPD,
)


def format_image_url_pd(image_url_pd: BaseModel) -> ImageURL:
    """Format the ImageURLPD object into an ImageURL object."""
    model_dump = image_url_pd.model_dump()
    if not model_dump["detail"]:
        return ImageURL(url=model_dump["url"])
    return ImageURL(url=model_dump["url"], detail=model_dump["detail"])


class SparkChatCompletionMessage(BaseModel):
    """A Spark-serializable chat completion message that can be used for any role in a conversation.

    This class provides a flexible structure for representing messages in a chat completion context,
    supporting various roles (user, assistant, system, tool) and additional attributes like image URLs
    and tool calls.

    Attributes:
        role (Literal["user", "assistant", "system", "tool"]): The role of the message sender.
        content (Optional[str]): The text content of the message.
        image_urls (Optional[List[ImageURLPD]]): List of image URLs associated with the message.
        name (Optional[str]): The name of the entity associated with the message.
        tool_calls (Optional[List[ChatCompletionMessageToolCallParamPD]]): Tool calls made in the message (for 'assistant' role).
        tool_call_id (Optional[str]): The ID of the tool call (for 'tool' role).
        cache_control (Optional[bool]): Whether to use Anthropic's prompt caching feature (beta).

    The class provides methods to format the message for different roles and serialize it to OpenAI-compatible types.
    """  # noqa: E501

    role: Literal["user", "assistant", "system", "tool"] = Field("user", description="The role of the message.")
    content: Optional[str] = Field(None, description="The text content of the message.")
    image_urls: Optional[List[ImageURLPD]] = Field(None, description="The image urls of the message.")  # type: ignore
    name: Optional[str] = Field(None, description="The name of the relevant chat entity")
    tool_calls: Optional[List[ChatCompletionMessageToolCallParamPD]] = Field(  # type: ignore
        None, description="The tool calls of the message (`assistant` role only)"
    )
    tool_call_id: Optional[str] = Field(None, description="The tool call id of the message (`tool` role only).")
    cache_control: Optional[bool] = Field(None, description="Whether to use Anthropic cache control.")

    def content_formatted(self) -> List[Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]]:
        """Format the message content, including any images, into a list of content parts.

        This method prepares the message content for use in API calls, handling both text and image content.

        Returns:
            List[Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]]:
                A list of formatted content parts, including text and images.
        """
        results: List[Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]] = (
            []
            if not self.image_urls
            else [
                ChatCompletionContentPartImageParam(image_url=format_image_url_pd(image_url), type="image_url")
                for image_url in self.image_urls
            ]
        )
        if self.content:
            results.append(ChatCompletionContentPartTextParam(text=self.content, type="text"))
        return results

    def as_user(self, string_only: bool = False) -> ChatCompletionUserMessageParam:
        """Format the message as a user message.

        This method prepares the message for use as a user input in a chat completion.

        Args:
            string_only (bool): If True, return only the text content for messages with images.

        Returns:
            ChatCompletionUserMessageParam: The formatted user message.
        """
        if self.name:
            if string_only and self.content:
                return ChatCompletionUserMessageParam(content=self.content, role="user", name=self.name)
            return ChatCompletionUserMessageParam(content=self.content_formatted(), role="user", name=self.name)
        if string_only and self.content:
            return ChatCompletionUserMessageParam(content=self.content, role="user")
        return ChatCompletionUserMessageParam(content=self.content_formatted(), role="user")

    def as_assistant(self) -> ChatCompletionAssistantMessageParam:
        """Format the message as an assistant message.

        This method prepares the message for use as an assistant response in a chat completion,
        including any tool calls if present.

        Returns:
            ChatCompletionAssistantMessageParam: The formatted assistant message.
        """
        if self.name:
            if self.tool_calls:
                return ChatCompletionAssistantMessageParam(
                    content=self.content,
                    role="assistant",
                    name=self.name,
                    tool_calls=[call.model_dump() for call in self.tool_calls],  # type: ignore
                )
            return ChatCompletionAssistantMessageParam(content=self.content, role="assistant", name=self.name)
        if self.tool_calls:
            return ChatCompletionAssistantMessageParam(
                content=self.content,
                role="assistant",
                tool_calls=[call.model_dump() for call in self.tool_calls],  # type: ignore
            )
        return ChatCompletionAssistantMessageParam(content=self.content, role="assistant")

    def as_system(self) -> ChatCompletionSystemMessageParam:
        """Format the message as a system message.

        This method prepares the message for use as a system instruction in a chat completion.

        Returns:
            ChatCompletionSystemMessageParam: The formatted system message.

        Raises:
            AssertionError: If the content is empty.
        """
        assert self.content, "`content` must not be empty"
        if self.name:
            return ChatCompletionSystemMessageParam(content=self.content, role="system", name=self.name)
        return ChatCompletionSystemMessageParam(content=self.content, role="system")

    def as_tool(self) -> ChatCompletionToolMessageParam:
        """Format the message as a tool message.

        This method prepares the message for use as a tool response in a chat completion.

        Returns:
            ChatCompletionToolMessageParam: The formatted tool message.

        Raises:
            AssertionError: If either content or tool_call_id is empty.
        """
        assert self.content and self.tool_call_id, "`content` and `tool_call_id` must not be empty"
        return ChatCompletionToolMessageParam(content=self.content, role="tool", tool_call_id=self.tool_call_id)

    def __call__(self, *args, string_only: bool = False, **kwargs) -> ChatCompletionMessageParam:
        """Serialize the message to an OpenAI-compatible type.

        This method allows the object to be called directly, returning the appropriate
        message type based on the role.

        Args:
            string_only (bool): If True, return only the text content.
                The resulting user message will only have string content.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ChatCompletionMessageParam: The serialized message in OpenAI-compatible format.
        """
        if self.role == "system":
            return self.as_system()
        if self.role == "user":
            return self.as_user(string_only=string_only)
        if self.role == "assistant":
            return self.as_assistant()
        return self.as_tool()


SparkChatCompletionMessages = RootModel[List[SparkChatCompletionMessage]]
"""A root model representing a list of SparkChatCompletionMessage objects.

This model is used to represent an entire conversation or a series of messages
in a chat completion context. It provides a convenient way to handle multiple
messages as a single entity while maintaining Pydantic's validation and serialization capabilities.

Usage:
    ```python
    messages = SparkChatCompletionMessages(root=[
        SparkChatCompletionMessage(role="user", content="Hello"),
        SparkChatCompletionMessage(role="assistant", content="Hi there!")
    ])
    ```
"""
