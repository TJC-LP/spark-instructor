import pytest

from spark_instructor.types.base import (
    ChatCompletionMessageToolCallParamPD,
    ImageURLPD,
    SparkChatCompletionMessage,
)


# Fixtures
@pytest.fixture
def user_message():
    return SparkChatCompletionMessage(role="user", content="Hello, AI!")


@pytest.fixture
def user_message_with_image():
    return SparkChatCompletionMessage(
        role="user", content="Check this image", image_urls=[ImageURLPD(url="https://example.com/image.jpg")]
    )


@pytest.fixture
def assistant_message():
    return SparkChatCompletionMessage(role="assistant", content="Hello, human!")


@pytest.fixture
def assistant_message_with_tool_calls():
    tool_call = ChatCompletionMessageToolCallParamPD(
        id="call1", type="function", function={"name": "test_function", "arguments": "{}"}
    )
    return SparkChatCompletionMessage(role="assistant", content="Using a tool", tool_calls=[tool_call])


@pytest.fixture
def system_message():
    return SparkChatCompletionMessage(role="system", content="You are a helpful assistant.")


@pytest.fixture
def tool_message():
    return SparkChatCompletionMessage(role="tool", content="Tool result", tool_call_id="call1")


# Tests
def test_content_formatted(user_message, user_message_with_image):
    assert user_message.content_formatted() == [{"text": "Hello, AI!", "type": "text"}]
    assert user_message_with_image.content_formatted() == [
        {"image_url": {"url": "https://example.com/image.jpg"}, "type": "image_url"},
        {"text": "Check this image", "type": "text"},
    ]


def test_as_user(user_message, user_message_with_image):
    assert user_message.as_user() == {"content": [{"text": "Hello, AI!", "type": "text"}], "role": "user"}
    assert user_message.as_user(string_only=True) == {"content": "Hello, AI!", "role": "user"}
    assert user_message_with_image.as_user() == {
        "content": [
            {"image_url": {"url": "https://example.com/image.jpg"}, "type": "image_url"},
            {"text": "Check this image", "type": "text"},
        ],
        "role": "user",
    }


def test_as_assistant(assistant_message, assistant_message_with_tool_calls):
    assert assistant_message.as_assistant() == {"content": "Hello, human!", "role": "assistant"}
    assert assistant_message_with_tool_calls.as_assistant() == {
        "content": "Using a tool",
        "role": "assistant",
        "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "test_function", "arguments": "{}"}}],
    }


def test_as_system(system_message):
    assert system_message.as_system() == {"content": "You are a helpful assistant.", "role": "system"}


def test_as_tool(tool_message):
    assert tool_message.as_tool() == {"content": "Tool result", "role": "tool", "tool_call_id": "call1"}


def test_call_method(user_message, assistant_message, system_message, tool_message):
    assert user_message() == user_message.as_user()
    assert assistant_message() == assistant_message.as_assistant()
    assert system_message() == system_message.as_system()
    assert tool_message() == tool_message.as_tool()


def test_error_cases():
    with pytest.raises(AssertionError):
        SparkChatCompletionMessage(role="system", content=None).as_system()

    with pytest.raises(AssertionError):
        SparkChatCompletionMessage(role="tool", content="Tool result", tool_call_id=None).as_tool()


def test_with_name():
    named_user = SparkChatCompletionMessage(role="user", content="Hello", name="Alice")
    assert named_user.as_user() == {"content": [{"text": "Hello", "type": "text"}], "role": "user", "name": "Alice"}

    named_assistant = SparkChatCompletionMessage(role="assistant", content="Hi", name="Bot")
    assert named_assistant.as_assistant() == {"content": "Hi", "role": "assistant", "name": "Bot"}

    named_system = SparkChatCompletionMessage(role="system", content="System message", name="System")
    assert named_system.as_system() == {"content": "System message", "role": "system", "name": "System"}


def test_string_only_with_images():
    message = SparkChatCompletionMessage(
        role="user", content="Check this image", image_urls=[ImageURLPD(url="https://example.com/image.jpg")]
    )
    assert message.as_user(string_only=True) == {"content": "Check this image", "role": "user"}
