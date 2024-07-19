from unittest.mock import Mock, patch

import pytest

from spark_instructor.completions.anthropic_completions import Message
from spark_instructor.factory.anthropic_factory import AnthropicFactory
from spark_instructor.types.base import SparkChatCompletionMessages


# Mock external dependencies
@pytest.fixture
def mock_get_anthropic_aclient():
    with patch("spark_instructor.factory.anthropic_factory.get_anthropic_aclient") as mock:
        yield mock


@pytest.fixture
def mock_transform_message_to_chat_completion():
    with patch("spark_instructor.factory.anthropic_factory.transform_message_to_chat_completion") as mock:
        yield mock


@pytest.fixture
def anthropic_factory(mock_get_anthropic_aclient):
    mock_get_anthropic_aclient.return_value = Mock()
    return AnthropicFactory.from_config()


@pytest.fixture
def sample_messages(valid_image_url):
    return SparkChatCompletionMessages(
        root=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Check this image", "image_urls": [{"url": valid_image_url}]},
        ]
    )


def test_from_config(mock_get_anthropic_aclient):
    factory = AnthropicFactory.from_config(base_url="https://api.anthropic.com", api_key="test_key")
    assert isinstance(factory, AnthropicFactory)
    mock_get_anthropic_aclient.assert_called_once()


def test_format_completion(anthropic_factory, mock_transform_message_to_chat_completion):
    mock_message = Mock(spec=Message)
    mock_transform_message_to_chat_completion.return_value = {"id": "test", "choices": []}
    result = anthropic_factory.format_completion(mock_message)

    mock_transform_message_to_chat_completion.assert_called_once_with(mock_message, enable_created_at=True)
    assert result == {"id": "test", "choices": []}


def test_format_messages(anthropic_factory, sample_messages, valid_image_url, valid_base64):
    result = anthropic_factory.format_messages(sample_messages)

    assert len(result) == 3
    assert result[0] == {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    assert result[1] == {"role": "assistant", "content": "Hi there!"}
    assert result[2]["role"] == "user"
    assert isinstance(result[2]["content"], list)
    assert result[2]["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "data": valid_base64.split(",")[-1], "media_type": "image/jpeg"},
    }
    assert result[2]["content"][1] == {"type": "text", "text": "Check this image"}


def test_format_messages_no_images(anthropic_factory):
    messages = SparkChatCompletionMessages(
        root=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )

    result = anthropic_factory.format_messages(messages)

    assert len(result) == 2
    assert result[0] == {"role": "user", "content": [{"text": "Hello", "type": "text"}]}
    assert result[1] == {"role": "assistant", "content": "Hi there!"}
