from unittest.mock import Mock, patch

import pytest

from spark_instructor.factory import O1Factory
from spark_instructor.types.base import SparkChatCompletionMessages


@pytest.fixture
def mock_get_o1_aclient():
    with patch("spark_instructor.factory.openai_factory.get_openai_aclient") as mock:
        yield mock


@pytest.fixture
def o1_factory(mock_get_o1_aclient):
    mock_get_o1_aclient.return_value = Mock()
    return O1Factory.from_config()


@pytest.fixture
def sample_messages(valid_image_url):
    return SparkChatCompletionMessages(
        root=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )


def test_o1_factory(sample_messages, o1_factory):
    assert o1_factory.format_messages(sample_messages) == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
