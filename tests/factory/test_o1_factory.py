import pytest

from spark_instructor.factory import O1Factory
from spark_instructor.types.base import SparkChatCompletionMessages


@pytest.fixture
def sample_messages(valid_image_url):
    return SparkChatCompletionMessages(
        root=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )


def test_o1_factory(sample_messages):
    o1_factory = O1Factory.from_config()
    assert o1_factory.format_messages(sample_messages) == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
