import pytest

from spark_instructor.completions import (
    AnthropicCompletion,
    DatabricksCompletion,
    OpenAICompletion,
)
from spark_instructor.completions.anthropic_completions import (
    Message,
    transform_message_to_chat_completion,
)


@pytest.fixture
def anthropic_tools_message() -> Message:
    raw_tools_message = {
        "id": "msg_01KF8Mhuze5KccdvbCT9ebrB",
        "content": [
            {
                "id": "toolu_01NtJx2hrzLJMR3vZF1xUw6t",
                "input": {"name": "<UNKNOWN>", "age": 30},
                "name": "User",
                "type": "tool_use",
            }
        ],
        "model": "claude-3-5-sonnet-20240620",
        "role": "assistant",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 919, "output_tokens": 102},
    }
    return Message(**raw_tools_message)  # type: ignore


@pytest.fixture
def anthropic_json_message() -> Message:
    raw_json_message = {
        "id": "msg_01PPz1sb8XEJfT3NBLxn6C7y",
        "content": [{"text": '{\n  "name": "John Doe",\n  "age": 30\n}', "type": "text"}],
        "model": "claude-3-5-sonnet-20240620",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 156, "output_tokens": 23},
    }
    return Message(**raw_json_message)  # type: ignore


def test_anthropic_conversion(anthropic_tools_message, anthropic_json_message):
    transformed_tools_message = transform_message_to_chat_completion(anthropic_tools_message)
    assert isinstance(transformed_tools_message, OpenAICompletion)
    assert transformed_tools_message.model_dump() == {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": None,
                    "function_call": None,
                    "refusal": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {"arguments": '{"name": ' '"<UNKNOWN>", ' '"age": ' "30}", "name": "User"},
                            "id": "toolu_01NtJx2hrzLJMR3vZF1xUw6t",
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 0,
        "id": "msg_01KF8Mhuze5KccdvbCT9ebrB",
        "model": "claude-3-5-sonnet-20240620",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {"completion_tokens": 102, "prompt_tokens": 919, "total_tokens": 1021},
    }

    transformed_json_message = transform_message_to_chat_completion(anthropic_json_message)
    assert isinstance(transformed_json_message, OpenAICompletion)
    assert transformed_json_message.model_dump() == {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": '{\n  "name": "John Doe",\n  "age": 30\n}',
                    "function_call": None,
                    "refusal": None,
                    "role": "assistant",
                    "tool_calls": None,
                },
            }
        ],
        "created": 0,
        "id": "msg_01PPz1sb8XEJfT3NBLxn6C7y",
        "model": "claude-3-5-sonnet-20240620",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {"completion_tokens": 23, "prompt_tokens": 156, "total_tokens": 179},
    }


def test_anthropic_completion(anthropic_json_message):
    raw_data = anthropic_json_message.model_dump()
    ac = AnthropicCompletion(**raw_data)
    assert ac.model_dump() == raw_data


def test_databricks_completion():
    raw_data = {
        "id": "chatcmpl_bc431f25-524a-4557-a2be-cd769c23a7ee",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": '```json\n{\n  "name": "John Doe",\n  "age": 30\n}\n```',
                    "role": "assistant",
                    "function_call": None,
                    "refusal": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1720553141,
        "model": "dbrx-instruct-032724",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {"completion_tokens": 21, "prompt_tokens": 169, "total_tokens": 190},
    }
    dc = DatabricksCompletion(**raw_data)
    assert dc.model_dump() == raw_data


def test_openai_completion():
    raw_data = {
        "id": "chatcmpl-9jAl2c3a8EiSPQPKAw0JcRpTy7i3E",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "function_call": None,
                    "refusal": None,
                    "tool_calls": [
                        {
                            "id": "call_FZ3fXr1wUlJvV2hvRY7EzdA5",
                            "function": {"arguments": '{"name":"John Doe","age":30}', "name": "User"},
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 1720553140,
        "model": "gpt-4o-2024-05-13",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": "fp_d576307f90",
        "usage": {"completion_tokens": 10, "prompt_tokens": 64, "total_tokens": 74},
    }
    oc = OpenAICompletion(**raw_data)
    assert oc.model_dump() == raw_data
