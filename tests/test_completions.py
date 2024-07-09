from spark_instructor.completions import (
    AnthropicCompletion,
    DatabricksCompletion,
    OpenAICompletion,
)


def test_anthropic_completion():
    raw_data = {
        "id": "msg_01PPz1sb8XEJfT3NBLxn6C7y",
        "content": [{"text": '{\n  "name": "John Doe",\n  "age": 30\n}', "type": "text"}],
        "model": "claude-3-5-sonnet-20240620",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 156, "output_tokens": 23},
    }
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
