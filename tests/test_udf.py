from typing import Type

import pytest
from pydantic import BaseModel
from pyspark.sql.types import IntegerType, StringType, StructType

from spark_instructor.client import MODEL_CLASS_ROUTE, ModelClass
from spark_instructor.udf import MessageRouter, ModelSerializer, OpenAICompletion
from spark_instructor.utils import zero_shot_prompt


# Mock classes for testing
@pytest.fixture
def test_response_model() -> Type[BaseModel]:
    class TestResponseModel(BaseModel):
        name: str
        age: int

    return TestResponseModel


@pytest.fixture
def test_completion_model() -> Type[BaseModel]:
    class TestCompletionModel(BaseModel):
        token_count: int
        model: str

    return TestCompletionModel


# Fixture to create ModelSerializer instance
@pytest.fixture
def model_serializer(test_response_model, test_completion_model):
    return ModelSerializer(test_response_model, test_completion_model)


# Tests
def test_to_snake_case():
    assert ModelSerializer.to_snake_case("CamelCase") == "camel_case"
    assert ModelSerializer.to_snake_case("snake_case") == "snake_case"
    assert ModelSerializer.to_snake_case("ThisIsALongString") == "this_is_a_long_string"


def test_response_model_name(model_serializer):
    assert model_serializer.response_model_name == "test_response_model"


def test_completion_model_name(model_serializer):
    assert model_serializer.completion_model_name == "test_completion_model"


def test_response_model_spark_schema(model_serializer):
    schema = model_serializer.response_model_spark_schema
    assert isinstance(schema, StructType)
    assert len(schema.fields) == 2
    assert schema.fields[0].name == "name"
    assert schema.fields[0].dataType == StringType()
    assert schema.fields[1].name == "age"
    assert schema.fields[1].dataType == IntegerType()


def test_completion_model_spark_schema(model_serializer):
    schema = model_serializer.completion_model_spark_schema
    assert isinstance(schema, StructType)
    assert len(schema.fields) == 2
    assert schema.fields[0].name == "token_count"
    assert schema.fields[0].dataType == IntegerType()
    assert schema.fields[1].name == "model"
    assert schema.fields[1].dataType == StringType()


def test_spark_schema(model_serializer):
    schema = model_serializer.spark_schema
    assert isinstance(schema, StructType)
    assert len(schema.fields) == 2
    assert schema.fields[0].name == "test_response_model"
    assert schema.fields[1].name == "test_completion_model"
    assert isinstance(schema.fields[0].dataType, StructType)
    assert isinstance(schema.fields[1].dataType, StructType)


class MockModel(BaseModel):
    result: str


@pytest.fixture
def mock_openai_completion():
    return OpenAICompletion(
        **{
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
                                "function": {"arguments": '{"result":"test"}', "name": "MockModel"},
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
    )


@pytest.fixture
def mock_client(mocker, mock_openai_completion):
    client = mocker.Mock()
    client.chat.completions.create.return_value = MockModel(result="test")
    client.chat.completions.create_with_completion.return_value = (MockModel(result="test"), mock_openai_completion)
    return client


@pytest.fixture
def mock_model_class_route(mocker, mock_client):

    def get_mock_client(*args, **kwargs):
        return mock_client

    mocker.patch.dict(MODEL_CLASS_ROUTE, {ModelClass.OPENAI: get_mock_client})


@pytest.fixture
def message_router() -> MessageRouter:
    return MessageRouter(model="gpt-3.5-turbo", response_model_type=MockModel)


def test_create_object_from_messages(mock_model_class_route, message_router):
    result = message_router.create_object_from_messages(
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert result.model_dump() == {"result": "test"}
    MODEL_CLASS_ROUTE[ModelClass.OPENAI]().chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo", response_model=MockModel, messages=[{"role": "user", "content": "Hello"}]
    )


def test_create_object_from_messages_udf(spark, mock_model_class_route, message_router):
    data = [
        ("Hello",),
    ]
    parser_udf = message_router.create_object_from_messages_udf(max_tokens=400)
    df = (
        spark.createDataFrame(data, ["content"])
        .withColumn("messages", zero_shot_prompt(user_message_column="content"))
        .withColumn("response", parser_udf("messages"))
    )
    assert df.schema.jsonValue() == {
        "fields": [
            {"metadata": {}, "name": "content", "nullable": True, "type": "string"},
            {
                "metadata": {},
                "name": "messages",
                "nullable": False,
                "type": {
                    "containsNull": False,
                    "elementType": {
                        "keyType": "string",
                        "type": "map",
                        "valueContainsNull": True,
                        "valueType": "string",
                    },
                    "type": "array",
                },
            },
            {
                "metadata": {},
                "name": "response",
                "nullable": True,
                "type": {
                    "fields": [{"metadata": {}, "name": "result", "nullable": False, "type": "string"}],
                    "type": "struct",
                },
            },
        ],
        "type": "struct",
    }


def test_create_object_and_completion_from_messages(mock_model_class_route, message_router, mock_openai_completion):
    result, completion = message_router.create_object_and_completion_from_messages(
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert result.model_dump() == {"result": "test"}
    assert completion == mock_openai_completion
    MODEL_CLASS_ROUTE[ModelClass.OPENAI]().chat.completions.create.create_object_and_completion_from_messages(
        model="gpt-3.5-turbo", response_model=MockModel, messages=[{"role": "user", "content": "Hello"}]
    )


def test_create_object_and_completion_from_messages_udf(spark, mock_model_class_route, message_router):
    data = [
        ("Hello",),
    ]
    parser_udf = message_router.create_object_and_completion_from_messages_udf(max_tokens=400)
    df = (
        spark.createDataFrame(data, ["content"])
        .withColumn("messages", zero_shot_prompt(user_message_column="content"))
        .withColumn("response", parser_udf("messages"))
    )
    assert df.schema.jsonValue() == {
        "fields": [
            {"metadata": {}, "name": "content", "nullable": True, "type": "string"},
            {
                "metadata": {},
                "name": "messages",
                "nullable": False,
                "type": {
                    "containsNull": False,
                    "elementType": {
                        "keyType": "string",
                        "type": "map",
                        "valueContainsNull": True,
                        "valueType": "string",
                    },
                    "type": "array",
                },
            },
            {
                "metadata": {},
                "name": "response",
                "nullable": True,
                "type": {
                    "fields": [
                        {
                            "metadata": {},
                            "name": "mock_model",
                            "nullable": True,
                            "type": {
                                "fields": [{"metadata": {}, "name": "result", "nullable": False, "type": "string"}],
                                "type": "struct",
                            },
                        },
                        {
                            "metadata": {},
                            "name": "chat_completion",
                            "nullable": True,
                            "type": {
                                "fields": [
                                    {"metadata": {}, "name": "id", "nullable": False, "type": "string"},
                                    {
                                        "metadata": {},
                                        "name": "choices",
                                        "nullable": False,
                                        "type": {
                                            "containsNull": False,
                                            "elementType": {
                                                "fields": [
                                                    {
                                                        "metadata": {},
                                                        "name": "finish_reason",
                                                        "nullable": False,
                                                        "type": "string",
                                                    },
                                                    {
                                                        "metadata": {},
                                                        "name": "index",
                                                        "nullable": False,
                                                        "type": "integer",
                                                    },
                                                    {
                                                        "metadata": {},
                                                        "name": "logprobs",
                                                        "nullable": True,
                                                        "type": {
                                                            "fields": [
                                                                {
                                                                    "metadata": {},
                                                                    "name": "content",
                                                                    "nullable": True,
                                                                    "type": {
                                                                        "containsNull": True,
                                                                        "elementType": {
                                                                            "fields": [
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "token",
                                                                                    "nullable": False,
                                                                                    "type": "string",
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "bytes",
                                                                                    "nullable": True,
                                                                                    "type": {
                                                                                        "containsNull": False,
                                                                                        "elementType": "integer",
                                                                                        "type": "array",
                                                                                    },
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "logprob",
                                                                                    "nullable": False,
                                                                                    "type": "double",
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "top_logprobs",
                                                                                    "nullable": False,
                                                                                    "type": {
                                                                                        "containsNull": False,
                                                                                        "elementType": {
                                                                                            "fields": [
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "token",
                                                                                                    "nullable": False,
                                                                                                    "type": "string",
                                                                                                },
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "bytes",
                                                                                                    "nullable": True,
                                                                                                    "type": {
                                                                                                        "containsNull": False,  # noqa: E501
                                                                                                        "elementType": "integer",  # noqa: E501
                                                                                                        "type": "array",
                                                                                                    },
                                                                                                },
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "logprob",
                                                                                                    "nullable": False,
                                                                                                    "type": "double",
                                                                                                },
                                                                                            ],
                                                                                            "type": "struct",
                                                                                        },
                                                                                        "type": "array",
                                                                                    },
                                                                                },
                                                                            ],
                                                                            "type": "struct",
                                                                        },
                                                                        "type": "array",
                                                                    },
                                                                }
                                                            ],
                                                            "type": "struct",
                                                        },
                                                    },
                                                    {
                                                        "metadata": {},
                                                        "name": "message",
                                                        "nullable": False,
                                                        "type": {
                                                            "fields": [
                                                                {
                                                                    "metadata": {},
                                                                    "name": "content",
                                                                    "nullable": True,
                                                                    "type": "string",
                                                                },
                                                                {
                                                                    "metadata": {},
                                                                    "name": "role",
                                                                    "nullable": False,
                                                                    "type": "string",
                                                                },
                                                                {
                                                                    "metadata": {},
                                                                    "name": "function_call",
                                                                    "nullable": True,
                                                                    "type": {
                                                                        "fields": [
                                                                            {
                                                                                "metadata": {},
                                                                                "name": "arguments",
                                                                                "nullable": False,
                                                                                "type": "string",
                                                                            },
                                                                            {
                                                                                "metadata": {},
                                                                                "name": "name",
                                                                                "nullable": False,
                                                                                "type": "string",
                                                                            },
                                                                        ],
                                                                        "type": "struct",
                                                                    },
                                                                },
                                                                {
                                                                    "metadata": {},
                                                                    "name": "tool_calls",
                                                                    "nullable": True,
                                                                    "type": {
                                                                        "containsNull": True,
                                                                        "elementType": {
                                                                            "fields": [
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "id",
                                                                                    "nullable": False,
                                                                                    "type": "string",
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "function",
                                                                                    "nullable": False,
                                                                                    "type": {
                                                                                        "fields": [
                                                                                            {
                                                                                                "metadata": {},
                                                                                                "name": "arguments",
                                                                                                "nullable": False,
                                                                                                "type": "string",
                                                                                            },
                                                                                            {
                                                                                                "metadata": {},
                                                                                                "name": "name",
                                                                                                "nullable": False,
                                                                                                "type": "string",
                                                                                            },
                                                                                        ],
                                                                                        "type": "struct",
                                                                                    },
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "type",
                                                                                    "nullable": False,
                                                                                    "type": "string",
                                                                                },
                                                                            ],
                                                                            "type": "struct",
                                                                        },
                                                                        "type": "array",
                                                                    },
                                                                },
                                                            ],
                                                            "type": "struct",
                                                        },
                                                    },
                                                ],
                                                "type": "struct",
                                            },
                                            "type": "array",
                                        },
                                    },
                                    {"metadata": {}, "name": "created", "nullable": False, "type": "integer"},
                                    {"metadata": {}, "name": "model", "nullable": False, "type": "string"},
                                    {"metadata": {}, "name": "object", "nullable": False, "type": "string"},
                                    {"metadata": {}, "name": "service_tier", "nullable": True, "type": "string"},
                                    {"metadata": {}, "name": "system_fingerprint", "nullable": True, "type": "string"},
                                    {
                                        "metadata": {},
                                        "name": "usage",
                                        "nullable": True,
                                        "type": {
                                            "fields": [
                                                {
                                                    "metadata": {},
                                                    "name": "completion_tokens",
                                                    "nullable": False,
                                                    "type": "integer",
                                                },
                                                {
                                                    "metadata": {},
                                                    "name": "prompt_tokens",
                                                    "nullable": False,
                                                    "type": "integer",
                                                },
                                                {
                                                    "metadata": {},
                                                    "name": "total_tokens",
                                                    "nullable": False,
                                                    "type": "integer",
                                                },
                                            ],
                                            "type": "struct",
                                        },
                                    },
                                ],
                                "type": "struct",
                            },
                        },
                    ],
                    "type": "struct",
                },
            },
        ],
        "type": "struct",
    }
