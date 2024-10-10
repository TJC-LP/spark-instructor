from pyspark.sql.functions import array, col, lit, struct

from spark_instructor.response_models import TextResponse
from spark_instructor.udf.instruct import instruct


def test_instruct_creation(mock_registry):
    udf = instruct(TextResponse, default_model="test-model", registry=mock_registry)
    assert callable(udf)


def test_instruct_execution(spark, mock_registry):
    udf = instruct(default_model="gpt-4o")

    # Create a test DataFrame
    data = [("What is the capital of France?",)]
    df = spark.createDataFrame(data, ["question"])
    df = df.withColumn("conversation", array(struct(lit("user").alias("role"), df.question.alias("content"))))

    # Apply the UDF
    result_df = df.withColumn("response", udf(col("conversation")))
    json_schema = result_df.schema.jsonValue()
    assert json_schema == {
        "fields": [
            {"metadata": {}, "name": "question", "nullable": True, "type": "string"},
            {
                "metadata": {},
                "name": "conversation",
                "nullable": False,
                "type": {
                    "containsNull": False,
                    "elementType": {
                        "fields": [
                            {"metadata": {}, "name": "role", "nullable": False, "type": "string"},
                            {"metadata": {}, "name": "content", "nullable": True, "type": "string"},
                        ],
                        "type": "struct",
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
                            "name": "chat_completion",
                            "nullable": True,
                            "type": {
                                "fields": [
                                    {"metadata": {}, "name": "id", "nullable": True, "type": "string"},
                                    {
                                        "metadata": {},
                                        "name": "choices",
                                        "nullable": True,
                                        "type": {
                                            "containsNull": True,
                                            "elementType": {
                                                "fields": [
                                                    {
                                                        "metadata": {},
                                                        "name": "finish_reason",
                                                        "nullable": True,
                                                        "type": "string",
                                                    },
                                                    {
                                                        "metadata": {},
                                                        "name": "index",
                                                        "nullable": True,
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
                                                                                    "nullable": True,
                                                                                    "type": "string",
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "bytes",
                                                                                    "nullable": True,
                                                                                    "type": {
                                                                                        "containsNull": True,
                                                                                        "elementType": "integer",
                                                                                        "type": "array",
                                                                                    },
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "logprob",
                                                                                    "nullable": True,
                                                                                    "type": "double",
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "top_logprobs",
                                                                                    "nullable": True,
                                                                                    "type": {
                                                                                        "containsNull": True,
                                                                                        "elementType": {
                                                                                            "fields": [
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "token",
                                                                                                    "nullable": True,
                                                                                                    "type": "string",
                                                                                                },
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "bytes",
                                                                                                    "nullable": True,
                                                                                                    "type": {
                                                                                                        "containsNull": True,  # noqa: E501
                                                                                                        "elementType": "integer",  # noqa: E501
                                                                                                        "type": "array",
                                                                                                    },
                                                                                                },
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "logprob",
                                                                                                    "nullable": True,
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
                                                                },
                                                                {
                                                                    "metadata": {},
                                                                    "name": "refusal",
                                                                    "nullable": True,
                                                                    "type": {
                                                                        "containsNull": True,
                                                                        "elementType": {
                                                                            "fields": [
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "token",
                                                                                    "nullable": True,
                                                                                    "type": "string",
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "bytes",
                                                                                    "nullable": True,
                                                                                    "type": {
                                                                                        "containsNull": True,
                                                                                        "elementType": "integer",
                                                                                        "type": "array",
                                                                                    },
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "logprob",
                                                                                    "nullable": True,
                                                                                    "type": "double",
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "top_logprobs",
                                                                                    "nullable": True,
                                                                                    "type": {
                                                                                        "containsNull": True,
                                                                                        "elementType": {
                                                                                            "fields": [
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "token",
                                                                                                    "nullable": True,
                                                                                                    "type": "string",
                                                                                                },
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "bytes",
                                                                                                    "nullable": True,
                                                                                                    "type": {
                                                                                                        "containsNull": True,  # noqa: E501
                                                                                                        "elementType": "integer",  # noqa: E501
                                                                                                        "type": "array",
                                                                                                    },
                                                                                                },
                                                                                                {
                                                                                                    "metadata": {},
                                                                                                    "name": "logprob",
                                                                                                    "nullable": True,
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
                                                                },
                                                            ],
                                                            "type": "struct",
                                                        },
                                                    },
                                                    {
                                                        "metadata": {},
                                                        "name": "message",
                                                        "nullable": True,
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
                                                                    "name": "refusal",
                                                                    "nullable": True,
                                                                    "type": "string",
                                                                },
                                                                {
                                                                    "metadata": {},
                                                                    "name": "role",
                                                                    "nullable": True,
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
                                                                                "nullable": True,
                                                                                "type": "string",
                                                                            },
                                                                            {
                                                                                "metadata": {},
                                                                                "name": "name",
                                                                                "nullable": True,
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
                                                                                    "nullable": True,
                                                                                    "type": "string",
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "function",
                                                                                    "nullable": True,
                                                                                    "type": {
                                                                                        "fields": [
                                                                                            {
                                                                                                "metadata": {},
                                                                                                "name": "arguments",
                                                                                                "nullable": True,
                                                                                                "type": "string",
                                                                                            },
                                                                                            {
                                                                                                "metadata": {},
                                                                                                "name": "name",
                                                                                                "nullable": True,
                                                                                                "type": "string",
                                                                                            },
                                                                                        ],
                                                                                        "type": "struct",
                                                                                    },
                                                                                },
                                                                                {
                                                                                    "metadata": {},
                                                                                    "name": "type",
                                                                                    "nullable": True,
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
                                    {"metadata": {}, "name": "created", "nullable": True, "type": "integer"},
                                    {"metadata": {}, "name": "model", "nullable": True, "type": "string"},
                                    {"metadata": {}, "name": "object", "nullable": True, "type": "string"},
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
                                                    "nullable": True,
                                                    "type": "integer",
                                                },
                                                {
                                                    "metadata": {},
                                                    "name": "prompt_tokens",
                                                    "nullable": True,
                                                    "type": "integer",
                                                },
                                                {
                                                    "metadata": {},
                                                    "name": "total_tokens",
                                                    "nullable": True,
                                                    "type": "integer",
                                                },
                                                {
                                                    "metadata": {},
                                                    "name": "completion_tokens_details",
                                                    "nullable": True,
                                                    "type": {
                                                        "fields": [
                                                            {
                                                                "metadata": {},
                                                                "name": "audio_tokens",
                                                                "nullable": True,
                                                                "type": "integer",
                                                            },
                                                            {
                                                                "metadata": {},
                                                                "name": "reasoning_tokens",
                                                                "nullable": True,
                                                                "type": "integer",
                                                            },
                                                        ],
                                                        "type": "struct",
                                                    },
                                                },
                                                {
                                                    "metadata": {},
                                                    "name": "prompt_tokens_details",
                                                    "nullable": True,
                                                    "type": {
                                                        "fields": [
                                                            {
                                                                "metadata": {},
                                                                "name": "audio_tokens",
                                                                "nullable": True,
                                                                "type": "integer",
                                                            },
                                                            {
                                                                "metadata": {},
                                                                "name": "cached_tokens",
                                                                "nullable": True,
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
                        }
                    ],
                    "type": "struct",
                },
            },
        ],
        "type": "struct",
    }
