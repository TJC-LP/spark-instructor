"""Utilities for prompt generation."""

from typing import Literal, Optional, Union

import pyspark.sql.functions as f
from pyspark.sql.column import Column
from pyspark.sql.types import ArrayType, BooleanType, StringType
from sparkdantic.model import create_spark_schema
from typing_extensions import Required, TypedDict

from spark_instructor.types.base import (
    ChatCompletionMessageToolCallParamPD,
    ImageURLPD,
    SparkChatCompletionMessage,
)
from spark_instructor.utils.types import make_spark_schema_nullable

ColumnOrName = Union[Column, str]

__all__ = ["zero_shot_prompt", "create_chat_completion_messages"]


def zero_shot_prompt(
    user_message_column: ColumnOrName,
    system_message_column: Optional[ColumnOrName] = None,
) -> Column:
    """Generate a zero-shot prompt for language models in Spark DataFrames.

    This function creates a structured array of messages suitable for zero-shot prompting
    in language models. It always includes a user message and optionally a system message.

    Args:
        user_message_column (Union[Column, str]): The column containing user messages.
            Can be either a Column object or a string column name.
        system_message_column (Optional[Union[Column, str]], optional): The column containing
            system messages. Can be either a Column object or a string column name. Defaults to None.

    Returns:
        Column: A Spark SQL Column containing an array of message structures.
            Each structure is a map with 'role' and 'content' keys.

    Notes:
        - If system_message_column is None, only the user message is included.
        - If system_message_column is provided, the system message and user message are included.

    Example:
        ```python

        >>> from databricks.connect import DatabricksSession
        >>> spark = DatabricksSession.builder.serverless().getOrCreate()
        >>> df = spark.createDataFrame([("Hello", "Be helpful")], ["user_msg", "sys_msg"])
        >>> prompt_col = zero_shot_prompt("user_msg", system_message_column="sys_msg")
        >>> df.select(prompt_col.alias("prompt")).show(truncate=False)
        +---------------------------------------------------------------------------+
        |prompt                                                                     |
        +---------------------------------------------------------------------------+
        |[{role -> system, content -> Be helpful}, {role -> user, content -> Hello}]|
        +---------------------------------------------------------------------------+
        <BLANKLINE>

        ```
    """
    role_column = f.lit("role")
    content_column = f.lit("content")
    user_map = f.create_map(role_column, f.lit("user"), content_column, user_message_column)

    if system_message_column is not None:
        system_map = f.create_map(role_column, f.lit("system"), content_column, system_message_column)
        return f.array(system_map, user_map)
    return f.array(user_map)


def get_column_or_null(column: ColumnOrName | None = None) -> Column:
    """Format optional column."""
    if column is None:
        return f.lit(None)
    if isinstance(column, str):
        return f.col(column)
    return column


class SparkChatCompletionColumns(TypedDict, total=False):
    role: Required[ColumnOrName]
    content: ColumnOrName
    image_urls: ColumnOrName
    name: ColumnOrName
    tool_calls: ColumnOrName
    tool_call_id: ColumnOrName
    cache_control: ColumnOrName


def create_chat_completion_messages(messages: list[SparkChatCompletionColumns], strict: bool = True) -> Column:
    """Create an array of chat completion message structures from a list of column specifications.

    This function generates a Spark SQL Column containing an array of structured messages
    suitable for chat completion tasks. It handles all possible fields of a chat message,
    including role, content, image URLs, name, tool calls, and tool call IDs. Note that ``image_urls``
    are NOT included in the ``content`` due to spark serialization requiring a static schema.

    Args:
        messages (list[SparkChatCompletionColumns]): A list of dictionaries, where each dictionary
            specifies the columns or literal values for different parts of a chat message.
            The dictionary keys can include:
            ```markdown
            - role (Required[ColumnOrName]): The role of the message (e.g., "user", "assistant", "system").
            - content (ColumnOrName, optional): The text content of the message.
            - image_urls (ColumnOrName, optional): URLs of images associated with the message.
            - name (ColumnOrName, optional): Name associated with the message.
            - tool_calls (ColumnOrName, optional): Tool calls made in the message.
            - tool_call_id (ColumnOrName, optional): ID of the tool call.
            ```
        strict (bool): Whether to make the schema nullability strict. Useful when columns are UDF generated.
            Recommended when using ``image_urls``.

    Returns:
        Column: A Spark SQL Column containing an array of structured messages. Each message
        is cast to the SparkChatCompletionMessage schema.

    Notes:
        - The function uses the SparkChatCompletionMessage schema to ensure type consistency.
        - Fields not specified in the input will be set to None in the output.
        - This function is particularly useful for creating complex, multi-message prompts
          for chat-based language models in a Spark environment.

    Example:
        ```python

        >>> from pyspark.sql import functions as f
        >>> from databricks.connect import DatabricksSession
        >>> spark = DatabricksSession.builder.serverless().getOrCreate()
        >>> df = spark.createDataFrame([("Hello", "Be helpful")], ["user_msg", "sys_msg"])
        >>> messages = [
        ...     {"role": f.lit("system"), "content": "sys_msg"},
        ...     {"role": f.lit("user"), "content": "user_msg"}
        ... ]
        >>> chat_messages = create_chat_completion_messages(messages)
        >>> df.select(chat_messages.alias("messages")).show(truncate=False)
        +-------------------------------------------------------------------------------------+
        |messages                                                                             |
        +-------------------------------------------------------------------------------------+
        |[{system, Be helpful, NULL, NULL, NULL, NULL}, {user, Hello, NULL, NULL, NULL, NULL}]|
        +-------------------------------------------------------------------------------------+
        <BLANKLINE>

        ```
    Raises:
        ValueError: If a required field (e.g., 'role') is missing from any message specification.
    """
    all_keys: list[Literal["role", "content", "image_urls", "name", "tool_calls", "tool_call_id", "cache_control"]] = [
        "role",
        "content",
        "image_urls",
        "name",
        "tool_calls",
        "tool_call_id",
        "cache_control",
    ]
    cast_schema = create_spark_schema(SparkChatCompletionMessage)
    image_urls_type = ArrayType(create_spark_schema(ImageURLPD))
    tool_calls_type = ArrayType(create_spark_schema(ChatCompletionMessageToolCallParamPD))
    if not strict:
        image_urls_type = make_spark_schema_nullable(image_urls_type)
        tool_calls_type = make_spark_schema_nullable(tool_calls_type)
        cast_schema = make_spark_schema_nullable(cast_schema)

    def create_struct(message: SparkChatCompletionColumns):
        struct_fields = []
        for key in all_keys:
            if key in message:
                val = message[key]
                struct_fields.append(f.col(val).alias(key) if isinstance(val, str) else val.alias(key))
            else:
                if key == "image_urls":
                    struct_fields.append(f.lit(None).cast(image_urls_type).alias(key))
                elif key == "tool_calls":
                    struct_fields.append(f.lit(None).cast(tool_calls_type).alias(key))
                elif key == "cache_control":
                    struct_fields.append(f.lit(None).cast(BooleanType()).alias(key))
                else:
                    struct_fields.append(f.lit(None).cast(StringType()).alias(key))
        return f.struct(*struct_fields).cast(cast_schema)

    return f.array(*[create_struct(message) for message in messages])
