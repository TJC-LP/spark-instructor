"""Utilities for prompt generation."""

from typing import Optional, Union

import pyspark.sql.functions as f
from pyspark.sql.column import Column

ColumnOrName = Union[Column, str]


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
    """
    role_column = f.lit("role")
    content_column = f.lit("content")
    user_map = f.create_map(role_column, f.lit("user"), content_column, user_message_column)

    if system_message_column is not None:
        system_map = f.create_map(role_column, f.lit("system"), content_column, system_message_column)
        return f.array(system_map, user_map)
    return f.array(user_map)
