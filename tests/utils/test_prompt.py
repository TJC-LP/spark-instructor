from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    ArrayType,
    MapType,
    NullType,
    Row,
    StringType,
    StructField,
    StructType,
)

from spark_instructor.utils.prompt import (
    create_chat_completion_messages,
    get_column_or_null,
    zero_shot_prompt,
)


def test_user_message_only(spark):
    df = spark.createDataFrame([("Hello",)], ["user_msg"])
    result = df.select(zero_shot_prompt("user_msg").alias("prompt"))
    assert result.schema == StructType(
        [StructField("prompt", ArrayType(MapType(StringType(), StringType()), False), False)]
    )
    assert result.collect()[0]["prompt"] == [{"role": "user", "content": "Hello"}]


def test_user_and_system_message(spark):
    df = spark.createDataFrame([("Hello", "Be helpful")], ["user_msg", "sys_msg"])
    result = df.select(zero_shot_prompt("user_msg", system_message_column="sys_msg").alias("prompt"))
    assert result.collect()[0]["prompt"] == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]


def test_user_and_static_system_message(spark):
    df = spark.createDataFrame([("Hello",)], ["user_msg"])
    result = df.select(zero_shot_prompt("user_msg", system_message_column=lit("Be helpful")).alias("prompt"))
    assert result.collect()[0]["prompt"] == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]


def test_column_objects(spark):
    df = spark.createDataFrame([("Hello", "Be helpful")], ["user_msg", "sys_msg"])
    result = df.select(zero_shot_prompt(col("user_msg"), system_message_column=col("sys_msg")).alias("prompt"))
    assert result.collect()[0]["prompt"] == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]


def test_multiple_rows(spark):
    df = spark.createDataFrame([("Hello", "Be helpful"), ("Hi", "Be concise")], ["user_msg", "sys_msg"])
    result = df.select(zero_shot_prompt("user_msg", system_message_column="sys_msg").alias("prompt"))
    assert result.collect() == [
        Row(prompt=[{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hello"}]),
        Row(prompt=[{"role": "system", "content": "Be concise"}, {"role": "user", "content": "Hi"}]),
    ]


def test_empty_dataframe(spark):
    df = spark.createDataFrame(
        [], schema=StructType([StructField("user_msg", StringType()), StructField("sys_msg", StringType())])
    )
    result = df.select(zero_shot_prompt("user_msg", system_message_column="sys_msg").alias("prompt"))
    assert result.count() == 0


def test_docstring_example(spark):
    df = spark.createDataFrame([("Hello", "Be helpful")], ["user_msg", "sys_msg"])
    prompt_col = zero_shot_prompt("user_msg", system_message_column="sys_msg")
    result = df.select(prompt_col.alias("prompt"))
    assert result.collect()[0]["prompt"] == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]


def test_create_chat_completion_messages(spark):
    # Test with minimal required fields
    df = spark.createDataFrame([("Hello", "Be helpful")], ["user_msg", "sys_msg"])
    messages = [{"role": lit("system"), "content": "sys_msg"}, {"role": lit("user"), "content": "user_msg"}]
    result = create_chat_completion_messages(messages)
    df = df.withColumn("messages", result)
    schema = df.select(result.alias("messages")).schema

    assert isinstance(schema, StructType)

    result_data = df.collect()[0]["messages"]
    assert len(result_data) == 2
    assert result_data[0]["role"] == "system"
    assert result_data[0]["content"] == "Be helpful"
    assert result_data[1]["role"] == "user"
    assert result_data[1]["content"] == "Hello"


def test_get_column_or_null(spark):
    # Test with None
    result = get_column_or_null(None)
    df = spark.range(1).select(result.alias("result"))
    assert df.schema["result"].dataType == NullType()
    assert df.collect()[0]["result"] is None

    # Test with string column name
    result = get_column_or_null("test_column")
    df = spark.createDataFrame([("value",)], ["test_column"]).select(result.alias("result"))
    assert df.schema["result"].dataType == StringType()
    assert df.collect()[0]["result"] == "value"

    # Test with Column object
    result = get_column_or_null(col("test_column"))
    df = spark.createDataFrame([("value",)], ["test_column"]).select(result.alias("result"))
    assert df.schema["result"].dataType == StringType()
    assert df.collect()[0]["result"] == "value"

    # Test with literal
    result = get_column_or_null(lit("literal_value"))
    df = spark.range(1).select(result.alias("result"))
    assert df.schema["result"].dataType == StringType()
    assert df.collect()[0]["result"] == "literal_value"
