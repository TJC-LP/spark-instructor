from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    ArrayType,
    MapType,
    Row,
    StringType,
    StructField,
    StructType,
)

from spark_instructor.utils.prompt import zero_shot_prompt


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
