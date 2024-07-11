from typing import Type

import pytest
from pydantic import BaseModel
from pyspark.sql.types import IntegerType, StringType, StructType

from spark_instructor.client import MODEL_CLASS_ROUTE, ModelClass
from spark_instructor.udf import MessageRouter, ModelSerializer


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
def mock_client(mocker):
    client = mocker.Mock()
    client.chat.completions.create.return_value = MockModel(result="test")
    return client


@pytest.fixture
def mock_model_class_route(mocker, mock_client):
    mocker.patch.dict(MODEL_CLASS_ROUTE, {ModelClass.OPENAI: lambda: mock_client})


@pytest.fixture
def message_router() -> MessageRouter:
    return MessageRouter(model="gpt-3.5-turbo", response_model_type=MockModel)


def test_create_completion(mock_model_class_route, message_router):
    result = message_router.create_object_from_messages(
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert result.model_dump() == {"result": "test"}
    MODEL_CLASS_ROUTE[ModelClass.OPENAI]().chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo", response_model=MockModel, messages=[{"role": "user", "content": "Hello"}]
    )
