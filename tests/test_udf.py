import pytest
from pydantic import BaseModel

from spark_instructor.client import MODEL_CLASS_ROUTE, ModelClass
from spark_instructor.udf import create_completion, create_serialized_completion


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


def test_create_completion(mock_model_class_route):
    result = create_completion(
        model="gpt-3.5-turbo",
        response_model=MockModel,
        messages=[{"role": "user", "content": "Hello"}],
        model_class=ModelClass.OPENAI,
    )

    assert result.model_dump() == {"result": "test"}
    MODEL_CLASS_ROUTE[ModelClass.OPENAI]().chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo", response_model=MockModel, messages=[{"role": "user", "content": "Hello"}]
    )


def test_process_messages(mocker, mock_model_class_route):
    mock_create_completion = mocker.patch("spark_instructor.udf.create_completion")
    mock_create_completion.return_value = MockModel(result="test")

    result = create_serialized_completion(
        model="gpt-3.5-turbo",
        response_model=MockModel,
        messages=[{"role": "user", "content": "Hello"}],
        model_class=ModelClass.OPENAI,
    )

    assert result == {"result": "test"}
    mock_create_completion.assert_called_once_with(
        "gpt-3.5-turbo", MockModel, [{"role": "user", "content": "Hello"}], ModelClass.OPENAI
    )
