import instructor
import pytest
from anthropic import Anthropic
from openai import OpenAI

from spark_instructor.client import (
    ModelClass,
    get_anthropic_client,
    get_databricks_client,
    get_instructor,
    get_ollama_client,
    get_openai_client,
    infer_model_class,
)

# Import the functions to be tested
from spark_instructor.utils import assert_env_is_set, get_env_variable


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "db_token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic_key")


def test_assert_env_is_set(mock_env_vars):
    assert_env_is_set("DATABRICKS_HOST")
    with pytest.raises(AssertionError):
        assert_env_is_set("NON_EXISTENT_VAR")


def test_get_env_variable(mock_env_vars):
    assert get_env_variable("DATABRICKS_HOST") == "https://databricks.com"
    with pytest.raises(AssertionError):
        get_env_variable("NON_EXISTENT_VAR")


@pytest.mark.parametrize(
    "model_name, expected_class",
    [
        ("databricks-dbrx-instruct", ModelClass.DATABRICKS),
        ("databricks-llama3-70b-instruct", ModelClass.DATABRICKS),
        ("gpt-4o", ModelClass.OPENAI),
        ("gpt-3.5-turbo", ModelClass.OPENAI),
        ("claude-3-5-sonnet-20240620", ModelClass.ANTHROPIC),
        ("llama3", ModelClass.OLLAMA),
        ("llama2", ModelClass.OLLAMA),
    ],
)
def test_infer_model_class_valid(model_name, expected_class):
    assert infer_model_class(model_name) == expected_class


@pytest.mark.parametrize(
    "invalid_model_name",
    [
        "bert",
        "t5",
        "roberta",
        "palm",
    ],
)
def test_infer_model_class_invalid(invalid_model_name):
    with pytest.raises(ValueError) as exc_info:
        infer_model_class(invalid_model_name)
    assert f"Model name `{invalid_model_name}` does not match any of the available model classes." in str(
        exc_info.value
    )


def test_get_databricks_client(mock_env_vars):
    client = get_databricks_client()
    assert isinstance(client, instructor.Instructor)
    assert isinstance(client.client, OpenAI)


def test_get_openai_client(mock_env_vars):
    client = get_openai_client()
    assert isinstance(client, instructor.Instructor)
    assert isinstance(client.client, OpenAI)


def test_get_openai_client_with_set_vars():
    client = get_openai_client(base_url="https://openai.com", api_key="openai_key")
    assert isinstance(client, instructor.Instructor)
    assert isinstance(client.client, OpenAI)


def test_get_anthropic_client(mock_env_vars):
    client = get_anthropic_client()
    assert isinstance(client, instructor.Instructor)
    assert isinstance(client.client, Anthropic)


def test_get_anthropic_client_with_set_vars():
    client = get_anthropic_client(base_url="https://anthropic.com", api_key="anthopic_key")
    assert isinstance(client, instructor.Instructor)
    assert isinstance(client.client, Anthropic)


def test_get_anthropic_client_import_error(mocker, mock_env_vars):
    # Mock the import of anthropic to raise an ImportError
    def mock_import(name, *args):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return __import__(name, *args)

    mocker.patch("builtins.__import__", side_effect=mock_import)

    with pytest.raises(ImportError) as exc_info:
        get_anthropic_client()

    assert "Please install ``anthropic``" in str(exc_info.value)


def test_get_ollama_client():
    client = get_ollama_client()
    assert isinstance(client, instructor.Instructor)
    assert isinstance(client.client, OpenAI)


@pytest.mark.parametrize(
    "model_class,expected_function",
    [
        (ModelClass.DATABRICKS, get_databricks_client),
        (ModelClass.OPENAI, get_openai_client),
        (ModelClass.ANTHROPIC, get_anthropic_client),
        (ModelClass.OLLAMA, get_ollama_client),
    ],
)
def test_get_instructor(mock_env_vars, model_class, expected_function):
    client = get_instructor(model_class)
    assert isinstance(client, instructor.Instructor)
    # We can't directly compare the functions, but we can check if the correct client type is returned
    assert type(client.client) is type(expected_function().client)


def test_get_instructor_mode_override(mock_env_vars):
    client = get_instructor(mode=instructor.Mode.MD_JSON)
    assert client.mode == instructor.Mode.MD_JSON


def test_get_instructor_default(mock_env_vars):
    client = get_instructor()
    assert isinstance(client, instructor.Instructor)
