"""Module for handling client routing.

Serves as a workaround for Spark serialization issues.
"""

import os
from enum import Enum
from typing import Any, Dict, Optional

import instructor
from openai import OpenAI


def assert_env_is_set(name: str):
    """Assert that an environment variable is set."""
    assert name in os.environ, f"``{name}`` is not set!"


def get_env_variable(name: str) -> str:
    """Get environment variable with the given name."""
    assert_env_is_set(name)
    return os.environ[name]


def get_databricks_client(
    mode: instructor.Mode = instructor.Mode.MD_JSON, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> instructor.Instructor:
    """Get the databricks client.

    Unless passed as arguments,
    ensure that the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables are set.
    """
    if base_url is None:
        base_url = f"{get_env_variable('DATABRICKS_HOST')}/serving-endpoints"
    if api_key is None:
        api_key = get_env_variable("DATABRICKS_TOKEN")
    return instructor.from_openai(
        OpenAI(api_key=api_key, base_url=base_url),
        mode=mode,
    )


def get_openai_client(
    mode: instructor.Mode = instructor.Mode.TOOLS, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> instructor.Instructor:
    """Get the OpenAI client.

    Unless passes as an argument,
    Ensure that ``OPENAI_API_KEY`` is set.
    """
    if not api_key:
        assert_env_is_set("OPENAI_API_KEY")
        return instructor.from_openai(OpenAI(base_url=base_url), mode=mode)
    return instructor.from_openai(OpenAI(base_url=base_url, api_key=api_key), mode=mode)


def get_anthropic_client(
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> instructor.Instructor:
    """Get the Anthropic client.

    Ensure that ``ANTHROPIC_API_KEY`` is set.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Please install ``anthropic`` by running ``pip install anthropic`` "
            "or ``pip install spark-instructor[anthropic]``"
        )

    if not api_key:
        assert_env_is_set("ANTHROPIC_API_KEY")
        return instructor.from_anthropic(Anthropic(), mode=mode)
    return instructor.from_anthropic(Anthropic(api_key=api_key, base_url=base_url), mode=mode)


def get_ollama_client(
    mode: instructor.Mode = instructor.Mode.JSON, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> instructor.Instructor:
    """Get the Ollama client."""
    if not base_url:
        host = "http://localhost:11434"
        base_url = f"{host}/v1"

    return instructor.from_openai(OpenAI(base_url=base_url, api_key="ollama" if not api_key else api_key), mode=mode)


class ModelClass(str, Enum):
    """Enumeration of available model classes."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DATABRICKS = "databricks"
    OLLAMA = "ollama"


def infer_model_class(model_name: str) -> ModelClass:
    """Attempt to infer the model class from the model name."""
    if "databricks" in model_name:
        return ModelClass.DATABRICKS
    elif "gpt" in model_name:
        return ModelClass.OPENAI
    elif "claude" in model_name:
        return ModelClass.ANTHROPIC
    elif "llama" in model_name:
        return ModelClass.OLLAMA
    raise ValueError(f"Model name `{model_name}` does not match any of the available model classes.")


MODEL_CLASS_ROUTE = {
    ModelClass.ANTHROPIC: get_anthropic_client,
    ModelClass.DATABRICKS: get_databricks_client,
    ModelClass.OPENAI: get_openai_client,
    ModelClass.OLLAMA: get_ollama_client,
}


def get_instructor(
    model_class: Optional[ModelClass] = None,
    mode: Optional[instructor.Mode] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> instructor.Instructor:
    """Get the instructor client based on the model class and mode."""
    if model_class is None:
        # Use OpenAI by default
        model_class = ModelClass.OPENAI
    kwargs: Dict[str, Any] = dict(api_key=api_key, base_url=base_url)
    if mode:
        kwargs |= dict(mode=mode)
    return MODEL_CLASS_ROUTE[model_class](**kwargs)
