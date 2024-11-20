"""Module for handling client routing.

Serves as a workaround for Spark serialization issues.
"""

from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional

import instructor
from openai import AsyncOpenAI, OpenAI

from spark_instructor.utils.env import assert_env_is_set, get_env_variable


@lru_cache
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


@lru_cache
def get_databricks_aclient(
    mode: instructor.Mode = instructor.Mode.MD_JSON, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> instructor.AsyncInstructor:
    """Get the async databricks client.

    Unless passed as arguments,
    ensure that the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables are set.
    """
    if base_url is None:
        base_url = f"{get_env_variable('DATABRICKS_HOST')}/serving-endpoints"
    if api_key is None:
        api_key = get_env_variable("DATABRICKS_TOKEN")
    return instructor.from_openai(
        AsyncOpenAI(api_key=api_key, base_url=base_url),
        mode=mode,
    )


@lru_cache
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


@lru_cache
def get_openai_aclient(
    mode: instructor.Mode = instructor.Mode.TOOLS, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> instructor.AsyncInstructor:
    """Get the async OpenAI client.

    Unless passes as an argument,
    Ensure that ``OPENAI_API_KEY`` is set.
    """
    if not api_key:
        assert_env_is_set("OPENAI_API_KEY")
        return instructor.from_openai(AsyncOpenAI(base_url=base_url), mode=mode)
    return instructor.from_openai(AsyncOpenAI(base_url=base_url, api_key=api_key), mode=mode)


@lru_cache
def get_anthropic_client(
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_prompt_caching: bool = False,
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
        return instructor.from_anthropic(Anthropic(), mode=mode, enable_prompt_caching=enable_prompt_caching)
    return instructor.from_anthropic(
        Anthropic(api_key=api_key, base_url=base_url), mode=mode, enable_prompt_caching=enable_prompt_caching
    )


@lru_cache
def get_anthropic_aclient(
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_prompt_caching: bool = False,
) -> instructor.AsyncInstructor:
    """Get the async Anthropic client.

    Ensure that ``ANTHROPIC_API_KEY`` is set.
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise ImportError(
            "Please install ``anthropic`` by running ``pip install anthropic`` "
            "or ``pip install spark-instructor[anthropic]``"
        )

    if not api_key:
        assert_env_is_set("ANTHROPIC_API_KEY")
        return instructor.from_anthropic(AsyncAnthropic(), mode=mode, enable_prompt_caching=enable_prompt_caching)
    return instructor.from_anthropic(
        AsyncAnthropic(api_key=api_key, base_url=base_url), mode=mode, enable_prompt_caching=enable_prompt_caching
    )


@lru_cache
def get_ollama_client(
    mode: instructor.Mode = instructor.Mode.JSON, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> instructor.Instructor:
    """Get the Ollama client."""
    if not base_url:
        host = "http://localhost:11434"
        base_url = f"{host}/v1"

    return instructor.from_openai(OpenAI(base_url=base_url, api_key="ollama" if not api_key else api_key), mode=mode)


@lru_cache
def get_ollama_aclient(
    mode: instructor.Mode = instructor.Mode.JSON, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> instructor.AsyncInstructor:
    """Get the async Ollama client."""
    if not base_url:
        host = "http://localhost:11434"
        base_url = f"{host}/v1"

    return instructor.from_openai(
        AsyncOpenAI(base_url=base_url, api_key="ollama" if not api_key else api_key), mode=mode
    )


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

MODEL_CLASS_ROUTE_ASYNC = {
    ModelClass.ANTHROPIC: get_anthropic_aclient,
    ModelClass.DATABRICKS: get_databricks_aclient,
    ModelClass.OPENAI: get_openai_aclient,
    ModelClass.OLLAMA: get_ollama_aclient,
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


def get_async_instructor(
    model_class: Optional[ModelClass] = None,
    mode: Optional[instructor.Mode] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> instructor.AsyncInstructor:
    """Get the instructor client based on the model class and mode."""
    if model_class is None:
        # Use OpenAI by default
        model_class = ModelClass.OPENAI
    kwargs: Dict[str, Any] = dict(api_key=api_key, base_url=base_url)
    if mode:
        kwargs |= dict(mode=mode)
    return MODEL_CLASS_ROUTE_ASYNC[model_class](**kwargs)
