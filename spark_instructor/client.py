"""Module for handling client routing.

Serves as a workaround for Spark serialization issues.
"""

import os
from enum import Enum

import instructor
from openai import OpenAI


def get_databricks_client() -> instructor.Instructor:
    """Get the databricks client.

    Ensure that the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables are set.
    """
    assert "DATABRICKS_HOST" in os.environ, "``DATABRICKS_HOST`` is not set!"
    assert "DATABRICKS_TOKEN" in os.environ, "``DATABRICKS_TOKEN`` is not set"
    return instructor.from_openai(
        OpenAI(api_key=os.getenv("DATABRICKS_TOKEN"), base_url=f"{os.getenv('DATABRICKS_HOST')}/serving-endpoints"),
        mode=instructor.Mode.MD_JSON,
    )


def get_openai_client() -> instructor.Instructor:
    """Get the OpenAI client.

    Ensure that ``OPENAI_API_KEY`` is set.
    """
    assert "OPENAI_API_KEY" in os.environ, "``OPENAI_API_KEY`` is not set!"
    return instructor.from_openai(OpenAI())


def get_anthropic_client() -> instructor.Instructor:
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

    assert "ANTHROPIC_API_KEY" in os.environ, "``ANTHROPIC_API_KEY`` is not set!"
    return instructor.from_anthropic(Anthropic())


class ModelClass(str, Enum):
    """Enumeration of available model classes."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DATABRICKS = "databricks"


def infer_model_class(model_name: str) -> ModelClass:
    """Attempt to infer the model class from the model name."""
    if "databricks" in model_name:
        return ModelClass.DATABRICKS
    elif "gpt" in model_name:
        return ModelClass.OPENAI
    elif "claude" in model_name:
        return ModelClass.ANTHROPIC
    raise ValueError(f"Model name `{model_name}` does not match any of the available model classes.")


MODEL_CLASS_ROUTE = {
    ModelClass.ANTHROPIC: get_anthropic_client,
    ModelClass.DATABRICKS: get_databricks_client,
    ModelClass.OPENAI: get_openai_client,
}
