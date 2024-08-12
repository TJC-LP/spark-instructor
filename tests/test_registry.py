import pytest

from spark_instructor.factory import (
    AnthropicFactory,
    DatabricksFactory,
    OllamaFactory,
    OpenAIFactory,
)
from spark_instructor.registry import ClientFactory, ClientRegistry


@pytest.fixture
def client_registry():
    return ClientRegistry()


def test_register(client_registry):
    class CustomFactory(ClientFactory):
        pass

    client_registry.register("custom", CustomFactory)
    assert client_registry.factories["custom"] == CustomFactory


def test_get_factory(client_registry):
    assert client_registry.get_factory("anthropic") == AnthropicFactory
    assert client_registry.get_factory("openai") == OpenAIFactory
    assert client_registry.get_factory("databricks") == DatabricksFactory
    assert client_registry.get_factory("ollama") == OllamaFactory


def test_get_factory_error(client_registry):
    with pytest.raises(ValueError, match="No factory registered for model class: nonexistent"):
        client_registry.get_factory("nonexistent")


def test_get_factory_from_model(client_registry):
    assert client_registry.get_factory_from_model("gpt-3.5-turbo") == OpenAIFactory
    assert client_registry.get_factory_from_model("claude-2") == AnthropicFactory
    assert client_registry.get_factory_from_model("databricks-llama") == DatabricksFactory
    assert client_registry.get_factory_from_model("llama2") == OllamaFactory


def test_get_factory_from_model_error(client_registry):
    with pytest.raises(
        ValueError, match="`nonexistent` does not have a mapped factory. Consider updating the `model_map`."
    ):
        client_registry.get_factory_from_model("nonexistent")


def test_case_insensitivity(client_registry):
    assert client_registry.get_factory("ANTHROPIC") == AnthropicFactory
    assert client_registry.get_factory("OpenAI") == OpenAIFactory


def test_model_map_consistency(client_registry):
    for model, factory in client_registry.model_map.items():
        assert factory in client_registry.factories
