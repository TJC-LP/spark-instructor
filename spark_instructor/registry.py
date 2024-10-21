"""Module for defining registries of clients."""

from dataclasses import dataclass, field
from typing import Dict, Type

from spark_instructor import is_anthropic_available
from spark_instructor.factory import (
    ClientFactory,
    DatabricksFactory,
    O1Factory,
    OllamaFactory,
    OpenAIFactory,
)


def get_default_factories() -> Dict[str, Type[ClientFactory]]:
    """Get default factories based on whether anthropic is available."""
    if is_anthropic_available():
        from spark_instructor.factory import AnthropicFactory

        return {
            "anthropic": AnthropicFactory,
            "openai": OpenAIFactory,
            "databricks": DatabricksFactory,
            "ollama": OllamaFactory,
            "o1": O1Factory,
        }
    return {"openai": OpenAIFactory, "databricks": DatabricksFactory, "ollama": OllamaFactory, "o1": O1Factory}


@dataclass
class ClientRegistry:
    """A registry for managing and accessing different client factories for various AI models.

    This class serves as a central repository for client factories, allowing easy registration,
    retrieval, and mapping of model names to their respective factories. It provides a flexible
    way to manage multiple AI model providers in a single application.

    Attributes:
        factories (Dict[str, Type[ClientFactory]]): A dictionary mapping model class names
            to their corresponding ClientFactory classes.
        model_map (Dict[str, str]): A dictionary mapping model name patterns to their
            corresponding model class names.

    The default configuration includes factories for Anthropic, OpenAI, Databricks, and Ollama,
    with mappings for common model name patterns.

    Usage:
        ```python
        registry = ClientRegistry()
        openai_factory = registry.get_factory_from_model("gpt-3.5-turbo")
        ```
    """

    factories: Dict[str, Type[ClientFactory]] = field(default_factory=get_default_factories)
    model_map: Dict[str, str] = field(
        default_factory=lambda: {
            "databricks": "databricks",
            "gpt": "openai",
            "claude": "anthropic",
            "llama": "ollama",
            "o1": "o1",
        }
    )

    def register(self, model_class: str, factory_class: Type[ClientFactory]):
        """Register a new factory for a given model class.

        This method allows adding new factory classes to the registry or overriding
        existing ones.

        Args:
            model_class (str): The name of the model class (e.g., "openai", "anthropic").
            factory_class (Type[ClientFactory]): The ClientFactory subclass to be registered.

        Note:
            The model_class is converted to lowercase before registration to ensure
            case-insensitive lookups.

        Example:
            ```python
            registry.register("custom_model", CustomModelFactory)
            ```
        """
        self.factories[model_class.lower()] = factory_class

    def get_factory(self, model_class: str) -> Type[ClientFactory]:
        """Retrieve a factory class for a given model class.

        This method looks up and returns the appropriate ClientFactory subclass
        for the specified model class.

        Args:
            model_class (str): The name of the model class to look up.

        Returns:
            Type[ClientFactory]: The corresponding ClientFactory subclass.

        Raises:
            ValueError: If no factory is registered for the given model class.

        Example:
            ```python
            openai_factory = registry.get_factory("openai")
            ```
        """
        factory_class = self.factories.get(model_class.lower())
        if not factory_class:
            raise ValueError(f"No factory registered for model class: {model_class}")
        return factory_class

    def get_factory_from_model(self, model: str) -> Type[ClientFactory]:
        """Retrieve a factory class based on a model name.

        This method uses the model_map to determine the appropriate model class
        for a given model name, then returns the corresponding factory.

        Args:
            model (str): The name of the model (e.g., "gpt-4o", "claude-3-5-sonnet-20240620").

        Returns:
            Type[ClientFactory]: The corresponding ClientFactory subclass.

        Raises:
            ValueError: If the model name doesn't match any known patterns in the model_map.

        Example:
            ```python
            claude_factory = registry.get_factory_from_model("claude-3-5-sonnet-20240620")
            ```

        Note:
            This method performs a partial match on the model name. For example,
            any model name containing "gpt" will be mapped to the OpenAI factory.
        """
        for key, val in self.model_map.items():
            if key in model:
                return self.get_factory(val)
        raise ValueError(f"`{model}` does not have a mapped factory. Consider updating the `model_map`.")
