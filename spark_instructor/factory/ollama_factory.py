"""Module for creating an Ollama factory."""

from typing import Optional

import instructor

from spark_instructor.client.base import get_ollama_aclient
from spark_instructor.factory.openai_factory import OpenAIFactory


class OllamaFactory(OpenAIFactory):
    """An Ollama factory."""

    @classmethod
    def from_config(
        cls,
        mode: Optional[instructor.Mode] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> "OllamaFactory":
        """Create an Ollama factory from custom entries."""
        return cls(get_ollama_aclient(mode or instructor.Mode.JSON, base_url, api_key))
