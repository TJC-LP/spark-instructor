"""Module for creating a Databricks factory."""

from typing import List, Optional

import instructor
from openai.types.chat import ChatCompletionMessageParam

from spark_instructor.client.base import get_databricks_aclient
from spark_instructor.factory.openai_factory import OpenAIFactory
from spark_instructor.types import SparkChatCompletionMessages


class DatabricksFactory(OpenAIFactory):
    """A databricks factory."""

    @classmethod
    def from_config(
        cls,
        mode: Optional[instructor.Mode] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> "DatabricksFactory":
        """Build a databricks factory from custom entries."""
        return cls(get_databricks_aclient(mode or instructor.Mode.MD_JSON, base_url, api_key))

    def format_messages(self, messages: SparkChatCompletionMessages) -> List[ChatCompletionMessageParam]:
        """Format messages by using default callable."""
        return [message(string_only=True) for message in messages.root]
