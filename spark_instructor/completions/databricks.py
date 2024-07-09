"""
This module defines Pydantic models for Databricks' chat completion API responses.

It extends the base models from spark_instructor.completions.base to provide
specific implementations for Databricks' API structure. These models can be used
to parse and validate Databricks API responses, ensuring type safety and
providing easy access to response data.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from spark_instructor.completions.base import BaseChoice, BaseCompletion, BaseMessage


class DatabricksMessage(BaseMessage):
    """
    Represents a message in a Databricks chat completion response.

    This class extends BaseMessage to include Databricks-specific fields such as
    function_call and tool_calls.

    Attributes:
        function_call (Optional[Dict[str, Any]]): Information about a function call,
            if applicable. Defaults to None.
        tool_calls (Optional[List[Dict[str, Any]]]): A list of tool calls made by the model,
            if any. Defaults to None. Note that this is a list of dictionaries, which may
            differ from other implementations.
    """

    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class DatabricksChoice(BaseChoice):
    """
    Represents a single choice in a Databricks chat completion response.

    This class extends BaseChoice to use the Databricks-specific message type.

    Attributes:
        message (DatabricksMessage): The message content for this choice.
    """

    message: DatabricksMessage


class DatabricksCompletion(BaseCompletion):
    """
    Represents a complete Databricks chat completion response.

    This class extends BaseCompletion to include all fields specific to
    Databricks' API response structure.

    Attributes:
        choices (List[DatabricksChoice]): A list of completion choices generated
            by the model.
        created (int): The Unix timestamp (in seconds) of when the completion
            was created.
        object (str): The object type, always "chat.completion" for chat completions.
        service_tier (Optional[str]): The service tier used for this completion,
            if applicable. Defaults to None.
        system_fingerprint (Optional[str]): A unique identifier for the system
            configuration used for this completion. Defaults to None.
    """

    choices: List[DatabricksChoice]
    created: int
    object: str = Field("chat.completion")
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None
