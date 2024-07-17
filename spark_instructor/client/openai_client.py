"""Module for OpenAI client."""

from typing import Optional, Type

import instructor
from openai import OpenAI

from spark_instructor.client.base import BaseClient, Mode
from spark_instructor.completions import OpenAICompletion
from spark_instructor.utils import assert_env_is_set


class OpenAIClient(BaseClient[OpenAICompletion]):
    """OpenAI instructor client."""

    completion_model: Type[OpenAICompletion] = OpenAICompletion

    def get_instructor(
        self, mode: Mode, api_key: Optional[str] = None, base_url: Optional[str] = None, *args, **kwargs
    ) -> instructor.Instructor:
        """Get the OpenAI client.

        Unless passes as an argument,
        Ensure that ``OPENAI_API_KEY`` is set.
        """
        if not api_key:
            assert_env_is_set("OPENAI_API_KEY")
            return instructor.from_openai(OpenAI(base_url=base_url), mode=mode)
        return instructor.from_openai(OpenAI(base_url=base_url, api_key=api_key), mode=mode)
