"""General utilities."""

from spark_instructor.utils.env import assert_env_is_set, get_env_variable
from spark_instructor.utils.prompt import zero_shot_prompt

__all__ = ["zero_shot_prompt", "assert_env_is_set", "get_env_variable"]
