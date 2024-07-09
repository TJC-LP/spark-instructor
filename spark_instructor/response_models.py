"""Sample response models for parsing."""

from pydantic_spark.base import SparkBase  # type: ignore


class User(SparkBase):
    """A user."""

    name: str
    age: int
