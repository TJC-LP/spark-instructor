"""A module for defining base classes for each completion type."""

from pydantic import BaseModel


class BaseCompletion(BaseModel):
    """Completion model for defining what model was used and number of tokens."""

    id: str
    model: str
