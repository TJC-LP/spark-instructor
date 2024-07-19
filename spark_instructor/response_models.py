"""Sample response models for parsing."""

from pydantic import BaseModel


class User(BaseModel):
    """A user."""

    name: str
    age: int


class TextResponse(BaseModel):
    """A text response."""

    text: str
