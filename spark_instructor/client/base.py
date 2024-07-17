"""A module for defining abstract classes for clients."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, Type

from instructor import Instructor, Mode

from spark_instructor.client import CompletionModel


class BaseClient(ABC, Generic[CompletionModel]):
    """An abstract class for instructor clients."""

    completion_model: Type[CompletionModel]

    @abstractmethod
    def get_instructor(
        self, mode: Mode, api_key: Optional[str] = None, base_url: Optional[str] = None, *args, **kwargs
    ) -> Instructor:
        """Abstract method for getting an instructor."""
        raise NotImplementedError("You must implement ``get_instructor``.")
