from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T", bound="ConfigSection")


class ConfigSection(ABC):
    """Base class for configuration sections with validation."""

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create a configuration section from a dictionary."""
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration section to a dictionary."""
        result = {}

        for key in dir(self.__class__):
            if (
                not key.startswith("_")
                and not callable(getattr(self.__class__, key))
                and key not in ["validate", "to_dict", "from_dict"]
            ):
                result[key] = getattr(self, key)

        for key, value in self.__dict__.items():
            if not key.startswith("_") and not callable(value):
                result[key] = value

        return result

    @abstractmethod
    def validate(self) -> None:
        """Validate the configuration section."""
