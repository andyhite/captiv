from typing import Any

from .defaults import (
    GenerationDefaults,
    GuiDefaults,
    ModelDefaults,
    RunPodDefaults,
    SystemDefaults,
)


class Config:
    """Main configuration class that holds all configuration sections."""

    def __init__(self):
        """Initialize the configuration with default values."""
        self.model = ModelDefaults()
        self.generation = GenerationDefaults()
        self.system = SystemDefaults()
        self.gui = GuiDefaults()
        self.runpod = RunPodDefaults()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create a configuration from a dictionary."""
        config = cls()

        if "model" in data and isinstance(data["model"], dict):
            config.model = ModelDefaults.from_dict(data["model"])

        if "generation" in data and isinstance(data["generation"], dict):
            config.generation = GenerationDefaults.from_dict(data["generation"])

        if "system" in data and isinstance(data["system"], dict):
            config.system = SystemDefaults.from_dict(data["system"])

        if "gui" in data and isinstance(data["gui"], dict):
            config.gui = GuiDefaults.from_dict(data["gui"])
        else:
            config.gui = GuiDefaults()

        if "runpod" in data and isinstance(data["runpod"], dict):
            config.runpod = RunPodDefaults.from_dict(data["runpod"])
        else:
            config.runpod = RunPodDefaults()

        config.validate()

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "model": self.model.to_dict(),
            "generation": self.generation.to_dict(),
            "system": self.system.to_dict(),
            "gui": self.gui.to_dict(),
            "runpod": self.runpod.to_dict(),
        }

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.model.validate()
        self.generation.validate()
        self.system.validate()
        self.gui.validate()
        self.runpod.validate()
