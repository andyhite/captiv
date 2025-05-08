"""
Configuration management for the Captiv CLI.

This module provides a centralized configuration system with nested sections,
validation, and default values for all configurable parameters.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import toml

from captiv.logging import logger

# Type variable for configuration sections
T = TypeVar("T", bound="ConfigSection")


class ConfigSection:
    """Base class for configuration sections with validation."""

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create a configuration section from a dictionary."""
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration section to a dictionary."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }

    def validate(self) -> None:
        """Validate the configuration section."""
        pass  # Base implementation does nothing


class ModelDefaults(ConfigSection):
    """Default configuration for models."""

    # Default model type to use
    default_model: str = "blip"

    # Default variants for each model type
    blip_variant: str = "blip-large"
    blip2_variant: str = "blip2-opt-2.7b"
    joycaption_variant: str = "joycaption-base"
    git_variant: str = "git-base"
    vit_gpt2_variant: str = "vit-gpt2"

    # Default modes for each model type
    blip_mode: Optional[str] = None  # None means unconditional captioning
    blip2_mode: str = "default"
    joycaption_mode: str = "default"
    git_mode: Optional[str] = None  # None means unconditional captioning
    vit_gpt2_mode: str = "default"

    def validate(self) -> None:
        """Validate model defaults."""
        valid_models = ["blip", "blip2", "joycaption", "git", "vit-gpt2"]
        if self.default_model not in valid_models:
            logger.warning(
                f"Invalid default_model: {self.default_model}. "
                f"Using 'blip' as fallback."
            )
            self.default_model = "blip"


class GenerationDefaults(ConfigSection):
    """Default configuration for text generation parameters."""

    # Common generation parameters with their default values
    max_length: int = 32
    min_length: int = 10
    num_beams: int = 3
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0

    # Model-specific overrides
    blip2_max_length: int = 75  # BLIP-2 can generate longer captions
    joycaption_max_length: int = 50  # JoyCaption default max length
    joycaption_num_beams: int = 5  # JoyCaption default beam count
    git_max_length: int = 50  # GIT default max length
    vit_gpt2_max_length: int = 50  # ViT+GPT2 default max length

    # JoyCaption-specific parameters
    joycaption_guidance_scale: float = 7.5  # Controls how closely to follow the prompt
    joycaption_quality_level: str = "standard"  # Options: "draft", "standard", "high"

    # WD14-specific parameters

    def validate(self) -> None:
        """Validate generation parameters."""
        if self.max_length < 1:
            logger.warning(f"Invalid max_length: {self.max_length}. Using default: 32")
            self.max_length = 32

        if self.min_length < 1:
            logger.warning(f"Invalid min_length: {self.min_length}. Using default: 10")
            self.min_length = 10

        if self.min_length > self.max_length:
            logger.warning(
                f"min_length ({self.min_length}) > max_length ({self.max_length}). "
                f"Setting min_length to max_length."
            )
            self.min_length = self.max_length

        if self.num_beams < 1:
            logger.warning(f"Invalid num_beams: {self.num_beams}. Using default: 3")
            self.num_beams = 3

        if self.temperature <= 0:
            logger.warning(
                f"Invalid temperature: {self.temperature}. Using default: 1.0"
            )
            self.temperature = 1.0

        if self.top_k < 1:
            logger.warning(f"Invalid top_k: {self.top_k}. Using default: 50")
            self.top_k = 50

        if self.top_p <= 0 or self.top_p > 1:
            logger.warning(f"Invalid top_p: {self.top_p}. Using default: 0.9")
            self.top_p = 0.9

        if self.repetition_penalty < 1:
            logger.warning(
                f"Invalid repetition_penalty: {self.repetition_penalty}. Using default: 1.0"
            )
            self.repetition_penalty = 1.0


class SystemDefaults(ConfigSection):
    """System-level configuration defaults."""

    # Supported torch dtypes
    supported_dtypes: List[str] = ["float16", "float32", "bfloat16"]

    # Default torch dtype (None means use model's default)
    default_torch_dtype: Optional[str] = None


class GuiDefaults(ConfigSection):
    """GUI configuration defaults."""

    # Server configuration
    host: str = "127.0.0.1"  # Default to localhost
    port: int = 7860  # Default Gradio port
    ssl_keyfile: Optional[str] = None  # Path to SSL key file
    ssl_certfile: Optional[str] = None  # Path to SSL certificate file

    def validate(self) -> None:
        """Validate GUI configuration."""
        # Ensure host has a sensible default
        if not self.host:
            logger.warning("Missing or invalid host. Using default: 127.0.0.1")
            self.host = "127.0.0.1"

        # Ensure port has a sensible default
        if self.port is None or self.port < 1 or self.port > 65535:
            logger.warning(f"Invalid port: {self.port}. Using default: 7860")
            self.port = 7860

        # Ensure ssl_keyfile and ssl_certfile have sensible defaults
        if self.ssl_keyfile == "":
            self.ssl_keyfile = None

        if self.ssl_certfile == "":
            self.ssl_certfile = None


class AppConfig:
    """Main configuration class that holds all configuration sections."""

    def __init__(self):
        """Initialize the configuration with default values."""
        self.model = ModelDefaults()
        self.generation = GenerationDefaults()
        self.system = SystemDefaults()
        self.gui = GuiDefaults()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create a configuration from a dictionary."""
        config = cls()

        # Process model section
        if "model" in data and isinstance(data["model"], dict):
            config.model = ModelDefaults.from_dict(data["model"])

        # Process generation section
        if "generation" in data and isinstance(data["generation"], dict):
            config.generation = GenerationDefaults.from_dict(data["generation"])

        # Process system section
        if "system" in data and isinstance(data["system"], dict):
            config.system = SystemDefaults.from_dict(data["system"])

        # Process GUI section
        if "gui" in data and isinstance(data["gui"], dict):
            config.gui = GuiDefaults.from_dict(data["gui"])
        else:
            # Ensure GUI section exists with default values
            logger.info("GUI section missing from config. Using default values.")
            config.gui = GuiDefaults()

        # Validate all sections
        config.validate()

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "model": self.model.to_dict(),
            "generation": self.generation.to_dict(),
            "system": self.system.to_dict(),
            "gui": self.gui.to_dict(),
        }

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.model.validate()
        self.generation.validate()
        self.system.validate()
        self.gui.validate()


# Global configuration instance with default values
DEFAULT_CONFIG = AppConfig()


def get_config_dir() -> Path:
    """
    Get the directory where configuration files are stored.

    Returns:
        Path to the configuration directory
    """
    config_dir = Path.home() / ".captiv"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_config_path(config_path: Optional[str] = None) -> Path:
    """
    Get the path to the configuration file.

    Args:
        config_path: Optional path to the configuration file

    Returns:
        Path to the configuration file
    """
    if config_path:
        return Path(config_path)
    return get_config_dir() / "config.toml"


def read_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Read the configuration from the configuration file.

    Args:
        config_path: Optional path to the configuration file

    Returns:
        AppConfig instance containing the configuration
    """
    path = get_config_path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG

    try:
        with open(path, "r") as f:
            config_dict = toml.load(f)
        return AppConfig.from_dict(config_dict)
    except (toml.TomlDecodeError, IOError) as e:
        logger.warning(f"Error reading config file: {e}. Using default configuration.")
        return DEFAULT_CONFIG


def write_config(config: AppConfig, config_path: Optional[str] = None) -> None:
    """
    Write the configuration to the configuration file.

    Args:
        config: AppConfig instance containing the configuration
        config_path: Optional path to the configuration file
    """
    path = get_config_path(config_path)

    # Ensure the directory exists
    path.parent.mkdir(exist_ok=True)

    with open(path, "w") as f:
        toml.dump(config.to_dict(), f)


def get_config_value(section: str, key: str, config_path: Optional[str] = None) -> Any:
    """
    Get a configuration value from a specific section.

    Args:
        section: Configuration section name
        key: Configuration key within the section
        config_path: Optional path to the configuration file

    Returns:
        Configuration value
    """
    config = read_config(config_path)
    if hasattr(config, section):
        section_obj = getattr(config, section)
        if hasattr(section_obj, key):
            return getattr(section_obj, key)

    # If the section or key doesn't exist, return None
    return None


def set_config_value(
    section: str, key: str, value: Any, config_path: Optional[str] = None
) -> None:
    """
    Set a configuration value in a specific section.

    Args:
        section: Configuration section name
        key: Configuration key within the section
        value: Configuration value to set
        config_path: Optional path to the configuration file
    """
    config = read_config(config_path)

    if hasattr(config, section):
        section_obj = getattr(config, section)
        if hasattr(section_obj, key):
            setattr(section_obj, key, value)
            config.validate()
            write_config(config, config_path)
        else:
            logger.warning(f"Unknown configuration key: {section}.{key}")
    else:
        logger.warning(f"Unknown configuration section: {section}")


def list_config(config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    List all configuration values.

    Args:
        config_path: Optional path to the configuration file

    Returns:
        Dictionary containing the configuration
    """
    config = read_config(config_path)
    return config.to_dict()


def get_legacy_config_value(key: str, config_path: Optional[str] = None) -> Any:
    """
    Get a configuration value using the legacy key format.

    This function maps legacy keys to the new section.key format.

    Args:
        key: Legacy configuration key
        config_path: Optional path to the configuration file

    Returns:
        Configuration value
    """
    # Map legacy keys to new section.key format
    legacy_key_mapping = {
        "model": ("model", "default_model"),
        "default_variant": ("model", "blip_variant"),  # This is a simplification
        "default_mode": ("model", "blip_mode"),  # This is a simplification
    }

    if key in legacy_key_mapping:
        section, new_key = legacy_key_mapping[key]
        return get_config_value(section, new_key, config_path)

    # If the key is not in the mapping, return None
    return None


def set_legacy_config_value(
    key: str, value: Any, config_path: Optional[str] = None
) -> None:
    """
    Set a configuration value using the legacy key format.

    This function maps legacy keys to the new section.key format.

    Args:
        key: Legacy configuration key
        value: Configuration value to set
        config_path: Optional path to the configuration file
    """
    # Map legacy keys to new section.key format
    legacy_key_mapping = {
        "model": ("model", "default_model"),
        "default_variant": ("model", "blip_variant"),  # This is a simplification
        "default_mode": ("model", "blip_mode"),  # This is a simplification
    }

    if key in legacy_key_mapping:
        section, new_key = legacy_key_mapping[key]
        set_config_value(section, new_key, value, config_path)
    else:
        logger.warning(f"Unknown legacy configuration key: {key}")
