"""
Set configuration command for the Captiv CLI.

This module provides the command logic for setting configuration values.
"""

import re
from typing import Annotated, Any

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services.config_manager import ConfigManager
from captiv.services.model_manager import ModelType


def smart_type_conversion(value: str) -> int | float | bool | str:
    """
    Automatically convert a string value to the most appropriate type.

    Args:
        value: String value to convert

    Returns:
        Converted value as int, float, bool, or str
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    if re.match(r"^-?\d+$", value):
        return int(value)

    if re.match(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$", value):
        return float(value)

    return value


def validate_special_values(section: str, key: str, value: Any) -> Any:
    """
    Validate special configuration values that have specific constraints.

    Args:
        section: Configuration section name
        key: Configuration key name
        value: Value to validate

    Returns:
        Validated value

    Raises:
        ValueError: If validation fails
    """
    if section == "model" and key == "default_model":
        try:
            ModelType(str(value))
        except ValueError:
            valid_models = ", ".join([m.value for m in ModelType])
            raise ValueError(
                f"Invalid model '{value}'. Valid models: {valid_models}"
            ) from None

    if key == "port" and isinstance(value, int) and not (1 <= value <= 65535):
        raise ValueError(f"Port must be between 1 and 65535, got: {value}")

    if (
        key in ["max_new_tokens", "min_new_tokens", "num_beams", "top_k"]
        and isinstance(value, int)
        and value <= 0
    ):
        raise ValueError(f"{key} must be a positive integer, got: {value}")

    if key in ["temperature", "top_p", "repetition_penalty"] and isinstance(
        value, int | float
    ):
        if key == "temperature" and value <= 0:
            raise ValueError(f"Temperature must be positive, got: {value}")
        elif key == "top_p" and not (0 < value <= 1):
            raise ValueError(f"top_p must be between 0 and 1, got: {value}")
        elif key == "repetition_penalty" and value <= 0:
            raise ValueError(f"Repetition penalty must be positive, got: {value}")

    return value


KeyValueArgument = Annotated[
    str,
    typer.Argument(
        ..., help="Configuration key-value pair in the format section.key=value"
    ),
]


@handle_cli_errors
def command(
    key_value: KeyValueArgument,
    config_file: ConfigFileOption = None,
) -> None:
    """Set a configuration value."""
    if "=" not in key_value:
        raise ValueError("Key-value pair must be in the format section.key=value")

    key_path, value = key_value.split("=", 1)

    if "." not in key_path:
        typer.echo("Error: Key path must be in the format section.key=value")
        typer.echo("Run 'captiv config list' to see available configuration options.")
        return

    config_manager = ConfigManager(config_file)
    section, key = key_path.split(".", 1)

    try:
        converted_value = smart_type_conversion(value)
        validated_value = validate_special_values(section, key, converted_value)

        config_manager.set_config_value(section, key, validated_value)
        typer.echo(f"Configuration updated: {key_path}={validated_value}")
    except ValueError as e:
        typer.echo(f"Error: {e}")
        typer.echo(f"Error: {e}")
