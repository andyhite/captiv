"""
Set configuration command for the Captiv CLI.

This module provides the command logic for setting configuration values.
"""

import typer
from typing_extensions import Annotated

from captiv import config
from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services.model_manager import ModelType


@handle_cli_errors
def command(
    key_value: Annotated[
        str,
        typer.Argument(
            ..., help="Configuration key-value pair in the format section.key=value"
        ),
    ],
    config_file: ConfigFileOption = None,
) -> None:
    """Set a configuration value."""
    if "=" not in key_value:
        raise ValueError("Key-value pair must be in the format section.key=value")

    key_path, value = key_value.split("=", 1)

    # Handle legacy format (no section)
    if "." not in key_path:
        # Handle special cases for certain legacy keys
        if key_path == "model":
            try:
                # Validate model value
                ModelType(value)
            except ValueError:
                valid_models = ", ".join([m.value for m in ModelType])
                raise ValueError(
                    f"Invalid model '{value}'. Valid models: {valid_models}"
                )

            # Set using legacy method
            config.set_legacy_config_value(key_path, value, config_file)
            typer.echo(f"Configuration updated: {key_path}={value}")
        else:
            typer.echo(f"Warning: Legacy configuration key '{key_path}' is deprecated.")
            typer.echo("Please use the new format: section.key=value")
            typer.echo(
                "Run 'captiv config list' to see available configuration options."
            )
    else:
        # New format: section.key=value
        section, key = key_path.split(".", 1)

        # Type conversion based on section and key
        if section == "model" and key == "default_model":
            try:
                # Validate model value
                ModelType(value)
            except ValueError:
                valid_models = ", ".join([m.value for m in ModelType])
                raise ValueError(
                    f"Invalid model '{value}'. Valid models: {valid_models}"
                )
        elif section == "generation":
            # Try to convert numeric values
            try:
                if key in ["max_length", "min_length", "num_beams", "top_k"]:
                    value = int(value)
                elif key in ["temperature", "top_p", "repetition_penalty"]:
                    value = float(value)
            except ValueError:
                raise ValueError(
                    f"Invalid value for {section}.{key}: {value}. Expected a number."
                )
        elif section == "gui":
            # Try to convert numeric values for GUI section
            try:
                if key == "port":
                    value = int(value)
            except ValueError:
                raise ValueError(
                    f"Invalid value for {section}.{key}: {value}. Expected a number."
                )

        config.set_config_value(section, key, value, config_file)
        typer.echo(f"Configuration updated: {key_path}={value}")
