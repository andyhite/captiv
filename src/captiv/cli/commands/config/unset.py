"""
Unset configuration command for the Captiv CLI.

This module provides the command logic for removing configuration values.
"""

from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services.config_manager import ConfigManager

KeyPathArgument = Annotated[
    str,
    typer.Argument(..., help="Configuration key path in format section.key"),
]


@handle_cli_errors
def command(
    key_path: KeyPathArgument,
    config_file: ConfigFileOption = None,
) -> None:
    """Remove a configuration value, resetting it to the default."""
    if "." not in key_path:
        typer.echo("Error: Key path must be in the format section.key")
        typer.echo("Run 'captiv config list' to see available configuration options.")
        return

    config_manager = ConfigManager(config_file)
    section, key = key_path.split(".", 1)

    try:
        default_value = config_manager._default_config
        if hasattr(default_value, section):
            section_obj = getattr(default_value, section)
            if hasattr(section_obj, key):
                default_val = getattr(section_obj, key)
            else:
                default_val = "default"
        else:
            default_val = "default"

        config_manager.unset_config_value(section, key)
        typer.echo(
            f"Configuration value {key_path} has been reset to default: {default_val}"
        )
    except ValueError as e:
        typer.echo(f"Error: {e}")
        typer.echo("Run 'captiv config list' to see available configuration options.")
