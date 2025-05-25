"""
Get configuration command for the Captiv CLI.

This module provides the command logic for getting configuration values.
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
    """Get a configuration value."""
    config_manager = ConfigManager(config_file)

    if "." not in key_path:
        typer.echo("Error: Key path must be in the format section.key")
        typer.echo("Run 'captiv config list' to see available configuration options.")
        return

    section, key = key_path.split(".", 1)
    try:
        value = config_manager.get_config_value(section, key)
        if value is not None:
            typer.echo(value)
        else:
            typer.echo(f"Configuration key '{key_path}' not found.")
    except ValueError as e:
        typer.echo(f"Error: {e}")
