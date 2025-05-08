"""
Get configuration command for the Captiv CLI.

This module provides the command logic for getting configuration values.
"""

import typer
from typing_extensions import Annotated

from captiv import config
from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption


@handle_cli_errors
def command(
    key_path: Annotated[
        str, typer.Argument(..., help="Configuration key path in format section.key")
    ],
    config_file: ConfigFileOption = None,
) -> None:
    """Get a configuration value."""
    if "." not in key_path:
        # For backward compatibility, try legacy key format
        value = config.get_legacy_config_value(key_path, config_file)
        typer.echo(f"{key_path}={value}")
    else:
        # New format: section.key
        section, key = key_path.split(".", 1)
        value = config.get_config_value(section, key, config_file)
        typer.echo(f"{key_path}={value}")
