"""
Clear configuration command for the Captiv CLI.

This module provides the command logic for clearing configuration values.
"""

from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services.config_manager import ConfigManager

SectionArgument = Annotated[
    str | None,
    typer.Argument(
        help="Configuration section to clear. If not provided, clears all sections.",
    ),
]


@handle_cli_errors
def command(
    section: SectionArgument = None,
    config_file: ConfigFileOption = None,
) -> None:
    """Clear configuration values for a section or the entire configuration."""
    config_manager = ConfigManager(config_file)

    try:
        config_manager.clear_config(section)
        if section:
            typer.echo(f"Configuration section '{section}' has been reset to defaults.")
        else:
            typer.echo("Configuration has been reset to defaults.")
    except ValueError as e:
        typer.echo(f"Error: {e}")
        typer.echo("Run 'captiv config list' to see available configuration sections.")
