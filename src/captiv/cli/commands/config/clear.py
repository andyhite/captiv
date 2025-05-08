"""
Clear configuration command for the Captiv CLI.

This module provides the command logic for clearing configuration values.
"""

from typing import Optional

import typer

from captiv import config
from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption


@handle_cli_errors
def command(
    section: Optional[str] = typer.Argument(
        None,
        help="Configuration section to clear. If not provided, clears all sections.",
    ),
    config_file: ConfigFileOption = None,
) -> None:
    """Clear configuration values for a section or the entire configuration."""
    # Read the current configuration
    cfg = config.read_config(config_file)
    config_path = config.get_config_path(config_file)

    if section:
        # Clear a specific section
        if hasattr(cfg, section):
            # Create a new instance of the section with default values
            section_class = getattr(cfg, section).__class__
            setattr(cfg, section, section_class())

            # Write the updated configuration
            config.write_config(cfg, config_file)
            typer.echo(f"Configuration section '{section}' has been reset to defaults.")
        else:
            typer.echo(f"Unknown configuration section: {section}")
            typer.echo(
                "Run 'captiv config list' to see available configuration sections."
            )
    else:
        # Clear the entire configuration by removing the config file
        if config_path.exists():
            config_path.unlink()
            typer.echo("Configuration has been reset to defaults.")
        else:
            typer.echo("No configuration file found. Already using defaults.")
