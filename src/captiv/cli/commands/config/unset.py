"""
Unset configuration command for the Captiv CLI.

This module provides the command logic for removing configuration values.
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
    """Remove a configuration value, resetting it to the default."""
    if "." not in key_path:
        typer.echo("Error: Key path must be in the format section.key")
        typer.echo("Run 'captiv config list' to see available configuration options.")
        return

    # Parse the section and key
    section, key = key_path.split(".", 1)

    # Read the current configuration
    cfg = config.read_config(config_file)

    # Check if the section exists
    if not hasattr(cfg, section):
        typer.echo(f"Unknown configuration section: {section}")
        typer.echo("Run 'captiv config list' to see available configuration sections.")
        return

    # Check if the key exists in the section
    section_obj = getattr(cfg, section)
    if not hasattr(section_obj, key):
        typer.echo(f"Unknown configuration key: {key_path}")
        typer.echo("Run 'captiv config list' to see available configuration options.")
        return

    # Get the default value for this key
    default_section = getattr(config.DEFAULT_CONFIG, section)
    if hasattr(default_section, key):
        default_value = getattr(default_section, key)

        # Reset to the default value
        setattr(section_obj, key, default_value)

        # Write the updated configuration
        config.write_config(cfg, config_file)
        typer.echo(
            f"Configuration value {key_path} has been reset to default: {default_value}"
        )
    else:
        typer.echo(f"Could not determine default value for {key_path}")
