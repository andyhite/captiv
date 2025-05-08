"""
List configuration command for the Captiv CLI.

This module provides the command logic for listing configuration values.
"""

import json
from typing import Optional

import typer

from captiv import config
from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption


@handle_cli_errors
def command(
    section: Optional[str] = typer.Argument(
        None, help="Configuration section to list. If not provided, lists all sections."
    ),
    config_file: ConfigFileOption = None,
    json_format: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """List configuration values for a section or the entire configuration."""
    cfg = config.list_config(config_file)

    if section:
        # List only the specified section
        if section in cfg:
            section_values = cfg[section]

            if json_format:
                # Output as JSON
                typer.echo(json.dumps({section: section_values}, indent=2))
            else:
                # Pretty print the section
                typer.echo(f"Configuration section: [{section}]")
                for key, value in section_values.items():
                    typer.echo(f"  {key} = {value}")
        else:
            typer.echo(f"Unknown configuration section: {section}")
            typer.echo("Available sections: " + ", ".join(cfg.keys()))
    else:
        # List all sections
        if json_format:
            # Output as JSON
            typer.echo(json.dumps(cfg, indent=2))
        else:
            # Pretty print the configuration
            typer.echo("Current configuration:")

            for section_name, section_values in cfg.items():
                typer.echo(f"\n[{section_name}]")
                for key, value in section_values.items():
                    typer.echo(f"  {key} = {value}")
