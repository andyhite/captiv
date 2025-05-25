"""
List configuration command for the Captiv CLI.

This module provides the command logic for listing configuration values.
"""

import json
from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services.config_manager import ConfigManager

SectionArgument = Annotated[
    str | None,
    typer.Argument(
        help="Configuration section to list. If not provided, lists all sections."
    ),
]

JsonFormatOption = Annotated[
    bool,
    typer.Option("--json", help="Output in JSON format"),
]


@handle_cli_errors
def command(
    section: SectionArgument = None,
    config_file: ConfigFileOption = None,
    json_format: JsonFormatOption = False,
) -> None:
    """List configuration values for a section or the entire configuration."""
    config_manager = ConfigManager(config_file)

    cfg = config_manager.get_config()

    if section:
        if section in cfg:
            section_values = cfg[section]

            if json_format:
                typer.echo(json.dumps({section: section_values}, indent=2))
            else:
                for key, value in section_values.items():
                    typer.echo(f"{key} = {value}")
        else:
            typer.echo(f"Unknown configuration section: {section}")
            typer.echo("Available sections: " + ", ".join(cfg.keys()))
    else:
        if json_format:
            typer.echo(json.dumps(cfg, indent=2))
        else:
            for section_name, section_values in cfg.items():
                typer.echo(f"\n[{section_name}]")
                for key, value in section_values.items():
                    typer.echo(f"  {key} = {value}")
