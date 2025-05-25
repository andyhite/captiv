"""
Shared CLI options for Captiv.

This module provides option definitions that are used by multiple CLI commands. Options
that are only used in one place should be defined inline in that command.
"""

from typing import Annotated

import typer

ConfigFileOption = Annotated[
    str | None,
    typer.Option(
        "--config-file",
        "-c",
        help="Path to the configuration file",
        envvar="CAPTIV_CONFIG_FILE",
    ),
]
