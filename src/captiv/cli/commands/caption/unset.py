"""
Unset image caption command for the Captiv CLI.

This module provides the command logic for removing the caption for a specific image file.
This command is registered as `captiv caption unset`.
"""

from pathlib import Path

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ImagePathArgument


@handle_cli_errors
def command(image_path: ImagePathArgument) -> None:
    """
    Remove the caption for a specific image file (deletes the sidecar .txt file).

    Usage: captiv caption unset IMAGE_PATH
    """
    # Get the caption file path
    caption_file = Path(str(image_path)).with_suffix(".txt")

    # Check if the caption file exists
    if not caption_file.exists():
        typer.echo(f"No caption found for {image_path.name}.")
        return

    # Remove the caption file
    caption_file.unlink()
    typer.echo(f"Caption removed for {image_path.name}.")
