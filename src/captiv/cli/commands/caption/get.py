"""
Get image caption command for the Captiv CLI.

This module provides the command logic for retrieving the caption for a specific image file.
This command is registered as `captiv caption get`.
"""

from pathlib import Path

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ImagePathArgument
from captiv.services.caption_manager import CaptionManager


@handle_cli_errors
def command(image_path: ImagePathArgument) -> None:
    """
    Get the caption for a specific image file (reads the sidecar .txt file).

    Usage: captiv caption get IMAGE_PATH
    """
    CaptionManager()

    # Get the caption file path
    caption_file = Path(str(image_path)).with_suffix(".txt")

    # Check if the caption file exists
    if not caption_file.exists():
        typer.echo(f"No caption found for {image_path.name}.")
        return

    # Read the caption
    caption = caption_file.read_text(encoding="utf-8").strip()

    if not caption:
        typer.echo(f"Caption file exists but is empty for {image_path.name}.")
    else:
        typer.echo(caption)
