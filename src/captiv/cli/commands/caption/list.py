"""
List image captions command for the Captiv CLI.

This module provides the command logic for listing image files and their captions in a directory.
This command is registered as `captiv caption list`.
"""

import os
from pathlib import Path
from typing import Optional

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services.caption_manager import CaptionManager


@handle_cli_errors
def command(
    directory: Optional[Path] = typer.Argument(
        None,
        help="Directory to list captions from. Defaults to current working directory.",
    ),
) -> None:
    """
    List all image files in a directory and show their captions if a sidecar .txt file exists and has text.

    Usage: captiv caption list [DIRECTORY]
    """
    # Default to current working directory if not specified
    if directory is None:
        directory = Path(os.getcwd())

    manager = CaptionManager()
    results = manager.list_images_with_captions(str(directory))

    if not results:
        typer.echo(f"No images found in {directory}.")
        return

    for img_name, caption in results:
        typer.echo(f"{img_name}: {caption if caption else 'No caption'}")
