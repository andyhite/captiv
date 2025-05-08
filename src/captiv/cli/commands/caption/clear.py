"""
Clear image captions command for the Captiv CLI.

This module provides the command logic for clearing captions for all images in a directory.
This command is registered as `captiv caption clear`.
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
        help="Directory to clear captions from. Defaults to current working directory.",
    ),
) -> None:
    """
    Clear all image captions in a directory (removes all sidecar .txt files).

    Usage: captiv caption clear [DIRECTORY]
    """
    # Default to current working directory if not specified
    if directory is None:
        directory = Path(os.getcwd())

    manager = CaptionManager()

    # Get all images with captions
    results = manager.list_images_with_captions(str(directory))

    if not results:
        typer.echo("No captioned images found.")
        return

    # Count of captions cleared
    cleared_count = 0

    # Clear each caption
    for img_name, caption in results:
        if caption:  # Only process images that have captions
            # Get the full path to the image
            img_path = Path(directory) / img_name
            # Get the caption file path
            caption_file = img_path.with_suffix(".txt")

            # Remove the caption file
            if caption_file.exists():
                caption_file.unlink()
                cleared_count += 1

    if cleared_count > 0:
        typer.echo(f"Cleared {cleared_count} caption(s).")
    else:
        typer.echo("No captions were cleared.")
