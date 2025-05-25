"""
Clear image captions command for the Captiv CLI.

This module provides the command logic for clearing captions for all images in a
directory. This command is registered as `captiv caption clear`.
"""

import os
from pathlib import Path
from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services import CaptionFileManager, FileManager, ImageFileManager

DirectoryOption = Annotated[
    Path | None,
    typer.Argument(
        help="Directory to clear captions from. Defaults to current working directory.",
    ),
]


@handle_cli_errors
def command(directory: DirectoryOption = None) -> None:
    """
    Clear all image captions in a directory (removes all sidecar .txt files).

    Usage: captiv caption clear [DIRECTORY]
    """
    if directory is None:
        directory = Path(os.getcwd())

    file_manager = FileManager()
    image_file_manager = ImageFileManager(file_manager)
    caption_manager = CaptionFileManager(file_manager, image_file_manager)

    try:
        caption_manager.clear_captions(directory)
        typer.echo("Captions cleared successfully.")
    except Exception as e:
        typer.echo(f"Error clearing captions: {e}")
