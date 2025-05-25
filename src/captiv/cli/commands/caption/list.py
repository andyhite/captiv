"""
List image captions command for the Captiv CLI.

This module provides the command logic for listing image files and their captions in a
directory. This command is registered as `captiv caption list`.
"""

import os
from pathlib import Path
from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services import CaptionFileManager, FileManager, ImageFileManager

DirectoryArgument = Annotated[
    Path | None,
    typer.Argument(
        help="Directory to list captions from. Defaults to current working directory.",
    ),
]


@handle_cli_errors
def command(directory: DirectoryArgument = None) -> None:
    """
    List all image files in a directory and show their captions if a sidecar .txt file
    exists and has text.

    Usage: captiv caption list [DIRECTORY]
    """
    if directory is None:
        directory = Path(os.getcwd())

    file_manager = FileManager()
    image_file_manager = ImageFileManager(file_manager)
    caption_manager = CaptionFileManager(file_manager, image_file_manager)

    results = caption_manager.list_images_and_captions(directory)

    if not results:
        typer.echo(f"No images found in {directory}.")
        return

    for img_path, caption in results:
        typer.echo(f"{img_path.name}: {caption if caption else 'No caption'}")
