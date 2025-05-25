"""
Set image caption command for the Captiv CLI.

This module provides the command logic for setting or updating the caption for a
specific image file. This command is registered as `captiv caption set`.
"""

from pathlib import Path
from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services import CaptionFileManager, FileManager, ImageFileManager

ImagePathArgument = Annotated[
    Path,
    typer.Argument(..., help="Path to the image file", exists=True, dir_okay=False),
]

CaptionArgument = Annotated[
    str,
    typer.Argument(..., help="Caption text to set for the image"),
]


@handle_cli_errors
def command(image_path: ImagePathArgument, caption: CaptionArgument) -> None:
    """
    Set or update the caption for a specific image file (writes/overwrites the sidecar
    .txt file).

    Usage: captiv caption set IMAGE_PATH CAPTION
    """
    file_manager = FileManager()
    image_file_manager = ImageFileManager(file_manager)
    caption_manager = CaptionFileManager(file_manager, image_file_manager)
    caption_manager.write_caption(image_path, caption)
    typer.echo(f"Caption updated for {image_path.name}.")
    typer.echo(f"Caption updated for {image_path.name}.")
