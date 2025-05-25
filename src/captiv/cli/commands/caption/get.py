"""
Get image caption command for the Captiv CLI.

This module provides the command logic for retrieving the caption for a specific image
file. This command is registered as `captiv caption get`.
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


@handle_cli_errors
def command(image_path: ImagePathArgument) -> None:
    """
    Get the caption for a specific image file (reads the sidecar .txt file).

    Usage: captiv caption get IMAGE_PATH
    """
    file_manager = FileManager()
    image_file_manager = ImageFileManager(file_manager)
    caption_manager = CaptionFileManager(file_manager, image_file_manager)

    try:
        caption = caption_manager.read_caption(image_path)
        if caption:
            typer.echo(caption)
        else:
            typer.echo(f"No caption found for {image_path.name}.")
    except FileNotFoundError:
        typer.echo(f"No caption found for {image_path.name}.")
    except Exception as e:
        typer.echo(f"Error reading caption for {image_path.name}: {e}")
        typer.echo(f"Error reading caption for {image_path.name}: {e}")
