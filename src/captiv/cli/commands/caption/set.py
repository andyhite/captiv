"""
Set image caption command for the Captiv CLI.

This module provides the command logic for setting or updating the caption for a specific image file.
This command is registered as `captiv caption set`.
"""

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ImagePathArgument
from captiv.services.caption_manager import CaptionManager


@handle_cli_errors
def command(image_path: ImagePathArgument, caption: str) -> None:
    """
    Set or update the caption for a specific image file (writes/overwrites the sidecar .txt file).

    Usage: captiv caption set IMAGE_PATH CAPTION
    """
    manager = CaptionManager()
    manager.set_caption(str(image_path), caption)
    typer.echo(f"Caption updated for {image_path.name}.")
