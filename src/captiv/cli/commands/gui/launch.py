"""
Launch command for the Captiv GUI.

This module provides the command logic for launching the Gradio GUI.
"""

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption


@handle_cli_errors
def command(
    config_file: ConfigFileOption = None,
    share: bool = typer.Option(
        False,
        "--share",
        help="Launch the GUI with a public URL (using Gradio's share feature)",
    ),
) -> None:
    """Launch the Gradio GUI for Captiv.

    By default, the GUI is only accessible on localhost. Use the --share flag
    to make it accessible via a public URL.
    """
    try:
        # Import here to avoid circular imports
        from captiv.gui.main import main

        # Launch the GUI with the config file
        main(share=share, config_path=config_file)
    except ImportError as e:
        if "gradio" in str(e):
            typer.echo(
                "Error: Gradio 4.44.1 is not installed. Please install it with:\n"
                "pip install gradio==4.44.1\n"
                "or\n"
                "poetry add gradio==4.44.1"
            )
        else:
            raise
