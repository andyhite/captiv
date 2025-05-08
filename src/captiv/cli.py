#!/usr/bin/env python
"""
Command-line interface for the Captiv image captioning library.

This module provides a CLI for generating image captions using various models, powered by Typer.
This is the entry point for the CLI, with the actual implementation in the captiv.cli package.
"""

import typer

# Import caption commands
from captiv.cli.commands.caption import clear as caption_clear
from captiv.cli.commands.caption import generate
from captiv.cli.commands.caption import get as caption_get
from captiv.cli.commands.caption import list as caption_list
from captiv.cli.commands.caption import set as caption_set
from captiv.cli.commands.caption import unset as caption_unset

# Import config commands
from captiv.cli.commands.config import clear, get, list, set, unset

# Import GUI commands
from captiv.cli.commands.gui import app as gui_app

# Import model commands
from captiv.cli.commands.model import app as model_app

# Import logging configuration
from captiv.logging import logger, setup_logging

# Create Typer apps
app = typer.Typer(help="Captiv - Image Captioning CLI")
config_app = typer.Typer(help="Manage Captiv configuration")
caption_app = typer.Typer(help="Generate and manage image captions")

# Register subapps
app.add_typer(config_app, name="config")
app.add_typer(caption_app, name="caption")
app.add_typer(model_app, name="model")
app.add_typer(gui_app, name="gui")

# Register caption commands
caption_app.command("generate")(generate.command)
caption_app.command("list")(caption_list.command)
caption_app.command("clear")(caption_clear.command)
caption_app.command("get")(caption_get.command)
caption_app.command("set")(caption_set.command)
caption_app.command("unset")(caption_unset.command)

# Register config commands
config_app.command("get")(get.command)
config_app.command("set")(set.command)
config_app.command("list")(list.command)
config_app.command("clear")(clear.command)
config_app.command("unset")(unset.command)


def run_app() -> None:
    """
    Run the Typer app with proper logging configuration.
    """
    # Configure logging with Loguru
    setup_logging(level="INFO", intercept_libraries=["gradio"])

    logger.info("Starting Captiv CLI")

    # Run the Typer app
    app()


def main() -> None:
    """
    Main entry point for the CLI.
    """
    run_app()


if __name__ == "__main__":
    main()
