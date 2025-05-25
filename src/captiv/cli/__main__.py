#!/usr/bin/env python
"""
Command-line interface for the Captiv image captioning library.

This module provides a CLI for generating image captions using various models, powered
by Typer. This is the entry point for the CLI, with the actual implementation in the
captiv.cli package.
"""

import typer

from .commands.caption import clear as caption_clear
from .commands.caption import generate
from .commands.caption import get as caption_get
from .commands.caption import list as caption_list
from .commands.caption import set as caption_set
from .commands.caption import unset as caption_unset
from .commands.config import clear, get, unset
from .commands.config import list as config_list
from .commands.config import set as config_set
from .commands.gui import app as gui_app
from .commands.model import app as model_app
from .commands.runpod import app as runpod_app

app = typer.Typer(help="Captiv - Image Captioning CLI")
config_app = typer.Typer(help="Manage Captiv configuration")
caption_app = typer.Typer(help="Generate and manage image captions")

app.add_typer(config_app, name="config")
app.add_typer(caption_app, name="caption")
app.add_typer(model_app, name="model")
app.add_typer(gui_app, name="gui")
app.add_typer(runpod_app, name="runpod")

caption_app.command("generate")(generate.command)
caption_app.command("list")(caption_list.command)
caption_app.command("clear")(caption_clear.command)
caption_app.command("get")(caption_get.command)
caption_app.command("set")(caption_set.command)
caption_app.command("unset")(caption_unset.command)

config_app.command("get")(get.command)
config_app.command("set")(config_set.command)
config_app.command("list")(config_list.command)
config_app.command("clear")(clear.command)
config_app.command("unset")(unset.command)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
