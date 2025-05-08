"""
GUI command for the Captiv CLI.

This module provides the command logic for launching the Gradio GUI.
"""

import typer

from captiv.cli.commands.gui import launch

app = typer.Typer(help="Launch the Gradio GUI")

# Register commands
app.command(name="launch")(launch.command)
