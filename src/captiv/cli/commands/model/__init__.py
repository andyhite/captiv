"""
Model commands for the Captiv CLI.

This package provides commands for managing and interacting with models and their
variants.
"""

from collections.abc import Callable

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services.model_manager import ModelType

from . import list as model_list
from . import show

app = typer.Typer(help="Manage and interact with models")

app.command("list")(model_list.command)
app.command("show")(show.command)


@handle_cli_errors
def model_command_handler(model_type: ModelType) -> Callable:
    """
    Factory function to create a command handler for a specific model.

    Args:
        model_type: The model to create a handler for.

    Returns:
        A command function that displays information about the specified model.
    """

    def command_func():
        """Display information about the model."""
        show.command(model_type.value)

    command_func.__doc__ = f"Display information about the {model_type.value} model."
    command_func.__name__ = f"{model_type.value}_command"

    return command_func


for model_type in ModelType:
    app.command(model_type.value)(model_command_handler(model_type))
