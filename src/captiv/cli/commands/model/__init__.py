"""
Model commands for the Captiv CLI.

This package provides commands for managing and interacting with models.
"""

from typing import Callable

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services.model_manager import ModelType

# Import model commands
from . import list as model_list
from . import show

# Create Typer app for model commands
app = typer.Typer(help="Manage and interact with models")

# Register static commands
app.command("list")(model_list.command)
app.command("show")(show.command)


# Dynamic command handler for model-specific commands
@handle_cli_errors
def model_command_handler(model_type: ModelType) -> Callable:
    """
    Factory function to create a command handler for a specific model type.

    Args:
        model_type: The model type to create a handler for.

    Returns:
        A command function that displays information about the specified model.
    """

    def command_func():
        """Display information about the model."""
        # Reuse the show command's functionality
        show.command(model_type.value)

    # Set the docstring and function name dynamically
    command_func.__doc__ = f"Display information about the {model_type.value} model."
    command_func.__name__ = f"{model_type.value}_command"

    return command_func


# Dynamically register a command for each model type
for model_type in ModelType:
    # Register the command with the model type as the command name
    app.command(model_type.value)(model_command_handler(model_type))
