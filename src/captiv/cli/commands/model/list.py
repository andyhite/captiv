"""
List models command for the Captiv CLI.

This module provides the command logic for listing all available models or variants of a specific model.
"""

from typing import Optional

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services.model_manager import ModelManager, ModelType


@handle_cli_errors
def command(
    model_type: Optional[str] = typer.Argument(
        None,
        help="Model type to list variants for. If not provided, lists all model types.",
    ),
) -> None:
    """
    List all available models or variants of a specific model.

    Usage: captiv model list [MODEL_TYPE]
    """
    manager = ModelManager()

    if model_type:
        # List variants for a specific model type
        try:
            model_enum = ModelType(model_type)
        except ValueError:
            valid_models = ", ".join([m.value for m in ModelType])
            typer.echo(f"Error: Invalid model type '{model_type}'")
            typer.echo(f"Valid model types: {valid_models}")
            raise typer.Exit(1)

        # Get variants and modes
        variants = manager.get_variant_details(model_enum)
        modes = manager.get_mode_details(model_enum)

        # Display model information
        typer.echo(f"\n=== {model_enum.value.upper()} Model ===\n")

        # Display available variants
        typer.echo("Available Variants:")
        for variant_name in variants.keys():
            typer.echo(f"  {variant_name}")

        # Display available modes
        typer.echo("\nAvailable Modes:")
        if modes:
            for mode_name in modes.keys():
                typer.echo(f"  {mode_name}")
        else:
            typer.echo("  No specific modes available for this model.")

        # Display usage example
        typer.echo("\nFor more details:")
        typer.echo(f"  captiv model show {model_enum.value}")
    else:
        # List all model types
        typer.echo("Available models:")
        for model in ModelType:
            typer.echo(f"  {model.value}")

        typer.echo("\nFor more details about a specific model:")
        typer.echo("  captiv model show MODEL_TYPE")
        typer.echo("  captiv model list MODEL_TYPE")
