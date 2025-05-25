"""
List models command for the Captiv CLI.

This module provides the command logic for listing all available models or model
variants of a specific model.
"""

from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services.model_manager import ModelManager, ModelType

ModelTypeArgument = Annotated[
    str | None,
    typer.Argument(
        help="Model to list model variants for. If not provided, lists all models.",
    ),
]


@handle_cli_errors
def command(
    model_type: ModelTypeArgument = None,
) -> None:
    """
    List all available models or model variants of a specific model.

    Usage: captiv model list [MODEL]
    """
    manager = ModelManager()

    if model_type:
        try:
            model_enum = ModelType(model_type)
        except ValueError:
            valid_models = ", ".join([m.value for m in ModelType])
            typer.echo(f"Error: Invalid model '{model_type}'")
            typer.echo(f"Valid models: {valid_models}")
            raise typer.Exit(1) from None

        variants = manager.get_variant_details(model_enum)
        modes = manager.get_mode_details(model_enum)

        typer.echo(f"\n=== {model_enum.value.upper()} Model ===\n")

        typer.echo("Available Model Variants:")
        for variant_name in variants:
            typer.echo(f"  {variant_name}")

        typer.echo("\nAvailable Modes:")
        if modes:
            for mode_name in modes:
                typer.echo(f"  {mode_name}")
        else:
            typer.echo("  No specific modes available for this model.")

        typer.echo("\nFor more details:")
        typer.echo(f"  captiv model show {model_enum.value}")
    else:
        typer.echo("Available models:")
        for model in ModelType:
            typer.echo(f"  {model.value}")

        typer.echo("\nFor more details about a specific model:")
        typer.echo("  captiv model show MODEL")
        typer.echo("  captiv model list MODEL")
