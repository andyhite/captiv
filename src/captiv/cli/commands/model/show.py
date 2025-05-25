"""
Show model details command for the Captiv CLI.

This module provides the command logic for displaying detailed information about a
specific model and its variants.
"""

from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services.model_manager import ModelManager, ModelType

ModelArgument = Annotated[
    str,
    typer.Argument(..., help="Model to show details for"),
]


@handle_cli_errors
def command(
    model: ModelArgument,
) -> None:
    """
    Show detailed information about a specific model, including available model
    variants, modes, and supported options.

    Usage: captiv model show MODEL
    """
    manager = ModelManager()

    try:
        model_type = ModelType(model)
    except ValueError:
        valid_models = ", ".join([m.value for m in ModelType])
        typer.echo(f"Error: Invalid model '{model}'")
        typer.echo(f"Valid models: {valid_models}")
        raise typer.Exit(1) from None

    model_class = manager.get_model_class(model_type)

    typer.echo(f"\n=== {model_type.value.upper()} Model ===\n")

    typer.echo("Description:")
    typer.echo(
        f"  {model_class.__doc__.strip() if model_class.__doc__ else 'No description available.'}"  # noqa: E501
    )

    variants = manager.get_variant_details(model_type)
    typer.echo("\nAvailable Model Variants:")
    for variant_name, variant_info in variants.items():
        typer.echo(f"  {variant_name}:")
        if "description" in variant_info:
            typer.echo(f"    Description: {variant_info['description']}")
        if "checkpoint" in variant_info:
            typer.echo(f"    Checkpoint: {variant_info['checkpoint']}")

    modes = manager.get_mode_details(model_type)
    typer.echo("\nAvailable Modes:")
    if modes:
        for mode_name, mode_description in modes.items():
            typer.echo(f"  {mode_name}: {mode_description}")
    else:
        typer.echo("  No specific modes available for this model.")

    prompt_options = manager.get_prompt_option_details(model_type)
    typer.echo("\nAvailable Prompt Options:")
    if prompt_options:
        for option_name, option_description in prompt_options.items():
            typer.echo(f"  {option_name}: {option_description}")
    else:
        typer.echo("  No prompt options available for this model.")

    typer.echo("\nSupported Generation Parameters:")
    typer.echo("  max_new_tokens: Maximum number of tokens in the generated caption")
    typer.echo("  min_new_tokens: Minimum number of tokens in the generated caption")
    typer.echo("  num_beams: Number of beams for beam search")
    typer.echo("  temperature: Temperature for sampling")
    typer.echo("  top_k: Top-k sampling parameter")
    typer.echo("  top_p: Top-p sampling parameter")
    typer.echo("  repetition_penalty: Repetition penalty parameter")
    typer.echo("  prompt_options: Comma-separated list of prompt options")
    typer.echo(
        "  prompt_variables: Comma-separated key=value pairs for prompt variables"
    )

    if prompt_options:
        typer.echo(f"\nNote: This model supports {len(prompt_options)} prompt options.")
        typer.echo("Use --prompt-options to include them in your captions.")

    typer.echo("\nUsage Examples:")
    typer.echo(
        f"  captiv caption generate image.jpg --model {model_type.value} --variant {list(variants.keys())[0]}"  # noqa: E501
    )
    if modes:
        typer.echo(
            f"  captiv caption generate image.jpg --model {model_type.value} --mode {list(modes.keys())[0]}"  # noqa: E501
        )
    if prompt_options:
        first_option = list(prompt_options.keys())[0]
        typer.echo(
            f"  captiv caption generate image.jpg --model {model_type.value} --prompt-options {first_option}"  # noqa: E501
        )
    typer.echo(
        f"  captiv caption generate image.jpg --model {model_type.value} --prompt-variables character_name=Alice,setting=forest"  # noqa: E501
    )
