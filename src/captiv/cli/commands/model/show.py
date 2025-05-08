"""
Show model details command for the Captiv CLI.

This module provides the command logic for displaying detailed information about a specific model.
"""

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.services.model_manager import ModelManager, ModelType


@handle_cli_errors
def command(
    model: str = typer.Argument(..., help="Model type to show details for"),
) -> None:
    """
    Show detailed information about a specific model, including available variants, modes, and supported options.

    Usage: captiv model show MODEL_TYPE
    """
    manager = ModelManager()

    # Validate model type
    try:
        model_type = ModelType(model)
    except ValueError:
        valid_models = ", ".join([m.value for m in ModelType])
        typer.echo(f"Error: Invalid model type '{model}'")
        typer.echo(f"Valid model types: {valid_models}")
        raise typer.Exit(1)

    # Get model class
    model_class = manager.get_model_class(model_type)

    # Display model information
    typer.echo(f"\n=== {model_type.value.upper()} Model ===\n")

    # Display model description
    typer.echo("Description:")
    typer.echo(
        f"  {model_class.__doc__.strip() if model_class.__doc__ else 'No description available.'}"
    )

    # Display available variants
    variants = manager.get_variant_details(model_type)
    typer.echo("\nAvailable Variants:")
    for variant_name, variant_info in variants.items():
        typer.echo(f"  {variant_name}:")
        if "description" in variant_info:
            typer.echo(f"    Description: {variant_info['description']}")
        if "checkpoint" in variant_info:
            typer.echo(f"    Checkpoint: {variant_info['checkpoint']}")

    # Display available modes
    modes = manager.get_mode_details(model_type)
    typer.echo("\nAvailable Modes:")
    if modes:
        for mode_name, mode_description in modes.items():
            typer.echo(f"  {mode_name}: {mode_description}")
    else:
        typer.echo("  No specific modes available for this model.")

    # Display available prompt options
    prompt_options = manager.get_prompt_option_details(model_type)
    typer.echo("\nAvailable Prompt Options:")
    if prompt_options:
        for option_name, option_description in prompt_options.items():
            typer.echo(f"  {option_name}: {option_description}")
    else:
        typer.echo("  No prompt options available for this model.")

    # Display generation parameters
    typer.echo("\nSupported Generation Parameters:")
    typer.echo("  max_length: Maximum length of the generated caption")
    typer.echo("  min_length: Minimum length of the generated caption")
    typer.echo("  num_beams: Number of beams for beam search")
    typer.echo("  temperature: Temperature for sampling")
    typer.echo("  top_k: Top-k sampling parameter")
    typer.echo("  top_p: Top-p sampling parameter")
    typer.echo("  repetition_penalty: Repetition penalty parameter")

    # Display model-specific parameters if any
    if model_type == ModelType.JOYCAPTION:
        typer.echo("\nJoyCaption-specific Parameters:")
        typer.echo("  guidance_scale: Controls how closely to follow the prompt")
        typer.echo("  quality_level: Quality level (draft, standard, high)")
        typer.echo("  negative_prompt: Text to avoid in the generation")
        typer.echo("  character_name: Name of the character to focus on")

    # Display usage example
    typer.echo("\nUsage Example:")
    typer.echo(
        f"  captiv caption generate image.jpg --model {model_type.value} --variant {list(variants.keys())[0]}"
    )
    if modes:
        typer.echo(
            f"  captiv caption generate image.jpg --model {model_type.value} --mode {list(modes.keys())[0]}"
        )
