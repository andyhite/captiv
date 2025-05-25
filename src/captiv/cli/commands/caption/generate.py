"""
Generate command for the Captiv CLI.

This module provides the command logic for generating image captions using any of the
supported models and their variants.
"""

import os
import time
from pathlib import Path
from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services import (
    CaptionFileManager,
    ConfigManager,
    FileManager,
    ImageFileManager,
    ModelManager,
    ModelType,
)
from captiv.utils.error_handling import EnhancedError

ModelTypeOption = Annotated[
    ModelType | None,
    typer.Option(
        "--model",
        "-m",
        help="Model to use for captioning (overrides configured default)",
        case_sensitive=False,
    ),
]

ModelVariantOption = Annotated[
    str | None,
    typer.Option(
        "--variant",
        "-v",
        help="Model variant to use",
    ),
]

CaptioningModeOption = Annotated[
    str | None,
    typer.Option(
        "--mode",
        help="Captioning mode to use",
    ),
]

PromptOption = Annotated[
    str | None,
    typer.Option(
        "--prompt",
        "-p",
        help="Custom prompt to use (overrides --mode)",
    ),
]

PromptOptionsOption = Annotated[
    str | None,
    typer.Option(
        "--prompt-options",
        help="Comma-separated list of prompt options to include",
    ),
]

PromptVariablesOption = Annotated[
    str | None,
    typer.Option(
        "--prompt-variables",
        help=(
            "Comma-separated key=value pairs for prompt variables "
            "(e.g., 'character_name=Alice,setting=forest')"
        ),
    ),
]

MaxNewTokensOption = Annotated[
    int | None,
    typer.Option(
        "--max-new-tokens",
        help="Maximum number of tokens in the generated caption",
    ),
]

MinNewTokensOption = Annotated[
    int | None,
    typer.Option(
        "--min-new-tokens",
        help="Minimum number of tokens in the generated caption (default: 10)",
    ),
]

NumBeamsOption = Annotated[
    int | None,
    typer.Option(
        "--num-beams",
        help="Number of beams for beam search (default: 3)",
    ),
]

TemperatureOption = Annotated[
    float | None,
    typer.Option(
        "--temperature",
        help="Temperature for sampling (default: 1.0)",
    ),
]

TopKOption = Annotated[
    int | None,
    typer.Option(
        "--top-k",
        help="Top-k sampling parameter (default: 50)",
    ),
]

TopPOption = Annotated[
    float | None,
    typer.Option(
        "--top-p",
        help="Top-p sampling parameter (default: 0.9)",
    ),
]

RepetitionPenaltyOption = Annotated[
    float | None,
    typer.Option(
        "--repetition-penalty",
        help="Repetition penalty parameter (default: 1.0)",
    ),
]

TorchDtypeOption = Annotated[
    str | None,
    typer.Option(
        "--torch-dtype",
        help="PyTorch data type to use for model loading",
        case_sensitive=False,
    ),
]

RunPodOption = Annotated[
    bool,
    typer.Option(
        "--runpod/--no-runpod",
        help="Use RunPod for remote GPU inference",
    ),
]

ImagePathArgument = Annotated[
    Path | None,
    typer.Argument(
        help=(
            "Image file or directory to generate captions for. "
            "Defaults to current working directory."
        ),
    ),
]

SaveOption = Annotated[
    bool,
    typer.Option(
        "--save/--no-save",
        help="Save generated captions to .txt files alongside images",
    ),
]


@handle_cli_errors
def command(
    image_path: ImagePathArgument = None,
    model: ModelTypeOption = None,
    variant: ModelVariantOption = None,
    mode: CaptioningModeOption = None,
    prompt: PromptOption = None,
    prompt_options: PromptOptionsOption = None,
    prompt_variables: PromptVariablesOption = None,
    max_new_tokens: MaxNewTokensOption = None,
    min_new_tokens: MinNewTokensOption = None,
    num_beams: NumBeamsOption = None,
    temperature: TemperatureOption = None,
    top_k: TopKOption = None,
    top_p: TopPOption = None,
    repetition_penalty: RepetitionPenaltyOption = None,
    torch_dtype: TorchDtypeOption = None,
    runpod: RunPodOption = False,
    config_file: ConfigFileOption = None,
    save: SaveOption = True,
) -> None:
    """Generate a caption for an image using the specified model."""
    if image_path is None:
        image_path = Path(os.getcwd())

    config_manager = ConfigManager(config_file)
    model_manager = ModelManager(config_manager)
    file_manager = FileManager()
    image_file_manager = ImageFileManager(file_manager)
    caption_manager = CaptionFileManager(file_manager, image_file_manager)

    selected_model = model if model is not None else model_manager.get_default_model()

    if not variant:
        model_class = model_manager.get_model_class(selected_model)
        default_variant = model_class.DEFAULT_VARIANT
        if default_variant:
            variant = default_variant
        else:
            variants = model_manager.get_variants_for_model(selected_model)
            if variants:
                variant = variants[0]
            else:
                raise ValueError(
                    f"No model variants available for {selected_model.value} model"
                )

    model_manager.validate_variant(selected_model, variant)

    if mode:
        model_manager.validate_mode(selected_model, mode)

    parsed_prompt_options = model_manager.parse_prompt_options(prompt_options)
    parsed_prompt_variables = model_manager.parse_prompt_variables(prompt_variables)

    generation_params = model_manager.build_generation_params(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        prompt_options=parsed_prompt_options,
        prompt_variables=parsed_prompt_variables,
    )

    try:
        if runpod:
            typer.echo(
                f"Loading {selected_model.value}/{variant} with RunPod support..."
            )
        else:
            typer.echo(f"Loading {selected_model.value}/{variant}...")
        model_instance = model_manager.create_model_instance(
            model_type=selected_model,
            variant=variant,
            torch_dtype=torch_dtype,
            use_runpod=runpod,
        )
    except EnhancedError as e:
        typer.echo(f"Error loading model: {e.message}")
        if e.troubleshooting_tips:
            typer.echo("Troubleshooting tips:")
            for i, tip in enumerate(e.troubleshooting_tips, 1):
                typer.echo(f"  {i}. {tip}")
        raise typer.Exit(1) from None

    if image_path.is_dir():
        typer.echo(f"Scanning directory {image_path}...")
        images = image_file_manager.list_image_files(image_path)

        if not images:
            typer.echo(f"No images found in {image_path}")
            return

        total_images = len(images)
        typer.echo(f"Found {total_images} images in {image_path}")
        typer.echo(f"Generating captions using {selected_model.value}/{variant}...")

        success_count = 0
        error_count = 0
        start_time = time.time()

        for i, img_path in enumerate(images):
            try:
                typer.echo(f"Processing {img_path.name} ({i + 1}/{total_images})")

                caption = model_manager.generate_caption(
                    model_instance=model_instance,
                    image_path=img_path,
                    mode=mode,
                    prompt=prompt,
                    generation_params=generation_params,
                )

                typer.echo(f"\n{img_path.name}: {caption}")

                if save:
                    caption_manager.write_caption(img_path, caption)
                    caption_file_path = caption_manager.get_caption_file_path(img_path)
                    typer.echo(f"Caption saved to {caption_file_path}")

                success_count += 1

            except EnhancedError as e:
                typer.echo(f"\nError processing {img_path.name}: {e.message}")
                if e.troubleshooting_tips:
                    typer.echo("Troubleshooting tips:")
                    for j, tip in enumerate(e.troubleshooting_tips, 1):
                        typer.echo(f"  {j}. {tip}")
                error_count += 1

            except Exception as e:
                typer.echo(f"\nError processing {img_path.name}: {str(e)}")
                error_count += 1

        elapsed_time = time.time() - start_time
        typer.echo(f"\nCaptioning complete in {elapsed_time:.2f} seconds")
        typer.echo(f"Successfully captioned: {success_count}/{total_images} images")
        if error_count > 0:
            typer.echo(f"Failed to caption: {error_count}/{total_images} images")
    else:
        try:
            model_info = f"{selected_model.value}/{variant}"
            typer.echo(
                f"Generating caption for {image_path.name} using {model_info}..."
            )

            start_time = time.time()
            caption = model_manager.generate_caption(
                model_instance=model_instance,
                image_path=image_path,
                mode=mode,
                prompt=prompt,
                generation_params=generation_params,
            )

            elapsed_time = time.time() - start_time
            typer.echo(f"Caption generated in {elapsed_time:.2f} seconds")

            typer.echo(f"\nCaption: {caption}")

            if save:
                caption_manager.write_caption(image_path, caption)
                caption_file_path = caption_manager.get_caption_file_path(image_path)
                typer.echo(f"Caption saved to {caption_file_path}")

        except EnhancedError as e:
            typer.echo(f"\nError: {e.message}")
            if e.troubleshooting_tips:
                typer.echo("Troubleshooting tips:")
                for i, tip in enumerate(e.troubleshooting_tips, 1):
                    typer.echo(f"  {i}. {tip}")
            raise typer.Exit(1) from None

        except Exception:
            raise
