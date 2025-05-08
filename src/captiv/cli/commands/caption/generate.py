"""
Generate command for the Captiv CLI.

This module provides the command logic for generating image captions
using any of the supported models.
"""

import os
import time
from pathlib import Path
from typing import Optional

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import (  # JoyCaption-specific options; Extra options flags
    AvoidMetaPhrasesOption,
    CaptioningModeOption,
    CharacterNameOption,
    ConfigFileOption,
    ExcludeAmbiguityOption,
    ExcludeArtistInfoOption,
    ExcludeImmutableOption,
    ExcludeMoodOption,
    ExcludeResolutionOption,
    ExcludeTextOption,
    FocusImportantElementsOption,
    GuidanceScaleOption,
    ImagePathArgument,
    IncludeAgesOption,
    IncludeCameraAngleOption,
    IncludeCameraDetailsOption,
    IncludeCompositionOption,
    IncludeContentRatingOption,
    IncludeDepthOfFieldOption,
    IncludeJpegArtifactsOption,
    IncludeLightingOption,
    IncludeLightingSourceOption,
    IncludeOrientationOption,
    IncludeQualityOption,
    IncludeShotTypeOption,
    IncludeVantageHeightOption,
    IncludeWatermarkOption,
    KeepPgOption,
    MaxLengthOption,
    MentionWatermarkOption,
    MinLengthOption,
    ModelTypeOption,
    ModelVariantOption,
    NegativePromptOption,
    NumBeamsOption,
    PromptOption,
    QualityLevelOption,
    RepetitionPenaltyOption,
    TemperatureOption,
    TopKOption,
    TopPOption,
    TorchDtypeOption,
    UseBluntPhrasingOption,
    UseVulgarLanguageOption,
)
from captiv.services.caption_manager import CaptionManager
from captiv.services.model_manager import ModelManager
from captiv.utils.error_handling import EnhancedError
from captiv.utils.progress import ProgressTracker, cli_progress_callback


@handle_cli_errors
def command(
    image_path: Optional[Path] = typer.Argument(
        None,
        help="Image file or directory to generate captions for. Defaults to current working directory.",
    ),
    model: ModelTypeOption = None,
    variant: ModelVariantOption = None,
    mode: CaptioningModeOption = None,
    prompt: PromptOption = None,
    max_length: MaxLengthOption = None,
    min_length: MinLengthOption = None,
    num_beams: NumBeamsOption = None,
    temperature: TemperatureOption = None,
    top_k: TopKOption = None,
    top_p: TopPOption = None,
    repetition_penalty: RepetitionPenaltyOption = None,
    torch_dtype: TorchDtypeOption = None,
    # JoyCaption-specific options
    guidance_scale: GuidanceScaleOption = None,
    quality_level: QualityLevelOption = None,
    negative_prompt: NegativePromptOption = None,
    character_name: CharacterNameOption = None,
    # Extra options flags
    exclude_immutable: ExcludeImmutableOption = None,
    include_lighting: IncludeLightingOption = None,
    include_camera_angle: IncludeCameraAngleOption = None,
    include_watermark: IncludeWatermarkOption = None,
    include_jpeg_artifacts: IncludeJpegArtifactsOption = None,
    include_camera_details: IncludeCameraDetailsOption = None,
    keep_pg: KeepPgOption = None,
    exclude_resolution: ExcludeResolutionOption = None,
    include_quality: IncludeQualityOption = None,
    include_composition: IncludeCompositionOption = None,
    exclude_text: ExcludeTextOption = None,
    include_depth_of_field: IncludeDepthOfFieldOption = None,
    include_lighting_source: IncludeLightingSourceOption = None,
    exclude_ambiguity: ExcludeAmbiguityOption = None,
    include_content_rating: IncludeContentRatingOption = None,
    focus_important_elements: FocusImportantElementsOption = None,
    exclude_artist_info: ExcludeArtistInfoOption = None,
    include_orientation: IncludeOrientationOption = None,
    use_vulgar_language: UseVulgarLanguageOption = None,
    use_blunt_phrasing: UseBluntPhrasingOption = None,
    include_ages: IncludeAgesOption = None,
    include_shot_type: IncludeShotTypeOption = None,
    exclude_mood: ExcludeMoodOption = None,
    include_vantage_height: IncludeVantageHeightOption = None,
    mention_watermark: MentionWatermarkOption = None,
    avoid_meta_phrases: AvoidMetaPhrasesOption = None,
    config_file: ConfigFileOption = None,
) -> None:
    """Generate a caption for an image using the specified model."""
    # Default to current working directory if not specified
    if image_path is None:
        image_path = Path(os.getcwd())

    # Initialize service managers
    model_manager = ModelManager()
    caption_manager = CaptionManager()

    # Use the specified model or the default from config
    model_type = (
        model if model is not None else model_manager.get_default_model(config_file)
    )

    # Get default variant if not provided
    if not variant:
        model_class = model_manager.get_model_class(model_type)
        default_variant = model_class.get_default_variant()
        if default_variant:
            variant = default_variant
        else:
            # If no default variant is available, use the first variant
            variants = model_manager.get_variants_for_model(model_type)
            if variants:
                variant = variants[0]
            else:
                raise ValueError(f"No variants available for {model_type.value} model")

    # Validate variant
    model_manager.validate_variant(model_type, variant)

    # Validate mode if provided
    if mode:
        model_manager.validate_mode(model_type, mode)

    # Check if the path is a directory
    if image_path.is_dir():
        # Process all images in the directory
        file_manager = caption_manager.file_manager

        # Show progress while listing images
        typer.echo(f"Scanning directory {image_path}...")
        images_with_captions = file_manager.list_images_with_captions(str(image_path))

        if not images_with_captions:
            typer.echo(f"No images found in {image_path}")
            return

        # Filter out images that already have captions if needed
        images_to_process = [(name, caption) for name, caption in images_with_captions]
        total_images = len(images_to_process)

        typer.echo(f"Found {total_images} images in {image_path}")
        typer.echo(f"Generating captions using {model_type.value} model ({variant})...")

        # Create a progress tracker for the bulk captioning process
        progress = ProgressTracker(
            total=total_images,
            description="Generating captions",
            callback=cli_progress_callback,
        )

        # Process images with progress tracking
        success_count = 0
        error_count = 0
        start_time = time.time()

        for i, (img_name, existing_caption) in enumerate(images_to_process):
            full_path = image_path / img_name
            try:
                # Update progress with current image
                progress.update(0, f"Processing {img_name} ({i + 1}/{total_images})")

                # Generate caption for each image
                caption = caption_manager.generate_caption(
                    model_type=model_type,
                    image_path=str(full_path),
                    variant=variant,
                    mode=mode,
                    prompt=prompt,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    torch_dtype=torch_dtype,
                    # Model-specific parameters
                    guidance_scale=guidance_scale,
                    quality_level=quality_level,
                    negative_prompt=negative_prompt,
                    character_name=character_name,
                    # Extra options flags
                    exclude_immutable=exclude_immutable,
                    include_lighting=include_lighting,
                    include_camera_angle=include_camera_angle,
                    include_watermark=include_watermark,
                    include_jpeg_artifacts=include_jpeg_artifacts,
                    include_camera_details=include_camera_details,
                    keep_pg=keep_pg,
                    exclude_resolution=exclude_resolution,
                    include_quality=include_quality,
                    include_composition=include_composition,
                    exclude_text=exclude_text,
                    include_depth_of_field=include_depth_of_field,
                    include_lighting_source=include_lighting_source,
                    exclude_ambiguity=exclude_ambiguity,
                    include_content_rating=include_content_rating,
                    focus_important_elements=focus_important_elements,
                    exclude_artist_info=exclude_artist_info,
                    include_orientation=include_orientation,
                    use_vulgar_language=use_vulgar_language,
                    use_blunt_phrasing=use_blunt_phrasing,
                    include_ages=include_ages,
                    include_shot_type=include_shot_type,
                    exclude_mood=exclude_mood,
                    include_vantage_height=include_vantage_height,
                    mention_watermark=mention_watermark,
                    avoid_meta_phrases=avoid_meta_phrases,
                )

                # Update progress and show caption
                progress.update(1, f"Caption generated for {img_name}")
                typer.echo(f"\n{img_name}: {caption}")
                success_count += 1

            except EnhancedError as e:
                # For enhanced errors, show the detailed message
                typer.echo(f"\nError processing {img_name}: {e.message}")
                if e.troubleshooting_tips:
                    typer.echo("Troubleshooting tips:")
                    for i, tip in enumerate(e.troubleshooting_tips, 1):
                        typer.echo(f"  {i}. {tip}")
                error_count += 1
                progress.update(1, f"Error processing {img_name}")

            except Exception as e:
                # For other exceptions, show a simple error message
                typer.echo(f"\nError processing {img_name}: {str(e)}")
                error_count += 1
                progress.update(1, f"Error processing {img_name}")

        # Show summary
        elapsed_time = time.time() - start_time
        typer.echo(f"\nCaptioning complete in {elapsed_time:.2f} seconds")
        typer.echo(f"Successfully captioned: {success_count}/{total_images} images")
        if error_count > 0:
            typer.echo(f"Failed to caption: {error_count}/{total_images} images")
    else:
        # Process a single image file
        try:
            # Show progress message
            typer.echo(
                f"Generating caption for {image_path.name} using {model_type.value} model ({variant})..."
            )

            # Create a simple progress tracker for model loading and caption generation
            progress = ProgressTracker(
                total=2,
                description="Generating caption",
                callback=cli_progress_callback,
            )

            # Update progress for model loading
            progress.update(1, "Loading model and generating caption")

            # Generate caption using the service
            start_time = time.time()
            caption = caption_manager.generate_caption(
                model_type=model_type,
                image_path=str(image_path),
                variant=variant,
                mode=mode,
                prompt=prompt,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                torch_dtype=torch_dtype,
                # Model-specific parameters
                guidance_scale=guidance_scale,
                quality_level=quality_level,
                negative_prompt=negative_prompt,
                character_name=character_name,
                # Extra options flags
                exclude_immutable=exclude_immutable,
                include_lighting=include_lighting,
                include_camera_angle=include_camera_angle,
                include_watermark=include_watermark,
                include_jpeg_artifacts=include_jpeg_artifacts,
                include_camera_details=include_camera_details,
                keep_pg=keep_pg,
                exclude_resolution=exclude_resolution,
                include_quality=include_quality,
                include_composition=include_composition,
                exclude_text=exclude_text,
                include_depth_of_field=include_depth_of_field,
                include_lighting_source=include_lighting_source,
                exclude_ambiguity=exclude_ambiguity,
                include_content_rating=include_content_rating,
                focus_important_elements=focus_important_elements,
                exclude_artist_info=exclude_artist_info,
                include_orientation=include_orientation,
                use_vulgar_language=use_vulgar_language,
                use_blunt_phrasing=use_blunt_phrasing,
                include_ages=include_ages,
                include_shot_type=include_shot_type,
                exclude_mood=exclude_mood,
                include_vantage_height=include_vantage_height,
                mention_watermark=mention_watermark,
                avoid_meta_phrases=avoid_meta_phrases,
            )

            # Update progress for completion
            elapsed_time = time.time() - start_time
            progress.complete(f"Caption generated in {elapsed_time:.2f} seconds")

            # Print the caption
            typer.echo(f"\nCaption: {caption}")

        except EnhancedError as e:
            # For enhanced errors, show the detailed message
            typer.echo(f"\nError: {e.message}")
            if e.troubleshooting_tips:
                typer.echo("Troubleshooting tips:")
                for i, tip in enumerate(e.troubleshooting_tips, 1):
                    typer.echo(f"  {i}. {tip}")
            raise typer.Exit(1)

        except Exception:
            # Let the exception propagate to the handle_cli_errors decorator
            raise
