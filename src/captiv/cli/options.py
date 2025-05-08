"""
Common CLI options for Captiv.

This module provides shared option definitions used by the CLI commands.
Options are grouped by category for better organization.
"""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from captiv.services.model_manager import ModelType

# Common option for config file path
ConfigFileOption = Annotated[
    Optional[str],
    typer.Option(
        "--config-file",
        "-c",
        help="Path to the configuration file",
        envvar="CAPTIV_CONFIG_FILE",
    ),
]

# Model and captioning options
ModelTypeOption = Annotated[
    Optional[ModelType],
    typer.Option(
        "--model",
        "-m",
        help="Model type to use for captioning (overrides configured default)",
        case_sensitive=False,
    ),
]

# Model variant option
ModelVariantOption = Annotated[
    Optional[str],
    typer.Option(
        "--variant",
        "-v",
        help="Model variant to use",
    ),
]

# Captioning mode option
CaptioningModeOption = Annotated[
    Optional[str],
    typer.Option(
        "--mode",
        help="Captioning mode to use",
    ),
]

# Custom prompt option
PromptOption = Annotated[
    Optional[str],
    typer.Option(
        "--prompt",
        "-p",
        help="Custom prompt to use (overrides --mode)",
    ),
]

# Generation parameters
MaxLengthOption = Annotated[
    Optional[int],
    typer.Option(
        "--max-length",
        help="Maximum length of the generated caption",
    ),
]

MinLengthOption = Annotated[
    Optional[int],
    typer.Option(
        "--min-length",
        help="Minimum length of the generated caption (default: 10)",
    ),
]

NumBeamsOption = Annotated[
    Optional[int],
    typer.Option(
        "--num-beams",
        help="Number of beams for beam search (default: 3)",
    ),
]

TemperatureOption = Annotated[
    Optional[float],
    typer.Option(
        "--temperature",
        help="Temperature for sampling (default: 1.0)",
    ),
]

TopKOption = Annotated[
    Optional[int],
    typer.Option(
        "--top-k",
        help="Top-k sampling parameter (default: 50)",
    ),
]

TopPOption = Annotated[
    Optional[float],
    typer.Option(
        "--top-p",
        help="Top-p sampling parameter (default: 0.9)",
    ),
]

RepetitionPenaltyOption = Annotated[
    Optional[float],
    typer.Option(
        "--repetition-penalty",
        help="Repetition penalty parameter (default: 1.0)",
    ),
]

TorchDtypeOption = Annotated[
    Optional[str],
    typer.Option(
        "--torch-dtype",
        help="PyTorch data type to use for model loading",
        case_sensitive=False,
    ),
]

# JoyCaption-specific options
GuidanceScaleOption = Annotated[
    Optional[float],
    typer.Option(
        "--guidance-scale",
        help="JoyCaption guidance scale parameter (default: 7.5)",
    ),
]

QualityLevelOption = Annotated[
    Optional[str],
    typer.Option(
        "--quality-level",
        help="JoyCaption quality level (draft, standard, high)",
    ),
]

NegativePromptOption = Annotated[
    Optional[str],
    typer.Option(
        "--negative-prompt",
        help="JoyCaption negative prompt to avoid certain content",
    ),
]

# Removed word count and length options

# Character name option
CharacterNameOption = Annotated[
    Optional[str],
    typer.Option(
        "--character-name",
        help="Name to use for characters in the image",
    ),
]

# Extra options flags
ExcludeImmutableOption = Annotated[
    Optional[bool],
    typer.Option(
        "--exclude-immutable",
        help="Exclude immutable characteristics of people/characters",
    ),
]

IncludeLightingOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-lighting",
        help="Include information about lighting",
    ),
]

IncludeCameraAngleOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-camera-angle",
        help="Include information about camera angle",
    ),
]

IncludeWatermarkOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-watermark",
        help="Include information about watermarks",
    ),
]

IncludeJpegArtifactsOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-jpeg-artifacts",
        help="Include information about JPEG artifacts",
    ),
]

IncludeCameraDetailsOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-camera-details",
        help="Include camera details for photos",
    ),
]

KeepPgOption = Annotated[
    Optional[bool],
    typer.Option(
        "--keep-pg",
        help="Keep content PG-rated",
    ),
]

ExcludeResolutionOption = Annotated[
    Optional[bool],
    typer.Option(
        "--exclude-resolution",
        help="Exclude image resolution information",
    ),
]

IncludeQualityOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-quality",
        help="Include subjective quality assessment",
    ),
]

IncludeCompositionOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-composition",
        help="Include composition style information",
    ),
]

ExcludeTextOption = Annotated[
    Optional[bool],
    typer.Option(
        "--exclude-text",
        help="Exclude text in the image from caption",
    ),
]

IncludeDepthOfFieldOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-depth-of-field",
        help="Include depth of field information",
    ),
]

IncludeLightingSourceOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-lighting-source",
        help="Include lighting source information",
    ),
]

ExcludeAmbiguityOption = Annotated[
    Optional[bool],
    typer.Option(
        "--exclude-ambiguity",
        help="Exclude ambiguous language",
    ),
]

IncludeContentRatingOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-content-rating",
        help="Include content rating (sfw, suggestive, nsfw)",
    ),
]

FocusImportantElementsOption = Annotated[
    Optional[bool],
    typer.Option(
        "--focus-important-elements",
        help="Focus only on important elements",
    ),
]

ExcludeArtistInfoOption = Annotated[
    Optional[bool],
    typer.Option(
        "--exclude-artist-info",
        help="Exclude artist name and artwork title",
    ),
]

IncludeOrientationOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-orientation",
        help="Include image orientation and aspect ratio",
    ),
]

UseVulgarLanguageOption = Annotated[
    Optional[bool],
    typer.Option(
        "--use-vulgar-language",
        help="Use vulgar language and profanity",
    ),
]

UseBluntPhrasingOption = Annotated[
    Optional[bool],
    typer.Option(
        "--use-blunt-phrasing",
        help="Use blunt, casual phrasing instead of euphemisms",
    ),
]

IncludeAgesOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-ages",
        help="Include ages of people/characters when applicable",
    ),
]

IncludeShotTypeOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-shot-type",
        help="Include shot type information",
    ),
]

ExcludeMoodOption = Annotated[
    Optional[bool],
    typer.Option(
        "--exclude-mood",
        help="Exclude mood/feeling information",
    ),
]

IncludeVantageHeightOption = Annotated[
    Optional[bool],
    typer.Option(
        "--include-vantage-height",
        help="Include vantage height information",
    ),
]

MentionWatermarkOption = Annotated[
    Optional[bool],
    typer.Option(
        "--mention-watermark",
        help="Mention watermark if present",
    ),
]

AvoidMetaPhrasesOption = Annotated[
    Optional[bool],
    typer.Option(
        "--avoid-meta-phrases",
        help="Avoid meta phrases like 'This image shows...'",
    ),
]

# Common path arguments
ImagePathArgument = Annotated[
    Path,
    typer.Argument(..., help="Path to the image file", exists=True, dir_okay=False),
]

DirectoryPathArgument = Annotated[
    Path,
    typer.Argument(
        ...,
        help="Path to the directory containing images",
        exists=True,
        file_okay=False,
    ),
]
