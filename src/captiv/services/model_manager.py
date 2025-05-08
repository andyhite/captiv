from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

import torch
from loguru import logger

from captiv import config
from captiv.models import (
    Blip2Model,
    BlipModel,
    GitModel,
    ImageCaptioningModel,
    JoyCaptionModel,
    KosmosModel,
    SigLIPModel,
    VitGpt2Model,
)
from captiv.services.exceptions import (
    InvalidModelModeError,
    InvalidModelTypeError,
    InvalidModelVariantError,
    ModelConfigurationError,
)
from captiv.utils.error_handling import EnhancedError, ErrorCategory, handle_errors
from captiv.utils.progress import ProgressTracker


# Define model type enum
class ModelType(str, Enum):
    BLIP = "blip"
    BLIP2 = "blip2"
    JOYCAPTION = "joycaption"
    GIT = "git"
    VIT_GPT2 = "vit-gpt2"
    KOSMOS = "kosmos"
    SIGLIP = "siglip"


class ModelManager:
    # Map of model types to model classes
    MODEL_CLASS_MAP = {
        ModelType.BLIP: BlipModel,
        ModelType.BLIP2: Blip2Model,
        ModelType.JOYCAPTION: JoyCaptionModel,
        ModelType.GIT: GitModel,
        ModelType.VIT_GPT2: VitGpt2Model,
        ModelType.KOSMOS: KosmosModel,
        ModelType.SIGLIP: SigLIPModel,
    }

    # Cache for model instances: (model_type, variant, dtype) -> instance
    _instance_cache: Dict[tuple, ImageCaptioningModel] = {}

    def get_model_class(self, model_type: ModelType) -> Type[ImageCaptioningModel]:
        """
        Return the model class for a given model type.

        Args:
            model_type: The type of model to get the class for.

        Returns:
            The model class.

        Raises:
            InvalidModelTypeError: If the model type is unknown.
        """
        if model_type in self.MODEL_CLASS_MAP:
            return self.MODEL_CLASS_MAP[model_type]
        else:
            raise InvalidModelTypeError(f"Unknown model type: {model_type}")

    def get_variants_for_model(self, model_type: ModelType) -> List[str]:
        """Return available variants for a given model type."""
        model_class = self.get_model_class(model_type)
        return list(model_class.get_variants().keys())

    def get_modes_for_model(self, model_type: ModelType) -> List[str]:
        """Return available modes for a given model type."""
        model_class = self.get_model_class(model_type)
        return list(model_class.get_modes().keys())

    def get_prompt_options_for_model(self, model_type: ModelType) -> List[str]:
        """Return available prompt options for a given model type."""
        model_class = self.get_model_class(model_type)
        return list(model_class.get_prompt_options().keys())

    def get_prompt_option_details(self, model_type: ModelType) -> Dict[str, str]:
        """Get detailed information about prompt options for a model type."""
        model_class = self.get_model_class(model_type)
        return model_class.get_prompt_options()

    def validate_variant(self, model_type: ModelType, variant: str) -> None:
        """
        Validate that the variant is valid for the model type.

        Args:
            model_type: The type of model.
            variant: The variant to validate.

        Raises:
            InvalidModelVariantError: If the variant is invalid for the model type.
        """
        variants = self.get_variants_for_model(model_type)
        if variant not in variants:
            raise InvalidModelVariantError(
                f"Invalid variant '{variant}' for {model_type.value} model. Available variants: {', '.join(variants)}"
            )

    def validate_mode(self, model_type: ModelType, mode: str) -> None:
        """
        Validate that the mode is valid for the model type.

        Args:
            model_type: The type of model.
            mode: The mode to validate.

        Raises:
            InvalidModelModeError: If the mode is invalid for the model type.
        """
        modes = self.get_modes_for_model(model_type)
        if mode not in modes:
            raise InvalidModelModeError(
                f"Invalid mode '{mode}' for {model_type.value} model. Available modes: {', '.join(modes)}"
            )

    def get_default_model(self, config_file: Optional[str] = None) -> ModelType:
        """
        Get the default model from the configuration.

        Args:
            config_file: Optional path to a config file.

        Returns:
            The default model type.
        """
        model_str = config.get_config_value("model", "default_model", config_file)
        try:
            return ModelType(model_str)
        except (ValueError, TypeError):
            # If the configured model is invalid or None, use BLIP as fallback
            return ModelType.BLIP

    def get_variant_details(self, model_type: ModelType) -> Dict[str, Dict[str, str]]:
        """Get detailed information about variants for a model type."""
        model_class = self.get_model_class(model_type)
        return model_class.get_variants()

    def get_mode_details(self, model_type: ModelType) -> Dict[str, str]:
        """Get detailed information about modes for a model type."""
        model_class = self.get_model_class(model_type)
        return model_class.get_modes()

    def parse_torch_dtype(self, torch_dtype: Optional[str]) -> Optional[torch.dtype]:
        """
        Parse a string representation of a torch dtype.

        Args:
            torch_dtype: String representation of a torch dtype.

        Returns:
            The corresponding torch.dtype or None if not specified.

        Raises:
            ModelConfigurationError: If the torch dtype is not supported.
        """
        if not torch_dtype:
            return None

        if torch_dtype == "float16":
            return torch.float16
        elif torch_dtype == "float32":
            return torch.float32
        elif torch_dtype == "bfloat16":
            return torch.bfloat16
        else:
            raise ModelConfigurationError(f"Unsupported torch_dtype '{torch_dtype}'")

    @handle_errors
    def create_model_instance(
        self,
        model_type: ModelType,
        variant: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], Any]] = None,
    ) -> ImageCaptioningModel:
        """
        Create an instance of a model with the specified configuration.
        Uses a cache to avoid reloading checkpoints if the same model/variant/dtype is requested.

        Args:
            model_type: The type of model to create.
            variant: The variant of the model to use. If None, uses the first available variant.
            torch_dtype: The torch dtype to use for the model.
            progress_callback: Optional callback function for progress updates.

        Returns:
            An instance of the model.

        Raises:
            InvalidModelTypeError: If the model type is unknown.
            InvalidModelVariantError: If the variant is invalid.
            ModelConfigurationError: If there's an error configuring the model.
            EnhancedError: Enhanced error with troubleshooting information.
        """
        model_class = self.get_model_class(model_type)

        # Parse torch dtype
        dtype = self.parse_torch_dtype(torch_dtype)

        # Use the specified variant or the default variant for the model
        if variant:
            self.validate_variant(model_type, variant)
            model_variant = variant
        else:
            # Use the already obtained model_class
            default_variant = model_class.get_default_variant()
            if default_variant is None:
                raise ModelConfigurationError(
                    f"No variants available for {model_type.value} model"
                )
            model_variant = default_variant

        # Use a cache key based on model_type, variant, dtype
        cache_key = (model_type, model_variant, str(dtype))
        if cache_key in self._instance_cache:
            logger.info(
                f"Using cached model instance for {model_type.value}/{model_variant}"
            )
            return self._instance_cache[cache_key]

        # Create a progress tracker for model loading
        # We use a fixed number of steps (5) as we can't know the exact progress
        # but we want to show some indication of activity
        logger.info(f"Loading model {model_type.value}/{model_variant}...")
        progress = ProgressTracker(
            total=5,
            description=f"Loading {model_type.value}/{model_variant} model",
            callback=progress_callback,
        )

        try:
            # Update progress to show we're starting
            progress.update(1, "Initializing model")

            # Update progress before initialization
            progress.update(1, "Loading model weights")

            # Create the model instance
            instance = model_class(model_variant, torch_dtype=dtype)

            # Update progress after initialization
            progress.update(2, "Model loaded successfully")
            progress.complete("Model ready for use")
            self._instance_cache[cache_key] = instance
            return instance
        except ImportError as e:
            error_msg = f"Failed to create {model_type.value} model"
            tips = ["Check if all required dependencies are installed"]

            if "accelerate" in str(e) and model_type == ModelType.JOYCAPTION:
                error_msg = f"Failed to create JoyCaption model: {str(e)}"
                tips = [
                    "The 'accelerate' package is required for JoyCaption",
                    "Install it with 'pip install accelerate'",
                    "Or use 'poetry install -E joycaption' to use JoyCaption models",
                ]

            # Create an enhanced error with troubleshooting tips
            raise EnhancedError(
                message=error_msg,
                category=ErrorCategory.MODEL_LOADING,
                original_error=e,
                troubleshooting_tips=tips,
                context={
                    "model_type": model_type.value,
                    "variant": model_variant,
                    "torch_dtype": str(dtype),
                },
            )
        except Exception as e:
            # Create an enhanced error for other exceptions
            raise EnhancedError(
                message=f"Failed to create {model_type.value} model instance",
                category=ErrorCategory.MODEL_LOADING,
                original_error=e,
                context={
                    "model_type": model_type.value,
                    "variant": model_variant,
                    "torch_dtype": str(dtype),
                },
            )

    def build_generation_params(
        self,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build a dictionary of generation parameters for the model.

        This method only includes parameters that are explicitly provided.
        Default values are handled by the model classes, which get them from the config.

        Args:
            max_length: Maximum length of the generated caption.
            min_length: Minimum length of the generated caption.
            num_beams: Number of beams for beam search.
            temperature: Temperature for sampling.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty parameter.

        Returns:
            A dictionary of generation parameters.
        """
        # Only include explicitly provided parameters
        # Model classes will get defaults from config for missing parameters
        gen_params = {}
        if max_length is not None:
            gen_params["max_length"] = max_length
        if min_length is not None:
            gen_params["min_length"] = min_length
        if num_beams is not None:
            gen_params["num_beams"] = num_beams
        if temperature is not None:
            gen_params["temperature"] = temperature
        if top_k is not None:
            gen_params["top_k"] = top_k
        if top_p is not None:
            gen_params["top_p"] = top_p
        if repetition_penalty is not None:
            gen_params["repetition_penalty"] = repetition_penalty

        return gen_params
