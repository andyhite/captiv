from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from loguru import logger

from captiv.models import (
    BaseModel,
    Blip2Model,
    BlipModel,
    JoyCaptionModel,
    KosmosModel,
    ModelVariant,
    VitGPT2Model,
)
from captiv.services.config_manager import ConfigManager
from captiv.services.exceptions import (
    InvalidModelModeError,
    InvalidModelTypeError,
    InvalidModelVariantError,
    ModelConfigurationError,
)
from captiv.utils.error_handling import EnhancedError, ErrorCategory, handle_errors


class ModelType(str, Enum):
    BLIP = "blip"
    BLIP2 = "blip2"
    JOYCAPTION = "joycaption"
    KOSMOS = "kosmos"
    VIT_GPT2 = "vit-gpt2"


class ModelManager:
    MODEL_CLASS_MAP = {
        ModelType.BLIP2: Blip2Model,
        ModelType.BLIP: BlipModel,
        ModelType.JOYCAPTION: JoyCaptionModel,
        ModelType.KOSMOS: KosmosModel,
        ModelType.VIT_GPT2: VitGPT2Model,
    }

    _instance_cache: dict[tuple, BaseModel] = {}

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        Initialize the ModelManager.

        Args:
            config_manager: Optional ConfigManager instance for configuration handling.
        """
        self.config_manager = config_manager or ConfigManager()
        self.runpod_service = None

    def get_model_class(self, model_type: ModelType) -> type[BaseModel]:
        """
        Return the model variant class for a given model.

        Args:
            model_type: The model to get the class for.

        Returns:
            The model variant class.

        Raises:
            InvalidModelTypeError: If the model is unknown.
        """
        if model_type in self.MODEL_CLASS_MAP:
            return self.MODEL_CLASS_MAP[model_type]
        else:
            raise InvalidModelTypeError(f"Unknown model: {model_type}")

    def get_variants_for_model(self, model_type: ModelType) -> list[str]:
        """Return available model variants for a given model."""
        model_class = self.get_model_class(model_type)
        return list(model_class.get_variants().keys())

    def get_modes_for_model(self, model_type: ModelType) -> list[str]:
        """Return available modes for a given model."""
        model_class = self.get_model_class(model_type)
        return list(model_class.get_modes().keys())

    def get_prompt_options_for_model(self, model_type: ModelType) -> list[str]:
        """Return available prompt options for a given model."""
        model_class = self.get_model_class(model_type)
        return list(model_class.get_prompt_options().keys())

    def get_prompt_option_details(self, model_type: ModelType) -> dict[str, str]:
        """Get detailed information about prompt options for a model."""
        model_class = self.get_model_class(model_type)
        return model_class.get_prompt_options()

    def get_variant_details(self, model_type: ModelType) -> dict[str, ModelVariant]:
        """Get detailed information about model variants for a model."""
        model_class = self.get_model_class(model_type)
        return model_class.get_variants()

    def get_mode_details(self, model_type: ModelType) -> dict[str, str | None]:
        """Get detailed information about modes for a model."""
        model_class = self.get_model_class(model_type)
        return model_class.get_modes()

    def validate_variant(self, model_type: ModelType, variant: str) -> None:
        """
        Validate that the model variant is valid for the model.

        Args:
            model_type: The model.
            variant: The model variant to validate.

        Raises:
            InvalidModelVariantError: If the model variant is invalid for the model.
        """
        variants = self.get_variants_for_model(model_type)
        if variant not in variants:
            raise InvalidModelVariantError(
                f"Invalid model variant '{variant}' for {model_type.value} model. Available model variants: {', '.join(variants)}"  # noqa: E501
            )

    def validate_mode(self, model_type: ModelType, mode: str) -> None:
        """
        Validate that the mode is valid for the model.

        Args:
            model_type: The model.
            mode: The mode to validate.

        Raises:
            InvalidModelModeError: If the mode is invalid for the model.
        """
        modes = self.get_modes_for_model(model_type)
        if mode not in modes:
            raise InvalidModelModeError(
                f"Invalid mode '{mode}' for {model_type.value} model. Available modes: {', '.join(modes)}"  # noqa: E501
            )

    def get_default_model(self) -> ModelType:
        """
        Get the default model from the configuration.

        Returns:
            The default model.
        """
        try:
            model_str = self.config_manager.get_config_value("model", "default_model")
            return ModelType(model_str)
        except (ValueError, TypeError):
            return ModelType.BLIP

    def parse_torch_dtype(self, torch_dtype: str | None) -> torch.dtype | None:
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

    def parse_prompt_options(self, prompt_options_str: str | None) -> list[str] | None:
        """
        Parse a comma-separated string of prompt options into a list.

        Args:
            prompt_options_str: Comma-separated string of prompt options

        Returns:
            List of prompt options or None if not provided
        """
        if not prompt_options_str:
            return None

        options = [option.strip() for option in prompt_options_str.split(",")]
        filtered_options = [option for option in options if option]
        return filtered_options if filtered_options else None

    def parse_prompt_variables(
        self, prompt_variables_str: str | None
    ) -> dict[str, str] | None:
        """
        Parse a comma-separated string of key=value pairs into a dictionary.

        Args:
            prompt_variables_str: Comma-separated string of key=value pairs

        Returns:
            Dictionary of prompt variables or None if not provided

        Raises:
            ValueError: If the format is invalid
        """
        if not prompt_variables_str:
            return None

        variables = {}
        pairs = [pair.strip() for pair in prompt_variables_str.split(",")]

        for pair in pairs:
            if not pair:
                continue

            if "=" not in pair:
                raise ValueError(
                    f"Invalid prompt variable format: '{pair}'. Expected 'key=value'"
                )

            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                raise ValueError(f"Empty key in prompt variable: '{pair}'")

            variables[key] = value

        return variables if variables else None

    @handle_errors
    def create_model_instance(
        self,
        model_type: ModelType,
        variant: str | None = None,
        torch_dtype: str | None = None,
        use_runpod: bool = False,
        progress_callback: Callable[[int, int, str], Any] | None = None,
    ) -> BaseModel:
        """
        Create an instance of a model variant with the specified configuration. Uses a
        cache to avoid reloading checkpoints if the same model/model variant/dtype is
        requested.

        Args:
            model_type: The model to create.
            variant: The model variant to use. If None, uses the first available.
            torch_dtype: The torch dtype to use for the model variant.
            use_runpod: Whether to use RunPod for remote inference.
            progress_callback: Optional callback function for progress updates.

        Returns:
            An instance of the model variant.

        Raises:
            InvalidModelTypeError: If the model is unknown.
            InvalidModelVariantError: If the model variant is invalid.
            ModelConfigurationError: If there's an error configuring the model variant.
            EnhancedError: Enhanced error with troubleshooting information.
        """
        model_class = self.get_model_class(model_type)

        dtype = self.parse_torch_dtype(torch_dtype)

        if variant:
            self.validate_variant(model_type, variant)
            model_variant = variant
        else:
            default_variant = model_class.DEFAULT_VARIANT
            if default_variant is None:
                raise ModelConfigurationError(
                    f"No model variants available for {model_type.value} model"
                )
            model_variant = default_variant

        cache_key = (model_type, model_variant, str(dtype), use_runpod)
        if cache_key in self._instance_cache:
            logger.debug(f"Using cached model: {model_type.value}/{model_variant}")
            return self._instance_cache[cache_key]

        logger.info(f"Loading model: {model_type.value}/{model_variant}")

        if progress_callback:
            progress_callback(1, 3, f"Loading {model_type.value}/{model_variant}")

        try:
            if progress_callback:
                progress_callback(2, 3, "Loading model weights")

            # Create model instance
            instance = model_class(model_variant, dtype=dtype)

            # Add RunPod support if requested
            if use_runpod:
                self._setup_runpod_for_instance(instance)

            if progress_callback:
                progress_callback(3, 3, "Model ready for use")
            self._instance_cache[cache_key] = instance
            return instance
        except ImportError as e:
            tips = []

            if "accelerate" in str(e) and model_type == ModelType.JOYCAPTION:
                tips = [
                    "The 'accelerate' package is required for JoyCaption",
                    "Install it with 'pip install accelerate'",
                    "Or use 'poetry install -E joycaption' to use JoyCaption models",
                ]

            raise EnhancedError(
                message=f"Failed to create {model_type.value} model variant",
                category=ErrorCategory.MODEL_LOADING,
                original_error=e,
                troubleshooting_tips=tips,
                context={
                    "model_type": model_type.value,
                    "variant": model_variant,
                    "torch_dtype": str(dtype),
                },
            ) from None
        except Exception as e:
            raise EnhancedError(
                message=f"Failed to create {model_type.value} model variant instance",
                category=ErrorCategory.MODEL_LOADING,
                original_error=e,
                context={
                    "model_type": model_type.value,
                    "variant": model_variant,
                    "torch_dtype": str(dtype),
                },
            ) from None

    def _setup_runpod_for_instance(self, instance: BaseModel) -> None:
        """Setup RunPod service for a model instance."""
        if not self.config_manager:
            return

        try:
            from captiv.services.runpod_service import RunPodService

            all_config = self.config_manager.get_config()
            runpod_config = all_config.get("runpod", {})

            api_key = runpod_config.get("api_key")
            if not api_key:
                logger.debug("No RunPod API key configured, using local inference")
                return

            template_id = runpod_config.get("template_id")
            auto_create = runpod_config.get("auto_create", True)

            if not auto_create:
                logger.info("RunPod configured but auto_create disabled")
                return

            self.runpod_service = RunPodService(api_key, template_id)

            # Store RunPod config and service on the instance for later use
            instance._runpod_config = runpod_config  # type: ignore
            instance._runpod_service = self.runpod_service  # type: ignore
            instance._use_runpod = True  # type: ignore

            # Override the caption_image method to use RunPod
            original_caption_image = instance.caption_image
            instance.caption_image = self._create_runpod_caption_method(  # type: ignore
                instance, original_caption_image, runpod_config
            )

            logger.info("RunPod service configured successfully for model instance")

        except Exception as e:
            logger.warning(f"Failed to setup RunPod service: {e}")

    def _create_runpod_caption_method(
        self, instance: BaseModel, original_method, runpod_config: dict
    ):
        """Create a RunPod-enabled caption method for the model instance."""

        def caption_image_with_runpod(image_input, prompt_or_mode, **kwargs):
            try:
                # Try RunPod first
                return self._caption_image_remote(
                    instance, image_input, prompt_or_mode, **kwargs
                )
            except Exception as e:
                logger.error(f"Remote caption generation failed: {e}")
                logger.info("Falling back to local inference")
                # Fallback to local inference
                return original_method(image_input, prompt_or_mode, **kwargs)

        return caption_image_with_runpod

    def _caption_image_remote(
        self, instance: BaseModel, image_input, prompt_or_mode, **kwargs
    ):
        """Generate caption using RunPod remote inference."""
        if not self.runpod_service:
            raise RuntimeError("RunPod service not configured")

        # Ensure RunPod is ready
        self._ensure_runpod_ready(instance)

        # Load and prepare image
        image = instance.load_image(image_input)
        image_bytes = self._image_to_bytes(image)

        # Extract prompt options and variables
        prompt_options = kwargs.pop("prompt_options", None)
        prompt_variables = kwargs.pop("prompt_variables", None)

        # Resolve the prompt
        prompt = instance.resolve_prompt(
            prompt_or_mode, prompt_options, prompt_variables
        )

        # Determine mode or use prompt
        if prompt_or_mode and prompt_or_mode in instance.MODES:
            mode = prompt_or_mode
            custom_prompt = None
        else:
            mode = "default"
            custom_prompt = prompt

        # Generate caption remotely
        caption = self.runpod_service.generate_caption_remote(
            image_data=image_bytes,
            model_variant=instance.variant_key,
            mode=mode,
            prompt=custom_prompt,
            **kwargs,
        )

        return caption

    def _ensure_runpod_ready(self, instance: BaseModel) -> None:
        """Ensure RunPod instance is ready for inference."""
        if not self.runpod_service:
            from captiv.services.exceptions import RunPodError

            raise RunPodError("RunPod service not configured")

        runpod_config = getattr(instance, "_runpod_config", {})

        # Check if we have an active pod
        if self.runpod_service.active_pod_id:
            if self.runpod_service.health_check():
                logger.debug("RunPod instance is healthy and ready")
                return
            else:
                logger.warning("RunPod instance is not healthy, recreating...")
                try:
                    self.runpod_service.terminate_pod(self.runpod_service.active_pod_id)
                except Exception as e:
                    logger.warning(f"Failed to terminate unhealthy pod: {e}")

        # Create new pod
        logger.info("Creating new RunPod instance...")
        pod_name = (
            f"captiv-{instance.__class__.__name__.lower()}-{instance.variant_key}"
        )

        gpu_type = runpod_config.get("default_gpu_type", "NVIDIA RTX A4000")

        pod_id = self.runpod_service.create_pod(name=pod_name, gpu_type=gpu_type)

        # Wait for pod to be ready
        startup_timeout = runpod_config.get("startup_timeout", 600)
        self.runpod_service.wait_for_pod_ready(pod_id, timeout=startup_timeout)

        logger.info(f"RunPod instance {pod_id} is ready for inference")

    def _image_to_bytes(self, image) -> bytes:
        """Convert PIL Image to bytes for transmission."""
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    def cleanup_runpod(self) -> None:
        """Clean up RunPod resources."""
        if self.runpod_service and self.runpod_service.active_pod_id:
            all_config = self.config_manager.get_config()
            runpod_config = all_config.get("runpod", {})
            auto_terminate = runpod_config.get("auto_terminate", True)

            if auto_terminate:
                try:
                    logger.info("Terminating RunPod instance...")
                    self.runpod_service.terminate_pod(self.runpod_service.active_pod_id)
                    logger.info("RunPod instance terminated successfully")
                except Exception as e:
                    logger.warning(f"Failed to terminate RunPod instance: {e}")
            else:
                try:
                    logger.info("Stopping RunPod instance...")
                    self.runpod_service.stop_pod(self.runpod_service.active_pod_id)
                    logger.info("RunPod instance stopped successfully")
                except Exception as e:
                    logger.warning(f"Failed to stop RunPod instance: {e}")

    def build_generation_params(
        self,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        num_beams: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        prompt_options: list[str] | None = None,
        prompt_variables: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Build a dictionary of generation parameters for the model variant.

        This method only includes parameters that are explicitly provided.

        Args:
            max_new_tokens: Maximum number of tokens in the generated caption.
            min_new_tokens: Minimum number of tokens in the generated caption.
            num_beams: Number of beams for beam search.
            temperature: Temperature for sampling.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty parameter.
            prompt_options: List of prompt options to include.
            prompt_variables: Dictionary of prompt variables.

        Returns:
            A dictionary of generation parameters.
        """
        gen_params = {}
        if max_new_tokens is not None:
            gen_params["max_new_tokens"] = max_new_tokens
        if min_new_tokens is not None:
            gen_params["min_new_tokens"] = min_new_tokens
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
        if prompt_options is not None:
            gen_params["prompt_options"] = prompt_options
        if prompt_variables is not None:
            gen_params["prompt_variables"] = prompt_variables

        return gen_params

    def generate_caption(
        self,
        model_instance: BaseModel,
        image_path: Any,
        mode: str | None = None,
        prompt: str | None = None,
        generation_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a caption for an image using the provided model instance.

        This method serves as a high-level interface for caption generation,
        handling the coordination between the model instance and any additional
        parameters or prompt customization.

        Args:
            model_instance: The model instance to use for generation
            image_path: Path to the image file or PIL Image object
            mode: Captioning mode to use
            prompt: Custom prompt to use (overrides mode)
            generation_params: Dictionary of generation parameters

        Returns:
            Generated caption text

        Raises:
            FileNotFoundError: If image file is not found
            ValueError: If image cannot be processed
        """
        prompt_or_mode = prompt if prompt else mode

        final_params = generation_params or {}

        return model_instance.caption_image(
            image_input=image_path, prompt_or_mode=prompt_or_mode, **final_params
        )
