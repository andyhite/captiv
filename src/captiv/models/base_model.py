from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from PIL import Image

from captiv.logging import logger


class ImageCaptioningModel(ABC):
    """
    Abstract base class for image captioning models.

    Handles device detection and model configuration.
    Provides shared utilities for prompt resolution and image loading.
    Implements template method pattern for caption_image().
    """

    # Subclasses should define these class variables
    MODES = {}
    VARIANTS = {}
    PROMPT_OPTIONS = {}  # Options that can be appended to prompts

    @classmethod
    def get_prompt_options(cls):
        """
        Return available prompt options for the model.

        By default, returns the class's PROMPT_OPTIONS dictionary.
        Subclasses can override this method for custom behavior.
        """
        return cls.PROMPT_OPTIONS

    def __init__(
        self, model_name_or_path: str, torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the captioning model and detect the available device.
        """
        self.model_name_or_path = model_name_or_path
        self.torch_dtype_requested = torch_dtype

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(
            f"{self.__class__.__name__}: Using device: {self.device} for model: {model_name_or_path}"
        )
        self.default_mode_key = None
        self.variant_key = None

    def __repr__(self) -> str:
        """
        Return a string representation of the model instance.
        """
        variant_info = f", variant='{self.variant_key}'" if self.variant_key else ""
        dtype_info = (
            f", dtype_req='{self.torch_dtype_requested}'"
            if self.torch_dtype_requested
            else ""
        )
        return (
            f"{self.__class__.__name__}(model='{self.model_name_or_path}'"
            f"{variant_info}, device='{self.device}'{dtype_info})"
        )

    def caption_image(
        self,
        image_input: Union[str, Image.Image],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Template method for image captioning.

        This method handles common operations like:
        - Loading and validating the image
        - Resolving the prompt
        - Error handling

        Subclasses must implement _process_image_and_generate_caption().

        Args:
            image_input: Path to image file or PIL Image.
            prompt: Optional prompt or mode key.
            Other arguments: Generation parameters.

        Returns:
            The generated caption.
        """
        try:
            # Import here to avoid circular imports
            from captiv import config

            # Load configuration defaults
            cfg = config.read_config()

            # Load and validate the image
            raw_image = self.load_image(image_input)

            # Resolve the prompt text, passing word_count and length if provided
            actual_prompt_text = self.resolve_prompt(
                prompt, self.default_mode_key, self.get_modes(), **kwargs
            )

            # Get model-specific defaults if available
            model_type = self.__class__.__name__.lower().replace("model", "")
            max_length_key = f"{model_type}_max_length"

            # Use provided values or get from config
            actual_max_length = (
                max_length
                if max_length is not None
                else (
                    getattr(cfg.generation, max_length_key)
                    if hasattr(cfg.generation, max_length_key)
                    else cfg.generation.max_length
                )
            )
            actual_min_length = (
                min_length if min_length is not None else cfg.generation.min_length
            )
            actual_num_beams = (
                num_beams if num_beams is not None else cfg.generation.num_beams
            )
            actual_temperature = (
                temperature if temperature is not None else cfg.generation.temperature
            )
            actual_top_k = top_k if top_k is not None else cfg.generation.top_k
            actual_top_p = top_p if top_p is not None else cfg.generation.top_p
            actual_repetition_penalty = (
                repetition_penalty
                if repetition_penalty is not None
                else cfg.generation.repetition_penalty
            )

            # Call the model-specific implementation
            return self._process_image_and_generate_caption(
                raw_image,
                actual_prompt_text,
                max_length=actual_max_length,
                min_length=actual_min_length,
                num_beams=actual_num_beams,
                temperature=actual_temperature,
                top_k=actual_top_k,
                top_p=actual_top_p,
                repetition_penalty=actual_repetition_penalty,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise

    @abstractmethod
    def _process_image_and_generate_caption(
        self,
        image: Image.Image,
        prompt_text: Optional[str],
        max_length: int = 32,
        min_length: int = 10,
        num_beams: int = 3,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Process the image and generate a caption using the specific model implementation.

        This method must be implemented by subclasses to handle model-specific processing.

        Args:
            image: Preprocessed PIL Image.
            prompt_text: Resolved prompt text.
            Other arguments: Generation parameters.

        Returns:
            The generated caption.
        """

    def caption_batch(
        self, image_inputs: List[Union[str, Image.Image]], **kwargs
    ) -> List[str]:
        """
        Generate captions for a batch of images (default: sequential).

        Args:
            image_inputs: List of image paths or PIL Images.
            **kwargs: Additional parameters to pass to caption_image.

        Returns:
            List of generated captions.
        """
        return [self.caption_image(img, **kwargs) for img in image_inputs]

    @staticmethod
    def resolve_prompt(
        prompt_or_mode: Optional[str],
        default_mode_key: Optional[str],
        modes: dict,
        **kwargs,
    ) -> Optional[str]:
        """
        Resolves the actual prompt text given a prompt or mode key.
        If prompt_or_mode is a key in modes, returns the corresponding prompt template.
        If prompt_or_mode is a string not in modes, returns it as a custom prompt.
        If prompt_or_mode is None, uses default_mode_key.
        """
        key = prompt_or_mode if prompt_or_mode is not None else default_mode_key
        if key is None:
            return None

        # If the key is in modes, get the prompt template
        if key in modes:
            return modes[key]

        # If not in modes, return as custom prompt
        return key

    @staticmethod
    def load_image(image_input: Union[str, Image.Image]) -> Image.Image:
        """
        Loads and returns a PIL Image from a file path or returns the image if already a PIL Image.
        Raises FileNotFoundError or ValueError on error.
        """
        if isinstance(image_input, str):
            try:
                return Image.open(image_input).convert("RGB")
            except FileNotFoundError:
                logger.error(f"Image file not found: {image_input}")
                raise
            except Exception as e:
                logger.error(f"Could not open image file '{image_input}': {e}")
                raise ValueError(f"Could not open image file '{image_input}': {e}")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            logger.error("Invalid image input. Must be a file path or PIL Image.")
            raise ValueError("Invalid image input. Must be a file path or PIL Image.")

    @classmethod
    def get_modes(cls) -> dict:
        """
        Return available captioning modes for the model.

        By default, returns the class's MODES dictionary.
        Subclasses can override this method for custom behavior.
        """
        return cls.MODES

    @classmethod
    def get_variants(cls) -> dict:
        """
        Return available model variants.

        By default, returns the class's VARIANTS dictionary.
        Subclasses can override this method for custom behavior.
        """
        return cls.VARIANTS

    @classmethod
    def get_default_variant(cls) -> Optional[str]:
        """
        Return the default variant for the model.

        By default, returns the class's DEFAULT_VARIANT if defined,
        otherwise returns the first variant from VARIANTS.
        Subclasses can override this method for custom behavior.

        Returns:
            The default variant name or None if no variants are available.
        """
        if hasattr(cls, "DEFAULT_VARIANT"):
            return getattr(cls, "DEFAULT_VARIANT")

        variants = list(cls.VARIANTS.keys())
        if variants:
            return variants[0]

        return None
