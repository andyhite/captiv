"""
Base model classes and utilities for image captioning models.

This module provides the foundational classes and type definitions for all image
captioning models in the Captiv library. It includes the abstract base class that all
models inherit from, as well as utility functions for creating model instances.
"""

import re
from typing import Any, NotRequired, TypedDict, Unpack

import torch
from PIL import Image
from transformers.image_utils import ImageInput


class ModelVariant(TypedDict):
    """
    Type definition for model variant configuration.

    Attributes:
        huggingface_id: The Hugging Face model identifier
        description: Optional description of the model variant
        default_mode: Optional default captioning mode for this variant
    """

    huggingface_id: str
    description: NotRequired[str]
    default_mode: NotRequired[str]


class GenerationOptions(TypedDict, total=False):
    """
    Type definition for text generation parameters.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate (caption length)
        min_new_tokens: Minimum number of new tokens to generate (caption length)
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for repeated tokens
    """

    max_new_tokens: int | None
    min_new_tokens: int | None
    num_beams: int | None
    prompt_options: NotRequired[list[str]]
    prompt_variables: NotRequired[dict[str, str]]
    repetition_penalty: float | None
    temperature: float | None
    top_k: int | None
    top_p: float | None


class BaseModel:
    """
    Abstract base class for all image captioning models.

    This class provides the common interface and functionality that all
    image captioning models must implement. It handles model loading,
    device management, and the core captioning workflow.

    Class Attributes:
        MODEL: The transformers model class to use
        TOKENIZER: The tokenizer class to use (optional)
        PROCESSOR: The processor class to use
        MODES: Available captioning modes for this model
        VARIANTS: Available model variants
        DEFAULT_VARIANT: Default variant to use if none specified
    """

    MODEL: Any = None
    TOKENIZER: Any = None
    PROCESSOR: Any = None

    MODES: dict[str, str | None] = {"default": None}
    VARIANTS: dict[str, ModelVariant] = {}
    PROMPT_OPTIONS: dict[str, str] = {}
    PROMPT_VARIABLES: dict[str, str] = {}
    DEFAULT_VARIANT: str | None = None

    def __init__(self, variant_key: str, dtype: torch.dtype | None = None):
        """
        Initialize the image captioning model.

        Args:
            variant_key: The model variant to load
            dtype: Optional torch data type to use for the model

        Raises:
            ValueError: If the specified variant is not available
        """
        self.variant_key = variant_key
        self._dtype = dtype

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        processor = self.load_processor()

        if isinstance(processor, tuple):
            self._processor = processor[0]
        else:
            self._processor = processor

        self._tokenizer = self.load_tokenizer()

        model = self.load_model()

        if dtype is None:
            self._model = model.to(self.device)
        else:
            self._model = model.to(self.device, dtype=dtype)

    def __repr__(self) -> str:
        """Return a string representation of the model instance."""
        return f"{self.__class__.__name__}(huggingface_id='{self.variant['huggingface_id']}', dtype='{self.dtype}, device='{self.device}')"  # noqa: E501

    def caption_image(
        self,
        image_input: ImageInput | str,
        prompt_or_mode: str | None,
        **kwargs: Unpack[GenerationOptions],
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image_input: Path to image file or PIL Image object
            prompt_or_mode: Custom prompt or predefined mode name
            **kwargs: Generation parameters (max_new_tokens, temperature, etc.)

        Returns:
            Generated caption text

        Raises:
            FileNotFoundError: If image file is not found
            ValueError: If image cannot be processed
        """
        with torch.no_grad():
            prompt_options = kwargs.pop("prompt_options", None)
            prompt_variables = kwargs.pop("prompt_variables", None)

            image = self.load_image(image_input)
            prompt = self.resolve_prompt(
                prompt_or_mode, prompt_options, prompt_variables
            )
            inputs = self.process_inputs(image, prompt)
            generated_ids = self.generate_ids(inputs, **kwargs)
            caption = self.decode_caption(generated_ids)

        return caption

    def resolve_prompt_options(
        self,
        prompt: str,
        prompt_options: list[str] | None,
    ) -> str:
        """
        Resolve prompt options into the final prompt text.

        Args:
            prompt: The prompt text
            prompt_options: List of options to include in the prompt
        Returns:
            Resolved prompt text with options included
        """
        if prompt_options:
            valid_options = [
                self.PROMPT_OPTIONS[option]
                for option in prompt_options
                if option in self.PROMPT_OPTIONS
            ]

            filtered_options = [
                str(option) for option in valid_options if option is not None
            ]

            return f"{prompt}\n\n{' '.join(filtered_options)}"

        return prompt

    def extract_required_variables(self, prompt: str) -> set[str]:
        """
        Extract required variable names from a prompt string.

        This method finds all variable placeholders in the format {variable_name}
        that would be used by str.format().

        Args:
            prompt: The prompt text with placeholders

        Returns:
            Set of variable names required by the prompt
        """
        if not prompt:
            return set()

        pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)[^}]*\}"
        matches = re.findall(pattern, prompt)
        return set(matches)

    def validate_prompt_variables(
        self, prompt: str, prompt_variables: dict[str, str] | None
    ) -> None:
        """
        Validate that all required variables are provided for a prompt.

        Args:
            prompt: The prompt text with placeholders
            prompt_variables: Dictionary of variables to replace in the prompt

        Raises:
            ValueError: If required variables are missing
        """
        if not prompt:
            return

        required_vars = self.extract_required_variables(prompt)
        if not required_vars:
            return

        provided_vars = set(prompt_variables.keys()) if prompt_variables else set()
        missing_vars = required_vars - provided_vars

        if missing_vars:
            missing_list = sorted(missing_vars)
            raise ValueError(
                f"Missing required prompt variables: {', '.join(missing_list)}. "
                f"Required variables: {', '.join(sorted(required_vars))}"
            )

    def resolve_prompt_variables(
        self, prompt: str, prompt_variables: dict[str, str] | None
    ) -> str:
        """
        Resolve prompt variables into the final prompt text.

        Args:
            prompt: The prompt text with placeholders
            prompt_variables: Dictionary of variables to replace in the prompt
        Returns:
            Resolved prompt text with variables replaced
        Raises:
            ValueError: If required variables are missing
        """
        if not prompt:
            return prompt

        self.validate_prompt_variables(prompt, prompt_variables)

        if prompt_variables:
            try:
                return prompt.format(**prompt_variables)
            except KeyError as e:
                raise ValueError(f"Missing prompt variable: {e}") from e
            except Exception as e:
                raise ValueError(f"Error formatting prompt: {e}") from e

        return prompt

    def resolve_prompt_mode(
        self,
        prompt_or_mode: str | None,
    ) -> str | None:
        """
        Resolve a prompt or mode into the final prompt text.

        Args:
            prompt_or_mode: Either a custom prompt or a predefined mode name
        Returns:
            Resolved prompt text or None for default behavior
        """
        if prompt_or_mode is None:
            return self.default_mode

        if prompt_or_mode in self.MODES:
            return self.MODES[prompt_or_mode]
        elif prompt_or_mode is not None:
            return prompt_or_mode

        return None

    def resolve_prompt(
        self,
        prompt_or_mode: str | None,
        prompt_options: list[str] | None,
        prompt_variables: dict[str, str] | None,
    ) -> str | None:
        """
        Resolve a prompt or mode into the final prompt text.

        Args:
            prompt_or_mode: Either a custom prompt or a predefined mode name

        Returns:
            Resolved prompt text or None for default behavior
        """
        prompt = self.resolve_prompt_mode(prompt_or_mode)

        if prompt:
            prompt = self.resolve_prompt_options(prompt, prompt_options)
            prompt = self.resolve_prompt_variables(prompt, prompt_variables)

        return prompt

    def process_inputs(
        self,
        image: Image.Image,
        prompt: str | None,
    ) -> Any:
        """
        Process image and prompt into model inputs.

        Args:
            image: PIL Image object
            prompt: Optional text prompt

        Returns:
            Processed inputs ready for model generation
        """
        inputs = self._processor(image, text=prompt, return_tensors="pt").to(
            self.device
        )

        for k, v in inputs.items():
            if hasattr(v, "to") and hasattr(v, "dtype"):
                try:
                    if hasattr(torch, "is_floating_point") and torch.is_floating_point(
                        v
                    ):
                        inputs[k] = v.to(
                            self.device,
                            dtype=torch.float16
                            if self.device != "cpu"
                            else torch.float32,
                        )
                    else:
                        inputs[k] = v.to(self.device)
                except (TypeError, AttributeError):
                    inputs[k] = v.to(self.device) if hasattr(v, "to") else v

        return inputs

    def generate_ids(self, inputs: Any, **kwargs: Unpack[GenerationOptions]) -> Any:
        """
        Generate token IDs from processed inputs.

        Args:
            inputs: Processed model inputs
            **kwargs: Generation parameters

        Returns:
            Generated token IDs
        """
        return self._model.generate(**inputs, **kwargs, do_sample=True)

    def decode_caption(self, generated_ids) -> str:
        """
        Decode generated token IDs into caption text.

        Args:
            generated_ids: Generated token IDs from the model

        Returns:
            Decoded caption text

        Raises:
            ValueError: If no tokenizer or processor is available
        """
        if self._tokenizer:
            return self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if self._processor:
            return self._processor.decode(generated_ids[0], skip_special_tokens=True)

        raise ValueError("No tokenizer or processor available for decoding.")

    def load_model(self) -> Any:
        """
        Load the model from Hugging Face.

        Returns:
            Loaded model instance

        Raises:
            ValueError: If MODEL class is not defined
        """
        if self.MODEL is None:
            raise ValueError("MODEL class is not defined for this model.")

        return self.MODEL.from_pretrained(
            self.variant["huggingface_id"],
            torch_dtype=self.dtype,
        )

    def load_tokenizer(self) -> Any:
        """
        Load the tokenizer from Hugging Face.

        Returns:
            Loaded tokenizer instance or None if not needed
        """
        if self.TOKENIZER is None:
            return None

        return self.TOKENIZER.from_pretrained(self.variant["huggingface_id"])

    def load_processor(self) -> Any:
        """
        Load the processor from Hugging Face.

        Returns:
            Loaded processor instance or None if not available
        """
        if self.PROCESSOR is None:
            return None

        return self.PROCESSOR.from_pretrained(self.variant["huggingface_id"])

    @classmethod
    def get_modes(cls) -> dict[str, str | None]:
        """Get available captioning modes for this model."""
        return cls.MODES

    @classmethod
    def get_variants(cls) -> dict[str, ModelVariant]:
        """Get available model variants for this model."""
        return cls.VARIANTS

    @classmethod
    def get_prompt_options(cls) -> dict[str, str]:
        """Get available prompt options for this model."""
        return cls.PROMPT_OPTIONS

    @property
    def variant(self):
        """
        Get the current model variant configuration.

        Returns:
            ModelVariant dictionary with configuration details

        Raises:
            ValueError: If the variant is not found
        """
        if self.variant_key in self.VARIANTS:
            return self.VARIANTS[self.variant_key]

        if self.DEFAULT_VARIANT in self.VARIANTS:
            return self.VARIANTS[self.DEFAULT_VARIANT]

        raise ValueError(
            f"Variant '{self.variant_key}' not found in available variants: {list(self.VARIANTS.keys())}"  # noqa: E501
        )

    @property
    def huggingface_id(self):
        """Get the Hugging Face model identifier for the current variant."""
        return self.variant["huggingface_id"]

    @property
    def description(self):
        """Get the description of the current model variant."""
        return self.variant.get("description", None)

    @property
    def default_mode(self):
        """Get the default captioning mode for the current variant."""
        return self.variant.get("default_mode", "default")

    @property
    def dtype(self):
        """
        Get the appropriate torch data type for the model.

        Returns:
            torch.dtype or None for automatic selection
        """
        if self._dtype is None:
            if self.device == "cuda" and torch.cuda.is_available():
                if torch.cuda.get_device_capability()[0] >= 8:
                    return torch.bfloat16
                else:
                    return torch.float16
            if self.device != "cpu":
                return torch.float16

            return None

        return self._dtype

    @staticmethod
    def load_image(image_input: ImageInput | str) -> Image.Image:
        """
        Load an image from various input types.

        Args:
            image_input: Path to image file or PIL Image object

        Returns:
            PIL Image object in RGB format

        Raises:
            FileNotFoundError: If image file is not found
            ValueError: If image cannot be processed
        """
        if isinstance(image_input, str):
            try:
                return Image.open(image_input).convert("RGB")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Image file '{image_input}' not found.") from e
            except Exception as e:
                raise ValueError(
                    f"Could not open image file '{image_input}': {e}"
                ) from e
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError("Invalid image input. Must be a file path or PIL Image.")


def create_model(
    model_class: Any,
    processor_class: Any,
    default_variant: str,
    variants: dict[str, ModelVariant],
    prompt_options: dict[str, str] | None = None,
    modes: dict[str, str | None] | None = None,
    tokenizer_class: Any | None = None,
):
    """
    Factory function to create a new image captioning model class.

    This function creates a new model class that inherits from BaseImageCaptioningModel
    with the specified configuration. It's used to define concrete model implementations
    without having to write boilerplate code.

    Args:
        model_class: The transformers model class to use
        processor_class: The processor class for handling inputs
        default_variant: The default model variant to use
        variants: Dictionary of available model variants
        modes: Available captioning modes
        tokenizer_class: Optional tokenizer class

    Returns:
        A new model class ready for instantiation

    Example:
        >>> MyModel = create_model(
        ...     model_class=BlipForConditionalGeneration,
        ...     processor_class=BlipProcessor,
        ...     default_variant="base",
        ...     variants={"base": {"huggingface_id": "Salesforce/blip-base"}}
        ... )
        >>> model = MyModel("base")
    """

    if modes is None:
        modes = {}
    if prompt_options is None:
        prompt_options = {}

    class ImageCaptioningModel(BaseModel):
        MODEL = model_class
        TOKENIZER = tokenizer_class
        PROCESSOR = processor_class
        VARIANTS = variants
        DEFAULT_VARIANT = default_variant
        PROMPT_OPTIONS = prompt_options
        MODES = modes

    return ImageCaptioningModel
