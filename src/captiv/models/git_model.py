from typing import Optional

import torch
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from captiv.logging import logger

from .base_model import ImageCaptioningModel


class GitModel(ImageCaptioningModel):
    """
    A class to encapsulate the GIT (GenerativeImage2Text) model for image captioning,
    inheriting from ImageCaptioningModel.
    It relies on the base class for device detection (CUDA, MPS, or CPU).
    """

    MODES = {
        "default": None,  # None means unconditional captioning
        "detailed": "a detailed description of the image",
        "concise": "a short description of the image",
        "artistic": "an artistic description of the image",
        "technical": "a technical description of the image",
    }

    VARIANTS = {
        "git-base": {
            "huggingface_id": "microsoft/git-base",
            "description": "Base GIT model for image captioning (110M parameters)",
            "default_mode": "default",  # Default mode for this variant
        },
        "git-large": {
            "huggingface_id": "microsoft/git-large",
            "description": "Large GIT model for image captioning (1.1B parameters)",
            "default_mode": "default",  # Default mode for this variant
        },
        "git-base-vqav2": {
            "huggingface_id": "microsoft/git-base-vqav2",
            "description": "GIT base model fine-tuned on VQAv2 for visual question answering",
            "default_mode": "default",
        },
    }

    # Define default variant
    DEFAULT_VARIANT = "git-large"

    def __init__(
        self,
        model_variant_or_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initializes the GitModel. The device (CUDA, MPS, or CPU) is determined
        by the ImageCaptioningModel base class.

        Args:
            model_variant_or_path (str, optional): A key from VARIANTS or a direct
                                                 Hugging Face model name/path.
                                                 If None, uses the default from config.
            torch_dtype (torch.dtype, optional): The desired torch data type.
                                                For memory optimization.
        """
        # Import here to avoid circular imports
        from captiv import config

        # Get default variant from config if not provided
        if model_variant_or_path is None:
            cfg = config.read_config()
            # Check if git_variant exists in config, otherwise use a default
            model_variant_or_path = getattr(cfg.model, "git_variant", "git-base")

        if model_variant_or_path in self.VARIANTS:
            huggingface_id = self.VARIANTS[model_variant_or_path]["huggingface_id"]
            self.variant_key = model_variant_or_path  # Set variant_key for __repr__
            self.default_mode_key = self.VARIANTS[model_variant_or_path].get(
                "default_mode"
            )
        else:
            huggingface_id = model_variant_or_path
            self.variant_key = (
                None  # Ensure variant_key is None if not a predefined variant
            )
            self.default_mode_key = None

        # Ensure huggingface_id is not None
        if not huggingface_id:
            huggingface_id = "microsoft/git-base"
            logger.warning(f"No model path provided, using default: {huggingface_id}")

        super().__init__(huggingface_id, torch_dtype=torch_dtype)

        # Determine the torch_dtype for model loading
        load_dtype = self.torch_dtype_requested
        if load_dtype is None and self.device != "cpu":
            load_dtype = torch.float16  # Default to float16 for GPU if not specified
            logger.info(
                f"No torch_dtype specified, defaulting to {load_dtype} for GPU."
            )
        elif (
            load_dtype is not None
            and self.device == "cpu"
            and load_dtype == torch.float16
        ):
            logger.warning(
                "torch.float16 requested but device is CPU. Loading in float32 instead."
            )
            load_dtype = (
                None  # Load in default (float32) for CPU if float16 was specified
            )

        # Initialize processor (tokenizer) and model
        logger.info(f"Initializing tokenizer for {self.model_name_or_path}")
        self.processor = AutoTokenizer.from_pretrained(self.model_name_or_path)

        logger.info(
            f"Initializing model for {self.model_name_or_path} with dtype: {load_dtype or 'default'}"
        )

        # Memory optimization options
        model_kwargs = {}
        if load_dtype:
            model_kwargs["torch_dtype"] = load_dtype

        # Add device_map for offloading if using large model on CUDA
        if "large" in self.model_name_or_path and self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            logger.info("Using device_map='auto' for large model on CUDA")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, **model_kwargs
            )
        except Exception as e:
            logger.warning(
                f"Failed to load model with specified options: {e}. "
                f"Attempting to load with default settings."
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

        # Only move model to device if not using device_map
        if "device_map" not in model_kwargs:
            logger.info(f"Moving GitModel to device: {self.device}")
            try:
                self.model.to(torch.device(self.device))
                logger.info(f"GitModel moved to {self.device}")
            except Exception as e:
                logger.warning(f"Could not move GitModel to {self.device}: {e}")

    def _process_image_and_generate_caption(
        self,
        image: Image.Image,
        prompt_text: Optional[str],
        max_length: int = 50,
        min_length: int = 10,
        num_beams: int = 3,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Process the image and generate a caption using the GIT model.

        This implements the abstract method from the base class.

        Args:
            image: Preprocessed PIL Image.
            prompt_text: Resolved prompt text.
            max_length: Maximum length of the generated caption.
            min_length: Minimum length of the generated caption.
            num_beams: Number of beams for beam search.
            temperature: Temperature for sampling.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty parameter.
            **kwargs: Additional model-specific parameters.
                - word_count: Number of words to limit the caption to.
                - length: Length description (short, medium, long).
        """
        # Process prompt text - simplified with direct string mapping
        # No additional processing needed since MODES is now a simple string->string dictionary

        # Prepare inputs for the model
        try:
            # Process the image
            # Note: GIT models typically need a processor that handles both image and text
            # Since we're using AutoTokenizer, we need to handle the image separately
            # This is a simplified implementation and may need adjustment based on the specific GIT model

            # For GIT models, we need to convert the image to pixel values
            # This is typically done by a processor or feature extractor
            # For simplicity, we'll assume the model can handle raw images
            # In a real implementation, you would use the appropriate processor

            # Tokenize the prompt if provided
            if prompt_text:
                inputs = self.processor(prompt_text, return_tensors="pt")
            else:
                # If no prompt, use an empty string or a default prompt
                inputs = self.processor("", return_tensors="pt")

            # Move inputs to the correct device
            inputs = {
                k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")
            }

            # Add the image to the inputs
            # This is a placeholder - actual implementation depends on the specific GIT model
            # inputs["pixel_values"] = preprocess_image(image).to(self.device)

            logger.warning(
                "GIT model implementation is simplified and may not work correctly with all GIT variants."
            )
            logger.warning(
                "For production use, refer to the specific GIT model documentation."
            )

            # Generate caption
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,  # Enable sampling when using temperature, top_k, and top_p
                **{
                    k: v for k, v in kwargs.items() if k not in ["word_count", "length"]
                },
            )

            # Decode the generated caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            logger.error(f"Error generating caption with GIT model: {e}")
            return f"Error generating caption: {str(e)}"
