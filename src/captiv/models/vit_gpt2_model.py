from typing import Optional

import torch
from PIL import Image

from captiv.logging import logger

from .base_model import ImageCaptioningModel


class VitGpt2Model(ImageCaptioningModel):
    """
    A class to encapsulate the ViT+GPT2 model for image captioning,
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
        "vit-gpt2": {
            "huggingface_id": "nlpconnect/vit-gpt2-image-captioning",
            "description": "ViT+GPT2 model for image captioning",
            "default_mode": "default",  # Default mode for this variant
        },
    }

    # Define default variant
    DEFAULT_VARIANT = "vit-gpt2"

    def __init__(
        self,
        model_variant_or_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initializes the VitGpt2Model. The device (CUDA, MPS, or CPU) is determined
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
            # Check if vit_gpt2_variant exists in config, otherwise use a default
            model_variant_or_path = getattr(cfg.model, "vit_gpt2_variant", "vit-gpt2")

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
            huggingface_id = "nlpconnect/vit-gpt2-image-captioning"
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

        # Initialize processor and model
        logger.info(f"Initializing processor and model for {self.model_name_or_path}")

        try:
            # Import here to avoid issues with static type checking
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            from transformers.models.vision_encoder_decoder import (
                VisionEncoderDecoderModel,
            )
            from transformers.models.vit import ViTImageProcessor

            # Memory optimization options
            model_kwargs = {}
            if load_dtype:
                model_kwargs["torch_dtype"] = load_dtype

            # Load processor for image processing
            self.processor = ViTImageProcessor.from_pretrained(self.model_name_or_path)

            # Load tokenizer for text decoding
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            # Load model
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name_or_path, **model_kwargs
            )

            # Move model to device
            # Use type ignore comment to suppress type checker errors
            # The actual runtime behavior is correct
            self.model = self.model.to(self.device)  # type: ignore
            logger.info(f"VitGpt2Model moved to {self.device}")

        except Exception as e:
            logger.error(f"Error initializing VitGpt2Model: {e}")
            # Re-raise to prevent using an uninitialized model
            raise RuntimeError(f"Failed to initialize VitGpt2Model: {e}")

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
        Process the image and generate a caption using the ViT+GPT2 model.

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
        """
        # Process prompt text using the simple string mapping

        try:
            # Process the image
            inputs = self.processor(images=image, return_tensors="pt")

            # Create a clean dictionary of inputs without any 'device' key
            # and move tensors to the correct device
            model_inputs = {}
            for k, v in inputs.items():
                if k != "device":  # Skip any 'device' key
                    if hasattr(v, "to"):
                        model_inputs[k] = v.to(self.device)
                    else:
                        model_inputs[k] = v

            # Note: ViT+GPT2 model doesn't typically use text prompts directly
            # The prompt is used more for post-processing guidance
            # But we'll log it for reference
            if prompt_text:
                logger.info(
                    f"Using prompt: '{prompt_text}' (Note: ViT+GPT2 may not directly use text prompts)"
                )

            # Generate caption - ensure we don't pass device parameter
            generation_kwargs = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": True,  # Enable sampling when using temperature, top_k, and top_p
            }

            # Add any additional kwargs, excluding device
            for k, v in kwargs.items():
                if k != "device":
                    generation_kwargs[k] = v

            # Generate caption
            outputs = self.model.generate(**model_inputs, **generation_kwargs)

            # Decode the generated caption using the tokenizer
            caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return caption

        except Exception as e:
            logger.error(f"Error generating caption with ViT+GPT2 model: {e}")
            return f"Error generating caption: {str(e)}"
