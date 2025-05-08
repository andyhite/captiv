from typing import Optional

import torch  # Already imported, but good to ensure
from PIL import Image
from transformers.models.blip_2 import Blip2ForConditionalGeneration, Blip2Processor

from captiv.logging import logger

from .base_model import ImageCaptioningModel


class Blip2Model(ImageCaptioningModel):
    """
    A class to encapsulate the BLIP-2 model for image captioning,
    inheriting from ImageCaptioningModel.
    It relies on the base class for device detection (CUDA, MPS, or CPU).
    """

    MODES = {
        "default": "a photo of",
        "detailed": "a detailed photo of",
        "concise": "a simple photo of",
        "artistic": "an artistic photo of",
        "technical": "a technical photo of",
        "question": "what is in this image?",  # This implies a question-answering setup
        "describe": "describe this image in detail",  # More direct for description
        "unconditional": None,  # Explicitly for models that support true unconditional generation
    }

    VARIANTS = {  # Renamed from AVAILABLE_VARIANTS
        "blip2-opt-2.7b": {  # Default, often referred to as blip2-base
            "huggingface_id": "Salesforce/blip2-opt-2.7b",
            "description": "BLIP-2 model with OPT 2.7B language model.",
            "default_mode": "default",
        },
        "blip2-flan-t5-xl": {
            "huggingface_id": "Salesforce/blip2-flan-t5-xl",
            "description": "BLIP-2 model with Flan-T5-XL language model (often better for VQA).",
            "default_mode": "question",
        },
        "blip2-opt-6.7b": {
            "huggingface_id": "Salesforce/blip2-opt-6.7b",
            "description": "BLIP-2 model with OPT 6.7B language model.",
            "default_mode": "default",
        },
    }

    # Define default variant
    DEFAULT_VARIANT = "blip2-opt-2.7b"

    def __init__(
        self,
        model_variant_or_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initializes the Blip2Model. The device (CUDA, MPS, or CPU) is determined
        by the ImageCaptioningModel base class.

        Args:
            model_variant_or_path (str, optional): A key from VARIANTS or a direct
                                                  Hugging Face model name/path.
                                                  If None, uses the default from config.
            torch_dtype (torch.dtype, optional): The desired torch data type for model loading
                                                 (e.g., torch.float16). If None, suitable defaults
                                                 based on device will be attempted.
        """
        # Import here to avoid circular imports
        from captiv import config

        # Get default variant from config if not provided
        if model_variant_or_path is None:
            cfg = config.read_config()
            model_variant_or_path = cfg.model.blip2_variant

        if model_variant_or_path in self.VARIANTS:
            huggingface_id = self.VARIANTS[model_variant_or_path]["huggingface_id"]
            self.variant_key = model_variant_or_path  # Set variant_key for __repr__
            self.default_mode_key = self.VARIANTS[model_variant_or_path].get(
                "default_mode"
            )
        else:
            huggingface_id = model_variant_or_path  # Assume it's a direct HF ID
            self.variant_key = None  # Ensure variant_key is None

        super().__init__(huggingface_id, torch_dtype=torch_dtype)
        # self.torch_dtype_requested is now set in the base class

        logger.info(f"Initializing Blip2Processor for {self.model_name_or_path}")
        processor_obj = Blip2Processor.from_pretrained(self.model_name_or_path)
        if isinstance(processor_obj, tuple):
            self.processor: Blip2Processor = processor_obj[0]
        else:
            self.processor: Blip2Processor = processor_obj

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

        logger.info(
            f"Initializing Blip2ForConditionalGeneration for {self.model_name_or_path} with dtype: {load_dtype or 'default'}"
        )
        try:
            self.model: Blip2ForConditionalGeneration = (
                Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=load_dtype,  # Use the determined load_dtype
                )
            )
        except Exception as e:
            logger.warning(
                f"Failed to load Blip2ForConditionalGeneration with dtype {load_dtype}: {e}. "
                f"Attempting to load with default dtype."
            )
            self.model: Blip2ForConditionalGeneration = (
                Blip2ForConditionalGeneration.from_pretrained(self.model_name_or_path)
            )  # Fallback to default precision

        logger.info(f"Moving Blip2Model to device: {self.device}")
        self.model.to(torch.device(self.device))  # type: ignore
        # If model loaded with a specific dtype, it's already set.
        # If it fell back, it's in default precision.
        # self.model.to(self.device) handles moving it to the correct device.

    # get_modes() and get_variants() are now inherited from the base class

    def _process_image_and_generate_caption(
        self,
        image: Image.Image,
        prompt_text: Optional[str],
        max_length: int = 75,
        min_length: int = 10,
        num_beams: int = 3,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Process the image and generate a caption using the BLIP-2 model.

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
        if prompt_text:
            inputs = self.processor(image, text=prompt_text, return_tensors="pt")
        else:
            inputs = self.processor(image, return_tensors="pt")

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        for k, v in inputs.items():
            if hasattr(v, "to"):
                # Only cast to float dtype if tensor is floating point, otherwise just move to device
                if torch.is_floating_point(v):
                    inputs[k] = v.to(self.device, dtype=dtype)
                else:
                    inputs[k] = v.to(self.device)

        # Create generation parameters
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

        # Add any additional kwargs
        for k, v in kwargs.items():
            generation_kwargs[k] = v

        generated_ids = self.model.generate(
            **inputs,
            **generation_kwargs,
        )
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()
        return caption
