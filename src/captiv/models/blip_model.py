from typing import Optional

import torch
from PIL import Image
from transformers.models.blip import BlipForConditionalGeneration, BlipProcessor

from captiv.logging import logger

from .base_model import ImageCaptioningModel


class BlipModel(ImageCaptioningModel):
    """
    A class to encapsulate the BLIP model for image captioning,
    inheriting from ImageCaptioningModel.
    It relies on the base class for device detection (CUDA, MPS, or CPU).
    """

    processor: BlipProcessor
    model: BlipForConditionalGeneration

    MODES = {
        "default": None,  # None means unconditional captioning
        "detailed": "a detailed description of the image",
        "concise": "a short description of the image",
        "artistic": "an artistic description of the image",
        "technical": "a technical description of the image",
    }

    VARIANTS = {
        "blip-base": {
            "huggingface_id": "Salesforce/blip-image-captioning-base",
            "description": "Salesforce BLIP image captioning base model",
            "default_mode": "default",
        },
        "blip-large": {
            "huggingface_id": "Salesforce/blip-image-captioning-large",
            "description": "Salesforce BLIP image captioning large model",
            "default_mode": "default",
        },
    }

    # Define default variant
    DEFAULT_VARIANT = "blip-large"

    # get_modes() and get_variants() are now inherited from the base class

    def __init__(
        self,
        model_variant_or_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initializes the BlipModel. The device (CUDA, MPS, or CPU) is determined
        by the ImageCaptioningModel base class.

        Args:
            model_variant_or_path (str, optional): A key from VARIANTS or a direct
                                                  Hugging Face model name/path.
                                                  If None, uses the default from config.
            torch_dtype (torch.dtype, optional): The desired torch data type.
                                                 For BLIP, this primarily informs the device choice
                                                 and is stored for informational purposes.
                                                 Actual model precision is typically handled by .to(device).
        """
        # Import here to avoid circular imports
        from captiv import config

        # Get default variant from config if not provided
        if model_variant_or_path is None:
            cfg = config.read_config()
            model_variant_or_path = cfg.model.blip_variant

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
            # self.default_mode_key remains None

        super().__init__(huggingface_id, torch_dtype=torch_dtype)
        # self.torch_dtype_requested is now set in the base class

        logger.info(f"Initializing BlipProcessor for {self.model_name_or_path}")
        processor_result = BlipProcessor.from_pretrained(self.model_name_or_path)
        if isinstance(processor_result, tuple):
            self.processor = processor_result[0]
        else:
            self.processor = processor_result

        logger.info(
            f"Initializing BlipForConditionalGeneration for {self.model_name_or_path}"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name_or_path
        )

        logger.info(f"Moving BlipModel to device: {self.device}")
        # Move model to device and dtype if requested
        try:
            if self.torch_dtype_requested and self.device != "cpu":
                self.model = self.model.to(
                    device=self.device, dtype=self.torch_dtype_requested
                )  # type: ignore
                logger.info(
                    f"BlipModel explicitly cast to {self.torch_dtype_requested} on {self.device}"
                )
            else:
                self.model = self.model.to(device=self.device)  # type: ignore
        except Exception as e:
            logger.warning(
                f"Could not move BlipModel to {self.device} with dtype {self.torch_dtype_requested}: {e}"
            )

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
        Process the image and generate a caption using the BLIP model.

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
            inputs = self.processor(image, text=prompt_text, return_tensors="pt").to(
                self.device
            )
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        tensor_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}

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

        out = self.model.generate(
            **tensor_inputs,
            **generation_kwargs,
        )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
