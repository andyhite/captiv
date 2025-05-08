from typing import Optional

import torch
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
from transformers.models.auto.processing_auto import AutoProcessor

from captiv.logging import logger

from .base_model import ImageCaptioningModel


class KosmosModel(ImageCaptioningModel):
    """
    Kosmos model for image captioning.
    """

    MODES = {
        "default": None,
        "detailed": "a detailed description of the image",
        "concise": "a short description of the image",
        "artistic": "an artistic description of the image",
        "technical": "a technical description of the image",
    }

    VARIANTS = {
        "kosmos-2": {
            "huggingface_id": "microsoft/kosmos-2-patch14-224",
            "description": "Kosmos-2 model",
            "default_mode": "default",
        },
        "kosmos-2-patch14-224": {
            "huggingface_id": "microsoft/kosmos-2-patch14-224",
            "description": "Kosmos-2 Patch14-224 model",
            "default_mode": "default",
        },
        "kosmos-2.5": {
            "huggingface_id": "microsoft/kosmos-2.5",
            "description": "Kosmos-2.5 model",
            "default_mode": "default",
        },
    }

    # Define default variant
    DEFAULT_VARIANT = "kosmos-2"

    @classmethod
    def get_variants(cls):
        return cls.VARIANTS

    def __init__(
        self,
        model_variant_or_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        from captiv import config

        if model_variant_or_path is None:
            cfg = config.read_config()
            model_variant_or_path = getattr(
                cfg.model, "kosmos_variant", "kosmos-2-patch14-224"
            )

        if model_variant_or_path in self.VARIANTS:
            huggingface_id = self.VARIANTS[model_variant_or_path]["huggingface_id"]
            self.variant_key = model_variant_or_path
            self.default_mode_key = self.VARIANTS[model_variant_or_path].get(
                "default_mode"
            )
        else:
            huggingface_id = model_variant_or_path
            self.variant_key = None
            self.default_mode_key = None

        if not huggingface_id:
            huggingface_id = "microsoft/kosmos-2-patch14-224"
        super().__init__(huggingface_id, torch_dtype=torch_dtype)

        logger.info(f"Initializing Kosmos processor for {self.model_name_or_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        logger.info(f"Initializing Kosmos model for {self.model_name_or_path}")
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name_or_path)

        logger.info(f"Moving KosmosModel to device: {self.device}")
        try:
            self.model = self.model.to(device=self.device)
        except Exception as e:
            logger.warning(f"Could not move KosmosModel to {self.device}: {e}")

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
        # Prepare inputs
        if prompt_text:
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}

        # Generation parameters
        generation_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
        }
        for k, v in kwargs.items():
            generation_kwargs[k] = v

        try:
            outputs = self.model.generate(**inputs, **generation_kwargs)
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[
                0
            ].strip()
            return caption
        except Exception as e:
            logger.error(f"Error generating caption with Kosmos model: {e}")
            return f"Error generating caption: {str(e)}"
