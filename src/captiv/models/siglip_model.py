from typing import Optional

import torch
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModelForImageClassification
from transformers.models.auto.processing_auto import AutoProcessor

from captiv.logging import logger

from .base_model import ImageCaptioningModel


class SigLIPModel(ImageCaptioningModel):
    """
    SigLIP model for image captioning.
    """

    MODES = {
        "default": None,
        "detailed": "a detailed description of the image",
        "concise": "a short description of the image",
        "artistic": "an artistic description of the image",
        "technical": "a technical description of the image",
    }

    VARIANTS = {
        "siglip-base": {
            "huggingface_id": "google/siglip-base-patch16-224",
            "description": "SigLIP base model",
            "default_mode": "default",
        },
        "siglip-base-patch16-224": {
            "huggingface_id": "google/siglip-base-patch16-224",
            "description": "SigLIP base patch16-224 model",
            "default_mode": "default",
        },
        "siglip-so400m-patch14-384": {
            "huggingface_id": "google/siglip-so400m-patch14-384",
            "description": "SigLIP SO400M patch14-384 model",
            "default_mode": "default",
        },
    }

    # Define default variant
    DEFAULT_VARIANT = "siglip-base"

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
                cfg.model, "siglip_variant", "siglip-base-patch16-224"
            )
            if not model_variant_or_path:
                model_variant_or_path = "siglip-base-patch16-224"

        if model_variant_or_path in self.VARIANTS:
            huggingface_id = self.VARIANTS[model_variant_or_path]["huggingface_id"]
            self.variant_key = model_variant_or_path
            self.default_mode_key = self.VARIANTS[model_variant_or_path].get(
                "default_mode"
            )
        else:
            huggingface_id = str(model_variant_or_path)
            self.variant_key = None
            self.default_mode_key = None

        if not huggingface_id:
            huggingface_id = "google/siglip-base-patch16-224"
        super().__init__(huggingface_id, torch_dtype=torch_dtype)

        logger.info(f"Initializing SigLIP processor for {self.model_name_or_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        logger.info(f"Initializing SigLIP model for {self.model_name_or_path}")
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name_or_path
        )

        logger.info(f"Moving SigLIPModel to device: {self.device}")
        try:
            self.model = self.model.to(device=self.device)
        except Exception as e:
            logger.warning(f"Could not move SigLIPModel to {self.device}: {e}")

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
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}

        try:
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            label = self.model.config.id2label.get(str(pred_idx), str(pred_idx))
            return label
        except Exception as e:
            logger.error(f"Error generating caption with SigLIP model: {e}")
            return f"Error generating caption: {str(e)}"
