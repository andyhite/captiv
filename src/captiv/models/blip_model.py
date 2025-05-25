"""
BLIP (Bootstrapping Language-Image Pre-training) model implementation.

This module provides the BLIP model for image captioning, which uses a vision
transformer encoder and a text decoder to generate captions. BLIP is known for its
strong performance on image captioning tasks.
"""

from transformers.models.blip import BlipForConditionalGeneration, BlipProcessor

from .base_model import create_model

BlipModel = create_model(
    model_class=BlipForConditionalGeneration,
    processor_class=BlipProcessor,
    default_variant="blip-large",
    variants={
        "blip-base": {
            "huggingface_id": "Salesforce/blip-image-captioning-base",
            "description": "Salesforce BLIP image captioning base model (129M parameters)",  # noqa: E501
        },
        "blip-large": {
            "huggingface_id": "Salesforce/blip-image-captioning-large",
            "description": "Salesforce BLIP image captioning large model (385M parameters)",  # noqa: E501
        },
    },
)
