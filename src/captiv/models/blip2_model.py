"""
BLIP-2 (Bootstrapping Language-Image Pre-training 2) model implementation.

This module provides the BLIP-2 model for image captioning, which improves upon the
original BLIP by using a lightweight Querying Transformer (Q-Former) to bridge the gap
between vision and language models. BLIP-2 achieves better performance with fewer
trainable parameters.
"""

from transformers.models.blip_2 import Blip2ForConditionalGeneration, Blip2Processor

from .base_model import create_model

Blip2Model = create_model(
    model_class=Blip2ForConditionalGeneration,
    processor_class=Blip2Processor,
    default_variant="blip2-opt-2.7b",
    variants={
        "blip2-opt-2.7b": {
            "huggingface_id": "Salesforce/blip2-opt-2.7b",
            "description": "BLIP-2 model with OPT 2.7B language model for balanced performance",  # noqa: E501
        },
        "blip2-opt-6.7b": {
            "huggingface_id": "Salesforce/blip2-opt-6.7b",
            "description": "BLIP-2 model with OPT 6.7B language model for higher quality captions",  # noqa: E501
        },
        "blip2-flan-t5-xl": {
            "huggingface_id": "Salesforce/blip2-flan-t5-xl",
            "description": "BLIP-2 model with Flan-T5-XL language model (often better for VQA tasks)",  # noqa: E501
        },
    },
)
