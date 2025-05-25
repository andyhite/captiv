"""
Kosmos model implementation for image captioning.

This module provides the Kosmos models from Microsoft, which are multimodal large
language models capable of perceiving general modalities, learning in context, and
following instructions. Kosmos models excel at various vision-language tasks including
image captioning.
"""

from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
from transformers.models.auto.processing_auto import AutoProcessor

from .base_model import create_model

KosmosModel = create_model(
    model_class=AutoModelForVision2Seq,
    processor_class=AutoProcessor,
    default_variant="kosmos-2.5",
    variants={
        "kosmos-2": {
            "huggingface_id": "microsoft/kosmos-2-patch14-224",
            "description": "Kosmos-2 multimodal model with grounding capabilities",
        },
        "kosmos-2.5": {
            "huggingface_id": "microsoft/kosmos-2.5",
            "description": "Kosmos-2.5 multimodal literate model with enhanced capabilities",  # noqa: E501
        },
    },
)
