"""
ViT-GPT2 model implementation for image captioning.

This module provides a Vision Transformer (ViT) encoder combined with a GPT-2
decoder for image captioning. This architecture uses the ViT model to encode
images into feature representations, which are then decoded into captions
using GPT-2's language generation capabilities.

The model follows an encoder-decoder architecture where:
- Encoder: Vision Transformer (ViT) processes the input image
- Decoder: GPT-2 generates the caption text
"""

from transformers.models.vision_encoder_decoder import VisionEncoderDecoderModel
from transformers.models.vit import ViTImageProcessor
from transformers.tokenization_utils import PreTrainedTokenizer

from .base_model import create_model

VitGPT2Model = create_model(
    model_class=VisionEncoderDecoderModel,
    processor_class=ViTImageProcessor,
    tokenizer_class=PreTrainedTokenizer,
    default_variant="vit-gpt2",
    variants={
        "vit-gpt2": {
            "huggingface_id": "nlpconnect/vit-gpt2-image-captioning",
            "description": "ViT encoder with GPT-2 decoder for image captioning",
            "default_mode": "default",
        },
    },
)
