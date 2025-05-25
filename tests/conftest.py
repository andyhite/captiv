"""Test configuration and fixtures."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_transformers_and_torch():
    """Mock transformers and torch to prevent model loading and GPU usage."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.return_value = "cpu"
    mock_torch.backends.mps.is_available.return_value = False

    mock_no_grad = MagicMock()
    mock_no_grad.__enter__ = MagicMock(return_value=None)
    mock_no_grad.__exit__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value = mock_no_grad

    mock_torch.float16 = MagicMock()
    mock_torch.float32 = MagicMock()
    mock_torch.bfloat16 = MagicMock()

    mock_torch.is_floating_point.return_value = True
    mock_torch.Tensor = MagicMock()

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor = MagicMock()
    mock_transformers.AutoModelForCausalLM = MagicMock()
    mock_transformers.BlipProcessor = MagicMock()
    mock_transformers.BlipForConditionalGeneration = MagicMock()
    mock_transformers.Blip2Processor = MagicMock()
    mock_transformers.Blip2ForConditionalGeneration = MagicMock()
    mock_transformers.VisionEncoderDecoderModel = MagicMock()
    mock_transformers.ViTImageProcessor = MagicMock()
    mock_transformers.AutoTokenizer = MagicMock()
    mock_transformers.GPT2TokenizerFast = MagicMock()

    mock_generation = MagicMock()
    mock_generation_utils = MagicMock()
    mock_generation_utils.GenerateOutput = MagicMock()
    mock_generation.utils = mock_generation_utils

    mock_image_utils = MagicMock()
    mock_image_utils.ImageInput = MagicMock()

    mock_tokenization_utils = MagicMock()
    mock_tokenization_utils.PreTrainedTokenizer = MagicMock()

    mock_models = MagicMock()

    mock_blip = MagicMock()
    mock_blip.BlipForConditionalGeneration = MagicMock()
    mock_blip.BlipProcessor = MagicMock()
    mock_models.blip = mock_blip

    mock_blip_2 = MagicMock()
    mock_blip_2.Blip2ForConditionalGeneration = MagicMock()
    mock_blip_2.Blip2Processor = MagicMock()
    mock_models.blip_2 = mock_blip_2

    mock_vit_gpt2 = MagicMock()
    mock_kosmos2 = MagicMock()
    mock_llava = MagicMock()
    mock_llava.LlavaForConditionalGeneration = MagicMock()
    mock_llava.LlavaProcessor = MagicMock()

    mock_auto = MagicMock()
    mock_modeling_auto = MagicMock()
    mock_modeling_auto.AutoModelForVision2Seq = MagicMock()
    mock_processing_auto = MagicMock()
    mock_processing_auto.AutoProcessor = MagicMock()
    mock_auto.modeling_auto = mock_modeling_auto
    mock_auto.processing_auto = mock_processing_auto

    mock_vision_encoder_decoder = MagicMock()
    mock_vision_encoder_decoder.VisionEncoderDecoderModel = MagicMock()

    mock_vit = MagicMock()
    mock_vit.ViTImageProcessor = MagicMock()

    mock_gpt2 = MagicMock()
    mock_gpt2.GPT2TokenizerFast = MagicMock()

    mock_models.vit_gpt2 = mock_vit_gpt2
    mock_models.kosmos2 = mock_kosmos2
    mock_models.llava = mock_llava
    mock_models.auto = mock_auto
    mock_models.vision_encoder_decoder = mock_vision_encoder_decoder
    mock_models.vit = mock_vit
    mock_models.gpt2 = mock_gpt2

    mock_accelerate = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "transformers.generation": mock_generation,
            "transformers.generation.utils": mock_generation_utils,
            "transformers.image_utils": mock_image_utils,
            "transformers.tokenization_utils": mock_tokenization_utils,
            "transformers.models": mock_models,
            "transformers.models.blip": mock_blip,
            "transformers.models.blip_2": mock_blip_2,
            "transformers.models.vit_gpt2": mock_vit_gpt2,
            "transformers.models.kosmos2": mock_kosmos2,
            "transformers.models.llava": mock_llava,
            "transformers.models.auto": mock_auto,
            "transformers.models.auto.modeling_auto": mock_modeling_auto,
            "transformers.models.auto.processing_auto": mock_processing_auto,
            "transformers.models.vision_encoder_decoder": mock_vision_encoder_decoder,
            "transformers.models.vit": mock_vit,
            "transformers.models.gpt2": mock_gpt2,
            "accelerate": mock_accelerate,
        },
    ):
        yield
