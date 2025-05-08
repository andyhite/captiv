import os
import sys
import unittest
from unittest.mock import Mock, patch

import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the required model classes

from src.captiv.models.vit_gpt2_model import VitGpt2Model


class TestVitGpt2Model(unittest.TestCase):
    """
    Unit tests for the VitGpt2Model class.
    These tests mock the transformers library components to avoid actual model loading.
    """

    @patch("transformers.models.vit.ViTImageProcessor.from_pretrained")
    @patch("transformers.models.auto.tokenization_auto.AutoTokenizer.from_pretrained")
    @patch(
        "transformers.models.vision_encoder_decoder.VisionEncoderDecoderModel.from_pretrained"
    )
    @patch.dict(
        "src.captiv.models.vit_gpt2_model.VitGpt2Model.VARIANTS",
        {
            "vit-gpt2": {
                "huggingface_id": "nlpconnect/vit-gpt2-image-captioning",
                "description": "ViT+GPT2 model for image captioning",
                "default_mode": "default",
            },
        },
    )
    def setUp(
        self,
        mock_model_from_pretrained,
        mock_tokenizer_from_pretrained,
        mock_processor_from_pretrained,
    ):
        """
        Set up the test environment with mocked components.
        """

        # Create a processor mock that returns an object with a to() method
        class ProcessorOutput(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self["pixel_values"] = torch.tensor([[[1.0, 2.0, 3.0]]])

            def to(self, device):
                # Return self to allow method chaining
                return self

        # Create a processor mock with a __call__ method that returns our ProcessorOutput
        self.mock_processor = Mock()
        self.mock_processor.return_value = ProcessorOutput()
        mock_processor_from_pretrained.return_value = self.mock_processor

        # Create a tokenizer mock for decoding
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.decode.return_value = "This is a test caption"
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer

        # Set up model mock
        self.mock_model = Mock()
        self.mock_model.generate.return_value = torch.tensor([[10, 20, 30]])
        self.mock_model.to.return_value = self.mock_model  # Return self for chaining
        mock_model_from_pretrained.return_value = self.mock_model

        # Create a real test image instead of a mock
        from tests.models.test_utils import create_test_image, get_test_image_path

        self.test_image = create_test_image()
        self.test_image_path = get_test_image_path()

        # Initialize the model with mocks in place
        with patch("torch.device", return_value="cpu"):
            self.vit_gpt2_model = VitGpt2Model(model_variant_or_path="vit-gpt2")

        # Manually set variant_key and default_mode_key for testing purposes
        if self.vit_gpt2_model.variant_key is None:
            self.vit_gpt2_model.variant_key = "vit-gpt2"
        if self.vit_gpt2_model.default_mode_key is None:
            self.vit_gpt2_model.default_mode_key = "default"

    def test_initialization(self):
        """
        Test that the VitGpt2Model initializes correctly with the expected mocks.
        """
        self.assertEqual(
            self.vit_gpt2_model.model_name_or_path,
            "nlpconnect/vit-gpt2-image-captioning",
        )
        self.assertEqual(self.vit_gpt2_model.variant_key, "vit-gpt2")
        self.assertEqual(self.vit_gpt2_model.default_mode_key, "default")

        # Verify the model was moved to device
        self.mock_model.to.assert_called()

    def test_caption_image_with_pil_image_no_prompt(self):
        """
        Test captioning with a PIL image and no prompt.
        """
        # Reset mocks to clear any calls from initialization
        self.mock_processor.reset_mock()
        self.mock_model.reset_mock()

        # Call the method under test
        caption = self.vit_gpt2_model.caption_image(self.test_image)

        # Verify the result
        self.assertEqual(caption, "This is a test caption")

        # Verify the processor was called correctly
        self.mock_processor.assert_called_once()
        args, kwargs = self.mock_processor.call_args
        self.assertEqual(kwargs.get("images"), self.test_image)
        self.assertEqual(kwargs.get("return_tensors"), "pt")

        # Verify generate was called with the expected parameters
        self.mock_model.generate.assert_called_once()

        # Verify decode was called with the expected parameters
        self.mock_tokenizer.decode.assert_called_once()

    def test_caption_image_with_prompt(self):
        """
        Test captioning with a custom prompt.
        """
        # Reset mocks
        self.mock_processor.reset_mock()
        self.mock_model.reset_mock()

        # Call the method with a custom prompt
        caption = self.vit_gpt2_model.caption_image(
            self.test_image, prompt="a detailed description of the image"
        )

        # Verify the result
        self.assertEqual(caption, "This is a test caption")

        # Verify the processor was called
        self.mock_processor.assert_called_once()

        # Verify generate and batch_decode were called
        self.mock_model.generate.assert_called_once()
        self.mock_tokenizer.decode.assert_called_once()

    def test_caption_image_with_mode_key(self):
        """
        Test captioning with a mode key instead of a direct prompt.
        """
        # Reset mocks
        self.mock_processor.reset_mock()
        self.mock_model.reset_mock()

        # Call the method with a mode key
        caption = self.vit_gpt2_model.caption_image(self.test_image, prompt="detailed")

        # Verify the result
        self.assertEqual(caption, "This is a test caption")

        # Verify the processor was called
        self.mock_processor.assert_called_once()

        # Verify generate and batch_decode were called
        self.mock_model.generate.assert_called_once()
        self.mock_tokenizer.decode.assert_called_once()

    def test_caption_image_with_generation_params(self):
        """
        Test captioning with custom generation parameters.
        """
        # Reset mocks
        self.mock_processor.reset_mock()
        self.mock_model.reset_mock()

        # Call the method with custom generation parameters
        caption = self.vit_gpt2_model.caption_image(
            self.test_image,
            max_length=50,
            min_length=5,
            num_beams=5,
            temperature=0.8,
            top_k=30,
            top_p=0.95,
            repetition_penalty=1.2,
        )

        # Verify the result
        self.assertEqual(caption, "This is a test caption")

        # Verify generate was called with the custom parameters
        self.mock_model.generate.assert_called_once()
        _, kwargs = self.mock_model.generate.call_args
        self.assertEqual(kwargs.get("max_length"), 50)
        self.assertEqual(kwargs.get("min_length"), 5)
        self.assertEqual(kwargs.get("num_beams"), 5)
        self.assertEqual(kwargs.get("temperature"), 0.8)
        self.assertEqual(kwargs.get("top_k"), 30)
        self.assertEqual(kwargs.get("top_p"), 0.95)
        self.assertEqual(kwargs.get("repetition_penalty"), 1.2)

    def test_caption_image_with_word_count(self):
        """
        Test captioning with word count parameter.
        Note: With the simplified MODES structure, word_count parameter
        is no longer used to trim the caption.
        """
        # Reset mocks
        self.mock_processor.reset_mock()
        self.mock_model.reset_mock()

        # Mock a caption
        self.mock_tokenizer.decode.return_value = (
            "This is a test caption with more words than allowed"
        )

        # Call the method with word count parameter
        caption = self.vit_gpt2_model.caption_image(
            self.test_image,
            prompt="detailed",
            word_count=3,
        )

        # Verify the result is not trimmed anymore
        self.assertEqual(caption, "This is a test caption with more words than allowed")

    def test_get_modes(self):
        """
        Test that get_modes returns the expected modes.
        """
        modes = VitGpt2Model.get_modes()
        self.assertIn("default", modes)
        self.assertIn("detailed", modes)
        self.assertIn("concise", modes)
        self.assertIn("artistic", modes)
        self.assertIn("technical", modes)

        # Verify the simplified string->string structure
        self.assertEqual(modes["default"], None)
        self.assertEqual(modes["detailed"], "a detailed description of the image")
        self.assertEqual(modes["concise"], "a short description of the image")
        self.assertEqual(modes["artistic"], "an artistic description of the image")
        self.assertEqual(modes["technical"], "a technical description of the image")

    def test_get_variants(self):
        """
        Test that get_variants returns the expected variants.
        """
        variants = VitGpt2Model.get_variants()
        self.assertEqual(
            variants["vit-gpt2"]["huggingface_id"],
            "nlpconnect/vit-gpt2-image-captioning",
        )


if __name__ == "__main__":
    unittest.main()
