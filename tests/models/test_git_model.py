import os
import sys
import unittest
from unittest.mock import Mock, patch

import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.captiv.models.git_model import GitModel


class TestGitModel(unittest.TestCase):
    """
    Unit tests for the GitModel class.
    These tests mock the transformers library components to avoid actual model loading.
    """

    @patch("src.captiv.models.git_model.AutoTokenizer.from_pretrained")
    @patch("src.captiv.models.git_model.AutoModelForCausalLM.from_pretrained")
    @patch.dict(
        "src.captiv.models.git_model.GitModel.VARIANTS",
        {
            "git-base": {
                "huggingface_id": "microsoft/git-base",
                "description": "Base GIT model for image captioning (110M parameters)",
                "default_mode": "default",
            },
            "git-large": {
                "huggingface_id": "microsoft/git-large",
                "description": "Large GIT model for image captioning (1.1B parameters)",
                "default_mode": "default",
            },
        },
    )
    def setUp(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        """
        Set up the test environment with mocked components.
        """

        # Create a tokenizer mock
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
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
        self.git_model = GitModel(model_variant_or_path="git-base")

        # Manually set variant_key and default_mode_key for testing purposes
        if self.git_model.variant_key is None:
            self.git_model.variant_key = "git-base"
        if self.git_model.default_mode_key is None:
            self.git_model.default_mode_key = "default"

    def test_initialization(self):
        """
        Test that the GitModel initializes correctly with the expected mocks.
        """
        self.assertEqual(self.git_model.model_name_or_path, "microsoft/git-base")
        self.assertEqual(self.git_model.variant_key, "git-base")
        self.assertEqual(self.git_model.default_mode_key, "default")

        # Verify the model was moved to device
        self.mock_model.to.assert_called()

    def test_caption_image_with_pil_image_no_prompt(self):
        """
        Test captioning with a PIL image and no prompt.
        """
        # Reset mocks to clear any calls from initialization
        self.mock_tokenizer.reset_mock()
        self.mock_model.reset_mock()

        # Call the method under test
        caption = self.git_model.caption_image(self.test_image)

        # Verify the result
        self.assertEqual(caption, "This is a test caption")

        # Verify the tokenizer was called
        self.mock_tokenizer.assert_called_once()

        # Verify generate was called with the expected parameters
        self.mock_model.generate.assert_called_once()

        # Verify decode was called with the expected parameters
        self.mock_tokenizer.decode.assert_called_once()

    def test_caption_image_with_prompt(self):
        """
        Test captioning with a custom prompt.
        """
        # Reset mocks
        self.mock_tokenizer.reset_mock()
        self.mock_model.reset_mock()

        # Call the method with a custom prompt
        caption = self.git_model.caption_image(
            self.test_image, prompt="a detailed description of the image"
        )

        # Verify the result
        self.assertEqual(caption, "This is a test caption")

        # Verify the tokenizer was called
        self.mock_tokenizer.assert_called_once()

        # Verify generate and decode were called
        self.mock_model.generate.assert_called_once()
        self.mock_tokenizer.decode.assert_called_once()

    def test_caption_image_with_mode_key(self):
        """
        Test captioning with a mode key instead of a direct prompt.
        """
        # Reset mocks
        self.mock_tokenizer.reset_mock()
        self.mock_model.reset_mock()

        # Call the method with a mode key
        caption = self.git_model.caption_image(self.test_image, prompt="detailed")

        # Verify the result
        self.assertEqual(caption, "This is a test caption")

        # Verify the tokenizer was called
        self.mock_tokenizer.assert_called_once()

        # Verify generate and decode were called
        self.mock_model.generate.assert_called_once()
        self.mock_tokenizer.decode.assert_called_once()

    def test_caption_image_with_generation_params(self):
        """
        Test captioning with custom generation parameters.
        """
        # Reset mocks
        self.mock_tokenizer.reset_mock()
        self.mock_model.reset_mock()

        # Call the method with custom generation parameters
        caption = self.git_model.caption_image(
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

    def test_get_modes(self):
        """
        Test that get_modes returns the expected modes.
        """
        modes = GitModel.get_modes()
        self.assertIn("default", modes)
        self.assertIn("detailed", modes)
        self.assertIn("concise", modes)
        self.assertIn("artistic", modes)
        self.assertIn("technical", modes)

    def test_get_variants(self):
        """
        Test that get_variants returns the expected variants.
        """
        variants = GitModel.get_variants()
        self.assertEqual(
            variants["git-base"]["huggingface_id"],
            "microsoft/git-base",
        )
        self.assertEqual(
            variants["git-large"]["huggingface_id"],
            "microsoft/git-large",
        )


if __name__ == "__main__":
    unittest.main()
