import os
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.captiv.models.kosmos_model import KosmosModel


class TestKosmosModel(unittest.TestCase):
    """
    Unit tests for the KosmosModel class.
    These tests mock the transformers library components to avoid actual model loading.
    """

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoModelForVision2Seq.from_pretrained")
    @patch.dict(
        "src.captiv.models.kosmos_model.KosmosModel.VARIANTS",
        {
            "kosmos-2": {
                "huggingface_id": "microsoft/kosmos-2-patch14-224",
                "description": "Kosmos-2 base model",
                "default_mode": "default",
            }
        },
    )
    def setUp(self, mock_model_from_pretrained, mock_processor_from_pretrained):
        # Create a processor mock that returns an object with a to() method
        class ProcessorOutput(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self["input_ids"] = torch.tensor([[1, 2, 3]])

            def to(self, device):
                return self

        self.mock_processor = Mock()
        self.mock_processor.return_value = ProcessorOutput()
        self.mock_processor.batch_decode = Mock(return_value=["Kosmos test caption"])
        mock_processor_from_pretrained.return_value = self.mock_processor

        self.mock_model = Mock()
        self.mock_model.generate.return_value = torch.tensor([[10, 20, 30]])
        self.mock_model.to.return_value = self.mock_model
        mock_model_from_pretrained.return_value = self.mock_model

        from tests.models.test_utils import create_test_image, get_test_image_path

        self.test_image = create_test_image()
        self.test_image_path = get_test_image_path()

        self.kosmos_model = KosmosModel(model_variant_or_path="kosmos-2")

        if self.kosmos_model.variant_key is None:
            self.kosmos_model.variant_key = "kosmos-2"
        if self.kosmos_model.default_mode_key is None:
            self.kosmos_model.default_mode_key = "default"

    def test_initialization(self):
        self.assertEqual(
            self.kosmos_model.model_name_or_path, "microsoft/kosmos-2-patch14-224"
        )
        self.assertEqual(self.kosmos_model.variant_key, "kosmos-2")
        self.assertEqual(self.kosmos_model.default_mode_key, "default")
        self.mock_model.to.assert_called()

    def test_caption_image_with_pil_image(self):
        self.mock_processor.reset_mock()
        self.mock_model.reset_mock()
        caption = self.kosmos_model.caption_image(self.test_image)
        self.assertEqual(caption, "Kosmos test caption")
        self.mock_processor.assert_called_once()
        self.mock_model.generate.assert_called_once()
        self.mock_processor.batch_decode.assert_called_once()

    def test_caption_image_with_file_path(self):
        self.mock_processor.reset_mock()
        self.mock_model.reset_mock()
        caption = self.kosmos_model.caption_image(self.test_image_path)
        self.assertEqual(caption, "Kosmos test caption")
        self.mock_processor.assert_called_once()
        args, kwargs = self.mock_processor.call_args
        self.assertEqual(kwargs.get("images").size, self.test_image.size)
        self.assertEqual(kwargs.get("images").mode, self.test_image.mode)
        self.assertIsInstance(kwargs.get("images"), Image.Image)

    def test_caption_image_with_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.kosmos_model.caption_image("nonexistent_image_that_does_not_exist.jpg")

    def test_get_modes(self):
        modes = KosmosModel.get_modes()
        self.assertIn("default", modes)

        # Verify the simplified string->string structure
        self.assertEqual(modes["default"], None)
        self.assertEqual(modes["detailed"], "a detailed description of the image")
        self.assertEqual(modes["concise"], "a short description of the image")
        self.assertEqual(modes["artistic"], "an artistic description of the image")
        self.assertEqual(modes["technical"], "a technical description of the image")

    def test_get_variants(self):
        variants = KosmosModel.get_variants()
        self.assertIn("kosmos-2", variants)
        self.assertEqual(
            variants["kosmos-2"]["huggingface_id"], "microsoft/kosmos-2-patch14-224"
        )


if __name__ == "__main__":
    unittest.main()
