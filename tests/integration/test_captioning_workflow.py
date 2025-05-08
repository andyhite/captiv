"""
Integration tests for the complete captioning workflow.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from captiv.models.base_model import ImageCaptioningModel
from captiv.services.caption_manager import CaptionManager
from captiv.services.image_file_manager import ImageFileManager
from captiv.services.model_manager import ModelManager, ModelType


def create_test_image(width=100, height=100):
    """Create a test image for testing."""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def test_image_path():
    """Create a temporary test image and return its path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test image
        img_path = os.path.join(tmpdir, "test_image.jpg")
        img = create_test_image()
        img.save(img_path)

        yield img_path


class TestCaptioningWorkflow:
    """Integration tests for the complete captioning workflow."""

    def test_end_to_end_captioning_workflow(self, test_image_path, mock_gpu):
        """
        Test the complete captioning workflow from end to end.

        This test verifies that:
        1. The image file manager can validate the image
        2. The model manager can create a model instance
        3. The model can generate a caption
        4. The caption manager can save the caption
        5. The caption can be read back from the file
        """
        # Create real service instances
        file_manager = ImageFileManager()
        model_manager = ModelManager()
        caption_manager = CaptionManager(file_manager, model_manager)

        # Mock the model instance to avoid actual model loading
        mock_model = MagicMock(spec=ImageCaptioningModel)
        mock_model.caption_image.return_value = "This is a test caption."

        # Mock the model manager's create_model_instance method
        with patch.object(
            model_manager, "create_model_instance", return_value=mock_model
        ):
            # Generate a caption
            caption = caption_manager.generate_caption(
                model_type=ModelType.BLIP,
                image_path=test_image_path,
                variant="blip-large",
                mode=None,
                prompt="Describe this image",
                max_length=50,
                min_length=10,
                num_beams=5,
                temperature=0.8,
                top_k=40,
                top_p=0.95,
                repetition_penalty=1.2,
            )

            # Verify the caption
            assert caption == "This is a test caption."

            # Verify the caption was saved to a file
            caption_file_path = Path(test_image_path).with_suffix(".txt")
            assert caption_file_path.exists()

            # Read the caption from the file
            with open(caption_file_path, "r") as f:
                saved_caption = f.read().strip()

            # Verify the saved caption
            assert saved_caption == "This is a test caption."

    def test_multiple_model_captioning(self, test_image_path, mock_gpu):
        """
        Test captioning with multiple models.

        This test verifies that:
        1. Different models can be used to generate captions
        2. The captions are correctly saved and retrieved
        """
        # Create real service instances
        file_manager = ImageFileManager()
        model_manager = ModelManager()
        caption_manager = CaptionManager(file_manager, model_manager)

        # Test with different models
        model_types = [ModelType.BLIP, ModelType.BLIP2, ModelType.GIT]
        expected_captions = {
            ModelType.BLIP: "Caption from BLIP model.",
            ModelType.BLIP2: "Caption from BLIP2 model.",
            ModelType.GIT: "Caption from GIT model.",
        }

        for model_type in model_types:
            # Mock the model instance
            mock_model = MagicMock(spec=ImageCaptioningModel)
            mock_model.caption_image.return_value = expected_captions[model_type]

            # Mock the model manager's create_model_instance method
            with patch.object(
                model_manager, "create_model_instance", return_value=mock_model
            ):
                # Generate a caption
                caption = caption_manager.generate_caption(
                    model_type=model_type, image_path=test_image_path
                )

                # Verify the caption
                assert caption == expected_captions[model_type]

                # Verify the caption was saved to a file
                caption_file_path = Path(test_image_path).with_suffix(".txt")
                assert caption_file_path.exists()

                # Read the caption from the file
                with open(caption_file_path, "r") as f:
                    saved_caption = f.read().strip()

                # Verify the saved caption
                assert saved_caption == expected_captions[model_type]

    def test_captioning_with_different_modes(self, test_image_path, mock_gpu):
        """
        Test captioning with different modes.

        This test verifies that:
        1. Different modes can be used to generate captions
        2. The mode is correctly passed to the model
        """
        # Create real service instances
        file_manager = ImageFileManager()
        model_manager = ModelManager()
        caption_manager = CaptionManager(file_manager, model_manager)

        # Mock the model instance
        mock_model = MagicMock(spec=ImageCaptioningModel)
        mock_model.caption_image.return_value = "Caption with specific mode."

        # Mock the model manager's create_model_instance method
        with patch.object(
            model_manager, "create_model_instance", return_value=mock_model
        ):
            # Generate a caption with a specific mode
            caption = caption_manager.generate_caption(
                model_type=ModelType.BLIP, image_path=test_image_path, mode="detailed"
            )

            # Verify the caption
            assert caption == "Caption with specific mode."

            # Verify the mode was passed to the model
            mock_model.caption_image.assert_called_once()
            args, kwargs = mock_model.caption_image.call_args
            assert kwargs.get("prompt") == "detailed"

    def test_captioning_with_custom_prompt(self, test_image_path, mock_gpu):
        """
        Test captioning with a custom prompt.

        This test verifies that:
        1. A custom prompt can be used to generate captions
        2. The prompt is correctly passed to the model
        """
        # Create real service instances
        file_manager = ImageFileManager()
        model_manager = ModelManager()
        caption_manager = CaptionManager(file_manager, model_manager)

        # Mock the model instance
        mock_model = MagicMock(spec=ImageCaptioningModel)
        mock_model.caption_image.return_value = "Caption with custom prompt."

        # Mock the model manager's create_model_instance method
        with patch.object(
            model_manager, "create_model_instance", return_value=mock_model
        ):
            # Generate a caption with a custom prompt
            custom_prompt = "Describe this image in detail"
            caption = caption_manager.generate_caption(
                model_type=ModelType.BLIP,
                image_path=test_image_path,
                prompt=custom_prompt,
            )

            # Verify the caption
            assert caption == "Caption with custom prompt."

            # Verify the prompt was passed to the model
            mock_model.caption_image.assert_called_once()
            args, kwargs = mock_model.caption_image.call_args
            assert kwargs.get("prompt") == custom_prompt

    def test_captioning_with_generation_params(self, test_image_path, mock_gpu):
        """
        Test captioning with custom generation parameters.

        This test verifies that:
        1. Custom generation parameters can be used to generate captions
        2. The parameters are correctly passed to the model
        """
        # Create real service instances
        file_manager = ImageFileManager()
        model_manager = ModelManager()
        caption_manager = CaptionManager(file_manager, model_manager)

        # Mock the model instance
        mock_model = MagicMock(spec=ImageCaptioningModel)
        mock_model.caption_image.return_value = (
            "Caption with custom generation parameters."
        )

        # Mock the model manager's create_model_instance method
        with patch.object(
            model_manager, "create_model_instance", return_value=mock_model
        ):
            # Generate a caption with custom generation parameters
            caption = caption_manager.generate_caption(
                model_type=ModelType.BLIP,
                image_path=test_image_path,
                max_length=100,
                min_length=20,
                num_beams=10,
                temperature=0.5,
                top_k=50,
                top_p=0.8,
                repetition_penalty=1.5,
            )

            # Verify the caption
            assert caption == "Caption with custom generation parameters."

            # Verify the generation parameters were passed to the model
            mock_model.caption_image.assert_called_once()
            args, kwargs = mock_model.caption_image.call_args
            assert kwargs.get("max_length") == 100
            assert kwargs.get("min_length") == 20
            assert kwargs.get("num_beams") == 10
            assert kwargs.get("temperature") == 0.5
            assert kwargs.get("top_k") == 50
            assert kwargs.get("top_p") == 0.8
            assert kwargs.get("repetition_penalty") == 1.5
