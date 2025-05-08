import os
import tempfile
from unittest.mock import MagicMock

import pytest

from captiv.services.caption_manager import CaptionManager
from captiv.services.exceptions import UnsupportedFileTypeError
from captiv.services.image_file_manager import ImageFileManager
from captiv.services.model_manager import ModelManager, ModelType


class TestCaptionManager:
    def setup_method(self):
        self.file_manager = ImageFileManager()
        self.model_manager = ModelManager()
        self.manager = CaptionManager(
            file_manager=self.file_manager, model_manager=self.model_manager
        )

    def test_get_supported_extensions(self):
        exts = self.manager.get_supported_extensions()
        assert isinstance(exts, (tuple, set))
        assert ".jpg" in exts or ".png" in exts

    def test_validate_image_file_supported(self):
        for ext in self.file_manager.get_supported_extensions():
            fname = f"test{ext}"
            with open(fname, "w") as f:
                f.write("dummy")
            try:
                assert self.file_manager.validate_image_file(fname) is None
            finally:
                os.remove(fname)

    def test_validate_image_file_unsupported(self):
        with pytest.raises(UnsupportedFileTypeError):
            self.file_manager.validate_image_file("test.txt")

    def test_set_and_list_captions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "image.jpg")
            cap_path = os.path.join(tmpdir, "image.txt")
            # Create a dummy image file
            with open(img_path, "w") as f:
                f.write("dummy")
            # Set caption
            self.manager.set_caption(img_path, "hello world")
            # Check caption file exists and content
            assert os.path.exists(cap_path)
            with open(cap_path) as f:
                assert f.read().strip() == "hello world"
            # List images with captions
            result = self.manager.list_images_with_captions(tmpdir)
            assert isinstance(result, list)
            # Print result for debugging
            print("DEBUG: list_images_with_captions result:", result)
            assert any(
                isinstance(r, tuple)
                and len(r) == 2
                and r[0] == "image.jpg"
                and r[1] == "hello world"
                for r in result
            )

    def test_generate_caption(self):
        # Mock the file manager's validate_image_file method
        self.file_manager.validate_image_file = MagicMock()

        # Mock the model manager's create_model_instance method
        dummy_model = MagicMock()
        dummy_model.caption_image.return_value = "generated caption"
        self.model_manager.create_model_instance = MagicMock(return_value=dummy_model)

        # Mock the model manager's build_generation_params method
        self.model_manager.build_generation_params = MagicMock(return_value={})

        caption = self.manager.generate_caption(
            model_type=ModelType.BLIP,
            image_path="fake.jpg",
            variant=None,
            mode=None,
            prompt=None,
            max_length=10,
            min_length=5,
            num_beams=1,
            temperature=1.0,
            top_k=5,
            top_p=0.9,
            repetition_penalty=1.0,
            torch_dtype=None,
        )

        # Verify the file manager's validate_image_file method was called
        self.file_manager.validate_image_file.assert_called_once_with("fake.jpg")

        # Verify the model manager's create_model_instance method was called
        self.model_manager.create_model_instance.assert_called_once_with(
            ModelType.BLIP, None, None
        )

        # Verify the model manager's build_generation_params method was called
        self.model_manager.build_generation_params.assert_called_once_with(
            10, 5, 1, 1.0, 5, 0.9, 1.0
        )

        # Verify the model's caption_image method was called
        dummy_model.caption_image.assert_called_once()

        assert caption == "generated caption"
