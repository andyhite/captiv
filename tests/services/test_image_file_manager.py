import os
import tempfile

import pytest

from captiv.services.exceptions import (
    DirectoryNotFoundError,
    FileNotFoundError,
    UnsupportedFileTypeError,
)
from captiv.services.image_file_manager import ImageFileManager


class TestImageFileManager:
    def setup_method(self):
        self.manager = ImageFileManager()

    def test_get_supported_extensions(self):
        exts = self.manager.get_supported_extensions()
        assert isinstance(exts, (tuple, set))
        assert ".jpg" in exts
        assert ".png" in exts

    def test_validate_image_file_supported(self):
        for ext in self.manager.get_supported_extensions():
            fname = f"test{ext}"
            with open(fname, "w") as f:
                f.write("dummy")
            try:
                assert self.manager.validate_image_file(fname) is None
            finally:
                os.remove(fname)

    def test_validate_image_file_unsupported(self):
        with pytest.raises(UnsupportedFileTypeError):
            self.manager.validate_image_file("test.txt")

    def test_validate_image_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.manager.validate_image_file("nonexistent.jpg")

    def test_list_images_with_captions_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.manager.list_images_with_captions(tmpdir)
            assert isinstance(result, list)
            assert len(result) == 0

    def test_list_images_with_captions_nonexistent_dir(self):
        with pytest.raises(DirectoryNotFoundError):
            self.manager.list_images_with_captions("/nonexistent/directory")

    def test_list_images_with_captions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some image files
            for i, ext in enumerate(list(self.manager.get_supported_extensions())[:3]):
                img_path = os.path.join(tmpdir, f"image{i}{ext}")
                with open(img_path, "w") as f:
                    f.write("dummy")

                # Add captions to some images
                if i % 2 == 0:
                    cap_path = os.path.join(tmpdir, f"image{i}.txt")
                    with open(cap_path, "w") as f:
                        f.write(f"Caption for image{i}")

            result = self.manager.list_images_with_captions(tmpdir)
            assert isinstance(result, list)
            assert len(result) == 3

            # Check that images with captions have them
            for i, (name, caption) in enumerate(sorted(result)):
                if i % 2 == 0:
                    assert caption is not None
                    assert f"Caption for image{i}" == caption
                else:
                    assert caption is None

    def test_write_caption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "image.jpg")
            cap_path = os.path.join(tmpdir, "image.txt")

            # Create a dummy image file
            with open(img_path, "w") as f:
                f.write("dummy")

            # Write caption
            self.manager.write_caption(img_path, "Test caption")

            # Check caption file exists and content
            assert os.path.exists(cap_path)
            with open(cap_path) as f:
                assert f.read().strip() == "Test caption"

    def test_write_caption_unsupported_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = os.path.join(tmpdir, "document.txt")

            # Create a dummy text file
            with open(txt_path, "w") as f:
                f.write("dummy")

            # Try to write caption to a non-image file
            with pytest.raises(UnsupportedFileTypeError):
                self.manager.write_caption(txt_path, "Test caption")

    def test_read_caption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "image.jpg")
            cap_path = os.path.join(tmpdir, "image.txt")

            # Create a dummy image file
            with open(img_path, "w") as f:
                f.write("dummy")

            # Create a caption file
            with open(cap_path, "w") as f:
                f.write("Test caption")

            # Read caption
            caption = self.manager.read_caption(img_path)
            assert caption == "Test caption"

            # Test reading from an image without a caption
            img2_path = os.path.join(tmpdir, "image2.jpg")
            with open(img2_path, "w") as f:
                f.write("dummy")

            caption = self.manager.read_caption(img2_path)
            assert caption is None
