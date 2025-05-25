"""Tests for CaptionFileManager service."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from captiv.services.caption_file_manager import CaptionFileManager
from captiv.services.exceptions import FileOperationError
from captiv.services.file_manager import FileManager
from captiv.services.image_file_manager import ImageFileManager


class TestCaptionFileManager:
    """Test cases for CaptionFileManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        self.mock_file_manager = MagicMock(spec=FileManager)
        self.mock_image_file_manager = MagicMock(spec=ImageFileManager)

        self.caption_manager = CaptionFileManager(
            file_manager=self.mock_file_manager,
            image_file_manager=self.mock_image_file_manager,
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        for file in self.temp_path.rglob("*"):
            if file.is_file():
                file.unlink()
        self.temp_path.rmdir()

    def test_init_with_dependencies(self):
        """Test CaptionFileManager initialization with dependencies."""
        file_manager = MagicMock(spec=FileManager)
        image_file_manager = MagicMock(spec=ImageFileManager)

        manager = CaptionFileManager(file_manager, image_file_manager)

        assert manager.file_manager == file_manager
        assert manager.image_file_manager == image_file_manager

    def test_get_caption_file_path_jpg_extension(self):
        """Test getting caption file path for JPG image."""
        image_path = self.temp_path / "image.jpg"
        caption_path = self.caption_manager.get_caption_file_path(image_path)

        expected_path = self.temp_path / "image.txt"
        assert caption_path == expected_path

    def test_get_caption_file_path_different_extensions(self):
        """Test getting caption file path for images with different extensions."""
        test_cases = [
            ("image.jpg", "image.txt"),
            ("photo.png", "photo.txt"),
            ("picture.jpeg", "picture.txt"),
            ("file.TIFF", "file.txt"),
            ("image.webp", "image.txt"),
        ]

        for image_name, expected_caption_name in test_cases:
            image_path = self.temp_path / image_name
            caption_path = self.caption_manager.get_caption_file_path(image_path)
            expected_path = self.temp_path / expected_caption_name
            assert caption_path == expected_path

    def test_get_caption_file_path_no_extension(self):
        """Test getting caption file path for file without extension."""
        image_path = self.temp_path / "image_no_ext"
        caption_path = self.caption_manager.get_caption_file_path(image_path)

        expected_path = self.temp_path / "image_no_ext.txt"
        assert caption_path == expected_path

    def test_get_caption_file_path_nested_directory(self):
        """Test getting caption file path for image in nested directory."""
        nested_dir = self.temp_path / "subdir" / "nested"
        image_path = nested_dir / "image.jpg"
        caption_path = self.caption_manager.get_caption_file_path(image_path)

        expected_path = nested_dir / "image.txt"
        assert caption_path == expected_path

    def test_read_caption_success(self):
        """Test reading caption successfully."""
        image_path = self.temp_path / "image.jpg"
        expected_caption = "A beautiful sunset over the mountains"

        self.mock_file_manager.read_file.return_value = expected_caption

        result = self.caption_manager.read_caption(image_path)

        assert result == expected_caption
        expected_caption_path = self.temp_path / "image.txt"
        self.mock_file_manager.read_file.assert_called_once_with(expected_caption_path)

    def test_read_caption_file_not_found(self):
        """Test reading caption when file not found."""
        image_path = self.temp_path / "image.jpg"

        self.mock_file_manager.read_file.side_effect = FileNotFoundError(
            "File not found"
        )

        result = self.caption_manager.read_caption(image_path)

        assert result == ""

    def test_read_caption_permission_error(self):
        """Test reading caption when permission error occurs."""
        image_path = self.temp_path / "image.jpg"

        self.mock_file_manager.read_file.side_effect = PermissionError(
            "Permission denied"
        )

        result = self.caption_manager.read_caption(image_path)

        assert result == ""

    def test_read_caption_general_exception(self):
        """Test reading caption when general exception occurs."""
        image_path = self.temp_path / "image.jpg"

        self.mock_file_manager.read_file.side_effect = Exception("General error")

        result = self.caption_manager.read_caption(image_path)

        assert result == ""

    def test_write_caption_success(self):
        """Test writing caption successfully."""
        image_path = self.temp_path / "image.jpg"
        caption_content = "A new caption for the image"

        self.caption_manager.write_caption(image_path, caption_content)

        expected_caption_path = self.temp_path / "image.txt"
        self.mock_file_manager.write_file.assert_called_once_with(
            expected_caption_path, caption_content
        )

    def test_write_caption_empty_content(self):
        """Test writing empty caption."""
        image_path = self.temp_path / "image.jpg"

        self.caption_manager.write_caption(image_path, "")

        expected_caption_path = self.temp_path / "image.txt"
        self.mock_file_manager.write_file.assert_called_once_with(
            expected_caption_path, ""
        )

    def test_write_caption_unicode_content(self):
        """Test writing caption with Unicode content."""
        image_path = self.temp_path / "image.jpg"
        caption_content = "Unicode caption with 世界 and émojis 🌍"

        self.caption_manager.write_caption(image_path, caption_content)

        expected_caption_path = self.temp_path / "image.txt"
        self.mock_file_manager.write_file.assert_called_once_with(
            expected_caption_path, caption_content
        )

    def test_write_caption_file_operation_error(self):
        """Test writing caption when file operation fails."""
        image_path = self.temp_path / "image.jpg"
        caption_content = "Test caption"

        self.mock_file_manager.write_file.side_effect = Exception("Write failed")

        with pytest.raises(FileOperationError, match="Failed to write caption"):
            self.caption_manager.write_caption(image_path, caption_content)

    def test_write_caption_permission_error(self):
        """Test writing caption when permission error occurs."""
        image_path = self.temp_path / "image.jpg"
        caption_content = "Test caption"

        self.mock_file_manager.write_file.side_effect = PermissionError(
            "Permission denied"
        )

        with pytest.raises(FileOperationError, match="Failed to write caption"):
            self.caption_manager.write_caption(image_path, caption_content)

    def test_delete_caption_success(self):
        """Test deleting caption successfully."""
        image_path = self.temp_path / "image.jpg"

        self.caption_manager.delete_caption(image_path)

        expected_caption_path = self.temp_path / "image.txt"
        self.mock_file_manager.delete_file.assert_called_once_with(
            expected_caption_path
        )

    def test_delete_caption_file_operation_error(self):
        """Test deleting caption when file operation fails."""
        image_path = self.temp_path / "image.jpg"

        self.mock_file_manager.delete_file.side_effect = Exception("Delete failed")

        with pytest.raises(FileOperationError, match="Failed to delete caption"):
            self.caption_manager.delete_caption(image_path)

    def test_delete_caption_permission_error(self):
        """Test deleting caption when permission error occurs."""
        image_path = self.temp_path / "image.jpg"

        self.mock_file_manager.delete_file.side_effect = PermissionError(
            "Permission denied"
        )

        with pytest.raises(FileOperationError, match="Failed to delete caption"):
            self.caption_manager.delete_caption(image_path)

    def test_clear_captions_success(self):
        """Test clearing all captions successfully."""
        directory = self.temp_path
        image_files = [
            self.temp_path / "image1.jpg",
            self.temp_path / "image2.png",
            self.temp_path / "image3.gif",
        ]

        self.mock_image_file_manager.list_image_files.return_value = image_files

        self.caption_manager.clear_captions(directory)

        self.mock_image_file_manager.list_image_files.assert_called_once_with(directory)

        expected_calls = [
            (self.temp_path / "image1.txt",),
            (self.temp_path / "image2.txt",),
            (self.temp_path / "image3.txt",),
        ]

        actual_calls = [
            call[0][0] for call in self.mock_file_manager.delete_file.call_args_list
        ]
        assert len(actual_calls) == 3
        for expected_path in [call[0] for call in expected_calls]:
            assert expected_path in actual_calls

    def test_clear_captions_empty_directory(self):
        """Test clearing captions in directory with no images."""
        directory = self.temp_path

        self.mock_image_file_manager.list_image_files.return_value = []

        self.caption_manager.clear_captions(directory)

        self.mock_image_file_manager.list_image_files.assert_called_once_with(directory)
        self.mock_file_manager.delete_file.assert_not_called()

    def test_clear_captions_list_images_error(self):
        """Test clearing captions when listing images fails."""
        directory = self.temp_path

        self.mock_image_file_manager.list_image_files.side_effect = Exception(
            "List failed"
        )

        with pytest.raises(FileOperationError, match="Failed to clear captions"):
            self.caption_manager.clear_captions(directory)

    def test_clear_captions_delete_error(self):
        """Test clearing captions when delete operation fails."""
        directory = self.temp_path
        image_files = [self.temp_path / "image1.jpg"]

        self.mock_image_file_manager.list_image_files.return_value = image_files
        self.mock_file_manager.delete_file.side_effect = Exception("Delete failed")

        with pytest.raises(FileOperationError, match="Failed to clear captions"):
            self.caption_manager.clear_captions(directory)

    def test_list_images_and_captions_success(self):
        """Test listing images and captions successfully."""
        directory = self.temp_path
        image_files = [
            self.temp_path / "image1.jpg",
            self.temp_path / "image2.png",
        ]

        self.mock_image_file_manager.list_image_files.return_value = image_files

        def mock_read_caption(image_path):
            if "image1" in str(image_path):
                return "Caption for image 1"
            elif "image2" in str(image_path):
                return "Caption for image 2"
            return ""

        with patch.object(
            self.caption_manager, "read_caption", side_effect=mock_read_caption
        ):
            result = self.caption_manager.list_images_and_captions(directory)

        expected_result = [
            (self.temp_path / "image1.jpg", "Caption for image 1"),
            (self.temp_path / "image2.png", "Caption for image 2"),
        ]

        assert result == expected_result
        self.mock_image_file_manager.list_image_files.assert_called_once_with(directory)

    def test_list_images_and_captions_empty_directory(self):
        """Test listing images and captions in empty directory."""
        directory = self.temp_path

        self.mock_image_file_manager.list_image_files.return_value = []

        result = self.caption_manager.list_images_and_captions(directory)

        assert result == []
        self.mock_image_file_manager.list_image_files.assert_called_once_with(directory)

    def test_list_images_and_captions_with_missing_captions(self):
        """Test listing images and captions when some captions are missing."""
        directory = self.temp_path
        image_files = [
            self.temp_path / "image1.jpg",
            self.temp_path / "image2.png",
        ]

        self.mock_image_file_manager.list_image_files.return_value = image_files

        def mock_read_caption(image_path):
            if "image1" in str(image_path):
                return "Caption for image 1"
            return ""

        with patch.object(
            self.caption_manager, "read_caption", side_effect=mock_read_caption
        ):
            result = self.caption_manager.list_images_and_captions(directory)

        expected_result = [
            (self.temp_path / "image1.jpg", "Caption for image 1"),
            (self.temp_path / "image2.png", ""),
        ]

        assert result == expected_result

    def test_integration_workflow(self):
        """Test complete workflow of caption operations."""
        image_path = self.temp_path / "workflow_image.jpg"
        caption_content = "Workflow test caption"

        self.caption_manager.write_caption(image_path, caption_content)
        expected_caption_path = self.temp_path / "workflow_image.txt"
        self.mock_file_manager.write_file.assert_called_with(
            expected_caption_path, caption_content
        )

        self.mock_file_manager.read_file.return_value = caption_content
        result = self.caption_manager.read_caption(image_path)
        assert result == caption_content

        self.caption_manager.delete_caption(image_path)
        self.mock_file_manager.delete_file.assert_called_with(expected_caption_path)

    def test_path_handling_with_pathlib(self):
        """Test that methods work correctly with pathlib.Path objects."""
        image_path = Path(self.temp_path / "pathlib_test.jpg")
        caption_content = "Pathlib test caption"

        self.caption_manager.write_caption(image_path, caption_content)

        expected_caption_path = Path(self.temp_path / "pathlib_test.txt")
        self.mock_file_manager.write_file.assert_called_with(
            expected_caption_path, caption_content
        )

    def test_get_caption_file_path_preserves_directory_structure(self):
        """Test that caption file path preserves directory structure."""
        nested_path = self.temp_path / "deep" / "nested" / "structure" / "image.jpg"
        caption_path = self.caption_manager.get_caption_file_path(nested_path)

        expected_path = self.temp_path / "deep" / "nested" / "structure" / "image.txt"
        assert caption_path == expected_path
        assert caption_path.parent == nested_path.parent

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across methods."""
        image_path = self.temp_path / "error_test.jpg"

        test_cases = [
            (
                lambda: self.caption_manager.write_caption(image_path, "test"),
                "write_file",
            ),
            (lambda: self.caption_manager.delete_caption(image_path), "delete_file"),
        ]

        for operation, mock_method in test_cases:
            getattr(self.mock_file_manager, mock_method).side_effect = Exception(
                "Test error"
            )

            with pytest.raises(FileOperationError):
                operation()

            getattr(self.mock_file_manager, mock_method).side_effect = None
