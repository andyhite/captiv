"""Tests for ImageFileManager service."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from captiv.services.file_manager import FileManager
from captiv.services.image_file_manager import ImageFileManager


class TestImageFileManager:
    """Test cases for ImageFileManager."""

    @pytest.fixture
    def mock_file_manager(self):
        """Create a mock FileManager."""
        return MagicMock(spec=FileManager)

    @pytest.fixture
    def image_file_manager(self, mock_file_manager):
        """Create an ImageFileManager instance with mock FileManager."""
        return ImageFileManager(mock_file_manager)

    def test_initialization(self, mock_file_manager):
        """Test ImageFileManager initialization."""
        manager = ImageFileManager(mock_file_manager)
        assert manager.file_manager is mock_file_manager

    def test_supported_extensions(self):
        """Test that SUPPORTED_EXTENSIONS contains expected image formats."""
        expected_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        }
        assert expected_extensions == ImageFileManager.SUPPORTED_EXTENSIONS

    def test_list_image_files(self, image_file_manager, mock_file_manager):
        """Test listing image files in a directory."""
        test_directory = Path("/test/directory")
        expected_files = [
            Path("/test/directory/image1.jpg"),
            Path("/test/directory/image2.png"),
            Path("/test/directory/image3.gif"),
        ]
        mock_file_manager.list_files.return_value = expected_files

        result = image_file_manager.list_image_files(test_directory)

        assert result == expected_files
        mock_file_manager.list_files.assert_called_once_with(
            test_directory, ImageFileManager.SUPPORTED_EXTENSIONS
        )

    def test_list_image_files_empty_directory(
        self, image_file_manager, mock_file_manager
    ):
        """Test listing image files in an empty directory."""
        test_directory = Path("/empty/directory")
        mock_file_manager.list_files.return_value = []

        result = image_file_manager.list_image_files(test_directory)

        assert result == []
        mock_file_manager.list_files.assert_called_once_with(
            test_directory, ImageFileManager.SUPPORTED_EXTENSIONS
        )

    def test_list_image_files_with_pathlib_path(
        self, image_file_manager, mock_file_manager
    ):
        """Test listing image files with Path object."""
        test_directory = Path("/test/path/object")
        expected_files = [Path("/test/path/object/photo.jpeg")]
        mock_file_manager.list_files.return_value = expected_files

        result = image_file_manager.list_image_files(test_directory)

        assert result == expected_files
        mock_file_manager.list_files.assert_called_once_with(
            test_directory, ImageFileManager.SUPPORTED_EXTENSIONS
        )

    def test_list_image_files_delegates_to_file_manager(
        self, image_file_manager, mock_file_manager
    ):
        """Test that list_image_files properly delegates to FileManager."""
        test_directory = Path("/delegation/test")

        image_file_manager.list_image_files(test_directory)

        mock_file_manager.list_files.assert_called_once_with(
            test_directory, ImageFileManager.SUPPORTED_EXTENSIONS
        )

    def test_supported_extensions_immutable(self):
        """Test that SUPPORTED_EXTENSIONS is a set (immutable for our purposes)."""
        extensions = ImageFileManager.SUPPORTED_EXTENSIONS
        assert isinstance(extensions, set)

        assert ".jpg" in extensions
        assert ".png" in extensions
        assert ".gif" in extensions

    def test_class_attribute_access(self):
        """Test accessing SUPPORTED_EXTENSIONS as class attribute."""
        extensions = ImageFileManager.SUPPORTED_EXTENSIONS
        assert len(extensions) == 7

    def test_instance_method_coverage(self, mock_file_manager):
        """Test complete method coverage including __init__."""
        manager = ImageFileManager(mock_file_manager)

        test_dir = Path("/test")
        mock_file_manager.list_files.return_value = []
        result = manager.list_image_files(test_dir)

        assert result == []
