"""Tests for the captiv.gui.utils module."""

import os
import tempfile
from unittest.mock import patch


def test_utils_module_imports():
    """Test that the utils module can be imported."""


def test_utils_module_functions_exist():
    """Test that all expected functions exist in the utils module."""
    import captiv.gui.utils

    assert hasattr(captiv.gui.utils, "is_image_file")
    assert hasattr(captiv.gui.utils, "get_subdirectories")
    assert hasattr(captiv.gui.utils, "normalize_path")

    assert callable(captiv.gui.utils.is_image_file)
    assert callable(captiv.gui.utils.get_subdirectories)
    assert callable(captiv.gui.utils.normalize_path)


def test_is_image_file_function():
    """Test the is_image_file function."""
    from captiv.gui.utils import is_image_file

    assert is_image_file("test.jpg") is True
    assert is_image_file("test.jpeg") is True
    assert is_image_file("test.png") is True
    assert is_image_file("test.gif") is True
    assert is_image_file("test.bmp") is True
    assert is_image_file("test.webp") is True

    assert is_image_file("test.JPG") is True
    assert is_image_file("test.PNG") is True
    assert is_image_file("test.GIF") is True

    assert is_image_file("test.txt") is False
    assert is_image_file("test.pdf") is False
    assert is_image_file("test.doc") is False
    assert is_image_file("test") is False
    assert is_image_file("") is False

    assert is_image_file("/path/to/image.jpg") is True
    assert is_image_file("/path/to/document.txt") is False

    assert is_image_file("my.file.name.png") is True
    assert is_image_file("my.file.name.txt") is False


def test_get_subdirectories_function():
    """Test the get_subdirectories function."""
    from captiv.gui.utils import get_subdirectories

    with tempfile.TemporaryDirectory() as temp_dir:
        subdir1 = os.path.join(temp_dir, "subdir1")
        subdir2 = os.path.join(temp_dir, "subdir2")
        hidden_dir = os.path.join(temp_dir, ".hidden")

        os.makedirs(subdir1)
        os.makedirs(subdir2)
        os.makedirs(hidden_dir)

        file_path = os.path.join(temp_dir, "file.txt")
        with open(file_path, "w") as f:
            f.write("test")

        subdirs = get_subdirectories(temp_dir)

        assert isinstance(subdirs, list)
        assert "subdir1" in subdirs
        assert "subdir2" in subdirs
        assert ".hidden" not in subdirs
        assert "file.txt" not in subdirs

        assert subdirs == sorted(subdirs)


def test_get_subdirectories_with_nonexistent_directory():
    """Test get_subdirectories with a non-existent directory."""
    from captiv.gui.utils import get_subdirectories

    subdirs = get_subdirectories("/nonexistent/directory")
    assert subdirs == []


def test_get_subdirectories_with_permission_error():
    """Test get_subdirectories when there's a permission error."""
    from captiv.gui.utils import get_subdirectories

    with patch(
        "captiv.gui.utils.os.listdir", side_effect=PermissionError("Permission denied")
    ):
        subdirs = get_subdirectories("/some/directory")
        assert subdirs == []


def test_normalize_path_function():
    """Test the normalize_path function."""
    from captiv.gui.utils import normalize_path

    assert normalize_path("/path/to/file") == "/path/to/file"
    assert normalize_path("relative/path") == "relative/path"
    assert normalize_path("") == ""

    path_dict = {"value": "/path/to/file"}
    assert normalize_path(path_dict) == "/path/to/file"

    path_dict_empty = {"value": ""}
    assert normalize_path(path_dict_empty) == ""

    other_dict = {"path": "/some/path", "name": "file"}
    result = normalize_path(other_dict)
    assert isinstance(result, str)

    assert normalize_path(123) == "123"
    assert normalize_path(None) == "None"

    numeric_dict = {"value": 123}
    assert normalize_path(numeric_dict) == "123"


def test_normalize_path_with_complex_dict():
    """Test normalize_path with more complex dictionary structures."""
    from captiv.gui.utils import normalize_path

    nested_dict = {"value": {"nested": "path"}}
    result = normalize_path(nested_dict)
    assert isinstance(result, str)

    list_dict = {"value": ["/path", "to", "file"]}
    result = normalize_path(list_dict)
    assert isinstance(result, str)


def test_normalize_path_edge_cases():
    """Test normalize_path with edge cases."""
    from captiv.gui.utils import normalize_path

    assert normalize_path(True) == "True"
    assert normalize_path(False) == "False"

    assert normalize_path(3.14) == "3.14"

    assert normalize_path({}) == "{}"

    none_dict = {"value": None}
    assert normalize_path(none_dict) == "None"


@patch("captiv.gui.utils.os.listdir")
@patch("captiv.gui.utils.os.path.isdir")
def test_get_subdirectories_mocked(mock_isdir, mock_listdir):
    """Test get_subdirectories with mocked filesystem operations."""
    from captiv.gui.utils import get_subdirectories

    mock_listdir.return_value = ["dir1", "dir2", ".hidden", "file.txt"]

    def mock_isdir_func(path):
        return not path.endswith("file.txt")

    mock_isdir.side_effect = mock_isdir_func

    subdirs = get_subdirectories("/test/directory")

    mock_listdir.assert_called_once_with("/test/directory")

    assert "dir1" in subdirs
    assert "dir2" in subdirs
    assert ".hidden" not in subdirs
    assert "file.txt" not in subdirs
    assert subdirs == sorted(subdirs)
