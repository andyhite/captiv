"""
Tests for the caption commands in the Captiv CLI.
"""

from unittest.mock import MagicMock, patch

import pytest
import typer

# Import the command functions directly
from captiv.cli.commands.caption import list as caption_list
from captiv.cli.commands.caption import set as caption_set


@pytest.fixture
def mock_services():
    """Fixture to mock the service classes at the import level."""
    caption_manager_mock = MagicMock()
    model_manager_mock = MagicMock()

    # Setup the caption manager mock
    caption_manager_instance = MagicMock()
    caption_manager_mock.return_value = caption_manager_instance

    # Setup the model manager mock
    model_manager_instance = MagicMock()
    model_manager_mock.return_value = model_manager_instance

    with (
        patch("captiv.cli.commands.caption.list.CaptionManager", caption_manager_mock),
        patch("captiv.cli.commands.caption.set.CaptionManager", caption_manager_mock),
    ):
        yield caption_manager_instance, model_manager_instance


def test_caption_list_empty(mock_services, capsys):
    """Test the caption list command when no images are found."""
    # Setup mock
    caption_manager, _ = mock_services
    caption_manager.list_images_with_captions.return_value = []

    # Run command
    caption_list.command("/fake/path")

    # Check output
    captured = capsys.readouterr()
    assert "No images found" in captured.out
    caption_manager.list_images_with_captions.assert_called_once_with("/fake/path")


def test_caption_list_with_images(mock_services, capsys):
    """Test the caption list command with images."""
    # Setup mock
    caption_manager, _ = mock_services
    caption_manager.list_images_with_captions.return_value = [
        ("image1.jpg", "Caption for image 1"),
        ("image2.jpg", None),
    ]

    # Run command
    caption_list.command("/fake/path")

    # Check output
    captured = capsys.readouterr()
    assert "image1.jpg: Caption for image 1" in captured.out
    assert "image2.jpg: No caption" in captured.out
    caption_manager.list_images_with_captions.assert_called_once_with("/fake/path")


def test_caption_list_error(mock_services, capsys):
    """Test the caption list command when an error occurs."""
    # Setup mock
    caption_manager, _ = mock_services
    caption_manager.list_images_with_captions.side_effect = ValueError(
        "Directory not found"
    )

    # Run command and check for exception
    with pytest.raises(typer.Exit):
        caption_list.command("/fake/path")

    # Check output
    captured = capsys.readouterr()
    assert "Error in list: Directory not found" in captured.err
    caption_manager.list_images_with_captions.assert_called_once_with("/fake/path")


def test_caption_set(mock_services, capsys):
    """Test the caption set command."""
    # Setup mock
    caption_manager, _ = mock_services

    # Create a mock Path object
    mock_path = MagicMock()
    mock_path.name = "image.jpg"
    mock_path.__str__ = MagicMock(return_value="/fake/path/image.jpg")

    # Run command with the mock Path
    with patch("pathlib.Path", return_value=mock_path):
        caption_set.command(mock_path, "Test caption")

    # Check that set_caption was called with the right arguments
    caption_manager.set_caption.assert_called_once_with(
        "/fake/path/image.jpg", "Test caption"
    )


def test_caption_set_error(mock_services, capsys):
    """Test the caption set command when an error occurs."""
    # Setup mock
    caption_manager, _ = mock_services
    caption_manager.set_caption.side_effect = ValueError("Image not found")

    # Run command and check for exception
    with pytest.raises(typer.Exit):
        caption_set.command("/fake/path/image.jpg", "Test caption")

    # Check output
    captured = capsys.readouterr()
    assert "Error in set: Image not found" in captured.err
    caption_manager.set_caption.assert_called_once_with(
        "/fake/path/image.jpg", "Test caption"
    )


# End of caption command tests
