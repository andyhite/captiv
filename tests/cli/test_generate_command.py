"""
Tests for the generate command in the Captiv CLI.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

# Import the command function directly
from captiv.cli.commands.caption import generate
from captiv.services.model_manager import ModelType


class MockPath:
    """Mock Path class for testing."""

    def __init__(self, path_str):
        self.path_str = path_str

    def is_file(self):
        return True

    def __str__(self):
        return self.path_str


@pytest.fixture
def mock_services():
    """Fixture to mock the service classes at the import level."""
    caption_manager_mock = MagicMock()
    model_manager_mock = MagicMock()

    # Setup the caption manager mock
    caption_manager_instance = MagicMock()
    caption_manager_mock.return_value = caption_manager_instance
    caption_manager_instance.generate_caption.return_value = "Generated caption"

    # Setup the model manager mock
    model_manager_instance = MagicMock()
    model_manager_mock.return_value = model_manager_instance
    model_manager_instance.get_model_class.return_value = MagicMock()

    # We don't need to patch Path since it's not directly imported in generate.py
    with (
        patch(
            "captiv.cli.commands.caption.generate.CaptionManager", caption_manager_mock
        ),
        patch("captiv.cli.commands.caption.generate.ModelManager", model_manager_mock),
    ):
        yield caption_manager_instance, model_manager_instance


def test_generate_single_image(mock_services, capsys):
    """Test generating a caption for a single image."""
    caption_manager, model_manager = mock_services

    # Run the command
    generate.command(
        image_path=Path("/fake/path/image.jpg"),
        model=ModelType.BLIP,
        variant=None,
        mode=None,
        prompt=None,
        max_length=None,
        min_length=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        torch_dtype=None,
        config_file=None,
    )

    # Check that the caption manager was called correctly
    caption_manager.generate_caption.assert_called_once()

    # Check the output
    captured = capsys.readouterr()
    assert "Generated caption" in captured.out


def test_generate_with_model_variant(mock_services, capsys):
    """Test generating a caption with a specific model variant."""
    caption_manager, model_manager = mock_services

    # Run the command with a specific variant
    generate.command(
        image_path=Path("/fake/path/image.jpg"),
        model=ModelType.BLIP,
        variant="blip-large",
        mode=None,
        prompt=None,
        max_length=None,
        min_length=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        torch_dtype=None,
        config_file=None,
    )

    # Check that the caption manager was called correctly
    caption_manager.generate_caption.assert_called_once()

    # Check the output
    captured = capsys.readouterr()
    assert "Generated caption" in captured.out


def test_generate_with_invalid_image(mock_services, capsys):
    """Test generating a caption for an invalid image."""
    caption_manager, model_manager = mock_services

    # Setup caption manager to raise an error
    caption_manager.generate_caption.side_effect = ValueError("Invalid image file")

    # Run the command and expect an error
    with pytest.raises(typer.Exit):
        generate.command(
            image_path=Path("/fake/path/invalid.jpg"),
            model=ModelType.BLIP,
            variant=None,
            mode=None,
            prompt=None,
            max_length=None,
            min_length=None,
            num_beams=None,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            torch_dtype=None,
            config_file=None,
        )

    # Check the error output
    captured = capsys.readouterr()
    assert "Error in generate: Invalid image file" in captured.err


def test_generate_with_all_parameters(mock_services, capsys):
    """Test generating a caption with all parameters specified."""
    caption_manager, model_manager = mock_services

    # Run the command with all parameters
    generate.command(
        image_path=Path("/fake/path/image.jpg"),
        model=ModelType.BLIP,
        variant="blip-large",
        mode="detailed",
        prompt="A custom prompt",
        max_length=50,
        min_length=10,
        num_beams=5,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.2,
        torch_dtype="float16",
        config_file=None,
    )

    # Check that the caption manager was called with all parameters
    caption_manager.generate_caption.assert_called_once()
    call_kwargs = caption_manager.generate_caption.call_args.kwargs

    assert call_kwargs["variant"] == "blip-large"
    assert call_kwargs["prompt"] == "A custom prompt"
    assert call_kwargs["max_length"] == 50
    assert call_kwargs["min_length"] == 10
    assert call_kwargs["num_beams"] == 5
    assert call_kwargs["temperature"] == 0.8
    assert call_kwargs["top_k"] == 40
    assert call_kwargs["top_p"] == 0.95
    assert call_kwargs["repetition_penalty"] == 1.2
    assert call_kwargs["torch_dtype"] == "float16"

    # Check the output
    captured = capsys.readouterr()
    assert "Generated caption" in captured.out


def test_generate_with_mode_validation_error(mock_services, capsys):
    """Test generating a caption with an invalid mode."""
    caption_manager, model_manager = mock_services

    # Setup model manager to raise an error on mode validation
    model_manager.validate_mode.side_effect = ValueError("Invalid mode")

    # Run the command and expect an error
    with pytest.raises(typer.Exit):
        generate.command(
            image_path=Path("/fake/path/image.jpg"),
            model=ModelType.BLIP,
            variant=None,
            mode="invalid_mode",
            prompt=None,
            max_length=None,
            min_length=None,
            num_beams=None,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            torch_dtype=None,
            config_file=None,
        )

    # Check the error output
    captured = capsys.readouterr()
    assert "Error in generate: Invalid mode" in captured.err
