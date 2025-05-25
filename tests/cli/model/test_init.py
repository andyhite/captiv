"""Tests for model CLI __init__ module."""

from unittest.mock import patch

from captiv.cli.commands.model import model_command_handler
from captiv.services.model_manager import ModelType


class TestModelInit:
    """Test model CLI __init__ module."""

    def test_model_command_handler_creates_function(self):
        """Test that model_command_handler creates a proper command function."""
        command_func = model_command_handler(ModelType.BLIP)

        assert callable(command_func)

        assert command_func.__name__ == "blip_command"
        assert "Display information about the blip model." in command_func.__doc__

    @patch("captiv.cli.commands.model.show.command")
    def test_model_command_handler_calls_show_command(self, mock_show_command):
        """Test that the generated command function calls show.command."""
        command_func = model_command_handler(ModelType.BLIP2)

        command_func()

        mock_show_command.assert_called_once_with("blip2")

    def test_model_command_handler_different_model_types(self):
        """Test model_command_handler with different model types."""
        for model_type in ModelType:
            command_func = model_command_handler(model_type)

            expected_name = f"{model_type.value}_command"
            assert command_func.__name__ == expected_name

            expected_doc = f"Display information about the {model_type.value} model."
            assert command_func.__doc__ == expected_doc
