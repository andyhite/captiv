"""
Tests for the config commands in the Captiv CLI.
"""

from unittest.mock import MagicMock, patch

import pytest

# Import the command functions directly
from captiv.cli.commands.config import get
from captiv.cli.commands.config import list as config_list
from captiv.cli.commands.config import set


@pytest.fixture
def mock_config():
    """Fixture to mock the config module."""
    # Create a mock AppConfig object
    mock_app_config = MagicMock()

    # Set up the model attribute
    mock_model = MagicMock()
    mock_model.default_model = "blip"
    mock_app_config.model = mock_model

    # Set up to_dict method to return a dictionary with our test data
    mock_app_config.to_dict.return_value = {
        "model": {
            "default_model": "blip",
            "blip_variant": "blip-large",
            "blip_mode": "detailed",
        }
    }

    # Create a mock for the config module
    with (
        patch("captiv.config.read_config", return_value=mock_app_config),
        patch("captiv.config.write_config") as mock_write,
    ):
        yield (
            {
                "default_mode": "detailed",
                "default_variant": "blip-large",
                "model": "blip",
            },
            mock_write,
        )


def test_config_get(mock_config, capsys):
    """Test the config get command."""
    mock_config_data, _ = mock_config

    # Test getting a specific config value
    get.command("model")

    # Check output
    captured = capsys.readouterr()
    assert "model=blip" in captured.out


def test_config_get_nonexistent(mock_config, capsys):
    """Test the config get command with a nonexistent key."""
    mock_config_data, _ = mock_config

    # Test getting a nonexistent config value
    get.command("nonexistent_key")

    # Check output
    captured = capsys.readouterr()
    assert "nonexistent_key=None" in captured.out


def test_config_set(mock_config, capsys):
    """Test the config set command."""
    mock_config_data, mock_write = mock_config

    # Test setting a config value
    set.command("model=blip2")

    # Check that write_config was called

    # Check that write_config was called
    mock_write.assert_called_once()


def test_config_set_new_key(mock_config, capsys):
    """Test the config set command with a new key."""
    mock_config_data, mock_write = mock_config

    # Test setting a new config value
    # We don't need to check if write_config was called since it's a legacy key
    # and the code just shows a warning
    set.command("new_key=new_value")

    # Check output
    captured = capsys.readouterr()
    assert "Warning: Legacy configuration key 'new_key' is deprecated" in captured.out


def test_config_list(mock_config, capsys):
    """Test the config list command."""
    mock_config_data, _ = mock_config

    # Test listing all config values with JSON format
    config_list.command(section=None, json_format=True)

    # Check output - we're using JSON format so we can check for specific values
    captured = capsys.readouterr()
    assert "model" in captured.out
    assert "default_model" in captured.out
    assert "blip" in captured.out
    assert "blip_variant" in captured.out
    assert "blip-large" in captured.out
