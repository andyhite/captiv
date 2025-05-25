"""Tests for config list CLI command."""

import json
from unittest.mock import MagicMock, patch


class TestConfigList:
    """Test the config list command."""

    @patch("captiv.cli.commands.config.list.ConfigManager")
    def test_config_list_all(self, mock_config_manager, runner, config_app):
        """Test listing all config values in normal and JSON formats."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_config.return_value = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"key3": "value3"},
        }

        result = runner.invoke(config_app, ["list"])
        assert result.exit_code == 0
        mock_manager.get_config.assert_called_with()

        result = runner.invoke(config_app, ["list", "--json"])
        assert result.exit_code == 0
        expected_json = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"key3": "value3"},
        }
        assert json.loads(result.stdout.strip()) == expected_json

    @patch("captiv.cli.commands.config.list.ConfigManager")
    def test_config_list_specific_section(
        self, mock_config_manager, runner, config_app
    ):
        """Test listing config values for a specific section in normal and JSON
        formats."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_config.return_value = {
            "test_section": {"key1": "value1", "key2": "value2"}
        }

        result = runner.invoke(config_app, ["list", "test_section"])
        assert result.exit_code == 0
        assert "key1 = value1" in result.stdout
        assert "key2 = value2" in result.stdout
        mock_manager.get_config.assert_called_with()

        result = runner.invoke(config_app, ["list", "test_section", "--json"])
        assert result.exit_code == 0
        expected_json = {"test_section": {"key1": "value1", "key2": "value2"}}
        assert json.loads(result.stdout.strip()) == expected_json

    @patch("captiv.cli.commands.config.list.ConfigManager")
    def test_config_list_unknown_section(self, mock_config_manager, runner, config_app):
        """Test listing config values for an unknown section."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_config.return_value = {
            "section1": {"key1": "value1"},
            "section2": {"key2": "value2"},
        }

        result = runner.invoke(config_app, ["list", "unknown_section"])
        assert result.exit_code == 0
        assert "Unknown configuration section: unknown_section" in result.stdout
        assert "Available sections: section1, section2" in result.stdout
        mock_manager.get_config.assert_called_once()
