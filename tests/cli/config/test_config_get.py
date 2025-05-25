"""Tests for config get CLI command."""

from unittest.mock import MagicMock, patch


class TestConfigGet:
    """Test the config get command."""

    @patch("captiv.cli.commands.config.get.ConfigManager")
    def test_config_get_valid_key_path(self, mock_config_manager, runner, config_app):
        """Test getting a config value with valid section.key format."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_config_value.return_value = "test_value"

        result = runner.invoke(config_app, ["get", "section.key"])

        assert result.exit_code == 0
        mock_manager.get_config_value.assert_called_once_with("section", "key")

    @patch("captiv.cli.commands.config.get.ConfigManager")
    def test_config_get_invalid_key_path(self, mock_config_manager, runner, config_app):
        """Test getting a config value with invalid key format."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        result = runner.invoke(config_app, ["get", "invalid_key"])

        assert result.exit_code == 0
        assert "Error: Key path must be in the format section.key" in result.stdout

    @patch("captiv.cli.commands.config.get.ConfigManager")
    def test_config_get_nonexistent_key(self, mock_config_manager, runner, config_app):
        """Test getting a non-existent config key."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_config_value.return_value = None

        result = runner.invoke(config_app, ["get", "section.nonexistent"])

        assert result.exit_code == 0
        assert "Configuration key 'section.nonexistent' not found." in result.stdout


class TestConfigGetMissingCoverage:
    """Test missing coverage for config get command."""

    @patch("captiv.cli.commands.config.get.ConfigManager")
    def test_config_get_key_path_with_multiple_dots(
        self, mock_config_manager, runner, config_app
    ):
        """Test key path with multiple dots (should split on first dot only)."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_config_value.return_value = "nested_value"

        result = runner.invoke(config_app, ["get", "section.nested.key.path"])

        assert result.exit_code == 0
        mock_manager.get_config_value.assert_called_once_with(
            "section", "nested.key.path"
        )

    def test_config_get_invalid_format_shows_help_message(self, runner, config_app):
        """Test that invalid format shows the help message."""
        result = runner.invoke(config_app, ["get", "no_dot_key"])

        assert result.exit_code == 0
        assert "Error: Key path must be in the format section.key" in result.stdout
        assert (
            "Run 'captiv config list' to see available configuration options."
            in result.stdout
        )

    @patch("captiv.cli.commands.config.get.ConfigManager")
    def test_config_get_value_error_handling(
        self, mock_config_manager, runner, config_app
    ):
        """Test handling of ValueError from ConfigManager."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager
        mock_manager.get_config_value.side_effect = ValueError("Invalid section name")

        result = runner.invoke(config_app, ["get", "invalid.section"])

        assert result.exit_code == 0
        assert "Error: Invalid section name" in result.stdout
