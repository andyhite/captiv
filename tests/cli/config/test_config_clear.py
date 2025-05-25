"""Tests for config clear CLI command."""

from unittest.mock import MagicMock, patch


class TestConfigClear:
    """Test the config clear command."""

    @patch("captiv.cli.commands.config.clear.ConfigManager")
    def test_config_clear_all(self, mock_config_manager, runner, config_app):
        """Test clearing all config values with success and error cases."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        result = runner.invoke(config_app, ["clear"])
        assert result.exit_code == 0
        assert "Configuration has been reset to defaults." in result.stdout
        mock_manager.clear_config.assert_called_with(None)

        mock_manager.clear_config.side_effect = ValueError("Cannot clear all sections")
        result = runner.invoke(config_app, ["clear"])
        assert result.exit_code == 0
        assert "Error: Cannot clear all sections" in result.stdout
        assert (
            "Run 'captiv config list' to see available configuration sections."
            in result.stdout
        )

    @patch("captiv.cli.commands.config.clear.ConfigManager")
    def test_config_clear_specific_section(
        self, mock_config_manager, runner, config_app
    ):
        """Test clearing config values for a specific section with success and error
        cases."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        result = runner.invoke(config_app, ["clear", "test_section"])
        assert result.exit_code == 0
        assert (
            "Configuration section 'test_section' has been reset to defaults."
            in result.stdout
        )
        mock_manager.clear_config.assert_called_with("test_section")

        mock_manager.clear_config.side_effect = ValueError("Invalid section name")
        result = runner.invoke(config_app, ["clear", "invalid_section"])
        assert result.exit_code == 0
        assert "Error: Invalid section name" in result.stdout
        assert (
            "Run 'captiv config list' to see available configuration sections."
            in result.stdout
        )
