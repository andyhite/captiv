"""Tests for config unset CLI command."""

from unittest.mock import MagicMock, patch


class TestConfigUnset:
    """Test the config unset command."""

    @patch("captiv.cli.commands.config.unset.ConfigManager")
    def test_config_unset_valid_key(self, mock_config_manager, runner, config_app):
        """Test unsetting a config key with valid format and default value lookup."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        mock_default_config = MagicMock()
        mock_section = MagicMock()
        mock_section.test_key = "default_value"
        mock_default_config.test_section = mock_section
        mock_manager._default_config = mock_default_config

        result = runner.invoke(config_app, ["unset", "test_section.test_key"])
        assert result.exit_code == 0
        assert (
            "Configuration value test_section.test_key has been reset to default: default_value"  # noqa: E501
            in result.stdout
        )
        mock_manager.unset_config_value.assert_called_with("test_section", "test_key")

        setattr(mock_section, "key.with.dots", "complex_default")
        result = runner.invoke(config_app, ["unset", "test_section.key.with.dots"])
        assert result.exit_code == 0
        assert (
            "Configuration value test_section.key.with.dots has been reset to default: complex_default"  # noqa: E501
            in result.stdout
        )
        mock_manager.unset_config_value.assert_called_with(
            "test_section", "key.with.dots"
        )

    @patch("captiv.cli.commands.config.unset.ConfigManager")
    def test_config_unset_missing_defaults(
        self, mock_config_manager, runner, config_app
    ):
        """Test unsetting config when section or key doesn't exist in defaults."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        class MockDefaultConfig:
            pass

        mock_manager._default_config = MockDefaultConfig()

        result = runner.invoke(config_app, ["unset", "missing_section.test_key"])
        assert result.exit_code == 0
        assert (
            "Configuration value missing_section.test_key has been reset to default: default"  # noqa: E501
            in result.stdout
        )

        mock_default_config = MagicMock()
        mock_section = MagicMock()
        del mock_section.missing_key
        mock_default_config.test_section = mock_section
        mock_manager._default_config = mock_default_config

        result = runner.invoke(config_app, ["unset", "test_section.missing_key"])
        assert result.exit_code == 0
        assert (
            "Configuration value test_section.missing_key has been reset to default: default"  # noqa: E501
            in result.stdout
        )

    @patch("captiv.cli.commands.config.unset.ConfigManager")
    def test_config_unset_error_cases(self, mock_config_manager, runner, config_app):
        """Test error handling in config unset command."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        result = runner.invoke(config_app, ["unset", "invalid_key"])
        assert result.exit_code == 0
        assert "Error: Key path must be in the format section.key" in result.stdout

        mock_manager.unset_config_value.side_effect = ValueError("Invalid key path")
        result = runner.invoke(config_app, ["unset", "section.key"])
        assert result.exit_code == 0
        assert "Error: Invalid key path" in result.stdout
        assert (
            "Run 'captiv config list' to see available configuration options."
            in result.stdout
        )
