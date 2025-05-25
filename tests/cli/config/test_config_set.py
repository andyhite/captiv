"""Tests for config set CLI command."""

from unittest.mock import MagicMock, patch

import pytest

from captiv.cli.commands.config.set import (
    smart_type_conversion,
    validate_special_values,
)
from captiv.services.model_manager import ModelType


class TestSmartTypeConversion:
    """Test the smart_type_conversion function."""

    def test_boolean_conversion(self):
        """Test conversion of boolean strings."""
        assert smart_type_conversion("true") is True
        assert smart_type_conversion("True") is True
        assert smart_type_conversion("TRUE") is True
        assert smart_type_conversion("false") is False
        assert smart_type_conversion("False") is False
        assert smart_type_conversion("FALSE") is False

    def test_numeric_conversion(self):
        """Test conversion of numeric strings."""
        assert smart_type_conversion("123") == 123
        assert smart_type_conversion("-456") == -456
        assert smart_type_conversion("0") == 0

        assert smart_type_conversion("123.45") == 123.45
        assert smart_type_conversion("-67.89") == -67.89
        assert smart_type_conversion("0.0") == 0.0
        assert smart_type_conversion(".5") == 0.5
        assert smart_type_conversion("1.") == 1.0

        assert smart_type_conversion("1e5") == 1e5
        assert smart_type_conversion("1.5e-3") == 1.5e-3
        assert smart_type_conversion("-2.5E+2") == -2.5e2

    def test_string_fallback(self):
        """Test that non-numeric strings remain as strings."""
        assert smart_type_conversion("hello") == "hello"
        assert smart_type_conversion("123abc") == "123abc"
        assert smart_type_conversion("") == ""
        assert smart_type_conversion("not_a_number") == "not_a_number"


class TestValidateSpecialValues:
    """Test the validate_special_values function."""

    def test_model_validation(self):
        """Test validation of model values."""
        for model in ModelType:
            result = validate_special_values("model", "default_model", model.value)
            assert result == model.value

        with pytest.raises(ValueError, match="Invalid model 'invalid_model'"):
            validate_special_values("model", "default_model", "invalid_model")

    def test_port_validation(self):
        """Test validation of port numbers."""
        assert validate_special_values("gui", "port", 8080) == 8080
        assert validate_special_values("gui", "port", 1) == 1
        assert validate_special_values("gui", "port", 65535) == 65535

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            validate_special_values("gui", "port", 0)
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            validate_special_values("gui", "port", 65536)

    def test_positive_integer_validation(self):
        """Test validation of positive integer fields."""
        fields = ["max_new_tokens", "min_new_tokens", "num_beams", "top_k"]

        for field in fields:
            assert validate_special_values("generation", field, 10) == 10
            with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
                validate_special_values("generation", field, 0)
            with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
                validate_special_values("generation", field, -5)

    def test_temperature_validation(self):
        """Test validation of temperature values."""
        assert validate_special_values("generation", "temperature", 1.0) == 1.0
        assert validate_special_values("generation", "temperature", 0.5) == 0.5

        with pytest.raises(ValueError, match="Temperature must be positive"):
            validate_special_values("generation", "temperature", 0)
        with pytest.raises(ValueError, match="Temperature must be positive"):
            validate_special_values("generation", "temperature", -0.5)

    def test_top_p_validation(self):
        """Test validation of top_p values."""
        assert validate_special_values("generation", "top_p", 0.9) == 0.9
        assert validate_special_values("generation", "top_p", 1.0) == 1.0
        assert validate_special_values("generation", "top_p", 0.1) == 0.1

        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            validate_special_values("generation", "top_p", 0)
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            validate_special_values("generation", "top_p", 1.5)

    def test_repetition_penalty_validation(self):
        """Test validation of repetition_penalty values."""
        assert validate_special_values("generation", "repetition_penalty", 1.2) == 1.2
        assert validate_special_values("generation", "repetition_penalty", 0.8) == 0.8

        with pytest.raises(ValueError, match="Repetition penalty must be positive"):
            validate_special_values("generation", "repetition_penalty", 0)
        with pytest.raises(ValueError, match="Repetition penalty must be positive"):
            validate_special_values("generation", "repetition_penalty", -1.0)

    def test_non_special_values_passthrough(self):
        """Test that non-special values pass through unchanged."""
        assert validate_special_values("other", "key", "value") == "value"
        assert validate_special_values("section", "other_key", 123) == 123
        assert validate_special_values("model", "other_key", "test") == "test"


class TestConfigSet:
    """Test the config set command."""

    @patch("captiv.cli.commands.config.set.ConfigManager")
    def test_config_set_valid_key_value(self, mock_config_manager, runner, config_app):
        """Test setting valid config key-value pairs with type conversion."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        result = runner.invoke(config_app, ["set", "section.key=test_value"])
        assert result.exit_code == 0
        mock_manager.set_config_value.assert_called_with("section", "key", "test_value")

        result = runner.invoke(config_app, ["set", "section.bool_key=true"])
        assert result.exit_code == 0
        mock_manager.set_config_value.assert_called_with("section", "bool_key", True)

        result = runner.invoke(config_app, ["set", "section.int_key=42"])
        assert result.exit_code == 0
        mock_manager.set_config_value.assert_called_with("section", "int_key", 42)

        result = runner.invoke(config_app, ["set", "section.float_key=3.14"])
        assert result.exit_code == 0
        mock_manager.set_config_value.assert_called_with("section", "float_key", 3.14)

    @patch("captiv.cli.commands.config.set.ConfigManager")
    def test_config_set_edge_cases(self, mock_config_manager, runner, config_app):
        """Test edge cases for config set command."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        result = runner.invoke(config_app, ["set", "section.nested.key=value"])
        assert result.exit_code == 0
        mock_manager.set_config_value.assert_called_with(
            "section", "nested.key", "value"
        )

        result = runner.invoke(config_app, ["set", "section.key="])
        assert result.exit_code == 0
        mock_manager.set_config_value.assert_called_with("section", "key", "")

        result = runner.invoke(config_app, ["set", "section.key=value=with=equals"])
        assert result.exit_code == 0
        mock_manager.set_config_value.assert_called_with(
            "section", "key", "value=with=equals"
        )

    @patch("captiv.cli.commands.config.set.ConfigManager")
    def test_config_set_with_custom_config_file(
        self, mock_config_manager, runner, config_app
    ):
        """Test setting config with custom config file."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        result = runner.invoke(
            config_app,
            ["set", "--config-file", "/custom/config.yaml", "section.key=value"],
        )
        assert result.exit_code == 0
        mock_config_manager.assert_called_with("/custom/config.yaml")

    @patch("captiv.cli.commands.config.set.ConfigManager")
    def test_config_set_error_cases(self, mock_config_manager, runner, config_app):
        """Test error handling in config set command."""
        mock_manager = MagicMock()
        mock_config_manager.return_value = mock_manager

        result = runner.invoke(config_app, ["set", "invalid_key=value"])
        assert result.exit_code == 0
        assert "Error: Key path must be in the format section.key" in result.stdout

        result = runner.invoke(config_app, ["set", "section.key"])
        assert result.exit_code == 1

        result = runner.invoke(config_app, ["set", "model.default_model=invalid_model"])
        assert result.exit_code == 0
        assert "Error: Invalid model 'invalid_model'" in result.stdout

        mock_manager.set_config_value.side_effect = ValueError("Config error")
        result = runner.invoke(config_app, ["set", "section.key=value"])
        assert result.exit_code == 0
        assert "Error: Config error" in result.stdout
