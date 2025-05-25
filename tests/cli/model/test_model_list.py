"""Tests for model list CLI command."""

from unittest.mock import patch

from captiv.services import ModelType


class TestModelList:
    """Test model list command."""

    @patch("captiv.cli.commands.model.list.ModelManager")
    def test_list_all_models(self, _mock_model_mgr, runner, model_app):
        """Test listing all available models."""
        result = runner.invoke(model_app, ["list"])

        assert result.exit_code == 0
        assert "Available models:" in result.stdout
        for model in ModelType:
            assert model.value in result.stdout
        assert "For more details about a specific model:" in result.stdout

    @patch("captiv.cli.commands.model.list.ModelManager")
    def test_list_specific_model_details(self, mock_model_mgr, runner, model_app):
        """Test listing variants and modes for a specific model."""
        mock_manager = mock_model_mgr.return_value

        mock_manager.get_variant_details.return_value = {
            "base": {"description": "Base variant"},
            "large": {"description": "Large variant"},
        }
        mock_manager.get_mode_details.return_value = {
            "default": "Default mode",
            "detailed": "Detailed mode",
        }

        result = runner.invoke(model_app, ["list", "blip"])

        assert result.exit_code == 0
        assert "=== BLIP Model ===" in result.stdout
        assert "Available Model Variants:" in result.stdout
        assert "base" in result.stdout
        assert "large" in result.stdout
        assert "Available Modes:" in result.stdout
        assert "default" in result.stdout
        assert "detailed" in result.stdout
        assert "For more details:" in result.stdout
        assert "captiv model show blip" in result.stdout

        mock_manager.get_variant_details.return_value = {
            "base": {"description": "Base variant"}
        }
        mock_manager.get_mode_details.return_value = {}

        result = runner.invoke(model_app, ["list", "blip"])

        assert result.exit_code == 0
        assert "No specific modes available for this model." in result.stdout

    def test_list_invalid_model(self, runner, model_app):
        """Test listing variants for an invalid model."""
        result = runner.invoke(model_app, ["list", "invalid_model"])

        assert result.exit_code == 1
        assert "Error: Invalid model 'invalid_model'" in result.stdout
        assert "Valid models:" in result.stdout
