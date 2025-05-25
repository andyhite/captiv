"""Tests for model show CLI command."""

from unittest.mock import MagicMock, patch


class TestModelShow:
    """Test model show command."""

    @patch("captiv.cli.commands.model.show.ModelManager")
    def test_show_model_details(self, mock_model_mgr, runner, model_app):
        """Test showing detailed model information."""
        mock_manager = mock_model_mgr.return_value
        mock_model_class = MagicMock()
        mock_model_class.__doc__ = "Test model description"
        mock_manager.get_model_class.return_value = mock_model_class

        mock_manager.get_variant_details.return_value = {
            "base": {
                "description": "Base variant description",
                "checkpoint": "model/base",
            },
            "large": {
                "description": "Large variant description",
                "checkpoint": "model/large",
            },
        }

        mock_manager.get_mode_details.return_value = {
            "default": "Default captioning mode",
            "detailed": "Detailed captioning mode",
        }

        mock_manager.get_prompt_option_details.return_value = {
            "include_lighting": "Include lighting information",
            "keep_pg": "Keep content PG-rated",
        }

        result = runner.invoke(model_app, ["show", "blip"])

        assert result.exit_code == 0
        assert "=== BLIP Model ===" in result.stdout
        assert "Description:" in result.stdout
        assert "Test model description" in result.stdout
        assert "Available Model Variants:" in result.stdout
        assert "base:" in result.stdout
        assert "Base variant description" in result.stdout
        assert "Checkpoint: model/base" in result.stdout
        assert "Available Modes:" in result.stdout
        assert "default: Default captioning mode" in result.stdout
        assert "Available Prompt Options:" in result.stdout
        assert "include_lighting: Include lighting information" in result.stdout
        assert "Supported Generation Parameters:" in result.stdout
        assert (
            "max_new_tokens: Maximum number of tokens in the generated caption"
            in result.stdout
        )
        assert "Usage Examples:" in result.stdout

    @patch("captiv.cli.commands.model.show.ModelManager")
    def test_show_model_no_description(self, mock_model_mgr, runner, model_app):
        """Test showing model with no description."""
        mock_manager = mock_model_mgr.return_value
        mock_model_class = MagicMock()
        mock_model_class.__doc__ = None
        mock_manager.get_model_class.return_value = mock_model_class

        mock_manager.get_variant_details.return_value = {"base": {}}
        mock_manager.get_mode_details.return_value = {}
        mock_manager.get_prompt_option_details.return_value = {}

        result = runner.invoke(model_app, ["show", "blip"])

        assert result.exit_code == 0
        assert "No description available." in result.stdout

    @patch("captiv.cli.commands.model.show.ModelManager")
    def test_show_model_no_modes(self, mock_model_mgr, runner, model_app):
        """Test showing model with no specific modes."""
        mock_manager = mock_model_mgr.return_value
        mock_model_class = MagicMock()
        mock_model_class.__doc__ = "Test description"
        mock_manager.get_model_class.return_value = mock_model_class

        mock_manager.get_variant_details.return_value = {"base": {}}
        mock_manager.get_mode_details.return_value = {}
        mock_manager.get_prompt_option_details.return_value = {}

        result = runner.invoke(model_app, ["show", "blip"])

        assert result.exit_code == 0
        assert "No specific modes available for this model." in result.stdout

    @patch("captiv.cli.commands.model.show.ModelManager")
    def test_show_model_no_prompt_options(self, mock_model_mgr, runner, model_app):
        """Test showing model with no prompt options."""
        mock_manager = mock_model_mgr.return_value
        mock_model_class = MagicMock()
        mock_model_class.__doc__ = "Test description"
        mock_manager.get_model_class.return_value = mock_model_class

        mock_manager.get_variant_details.return_value = {"base": {}}
        mock_manager.get_mode_details.return_value = {}
        mock_manager.get_prompt_option_details.return_value = {}

        result = runner.invoke(model_app, ["show", "blip"])

        assert result.exit_code == 0
        assert "No prompt options available for this model." in result.stdout

    @patch("captiv.cli.commands.model.show.ModelManager")
    def test_show_model_with_prompt_options_note(
        self, mock_model_mgr, runner, model_app
    ):
        """Test showing model with prompt options includes usage note."""
        mock_manager = mock_model_mgr.return_value
        mock_model_class = MagicMock()
        mock_model_class.__doc__ = "Test description"
        mock_manager.get_model_class.return_value = mock_model_class

        mock_manager.get_variant_details.return_value = {"base": {}}
        mock_manager.get_mode_details.return_value = {}
        mock_manager.get_prompt_option_details.return_value = {
            "option1": "Description 1",
            "option2": "Description 2",
        }

        result = runner.invoke(model_app, ["show", "blip"])

        assert result.exit_code == 0
        assert "Note: This model supports 2 prompt options." in result.stdout
        assert "Use --prompt-options to include them" in result.stdout

    @patch("captiv.cli.commands.model.show.ModelManager")
    def test_show_model_usage_examples_with_modes(
        self, mock_model_mgr, runner, model_app
    ):
        """Test that usage examples include mode examples when available."""
        mock_manager = mock_model_mgr.return_value
        mock_model_class = MagicMock()
        mock_model_class.__doc__ = "Test description"
        mock_manager.get_model_class.return_value = mock_model_class

        mock_manager.get_variant_details.return_value = {"base": {}, "large": {}}
        mock_manager.get_mode_details.return_value = {"detailed": "Detailed mode"}
        mock_manager.get_prompt_option_details.return_value = {"option1": "Description"}

        result = runner.invoke(model_app, ["show", "blip"])

        assert result.exit_code == 0
        assert (
            "captiv caption generate image.jpg --model blip --variant base"
            in result.stdout
        )
        assert (
            "captiv caption generate image.jpg --model blip --mode detailed"
            in result.stdout
        )
        assert (
            "captiv caption generate image.jpg --model blip --prompt-options option1"
            in result.stdout
        )
        assert (
            "captiv caption generate image.jpg --model blip --prompt-variables character_name=Alice,setting=forest"  # noqa: E501
            in result.stdout
        )

    def test_show_invalid_model(self, runner, model_app):
        """Test showing details for an invalid model."""
        result = runner.invoke(model_app, ["show", "invalid_model"])

        assert result.exit_code == 1
        assert "Error: Invalid model 'invalid_model'" in result.stdout
        assert "Valid models:" in result.stdout
