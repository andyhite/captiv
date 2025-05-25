"""Tests for caption generate CLI command."""

from typing import Any
from unittest.mock import Mock, patch

from captiv.services import ModelType
from captiv.utils.error_handling import EnhancedError


class TestCaptionGenerate:
    """Test caption generate command."""

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.CaptionFileManager")
    @patch("captiv.cli.commands.caption.generate.ImageFileManager")
    @patch("captiv.cli.commands.caption.generate.FileManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_single_image_success(
        self,
        _mock_config_mgr: Any,
        _mock_file_mgr: Any,
        _mock_img_mgr: Any,
        mock_caption_mgr: Any,
        mock_model_mgr: Any,
        temp_single_image: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test successful caption generation for single image."""
        mock_model_instance = Mock()
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = (
            "base"
        )
        mock_model_mgr.return_value.get_variants_for_model.return_value = ["base"]
        mock_model_mgr.return_value.validate_variant.return_value = None
        mock_model_mgr.return_value.parse_prompt_options.return_value = []
        mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
        mock_model_mgr.return_value.build_generation_params.return_value = {}
        mock_model_mgr.return_value.create_model_instance.return_value = (
            mock_model_instance
        )
        mock_model_mgr.return_value.generate_caption.return_value = "Generated caption"

        mock_caption_mgr.return_value.write_caption.return_value = None
        mock_caption_mgr.return_value.get_caption_file_path.return_value = (
            temp_single_image.with_suffix(".txt")
        )

        result = runner.invoke(caption_app, ["generate", str(temp_single_image)])

        assert result.exit_code == 0
        assert "Generated caption" in result.stdout
        assert "Caption saved to" in result.stdout

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.CaptionFileManager")
    @patch("captiv.cli.commands.caption.generate.ImageFileManager")
    @patch("captiv.cli.commands.caption.generate.FileManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_directory_success(
        self,
        _mock_config_mgr: Any,
        _mock_file_mgr: Any,
        mock_img_mgr: Any,
        mock_caption_mgr: Any,
        mock_model_mgr: Any,
        temp_image_dir: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test successful caption generation for directory."""
        mock_model_instance = Mock()
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = (
            "base"
        )
        mock_model_mgr.return_value.get_variants_for_model.return_value = ["base"]
        mock_model_mgr.return_value.validate_variant.return_value = None
        mock_model_mgr.return_value.parse_prompt_options.return_value = []
        mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
        mock_model_mgr.return_value.build_generation_params.return_value = {}
        mock_model_mgr.return_value.create_model_instance.return_value = (
            mock_model_instance
        )
        mock_model_mgr.return_value.generate_caption.return_value = "Generated caption"

        mock_img_mgr.return_value.list_image_files.return_value = [
            temp_image_dir / "test1.jpg",
            temp_image_dir / "test2.png",
        ]

        mock_caption_mgr.return_value.write_caption.return_value = None
        mock_caption_mgr.return_value.get_caption_file_path.return_value = (
            temp_image_dir / "test.txt"
        )

        result = runner.invoke(caption_app, ["generate", str(temp_image_dir)])

        assert result.exit_code == 0
        assert "Found 2 images" in result.stdout
        assert "Successfully captioned: 2/2 images" in result.stdout

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_no_default_variant_uses_first_available(
        self,
        _mock_config_mgr: Any,
        mock_model_mgr: Any,
        temp_single_image: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test generation when no default variant but variants available."""
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = None
        mock_model_mgr.return_value.get_variants_for_model.return_value = [
            "variant1",
            "variant2",
        ]
        mock_model_mgr.return_value.validate_variant.return_value = None
        mock_model_mgr.return_value.parse_prompt_options.return_value = []
        mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
        mock_model_mgr.return_value.build_generation_params.return_value = {}
        mock_model_mgr.return_value.create_model_instance.return_value = Mock()
        mock_model_mgr.return_value.generate_caption.return_value = "Generated caption"

        result = runner.invoke(
            caption_app, ["generate", str(temp_single_image), "--no-save"]
        )

        assert result.exit_code == 0

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_no_variants_available_error(
        self,
        _mock_config_mgr: Any,
        mock_model_mgr: Any,
        temp_single_image: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test error when no variants available for model."""
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = None
        mock_model_mgr.return_value.get_variants_for_model.return_value = []

        result = runner.invoke(caption_app, ["generate", str(temp_single_image)])

        assert result.exit_code == 1
        assert "No model variants available" in result.stdout

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_model_loading_error(
        self,
        _mock_config_mgr: Any,
        mock_model_mgr: Any,
        temp_single_image: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test error handling during model loading."""
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = (
            "base"
        )
        mock_model_mgr.return_value.validate_variant.return_value = None
        mock_model_mgr.return_value.parse_prompt_options.return_value = []
        mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
        mock_model_mgr.return_value.build_generation_params.return_value = {}

        error = EnhancedError(
            "Model loading failed", troubleshooting_tips=["Tip 1", "Tip 2"]
        )
        mock_model_mgr.return_value.create_model_instance.side_effect = error

        result = runner.invoke(caption_app, ["generate", str(temp_single_image)])

        assert result.exit_code == 1
        assert "Error loading model: Model loading failed" in result.stdout
        assert "Troubleshooting tips:" in result.stdout
        assert "1. Tip 1" in result.stdout

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.ImageFileManager")
    @patch("captiv.cli.commands.caption.generate.FileManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_directory_no_images(
        self,
        _mock_config_mgr: Any,
        _mock_file_mgr: Any,
        mock_img_mgr: Any,
        mock_model_mgr: Any,
        temp_image_dir: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test directory processing with no images found."""
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = (
            "base"
        )
        mock_model_mgr.return_value.validate_variant.return_value = None
        mock_model_mgr.return_value.parse_prompt_options.return_value = []
        mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
        mock_model_mgr.return_value.build_generation_params.return_value = {}
        mock_model_mgr.return_value.create_model_instance.return_value = Mock()

        mock_img_mgr.return_value.list_image_files.return_value = []

        result = runner.invoke(caption_app, ["generate", str(temp_image_dir)])

        assert result.exit_code == 0
        assert "No images found" in result.stdout

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.ImageFileManager")
    @patch("captiv.cli.commands.caption.generate.FileManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_directory_with_errors(
        self,
        _mock_config_mgr: Any,
        _mock_file_mgr: Any,
        mock_img_mgr: Any,
        mock_model_mgr: Any,
        temp_image_dir: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test directory processing with some images failing."""
        mock_model_instance = Mock()
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = (
            "base"
        )
        mock_model_mgr.return_value.validate_variant.return_value = None
        mock_model_mgr.return_value.parse_prompt_options.return_value = []
        mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
        mock_model_mgr.return_value.build_generation_params.return_value = {}
        mock_model_mgr.return_value.create_model_instance.return_value = (
            mock_model_instance
        )

        mock_model_mgr.return_value.generate_caption.side_effect = [
            "Success caption",
            EnhancedError("Enhanced error", troubleshooting_tips=["Tip"]),
            Exception("Generic error"),
        ]

        mock_img_mgr.return_value.list_image_files.return_value = [
            temp_image_dir / "test1.jpg",
            temp_image_dir / "test2.png",
            temp_image_dir / "test3.jpg",
        ]

        result = runner.invoke(
            caption_app, ["generate", str(temp_image_dir), "--no-save"]
        )

        assert result.exit_code == 0
        assert "Successfully captioned: 1/3 images" in result.stdout
        assert "Failed to caption: 2/3 images" in result.stdout

    def test_generate_default_to_current_directory(
        self, runner: Any, caption_app: Any
    ) -> None:
        """Test that command defaults to current directory when no path provided."""
        with patch("captiv.cli.commands.caption.generate.os.getcwd") as mock_getcwd:
            mock_getcwd.return_value = "/test/path"

            with patch(
                "captiv.cli.commands.caption.generate.ModelManager"
            ) as mock_model_mgr:
                mock_model_mgr.return_value.get_default_model.return_value = (
                    ModelType.BLIP
                )
                mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = "base"  # noqa: E501
                mock_model_mgr.return_value.validate_variant.return_value = None
                mock_model_mgr.return_value.parse_prompt_options.return_value = []
                mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
                mock_model_mgr.return_value.build_generation_params.return_value = {}
                mock_model_mgr.return_value.create_model_instance.return_value = Mock()
                mock_model_mgr.return_value.generate_caption.return_value = "Caption"

                with patch(
                    "captiv.cli.commands.caption.generate.ImageFileManager"
                ) as mock_img_mgr:
                    mock_img_mgr.return_value.list_image_files.return_value = []

                    result = runner.invoke(caption_app, ["generate"])

                    assert result.exit_code == 1
                    mock_getcwd.assert_called_once()

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_with_mode_validation(
        self,
        _mock_config_mgr: Any,
        mock_model_mgr: Any,
        temp_single_image: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test mode validation during generation."""
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = (
            "base"
        )
        mock_model_mgr.return_value.validate_variant.return_value = None
        mock_model_mgr.return_value.validate_mode.return_value = None
        mock_model_mgr.return_value.parse_prompt_options.return_value = []
        mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
        mock_model_mgr.return_value.build_generation_params.return_value = {}
        mock_model_mgr.return_value.create_model_instance.return_value = Mock()
        mock_model_mgr.return_value.generate_caption.return_value = "Caption"

        result = runner.invoke(
            caption_app,
            ["generate", str(temp_single_image), "--mode", "test_mode", "--no-save"],
        )

        assert result.exit_code == 0
        mock_model_mgr.return_value.validate_mode.assert_called_once_with(
            ModelType.BLIP, "test_mode"
        )


class TestCaptionGenerateMissingCoverage:
    """Test missing coverage for caption generate command."""

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.CaptionFileManager")
    @patch("captiv.cli.commands.caption.generate.ImageFileManager")
    @patch("captiv.cli.commands.caption.generate.FileManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_single_image_enhanced_error_fallback(
        self,
        _mock_config_mgr: Any,
        _mock_file_mgr: Any,
        _mock_img_mgr: Any,
        _mock_caption_mgr: Any,
        mock_model_mgr: Any,
        temp_single_image: Any,
        runner: Any,
        caption_app: Any,
    ) -> None:
        """Test single image processing with enhanced error fallback to generic
        exception."""
        mock_model_instance = Mock()
        mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
        mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = (
            "base"
        )
        mock_model_mgr.return_value.validate_variant.return_value = None
        mock_model_mgr.return_value.parse_prompt_options.return_value = []
        mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
        mock_model_mgr.return_value.build_generation_params.return_value = {}
        mock_model_mgr.return_value.create_model_instance.return_value = (
            mock_model_instance
        )

        mock_model_mgr.return_value.generate_caption.side_effect = Exception(
            "Generic error"
        )

        result = runner.invoke(
            caption_app, ["generate", str(temp_single_image), "--no-save"]
        )

        assert result.exit_code == 1

    @patch("captiv.cli.commands.caption.generate.ModelManager")
    @patch("captiv.cli.commands.caption.generate.ConfigManager")
    def test_generate_default_directory_path_resolution(
        self, _mock_config_mgr: Any, mock_model_mgr: Any, runner: Any, caption_app: Any
    ) -> None:
        """Test that default directory path is properly resolved when no argument
        provided."""
        with (
            patch("captiv.cli.commands.caption.generate.os.getcwd") as mock_getcwd,
            patch("captiv.cli.commands.caption.generate.Path") as mock_path,
        ):
            mock_getcwd.return_value = "/test/current/dir"
            mock_path_instance = Mock()
            mock_path_instance.is_dir.return_value = True
            mock_path.return_value = mock_path_instance

            mock_model_mgr.return_value.get_default_model.return_value = ModelType.BLIP
            mock_model_mgr.return_value.get_model_class.return_value.DEFAULT_VARIANT = (
                "base"  # noqa: E501
            )
            mock_model_mgr.return_value.validate_variant.return_value = None
            mock_model_mgr.return_value.parse_prompt_options.return_value = []
            mock_model_mgr.return_value.parse_prompt_variables.return_value = {}
            mock_model_mgr.return_value.build_generation_params.return_value = {}
            mock_model_mgr.return_value.create_model_instance.return_value = Mock()

            with patch(
                "captiv.cli.commands.caption.generate.ImageFileManager"
            ) as mock_img_mgr:
                mock_img_mgr.return_value.list_image_files.return_value = []

                runner.invoke(caption_app, ["generate"])

                mock_getcwd.assert_called_once()
                mock_path.assert_called_with("/test/current/dir")
