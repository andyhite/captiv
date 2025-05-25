"""Tests for caption unset CLI command."""

from unittest.mock import patch


class TestCaptionUnset:
    """Test caption unset command."""

    @patch("captiv.cli.commands.caption.unset.CaptionFileManager")
    @patch("captiv.cli.commands.caption.unset.ImageFileManager")
    @patch("captiv.cli.commands.caption.unset.FileManager")
    def test_unset_caption_success(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test successful caption unsetting."""
        result = runner.invoke(caption_app, ["unset", str(temp_single_image)])

        assert result.exit_code == 0
        assert "Caption removed for" in result.stdout
        mock_caption_mgr.return_value.delete_caption.assert_called_once_with(
            temp_single_image
        )

    @patch("captiv.cli.commands.caption.unset.CaptionFileManager")
    @patch("captiv.cli.commands.caption.unset.ImageFileManager")
    @patch("captiv.cli.commands.caption.unset.FileManager")
    def test_unset_caption_file_not_found(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test caption unsetting when file not found."""
        mock_caption_mgr.return_value.delete_caption.side_effect = FileNotFoundError()

        result = runner.invoke(caption_app, ["unset", str(temp_single_image)])

        assert result.exit_code == 0
        assert f"Error removing caption for {temp_single_image.name}:" in result.stdout

    @patch("captiv.cli.commands.caption.unset.CaptionFileManager")
    @patch("captiv.cli.commands.caption.unset.ImageFileManager")
    @patch("captiv.cli.commands.caption.unset.FileManager")
    def test_unset_caption_error(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test caption unsetting with error."""
        mock_caption_mgr.return_value.delete_caption.side_effect = Exception(
            "Delete error"
        )

        result = runner.invoke(caption_app, ["unset", str(temp_single_image)])

        assert result.exit_code == 0
        assert "Error removing caption" in result.stdout
