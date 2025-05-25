"""Tests for caption set CLI command."""

from unittest.mock import patch


class TestCaptionSet:
    """Test caption set command."""

    @patch("captiv.cli.commands.caption.set.CaptionFileManager")
    @patch("captiv.cli.commands.caption.set.ImageFileManager")
    @patch("captiv.cli.commands.caption.set.FileManager")
    def test_set_caption_success(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test successful caption setting."""
        mock_caption_mgr.return_value.get_caption_file_path.return_value = (
            temp_single_image.with_suffix(".txt")
        )

        result = runner.invoke(
            caption_app, ["set", str(temp_single_image), "New caption"]
        )

        assert result.exit_code == 0
        assert "Caption updated for" in result.stdout
        mock_caption_mgr.return_value.write_caption.assert_called_once_with(
            temp_single_image, "New caption"
        )

    @patch("captiv.cli.commands.caption.set.CaptionFileManager")
    @patch("captiv.cli.commands.caption.set.ImageFileManager")
    @patch("captiv.cli.commands.caption.set.FileManager")
    def test_set_caption_error(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test caption setting with error."""
        mock_caption_mgr.return_value.write_caption.side_effect = Exception(
            "Write error"
        )

        result = runner.invoke(
            caption_app, ["set", str(temp_single_image), "New caption"]
        )

        assert result.exit_code == 1
