"""Tests for caption get CLI command."""

from unittest.mock import patch


class TestCaptionGet:
    """Test caption get command."""

    @patch("captiv.cli.commands.caption.get.CaptionFileManager")
    @patch("captiv.cli.commands.caption.get.ImageFileManager")
    @patch("captiv.cli.commands.caption.get.FileManager")
    def test_get_caption_success(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test successful caption retrieval."""
        mock_caption_mgr.return_value.read_caption.return_value = "Test caption"

        result = runner.invoke(caption_app, ["get", str(temp_single_image)])

        assert result.exit_code == 0
        assert "Test caption" in result.stdout

    @patch("captiv.cli.commands.caption.get.CaptionFileManager")
    @patch("captiv.cli.commands.caption.get.ImageFileManager")
    @patch("captiv.cli.commands.caption.get.FileManager")
    def test_get_caption_empty(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test caption retrieval when caption is empty."""
        mock_caption_mgr.return_value.read_caption.return_value = ""

        result = runner.invoke(caption_app, ["get", str(temp_single_image)])

        assert result.exit_code == 0
        assert "No caption found" in result.stdout

    @patch("captiv.cli.commands.caption.get.CaptionFileManager")
    @patch("captiv.cli.commands.caption.get.ImageFileManager")
    @patch("captiv.cli.commands.caption.get.FileManager")
    def test_get_caption_file_not_found(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test caption retrieval when file not found."""
        mock_caption_mgr.return_value.read_caption.side_effect = FileNotFoundError()

        result = runner.invoke(caption_app, ["get", str(temp_single_image)])

        assert result.exit_code == 0
        assert "No caption found" in result.stdout

    @patch("captiv.cli.commands.caption.get.CaptionFileManager")
    @patch("captiv.cli.commands.caption.get.ImageFileManager")
    @patch("captiv.cli.commands.caption.get.FileManager")
    def test_get_caption_generic_error(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_single_image,
        runner,
        caption_app,
    ):
        """Test caption retrieval with generic error."""
        mock_caption_mgr.return_value.read_caption.side_effect = Exception(
            "Generic error"
        )

        result = runner.invoke(caption_app, ["get", str(temp_single_image)])

        assert result.exit_code == 0
        assert "Error reading caption" in result.stdout
        assert "Generic error" in result.stdout
