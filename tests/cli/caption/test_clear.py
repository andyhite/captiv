"""Tests for caption clear CLI command."""

from unittest.mock import patch


class TestCaptionClear:
    """Test caption clear command."""

    @patch("captiv.cli.commands.caption.clear.CaptionFileManager")
    @patch("captiv.cli.commands.caption.clear.ImageFileManager")
    @patch("captiv.cli.commands.caption.clear.FileManager")
    def test_clear_captions_success(
        self,
        _mock_file_mgr,
        mock_img_mgr,
        mock_caption_mgr,
        temp_image_dir,
        runner,
        caption_app,
    ):
        """Test successful caption clearing."""
        mock_img_mgr.return_value.list_image_files.return_value = [
            temp_image_dir / "test1.jpg",
            temp_image_dir / "test2.png",
        ]
        mock_caption_mgr.return_value.get_caption_file_path.side_effect = [
            temp_image_dir / "test1.txt",
            temp_image_dir / "test2.txt",
        ]

        result = runner.invoke(caption_app, ["clear", str(temp_image_dir)])

        assert result.exit_code == 0
        assert "Captions cleared successfully" in result.stdout

    @patch("captiv.cli.commands.caption.clear.CaptionFileManager")
    @patch("captiv.cli.commands.caption.clear.ImageFileManager")
    @patch("captiv.cli.commands.caption.clear.FileManager")
    def test_clear_captions_no_images(
        self,
        _mock_file_mgr,
        mock_img_mgr,
        _mock_caption_mgr,
        temp_image_dir,
        runner,
        caption_app,
    ):
        """Test caption clearing with no images."""
        mock_img_mgr.return_value.list_image_files.return_value = []

        result = runner.invoke(caption_app, ["clear", str(temp_image_dir)])

        assert result.exit_code == 0
        assert "Captions cleared successfully" in result.stdout

    @patch("captiv.cli.commands.caption.clear.CaptionFileManager")
    @patch("captiv.cli.commands.caption.clear.ImageFileManager")
    @patch("captiv.cli.commands.caption.clear.FileManager")
    def test_clear_captions_with_errors(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_image_dir,
        runner,
        caption_app,
    ):
        """Test caption clearing with some errors."""
        mock_caption_mgr.return_value.clear_captions.side_effect = Exception(
            "Clear error"
        )

        result = runner.invoke(caption_app, ["clear", str(temp_image_dir)])

        assert result.exit_code == 0
        assert "Error clearing captions: Clear error" in result.stdout


class TestCaptionClearMissingCoverage:
    """Test missing coverage for caption clear command."""

    def test_clear_default_directory_path_resolution(self, runner, caption_app):
        """Test that default directory is properly resolved when no argument
        provided."""
        with (
            patch("captiv.cli.commands.caption.clear.os.getcwd") as mock_getcwd,
            patch("captiv.cli.commands.caption.clear.Path") as mock_path,
        ):
            mock_getcwd.return_value = "/test/current/dir"
            mock_path_instance = mock_path.return_value

            with patch(
                "captiv.cli.commands.caption.clear.CaptionFileManager"
            ) as mock_caption_mgr:
                mock_caption_mgr.return_value.clear_captions.return_value = None

                result = runner.invoke(caption_app, ["clear"])

                assert result.exit_code == 0
                mock_getcwd.assert_called_once()
                mock_path.assert_called_with("/test/current/dir")
                mock_caption_mgr.return_value.clear_captions.assert_called_once_with(
                    mock_path_instance
                )
