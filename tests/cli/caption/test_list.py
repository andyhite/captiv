"""Tests for caption list CLI command."""

from unittest.mock import patch


class TestCaptionList:
    """Test caption list command."""

    @patch("captiv.cli.commands.caption.list.CaptionFileManager")
    @patch("captiv.cli.commands.caption.list.ImageFileManager")
    @patch("captiv.cli.commands.caption.list.FileManager")
    def test_list_captions_success(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_image_dir,
        runner,
        caption_app,
    ):
        """Test successful caption listing."""
        mock_caption_mgr.return_value.list_images_and_captions.return_value = [
            (temp_image_dir / "test1.jpg", "Caption 1"),
            (temp_image_dir / "test2.png", "Caption 2"),
        ]

        result = runner.invoke(caption_app, ["list", str(temp_image_dir)])

        assert result.exit_code == 0
        assert "test1.jpg: Caption 1" in result.stdout
        assert "test2.png: Caption 2" in result.stdout

    @patch("captiv.cli.commands.caption.list.CaptionFileManager")
    @patch("captiv.cli.commands.caption.list.ImageFileManager")
    @patch("captiv.cli.commands.caption.list.FileManager")
    def test_list_captions_no_images(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_image_dir,
        runner,
        caption_app,
    ):
        """Test caption listing with no images."""
        mock_caption_mgr.return_value.list_images_and_captions.return_value = []

        result = runner.invoke(caption_app, ["list", str(temp_image_dir)])

        assert result.exit_code == 0
        assert f"No images found in {temp_image_dir}." in result.stdout

    @patch("captiv.cli.commands.caption.list.CaptionFileManager")
    @patch("captiv.cli.commands.caption.list.ImageFileManager")
    @patch("captiv.cli.commands.caption.list.FileManager")
    def test_list_captions_with_missing_captions(
        self,
        _mock_file_mgr,
        _mock_img_mgr,
        mock_caption_mgr,
        temp_image_dir,
        runner,
        caption_app,
    ):
        """Test caption listing with some missing captions."""
        mock_caption_mgr.return_value.list_images_and_captions.return_value = [
            (temp_image_dir / "test1.jpg", "Caption 1"),
            (temp_image_dir / "test2.png", None),
        ]

        result = runner.invoke(caption_app, ["list", str(temp_image_dir)])

        assert result.exit_code == 0
        assert "test1.jpg: Caption 1" in result.stdout
        assert "test2.png: No caption" in result.stdout


class TestCaptionListMissingCoverage:
    """Test missing coverage for caption list command."""

    def test_list_default_directory_path_resolution(self, runner, caption_app):
        """Test that default directory is properly resolved when no argument
        provided."""
        with (
            patch("captiv.cli.commands.caption.list.os.getcwd") as mock_getcwd,
            patch("captiv.cli.commands.caption.list.Path") as mock_path,
        ):
            mock_getcwd.return_value = "/test/current/dir"
            mock_path_instance = mock_path.return_value

            with patch(
                "captiv.cli.commands.caption.list.CaptionFileManager"
            ) as mock_caption_mgr:
                mock_caption_mgr.return_value.list_images_and_captions.return_value = []  # noqa: E501

                result = runner.invoke(caption_app, ["list"])

                assert result.exit_code == 0
                mock_getcwd.assert_called_once()
                mock_path.assert_called_with("/test/current/dir")
                mock_caption_mgr.return_value.list_images_and_captions.assert_called_once_with(
                    mock_path_instance
                )
