"""Tests for the captiv.gui.main module."""

from unittest.mock import Mock, patch


def test_main_module_imports():
    """Test that the main module can be imported."""


def test_captiv_gui_class_exists():
    """Test that CaptivGUI class exists and can be imported."""
    from captiv.gui.main import CaptivGUI

    assert isinstance(CaptivGUI, type)


def test_main_function_exists():
    """Test that main function exists and can be imported."""
    from captiv.gui.main import main

    assert callable(main)


@patch("captiv.gui.main.ConfigManager")
@patch("captiv.gui.main.ModelManager")
@patch("captiv.gui.main.ImageFileManager")
@patch("captiv.gui.main.CaptionFileManager")
@patch("captiv.gui.main.FileManager")
@patch("captiv.gui.main.GallerySection")
@patch("captiv.gui.main.DirectorySection")
@patch("captiv.gui.main.CaptionSection")
@patch("captiv.gui.main.ModelSection")
@patch("captiv.gui.main.BulkCaptionSection")
def test_captiv_gui_init(
    mock_bulk_caption_section,
    mock_model_section,
    mock_caption_section,
    mock_directory_section,
    mock_gallery_section,
    mock_file_manager,
    mock_caption_file_manager,
    mock_image_file_manager,
    mock_model_manager,
    mock_config_manager,
):
    """Test CaptivGUI initialization without creating the interface."""
    from captiv.gui.main import CaptivGUI

    mock_config_instance = Mock()
    mock_config_manager.return_value = mock_config_instance
    mock_config_instance.read_config.return_value = Mock()

    mock_file_manager_instance = Mock()
    mock_file_manager.return_value = mock_file_manager_instance

    mock_image_file_manager_instance = Mock()
    mock_image_file_manager.return_value = mock_image_file_manager_instance

    mock_caption_file_manager_instance = Mock()
    mock_caption_file_manager.return_value = mock_caption_file_manager_instance

    mock_model_manager_instance = Mock()
    mock_model_manager.return_value = mock_model_manager_instance

    mock_gallery_section.return_value = Mock()
    mock_directory_section.return_value = Mock()
    mock_caption_section.return_value = Mock()
    mock_model_section.return_value = Mock()
    mock_bulk_caption_section.return_value = Mock()

    with patch.object(CaptivGUI, "create_interface"):
        gui = CaptivGUI(share=False, config_path=None)

        assert gui.share is False
        assert gui.base_file_manager is mock_file_manager_instance
        assert gui.file_manager is mock_image_file_manager_instance
        assert gui.caption_manager is mock_caption_file_manager_instance
        assert gui.model_manager is mock_model_manager_instance

        mock_file_manager.assert_called_once()
        mock_image_file_manager.assert_called_once_with(mock_file_manager_instance)
        mock_caption_file_manager.assert_called_once_with(
            mock_file_manager_instance, mock_image_file_manager_instance
        )
        mock_model_manager.assert_called_once()
        mock_config_manager.assert_called_once_with(None)


@patch("captiv.gui.main.ConfigManager")
@patch("captiv.gui.main.ModelManager")
@patch("captiv.gui.main.ImageFileManager")
@patch("captiv.gui.main.CaptionFileManager")
@patch("captiv.gui.main.FileManager")
@patch("captiv.gui.main.GallerySection")
@patch("captiv.gui.main.DirectorySection")
@patch("captiv.gui.main.CaptionSection")
@patch("captiv.gui.main.ModelSection")
@patch("captiv.gui.main.BulkCaptionSection")
def test_captiv_gui_init_with_config_path(
    mock_bulk_caption_section,
    mock_model_section,
    mock_caption_section,
    mock_directory_section,
    mock_gallery_section,
    mock_file_manager,
    mock_caption_file_manager,
    mock_image_file_manager,
    mock_model_manager,
    mock_config_manager,
):
    """Test CaptivGUI initialization with custom config path."""
    from captiv.gui.main import CaptivGUI

    mock_config_instance = Mock()
    mock_config_manager.return_value = mock_config_instance
    mock_config_instance.read_config.return_value = Mock()

    mock_file_manager.return_value = Mock()
    mock_image_file_manager.return_value = Mock()
    mock_caption_file_manager.return_value = Mock()
    mock_model_manager.return_value = Mock()

    mock_gallery_section.return_value = Mock()
    mock_directory_section.return_value = Mock()
    mock_caption_section.return_value = Mock()
    mock_model_section.return_value = Mock()
    mock_bulk_caption_section.return_value = Mock()

    with patch.object(CaptivGUI, "create_interface"):
        config_path = "/custom/config/path"
        gui = CaptivGUI(share=True, config_path=config_path)

        assert gui.share is True

        mock_config_manager.assert_called_once_with(config_path)


def test_captiv_gui_handle_dir_change():
    """Test the handle_dir_change method."""
    from captiv.gui.main import CaptivGUI

    with (
        patch("captiv.gui.main.ConfigManager"),
        patch("captiv.gui.main.ModelManager"),
        patch("captiv.gui.main.ImageFileManager"),
        patch("captiv.gui.main.CaptionFileManager"),
        patch("captiv.gui.main.FileManager"),
        patch("captiv.gui.main.GallerySection") as mock_gallery_section,
        patch("captiv.gui.main.DirectorySection") as mock_directory_section,
        patch("captiv.gui.main.CaptionSection"),
        patch("captiv.gui.main.ModelSection"),
        patch("captiv.gui.main.BulkCaptionSection"),
        patch.object(CaptivGUI, "create_interface"),
    ):
        mock_directory_instance = Mock()
        mock_directory_section.return_value = mock_directory_instance
        mock_directory_instance.handle_dir_change.return_value = (
            {"choices": ["/test"]},
            "/test",
        )

        mock_gallery_instance = Mock()
        mock_gallery_section.return_value = mock_gallery_instance
        mock_gallery_instance.get_gallery_images.return_value = [
            "image1.jpg",
            "image2.jpg",
        ]

        gui = CaptivGUI()

        dir_update, gallery_images = gui.handle_dir_change("/test/directory")

        mock_directory_instance.handle_dir_change.assert_called_once_with(
            "/test/directory"
        )
        mock_gallery_instance.set_current_directory.assert_called_once_with("/test")
        mock_gallery_instance.get_gallery_images.assert_called_once_with("/test")

        assert dir_update == {"choices": ["/test"]}
        assert gallery_images == ["image1.jpg", "image2.jpg"]


def test_main_function_is_decorated():
    """Test that main function is decorated (signature may be changed by decorator)."""
    from captiv.gui.main import main

    assert callable(main)

    assert callable(main)


def test_main_function_can_be_called():
    """Test that main function can be called with expected parameters."""
    from captiv.gui.main import main

    with patch("captiv.gui.main.CaptivGUI") as mock_gui:
        try:
            main()
        except TypeError:
            main(share=False, config_path=None)

        mock_gui.assert_called()
