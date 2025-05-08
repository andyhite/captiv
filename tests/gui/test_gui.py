"""
Tests for the Captiv GUI components.
"""

from unittest.mock import MagicMock, patch

import pytest

from captiv.gui.main import CaptivGUI
from captiv.services.caption_manager import CaptionManager
from captiv.services.image_file_manager import ImageFileManager
from captiv.services.model_manager import ModelManager, ModelType


@pytest.fixture
def mock_gradio():
    """Mock Gradio components."""
    with (
        patch("gradio.Blocks") as mock_blocks,
        patch("gradio.Row") as mock_row,
        patch("gradio.Column") as mock_column,
        patch("gradio.Tabs") as mock_tabs,
        patch("gradio.Tab") as mock_tab,
        patch("gradio.Gallery") as mock_gallery,
        patch("gradio.Dropdown") as mock_dropdown,
        patch("gradio.Textbox") as mock_textbox,
        patch("gradio.Button") as mock_button,
        patch("gradio.Markdown") as mock_markdown,
    ):
        # Configure mocks
        mock_blocks.return_value.__enter__.return_value = mock_blocks.return_value
        mock_row.return_value.__enter__.return_value = mock_row.return_value
        mock_column.return_value.__enter__.return_value = mock_column.return_value
        mock_tabs.return_value.__enter__.return_value = mock_tabs.return_value
        mock_tab.return_value.__enter__.return_value = mock_tab.return_value

        # Return all mocks
        yield {
            "blocks": mock_blocks,
            "row": mock_row,
            "column": mock_column,
            "tabs": mock_tabs,
            "tab": mock_tab,
            "gallery": mock_gallery,
            "dropdown": mock_dropdown,
            "textbox": mock_textbox,
            "button": mock_button,
            "markdown": mock_markdown,
        }


@pytest.fixture
def mock_services():
    """Mock service classes."""
    with (
        patch("captiv.gui.main.CaptionManager") as mock_caption_manager,
        patch("captiv.gui.main.ModelManager") as mock_model_manager,
        patch("captiv.gui.main.ImageFileManager") as mock_file_manager,
        patch("captiv.gui.main.read_config") as mock_read_config,
    ):
        # Configure mocks
        mock_caption_manager_instance = MagicMock(spec=CaptionManager)
        mock_model_manager_instance = MagicMock(spec=ModelManager)
        mock_file_manager_instance = MagicMock(spec=ImageFileManager)

        mock_caption_manager.return_value = mock_caption_manager_instance
        mock_model_manager.return_value = mock_model_manager_instance
        mock_file_manager.return_value = mock_file_manager_instance

        # Mock config
        mock_config = MagicMock()
        mock_config.gui.host = "localhost"
        mock_config.gui.port = 7860
        mock_config.gui.ssl_keyfile = None
        mock_config.gui.ssl_certfile = None
        mock_read_config.return_value = mock_config

        yield {
            "caption_manager": mock_caption_manager_instance,
            "model_manager": mock_model_manager_instance,
            "file_manager": mock_file_manager_instance,
            "config": mock_config,
        }


@pytest.fixture
def mock_gui_sections():
    """Mock GUI sections."""
    with (
        patch("captiv.gui.main.GallerySection") as mock_gallery_section,
        patch("captiv.gui.main.DirectorySection") as mock_directory_section,
        patch("captiv.gui.main.CaptionSection") as mock_caption_section,
        patch("captiv.gui.main.ModelSection") as mock_model_section,
        patch("captiv.gui.main.BulkCaptionSection") as mock_bulk_caption_section,
    ):
        # Configure mocks
        mock_gallery_section_instance = MagicMock()
        mock_directory_section_instance = MagicMock()
        mock_caption_section_instance = MagicMock()
        mock_model_section_instance = MagicMock()
        mock_bulk_caption_section_instance = MagicMock()

        mock_gallery_section.return_value = mock_gallery_section_instance
        mock_directory_section.return_value = mock_directory_section_instance
        mock_caption_section.return_value = mock_caption_section_instance
        mock_model_section.return_value = mock_model_section_instance
        mock_bulk_caption_section.return_value = mock_bulk_caption_section_instance

        # Mock section create_section methods
        mock_gallery = MagicMock()
        mock_selected_image = MagicMock()
        mock_gallery_section_instance.create_section.return_value = (
            mock_gallery,
            mock_selected_image,
        )

        mock_dir_dropdown = MagicMock()
        mock_directory_section_instance.create_section.return_value = mock_dir_dropdown

        mock_caption_textbox = MagicMock()
        mock_save_status = MagicMock()
        mock_save_caption_btn = MagicMock()
        mock_generate_caption_btn = MagicMock()
        mock_caption_section_instance.create_section.return_value = (
            mock_caption_textbox,
            mock_save_status,
            mock_save_caption_btn,
            mock_generate_caption_btn,
        )

        mock_bulk_caption_btn = MagicMock()
        mock_bulk_caption_status = MagicMock()
        mock_bulk_caption_section_instance.create_section.return_value = (
            mock_bulk_caption_btn,
            mock_bulk_caption_status,
        )

        mock_model_type_dropdown = MagicMock()
        mock_model_dropdown = MagicMock()
        mock_mode_dropdown = MagicMock()
        mock_prompt_textbox = MagicMock()
        mock_advanced_options = MagicMock()
        mock_model_section_instance.create_section.return_value = (
            mock_model_type_dropdown,
            mock_model_dropdown,
            mock_mode_dropdown,
            mock_prompt_textbox,
            mock_advanced_options,
        )

        yield {
            "gallery_section": mock_gallery_section_instance,
            "directory_section": mock_directory_section_instance,
            "caption_section": mock_caption_section_instance,
            "model_section": mock_model_section_instance,
            "bulk_caption_section": mock_bulk_caption_section_instance,
            "gallery": mock_gallery,
            "selected_image": mock_selected_image,
            "dir_dropdown": mock_dir_dropdown,
            "caption_textbox": mock_caption_textbox,
            "save_status": mock_save_status,
            "save_caption_btn": mock_save_caption_btn,
            "generate_caption_btn": mock_generate_caption_btn,
            "bulk_caption_btn": mock_bulk_caption_btn,
            "bulk_caption_status": mock_bulk_caption_status,
            "model_type_dropdown": mock_model_type_dropdown,
            "model_dropdown": mock_model_dropdown,
            "mode_dropdown": mock_mode_dropdown,
            "prompt_textbox": mock_prompt_textbox,
            "advanced_options": mock_advanced_options,
        }


@patch("captiv.gui.main.CaptivGUI.create_interface")
def test_gui_initialization(mock_create_interface, mock_services, mock_gui_sections):
    """Test that the GUI initializes correctly."""
    # Prevent the actual interface from being created
    mock_create_interface.return_value = None

    # Create the GUI
    gui = CaptivGUI(share=False, config_path=None)

    # Check that services were initialized
    assert gui.caption_manager == mock_services["caption_manager"]
    assert gui.model_manager == mock_services["model_manager"]
    assert gui.file_manager == mock_services["file_manager"]

    # Check that sections were initialized
    assert gui.gallery_section == mock_gui_sections["gallery_section"]
    assert gui.directory_section == mock_gui_sections["directory_section"]
    assert gui.caption_section == mock_gui_sections["caption_section"]
    assert gui.model_section == mock_gui_sections["model_section"]
    assert gui.bulk_caption_section == mock_gui_sections["bulk_caption_section"]

    # Check that create_interface was called
    mock_create_interface.assert_called_once()


def test_gui_event_handlers_setup(
    mock_services,
    mock_gui_sections,
    mock_gradio,
):
    """Test that the GUI event handlers are set up correctly."""
    # Prevent the actual interface from being created
    with patch("captiv.gui.main.CaptivGUI.create_interface"):
        gui = CaptivGUI(share=False, config_path=None)

    # Manually set up the GUI components that would be created by create_interface
    gui.gallery = mock_gui_sections["gallery"]
    gui.selected_image = mock_gui_sections["selected_image"]
    gui.dir_dropdown = mock_gui_sections["dir_dropdown"]
    gui.caption_textbox = mock_gui_sections["caption_textbox"]
    gui.save_status = mock_gui_sections["save_status"]
    gui.save_caption_btn = mock_gui_sections["save_caption_btn"]
    gui.generate_caption_btn = mock_gui_sections["generate_caption_btn"]
    gui.model_type_dropdown = mock_gui_sections["model_type_dropdown"]
    gui.model_dropdown = mock_gui_sections["model_dropdown"]
    gui.mode_dropdown = mock_gui_sections["mode_dropdown"]
    gui.prompt_textbox = mock_gui_sections["prompt_textbox"]
    gui.advanced_options = mock_gui_sections["advanced_options"]
    gui.bulk_caption_btn = mock_gui_sections["bulk_caption_btn"]
    gui.bulk_caption_status = mock_gui_sections["bulk_caption_status"]
    gui.caption_progress = mock_gui_sections.get("caption_progress", MagicMock())

    # Call setup_event_handlers manually
    gui.setup_event_handlers()

    # Check that event handlers were set up
    assert gui.gallery.select.called
    assert gui.selected_image.change.called
    assert gui.dir_dropdown.change.called
    assert gui.save_caption_btn.click.called
    assert gui.generate_caption_btn.click.called
    assert gui.model_type_dropdown.change.called
    assert gui.mode_dropdown.change.called
    assert gui.bulk_caption_btn.click.called


@patch("captiv.gui.main.CaptivGUI.create_interface")
def test_handle_dir_change(mock_create_interface, mock_services, mock_gui_sections):
    """Test the handle_dir_change method."""
    # Prevent the actual interface from being created
    mock_create_interface.return_value = None

    # Create the GUI
    gui = CaptivGUI(share=False, config_path=None)

    # Mock the directory section's handle_dir_change method
    new_path = "/fake/path"
    dir_update = {"choices": ["/fake/path"], "value": "/fake/path"}
    mock_gui_sections["directory_section"].handle_dir_change.return_value = (
        dir_update,
        new_path,
    )

    # Mock the gallery section's get_gallery_images method
    gallery_images = ["image1.jpg", "image2.jpg"]
    mock_gui_sections[
        "gallery_section"
    ].get_gallery_images.return_value = gallery_images

    # Call handle_dir_change
    result = gui.handle_dir_change("/fake/path")

    # Check that the directory section's handle_dir_change method was called
    mock_gui_sections["directory_section"].handle_dir_change.assert_called_once_with(
        "/fake/path"
    )

    # Check that the gallery section's set_current_directory method was called
    mock_gui_sections["gallery_section"].set_current_directory.assert_called_once_with(
        new_path
    )

    # Check that the gallery section's get_gallery_images method was called
    mock_gui_sections["gallery_section"].get_gallery_images.assert_called_once_with(
        new_path
    )

    # Check the result
    assert result == (dir_update, gallery_images)


@patch("captiv.gui.main.CaptivGUI.create_interface")
def test_on_generate_caption(mock_create_interface, mock_services, mock_gui_sections):
    """Test the on_generate_caption method."""
    # Prevent the actual interface from being created
    mock_create_interface.return_value = None

    # Create the GUI
    gui = CaptivGUI(share=False, config_path=None)

    # Mock the caption manager's generate_caption method
    mock_services["caption_manager"].generate_caption.return_value = "Generated caption"

    # Call on_generate_caption
    result = gui.on_generate_caption(
        image_path="/fake/path/image.jpg",
        model_type_str="blip",
        model="blip-large",
        mode="detailed",
        prompt="A custom prompt",
        max_length=50,
        min_length=10,
        num_beams=5,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    # Check that the caption manager's generate_caption method was called
    mock_services["caption_manager"].generate_caption.assert_called_once()
    call_kwargs = mock_services["caption_manager"].generate_caption.call_args.kwargs

    assert call_kwargs["model_type"] == ModelType.BLIP
    assert call_kwargs["image_path"] == "/fake/path/image.jpg"
    assert call_kwargs["variant"] == "blip-large"
    assert call_kwargs["mode"] == "detailed"
    assert call_kwargs["prompt"] == "A custom prompt"
    assert call_kwargs["max_length"] == 50
    assert call_kwargs["min_length"] == 10
    assert call_kwargs["num_beams"] == 5
    assert call_kwargs["temperature"] == 0.8
    assert call_kwargs["top_k"] == 40
    assert call_kwargs["top_p"] == 0.95
    assert call_kwargs["repetition_penalty"] == 1.2

    # Check the result
    # The method now returns three values: caption, status, progress_status
    caption, status, progress_status = result
    assert caption == "Generated caption"
    assert status == "Caption generated successfully"
    assert "Caption generated in" in progress_status


@patch("captiv.gui.main.CaptivGUI.create_interface")
def test_on_generate_caption_no_image(
    mock_create_interface, mock_services, mock_gui_sections
):
    """Test the on_generate_caption method with no image selected."""
    # Prevent the actual interface from being created
    mock_create_interface.return_value = None

    # Create the GUI
    gui = CaptivGUI(share=False, config_path=None)

    # Call on_generate_caption with no image
    with patch("gradio.Warning") as mock_warning:
        result = gui.on_generate_caption(
            image_path="",
            model_type_str="blip",
            model="blip-large",
            mode="detailed",
            prompt="A custom prompt",
            max_length=50,
            min_length=10,
            num_beams=5,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.2,
        )

    # Check that the caption manager's generate_caption method was not called
    mock_services["caption_manager"].generate_caption.assert_not_called()

    # Check that a warning was shown
    mock_warning.assert_called_once_with("No image selected")

    # Check the result
    # The method now returns three values: caption, status, progress_status
    caption, status, progress_status = result
    assert caption == "No image selected"
    assert status == "No image selected"
    assert progress_status == "No image selected"


@patch("captiv.gui.main.CaptivGUI.create_interface")
def test_on_generate_caption_error(
    mock_create_interface, mock_services, mock_gui_sections
):
    """Test the on_generate_caption method when an error occurs."""
    # Prevent the actual interface from being created
    mock_create_interface.return_value = None

    # Create the GUI
    gui = CaptivGUI(share=False, config_path=None)

    # Mock the caption manager's generate_caption method to raise an error
    mock_services["caption_manager"].generate_caption.side_effect = ValueError(
        "Test error"
    )

    # Call on_generate_caption
    with patch("gradio.Error") as mock_error:
        result = gui.on_generate_caption(
            image_path="/fake/path/image.jpg",
            model_type_str="blip",
            model="blip-large",
            mode="detailed",
            prompt="A custom prompt",
            max_length=50,
            min_length=10,
            num_beams=5,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.2,
        )

    # Check that an error was shown
    mock_error.assert_called_once()

    # Check the result
    assert "Error generating caption" in result[0]
    assert "Test error" in result[0]
