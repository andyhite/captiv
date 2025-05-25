"""Tests for the captiv.gui.sections.caption module."""

from pathlib import Path
from unittest.mock import Mock, patch


def test_caption_module_imports():
    """Test that the caption module can be imported."""


def test_caption_section_class_exists():
    """Test that CaptionSection class exists and can be imported."""
    from captiv.gui.sections.caption import CaptionSection

    assert isinstance(CaptionSection, type)


def test_caption_section_init():
    """Test CaptionSection initialization."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()

    section = CaptionSection(caption_manager=mock_caption_manager)

    assert section.caption_manager is mock_caption_manager
    assert section.caption_textbox is None
    assert section.save_caption_btn is None
    assert section.generate_caption_btn is None


@patch("captiv.gui.sections.caption.gr")
def test_caption_section_create_section(mock_gr):
    """Test CaptionSection create_section method."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()

    mock_textbox = Mock()
    mock_save_button = Mock()
    mock_generate_button = Mock()

    mock_gr.Textbox.return_value = mock_textbox
    mock_gr.Button.side_effect = [mock_save_button, mock_generate_button]
    mock_gr.Row.return_value.__enter__ = Mock()
    mock_gr.Row.return_value.__exit__ = Mock()

    section = CaptionSection(caption_manager=mock_caption_manager)

    result = section.create_section()

    assert result == (mock_textbox, mock_save_button, mock_generate_button)
    assert section.caption_textbox is mock_textbox
    assert section.save_caption_btn is mock_save_button
    assert section.generate_caption_btn is mock_generate_button

    mock_gr.Textbox.assert_called_once_with(
        label="Caption",
        placeholder="Select an image to view or edit its caption",
        lines=4,
        interactive=True,
    )
    mock_gr.Row.assert_called_once()
    assert mock_gr.Button.call_count == 2


def test_caption_section_on_image_select_valid_image():
    """Test on_image_select with valid image path."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()
    mock_caption_manager.read_caption.return_value = "Test caption"

    section = CaptionSection(caption_manager=mock_caption_manager)

    with patch("captiv.gui.sections.caption.os.path.isdir", return_value=False):
        result = section.on_image_select("/path/to/image.jpg")

    assert result == "Test caption"
    mock_caption_manager.read_caption.assert_called_once_with(
        Path("/path/to/image.jpg")
    )


def test_caption_section_on_image_select_no_caption():
    """Test on_image_select when no caption exists."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()
    mock_caption_manager.read_caption.return_value = None

    section = CaptionSection(caption_manager=mock_caption_manager)

    with patch("captiv.gui.sections.caption.os.path.isdir", return_value=False):
        result = section.on_image_select("/path/to/image.jpg")

    assert result == ""
    mock_caption_manager.read_caption.assert_called_once_with(
        Path("/path/to/image.jpg")
    )


def test_caption_section_on_image_select_empty_path():
    """Test on_image_select with empty image path."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()

    section = CaptionSection(caption_manager=mock_caption_manager)

    result = section.on_image_select("")

    assert result == ""
    mock_caption_manager.read_caption.assert_not_called()


def test_caption_section_on_image_select_directory():
    """Test on_image_select with directory path."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()

    section = CaptionSection(caption_manager=mock_caption_manager)

    with patch("captiv.gui.sections.caption.os.path.isdir", return_value=True):
        result = section.on_image_select("/path/to/directory")

    assert result == ""
    mock_caption_manager.read_caption.assert_not_called()


def test_caption_section_on_image_select_file_not_found():
    """Test on_image_select when caption file is not found."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()
    mock_caption_manager.read_caption.side_effect = FileNotFoundError(
        "Caption file not found"
    )

    section = CaptionSection(caption_manager=mock_caption_manager)

    with patch("captiv.gui.sections.caption.os.path.isdir", return_value=False):
        result = section.on_image_select("/path/to/image.jpg")

    assert result == ""
    mock_caption_manager.read_caption.assert_called_once_with(
        Path("/path/to/image.jpg")
    )


@patch("captiv.gui.sections.caption.gr")
def test_caption_section_on_save_caption_success(mock_gr):
    """Test on_save_caption with successful save."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()

    section = CaptionSection(caption_manager=mock_caption_manager)

    result = section.on_save_caption("/path/to/image.jpg", "Test caption")

    assert "Caption saved for image.jpg" in result
    mock_caption_manager.write_caption.assert_called_once_with(
        Path("/path/to/image.jpg"), "Test caption"
    )
    mock_gr.Info.assert_called_once()


@patch("captiv.gui.sections.caption.gr")
def test_caption_section_on_save_caption_no_image(mock_gr):
    """Test on_save_caption with no image selected."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()

    section = CaptionSection(caption_manager=mock_caption_manager)

    result = section.on_save_caption("", "Test caption")

    assert result == "No image selected"
    mock_caption_manager.write_caption.assert_not_called()
    mock_gr.Warning.assert_called_once_with("No image selected")


@patch("captiv.gui.sections.caption.gr")
def test_caption_section_on_save_caption_file_not_found(mock_gr):
    """Test on_save_caption when image file is not found."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()
    mock_caption_manager.write_caption.side_effect = FileNotFoundError(
        "Image file not found"
    )

    section = CaptionSection(caption_manager=mock_caption_manager)

    result = section.on_save_caption("/path/to/nonexistent.jpg", "Test caption")

    assert "Error: Image file not found." in result
    mock_caption_manager.write_caption.assert_called_once()
    mock_gr.Error.assert_called_once_with("Error: Image file not found.")


def test_caption_section_imports_and_dependencies():
    """Test that all required imports and dependencies are available."""
    from captiv.gui.sections.caption import (
        CaptionFileManager,
        CaptionSection,
        EnhancedError,
        gr,
        logger,
    )

    assert CaptionSection is not None
    assert gr is not None
    assert logger is not None
    assert CaptionFileManager is not None
    assert EnhancedError is not None


def test_caption_section_method_signatures():
    """Test that methods have the expected signatures."""
    import inspect

    from captiv.gui.sections.caption import CaptionSection

    init_sig = inspect.signature(CaptionSection.__init__)
    init_params = list(init_sig.parameters.keys())
    assert "self" in init_params
    assert "caption_manager" in init_params

    create_sig = inspect.signature(CaptionSection.create_section)
    create_params = list(create_sig.parameters.keys())
    assert "self" in create_params

    image_sig = inspect.signature(CaptionSection.on_image_select)
    image_params = list(image_sig.parameters.keys())
    assert "self" in image_params
    assert "image_path" in image_params

    save_sig = inspect.signature(CaptionSection.on_save_caption)
    save_params = list(save_sig.parameters.keys())
    assert "self" in save_params
    assert "image_path" in save_params
    assert "caption" in save_params


@patch("captiv.gui.sections.caption.gr")
@patch("captiv.gui.sections.caption.os.path.basename")
def test_caption_section_on_save_caption_basename_usage(mock_basename, mock_gr):
    """Test that on_save_caption uses basename for display."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()

    mock_basename.return_value = "test_image.jpg"

    section = CaptionSection(caption_manager=mock_caption_manager)

    result = section.on_save_caption("/long/path/to/test_image.jpg", "Test caption")

    mock_basename.assert_called_once_with("/long/path/to/test_image.jpg")

    assert "Caption saved for test_image.jpg" in result


def test_caption_section_exception_handling():
    """Test that CaptionSection handles exceptions properly."""
    from captiv.gui.sections.caption import CaptionSection

    mock_caption_manager = Mock()
    mock_caption_manager.read_caption.side_effect = Exception("Unexpected error")

    section = CaptionSection(caption_manager=mock_caption_manager)

    with patch("captiv.gui.sections.caption.os.path.isdir", return_value=False):
        result = section.on_image_select("/path/to/image.jpg")

    assert result == ""
