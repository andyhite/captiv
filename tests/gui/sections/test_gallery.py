"""Tests for the captiv.gui.sections.gallery module."""

from pathlib import Path
from unittest.mock import Mock, patch


def test_gallery_module_imports():
    """Test that the gallery module can be imported."""


def test_gallery_section_class_exists():
    """Test that GallerySection class exists and can be imported."""
    from captiv.gui.sections.gallery import GallerySection

    assert isinstance(GallerySection, type)


@patch("captiv.gui.sections.gallery.os.path.expanduser")
def test_gallery_section_init(mock_expanduser):
    """Test GallerySection initialization."""
    from captiv.gui.sections.gallery import GallerySection

    mock_expanduser.return_value = "/home/user"

    mock_caption_manager = Mock()

    section = GallerySection(caption_manager=mock_caption_manager)

    assert section.caption_manager is mock_caption_manager
    assert section.current_directory == "/home/user"
    assert section.current_image is None
    assert section.gallery is None
    assert section.selected_image is None


@patch("captiv.gui.sections.gallery.gr")
def test_gallery_section_create_section(mock_gr):
    """Test GallerySection create_section method."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()

    mock_gallery = Mock()
    mock_state = Mock()

    mock_gr.Gallery.return_value = mock_gallery
    mock_gr.State.return_value = mock_state

    section = GallerySection(caption_manager=mock_caption_manager)

    result = section.create_section()

    assert result == (mock_gallery, mock_state)
    assert section.gallery is mock_gallery
    assert section.selected_image is mock_state

    mock_gr.Gallery.assert_called_once_with(
        label="Images",
        show_label=True,
        columns=6,
        object_fit="cover",
        elem_classes="gallery",
        scale=1,
    )
    mock_gr.State.assert_called_once_with(value=None)


def test_gallery_section_on_gallery_select_with_index():
    """Test on_gallery_select with event containing index."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()
    mock_caption_manager.list_images_and_captions.return_value = [
        ("image1.jpg", None),
        ("image2.jpg", "caption"),
    ]

    section = GallerySection(caption_manager=mock_caption_manager)
    section.current_directory = "/test/directory"

    mock_event = Mock()
    mock_event.index = 1

    with (
        patch(
            "captiv.gui.sections.gallery.os.path.join",
            return_value="/test/directory/image2.jpg",
        ),
        patch("captiv.gui.sections.gallery.os.path.exists", return_value=True),
        patch("captiv.gui.sections.gallery.os.path.isdir", return_value=False),
        patch("captiv.gui.sections.gallery.is_image_file", return_value=True),
        patch(
            "captiv.gui.sections.gallery.os.path.abspath",
            return_value="/test/directory/image2.jpg",
        ),
    ):
        result = section.on_gallery_select(mock_event)

    assert result == "/test/directory/image2.jpg"
    assert section.current_image == "/test/directory/image2.jpg"


def test_gallery_section_on_gallery_select_with_value_string():
    """Test on_gallery_select with event containing string value."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()

    section = GallerySection(caption_manager=mock_caption_manager)

    mock_event = Mock()
    mock_event.value = "/test/directory/image.jpg"

    with (
        patch("captiv.gui.sections.gallery.os.path.exists", return_value=True),
        patch("captiv.gui.sections.gallery.os.path.isdir", return_value=False),
        patch("captiv.gui.sections.gallery.is_image_file", return_value=True),
        patch(
            "captiv.gui.sections.gallery.os.path.abspath",
            return_value="/test/directory/image.jpg",
        ),
    ):
        result = section.on_gallery_select(mock_event)

    assert result == "/test/directory/image.jpg"
    assert section.current_image == "/test/directory/image.jpg"


def test_gallery_section_on_gallery_select_with_dict_value():
    """Test on_gallery_select with event containing dict value."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()

    section = GallerySection(caption_manager=mock_caption_manager)

    mock_event = Mock()
    mock_event.value = {"image": {"path": "/test/directory/image.jpg"}}

    with (
        patch("captiv.gui.sections.gallery.os.path.exists", return_value=True),
        patch("captiv.gui.sections.gallery.os.path.isdir", return_value=False),
        patch("captiv.gui.sections.gallery.is_image_file", return_value=True),
        patch(
            "captiv.gui.sections.gallery.os.path.abspath",
            return_value="/test/directory/image.jpg",
        ),
    ):
        result = section.on_gallery_select(mock_event)

    assert result == "/test/directory/image.jpg"
    assert section.current_image == "/test/directory/image.jpg"


def test_gallery_section_on_gallery_select_none_event():
    """Test on_gallery_select with None event."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()

    section = GallerySection(caption_manager=mock_caption_manager)

    result = section.on_gallery_select(None)

    assert result == ""


def test_gallery_section_on_gallery_select_invalid_path():
    """Test on_gallery_select with invalid image path."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()

    section = GallerySection(caption_manager=mock_caption_manager)

    mock_event = Mock()
    mock_event.value = "/nonexistent/image.jpg"

    with patch("captiv.gui.sections.gallery.os.path.exists", return_value=False):
        result = section.on_gallery_select(mock_event)

    assert result == ""


def test_gallery_section_on_gallery_select_not_image_file():
    """Test on_gallery_select with non-image file."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()

    section = GallerySection(caption_manager=mock_caption_manager)

    mock_event = Mock()
    mock_event.value = "/test/directory/document.txt"

    with (
        patch("captiv.gui.sections.gallery.os.path.exists", return_value=True),
        patch("captiv.gui.sections.gallery.os.path.isdir", return_value=False),
        patch("captiv.gui.sections.gallery.is_image_file", return_value=False),
    ):
        result = section.on_gallery_select(mock_event)

    assert result == ""


def test_gallery_section_get_gallery_images_success():
    """Test get_gallery_images with successful execution."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()
    mock_caption_manager.list_images_and_captions.return_value = [
        ("image1.jpg", None),
        ("image2.jpg", "caption"),
        ("image3.png", None),
    ]

    section = GallerySection(caption_manager=mock_caption_manager)

    with (
        patch(
            "captiv.gui.sections.gallery.os.path.join",
            side_effect=lambda *args: "/".join(args),
        ),
        patch("captiv.gui.sections.gallery.os.path.isfile", return_value=True),
    ):
        result = section.get_gallery_images("/test/directory")

    expected = [
        "/test/directory/image1.jpg",
        "/test/directory/image2.jpg",
        "/test/directory/image3.png",
    ]
    assert result == expected
    mock_caption_manager.list_images_and_captions.assert_called_once_with(
        Path("/test/directory")
    )


def test_gallery_section_get_gallery_images_with_missing_files():
    """Test get_gallery_images when some files don't exist."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()
    mock_caption_manager.list_images_and_captions.return_value = [
        ("image1.jpg", None),
        ("missing.jpg", None),
        ("image3.png", None),
    ]

    section = GallerySection(caption_manager=mock_caption_manager)

    def mock_isfile(path):
        return path.endswith(("image1.jpg", "image3.png"))

    with (
        patch(
            "captiv.gui.sections.gallery.os.path.join",
            side_effect=lambda *args: "/".join(args),
        ),
        patch("captiv.gui.sections.gallery.os.path.isfile", side_effect=mock_isfile),
    ):
        result = section.get_gallery_images("/test/directory")

    expected = [
        "/test/directory/image1.jpg",
        "/test/directory/image3.png",
    ]
    assert result == expected


def test_gallery_section_get_gallery_images_exception():
    """Test get_gallery_images when an exception occurs."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()
    mock_caption_manager.list_images_and_captions.side_effect = Exception("Test error")

    section = GallerySection(caption_manager=mock_caption_manager)

    result = section.get_gallery_images("/test/directory")

    assert result == []


def test_gallery_section_set_current_directory():
    """Test set_current_directory method."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()

    section = GallerySection(caption_manager=mock_caption_manager)

    section.set_current_directory("/new/directory")

    assert section.current_directory == "/new/directory"


def test_gallery_section_imports_and_dependencies():
    """Test that all required imports and dependencies are available."""
    from captiv.gui.sections.gallery import (
        CaptionFileManager,
        GallerySection,
        gr,
        is_image_file,
        logger,
    )

    assert GallerySection is not None
    assert gr is not None
    assert logger is not None
    assert is_image_file is not None
    assert CaptionFileManager is not None


def test_gallery_section_method_signatures():
    """Test that methods have the expected signatures."""
    import inspect

    from captiv.gui.sections.gallery import GallerySection

    init_sig = inspect.signature(GallerySection.__init__)
    init_params = list(init_sig.parameters.keys())
    assert "self" in init_params
    assert "caption_manager" in init_params

    create_sig = inspect.signature(GallerySection.create_section)
    create_params = list(create_sig.parameters.keys())
    assert "self" in create_params

    select_sig = inspect.signature(GallerySection.on_gallery_select)
    select_params = list(select_sig.parameters.keys())
    assert "self" in select_params
    assert "evt" in select_params

    images_sig = inspect.signature(GallerySection.get_gallery_images)
    images_params = list(images_sig.parameters.keys())
    assert "self" in images_params
    assert "directory" in images_params

    set_dir_sig = inspect.signature(GallerySection.set_current_directory)
    set_dir_params = list(set_dir_sig.parameters.keys())
    assert "self" in set_dir_params
    assert "directory" in set_dir_params


def test_gallery_section_exception_handling_in_select():
    """Test that on_gallery_select handles exceptions properly."""
    from captiv.gui.sections.gallery import GallerySection

    mock_caption_manager = Mock()

    section = GallerySection(caption_manager=mock_caption_manager)

    mock_event = Mock()
    mock_event.value = "/test/path"

    with patch(
        "captiv.gui.sections.gallery.os.path.exists",
        side_effect=Exception("Test error"),
    ):
        result = section.on_gallery_select(mock_event)

    assert result == ""
