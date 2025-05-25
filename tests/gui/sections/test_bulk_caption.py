"""Tests for the captiv.gui.sections.bulk_caption module."""

from unittest.mock import Mock, patch


def test_bulk_caption_module_imports():
    """Test that the bulk_caption module can be imported."""


def test_bulk_caption_section_class_exists():
    """Test that BulkCaptionSection class exists and can be imported."""
    from captiv.gui.sections.bulk_caption import BulkCaptionSection

    assert isinstance(BulkCaptionSection, type)


def test_bulk_caption_section_init():
    """Test BulkCaptionSection initialization."""
    from captiv.gui.sections.bulk_caption import BulkCaptionSection

    mock_caption_manager = Mock()
    mock_file_manager = Mock()
    mock_model_manager = Mock()

    section = BulkCaptionSection(
        caption_manager=mock_caption_manager,
        file_manager=mock_file_manager,
        model_manager=mock_model_manager,
    )

    assert section.caption_manager is mock_caption_manager
    assert section.file_manager is mock_file_manager
    assert section.model_manager is mock_model_manager
    assert section.bulk_caption_btn is None


@patch("captiv.gui.sections.bulk_caption.gr")
def test_bulk_caption_section_create_section(mock_gr):
    """Test BulkCaptionSection create_section method."""
    from captiv.gui.sections.bulk_caption import BulkCaptionSection

    mock_caption_manager = Mock()
    mock_file_manager = Mock()
    mock_model_manager = Mock()

    mock_button = Mock()
    mock_gr.Button.return_value = mock_button
    mock_gr.Row.return_value.__enter__ = Mock()
    mock_gr.Row.return_value.__exit__ = Mock()

    section = BulkCaptionSection(
        caption_manager=mock_caption_manager,
        file_manager=mock_file_manager,
        model_manager=mock_model_manager,
    )

    result = section.create_section()

    assert result is mock_button
    assert section.bulk_caption_btn is mock_button

    mock_gr.Row.assert_called_once()
    mock_gr.Button.assert_called_once_with("Generate captions for all images", scale=2)


def test_bulk_caption_section_on_bulk_caption_method_exists():
    """Test that on_bulk_caption method exists and is decorated."""
    from captiv.gui.sections.bulk_caption import BulkCaptionSection

    mock_caption_manager = Mock()
    mock_file_manager = Mock()
    mock_model_manager = Mock()

    section = BulkCaptionSection(
        caption_manager=mock_caption_manager,
        file_manager=mock_file_manager,
        model_manager=mock_model_manager,
    )

    assert hasattr(section, "on_bulk_caption")
    assert callable(section.on_bulk_caption)


@patch("captiv.gui.sections.bulk_caption.gr")
@patch("captiv.gui.sections.bulk_caption.os")
def test_bulk_caption_section_on_bulk_caption_invalid_directory(mock_os, mock_gr):
    """Test on_bulk_caption with invalid directory."""
    from captiv.gui.sections.bulk_caption import BulkCaptionSection

    mock_caption_manager = Mock()
    mock_file_manager = Mock()
    mock_model_manager = Mock()

    mock_os.path.isdir.return_value = False

    section = BulkCaptionSection(
        caption_manager=mock_caption_manager,
        file_manager=mock_file_manager,
        model_manager=mock_model_manager,
    )

    mock_progress = Mock()

    result = section.on_bulk_caption(
        directory="",
        model_str="blip",
        model_variant="base",
        mode="descriptive",
        prompt="",
        max_new_tokens=100,
        min_new_tokens=10,
        num_beams=3,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        progress=mock_progress,
    )

    assert result == "Invalid directory"
    mock_gr.Warning.assert_called_once_with("Invalid directory")


@patch("captiv.gui.sections.bulk_caption.gr")
@patch("captiv.gui.sections.bulk_caption.os")
@patch("captiv.gui.sections.bulk_caption.ModelType")
def test_bulk_caption_section_on_bulk_caption_no_images(
    mock_model_type, mock_os, mock_gr
):
    """Test on_bulk_caption with no images in directory."""
    from captiv.gui.sections.bulk_caption import BulkCaptionSection

    mock_caption_manager = Mock()
    mock_file_manager = Mock()
    mock_model_manager = Mock()

    mock_os.path.isdir.return_value = True

    mock_caption_manager.list_images_and_captions.return_value = []

    mock_model_type.return_value = Mock()

    section = BulkCaptionSection(
        caption_manager=mock_caption_manager,
        file_manager=mock_file_manager,
        model_manager=mock_model_manager,
    )

    mock_progress = Mock()

    result = section.on_bulk_caption(
        directory="/test/directory",
        model_str="blip",
        model_variant="base",
        mode="descriptive",
        prompt="",
        max_new_tokens=100,
        min_new_tokens=10,
        num_beams=3,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        progress=mock_progress,
    )

    assert result == "No images found in the directory"
    mock_gr.Warning.assert_called_once_with("No images found in the directory")


@patch("captiv.gui.sections.bulk_caption.gr")
@patch("captiv.gui.sections.bulk_caption.os")
@patch("captiv.gui.sections.bulk_caption.ModelType")
@patch("captiv.gui.sections.bulk_caption.time")
def test_bulk_caption_section_on_bulk_caption_success(
    mock_time, mock_model_type, mock_os, mock_gr
):
    """Test on_bulk_caption successful execution."""
    from captiv.gui.sections.bulk_caption import BulkCaptionSection

    mock_caption_manager = Mock()
    mock_file_manager = Mock()
    mock_model_manager = Mock()

    mock_os.path.isdir.return_value = True
    mock_os.path.join.side_effect = lambda *args: "/".join(str(arg) for arg in args)

    mock_caption_manager.list_images_and_captions.return_value = [
        ("image1.jpg", None),
        ("image2.jpg", "existing caption"),
    ]

    mock_model = Mock()
    mock_model_type.return_value = mock_model

    mock_model_class = Mock()
    mock_model_class.DEFAULT_VARIANT = "base"
    mock_model_manager.get_model_class.return_value = mock_model_class
    mock_model_manager.get_variants_for_model.return_value = ["base"]

    mock_model_instance = Mock()
    mock_model_instance.caption_image.return_value = "Generated caption"
    mock_model_manager.create_model_instance.return_value = mock_model_instance
    mock_model_manager.build_generation_params.return_value = {}

    mock_time.time.side_effect = [0, 10]

    section = BulkCaptionSection(
        caption_manager=mock_caption_manager,
        file_manager=mock_file_manager,
        model_manager=mock_model_manager,
    )

    mock_progress = Mock()
    mock_progress.tqdm.return_value = enumerate(
        [
            ("image1.jpg", None),
            ("image2.jpg", "existing caption"),
        ]
    )

    result = section.on_bulk_caption(
        directory="/test/directory",
        model_str="blip",
        model_variant="base",
        mode="descriptive",
        prompt="",
        max_new_tokens=100,
        min_new_tokens=10,
        num_beams=3,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        progress=mock_progress,
    )

    assert "Captioning complete" in result
    assert "Generated captions for 0 images" in result
    assert "Skipped 2 images" in result

    mock_caption_manager.write_caption.assert_not_called()


def test_bulk_caption_section_imports_and_dependencies():
    """Test that all required imports and dependencies are available."""
    from captiv.gui.sections.bulk_caption import (
        BulkCaptionSection,
        CaptionFileManager,
        EnhancedError,
        ImageFileManager,
        ModelManager,
        ModelType,
        gr,
        handle_errors,
        logger,
    )

    assert BulkCaptionSection is not None
    assert gr is not None
    assert logger is not None
    assert CaptionFileManager is not None
    assert ImageFileManager is not None
    assert ModelManager is not None
    assert ModelType is not None
    assert EnhancedError is not None
    assert handle_errors is not None


def test_bulk_caption_section_method_signatures():
    """Test that methods have the expected signatures."""
    import inspect

    from captiv.gui.sections.bulk_caption import BulkCaptionSection

    init_sig = inspect.signature(BulkCaptionSection.__init__)
    init_params = list(init_sig.parameters.keys())
    assert "self" in init_params
    assert "caption_manager" in init_params
    assert "file_manager" in init_params
    assert "model_manager" in init_params

    create_sig = inspect.signature(BulkCaptionSection.create_section)
    create_params = list(create_sig.parameters.keys())
    assert "self" in create_params

    assert hasattr(BulkCaptionSection, "on_bulk_caption")
    assert callable(BulkCaptionSection.on_bulk_caption)
