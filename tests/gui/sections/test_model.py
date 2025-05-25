"""Tests for the captiv.gui.sections.model module."""

from unittest.mock import Mock, patch


def test_model_module_imports():
    """Test that the model module can be imported."""


def test_model_section_class_exists():
    """Test that ModelSection class exists and can be imported."""
    from captiv.gui.sections.model import ModelSection

    assert isinstance(ModelSection, type)


@patch("captiv.gui.sections.model.ModelType")
def test_model_section_init(mock_model_type):
    """Test ModelSection initialization."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_default_model = Mock()
    mock_default_model.value = "blip"
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    assert section.model_manager is mock_model_manager
    assert section.current_model is mock_default_model
    assert section.model_dropdown is None
    assert section.model_variant_dropdown is None
    assert section.mode_dropdown is None
    assert section.prompt_textbox is None


@patch("captiv.gui.sections.model.gr")
@patch("captiv.gui.sections.model.ModelType")
def test_model_section_create_section(mock_model_type, mock_gr):
    """Test ModelSection create_section method."""
    from captiv.gui.sections.model import ModelSection

    mock_model_type.__iter__ = Mock(
        return_value=iter([Mock(value="blip"), Mock(value="blip2")])
    )

    mock_model_manager = Mock()
    mock_default_model = Mock()
    mock_default_model.value = "blip"
    mock_model_manager.get_default_model.return_value = mock_default_model

    mock_model_class = Mock()
    mock_model_class.DEFAULT_VARIANT = "base"
    mock_model_manager.get_model_class.return_value = mock_model_class

    mock_dropdown1 = Mock()
    mock_dropdown2 = Mock()
    mock_dropdown3 = Mock()
    mock_textbox = Mock()
    mock_sliders = [Mock() for _ in range(7)]
    mock_accordion = Mock()
    mock_checkbox_group = Mock()
    mock_character_textbox = Mock()

    mock_gr.Dropdown.side_effect = [mock_dropdown1, mock_dropdown2, mock_dropdown3]
    mock_gr.Textbox.side_effect = [mock_textbox, mock_character_textbox]
    mock_gr.Slider.side_effect = mock_sliders
    mock_gr.Accordion.return_value.__enter__ = Mock(return_value=mock_accordion)
    mock_gr.Accordion.return_value.__exit__ = Mock()
    mock_gr.CheckboxGroup.return_value = mock_checkbox_group

    section = ModelSection(model_manager=mock_model_manager)

    section.get_variants_for_model = Mock(return_value=["base", "large"])
    section.get_modes_for_model = Mock(return_value=["descriptive", "custom"])
    section.get_prompt_options_for_model = Mock(return_value={})

    result = section.create_section()

    assert isinstance(result, tuple)
    assert len(result) == 14


def test_model_section_on_model_change():
    """Test on_model_change method."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_default_model = Mock()
    mock_default_model.value = "blip"
    mock_model_manager.get_default_model.return_value = mock_default_model

    mock_model_class = Mock()
    mock_model_class.DEFAULT_VARIANT = "base"
    mock_model_manager.get_model_class.return_value = mock_model_class

    section = ModelSection(model_manager=mock_model_manager)

    section.get_variants_for_model = Mock(return_value=["base", "large"])
    section.get_modes_for_model = Mock(return_value=["descriptive", "custom"])
    section.get_prompt_options_for_model = Mock(return_value={})
    section.update_prompt_options_for_model = Mock(return_value=(Mock(), Mock()))

    with patch("captiv.gui.sections.model.ModelType") as mock_model_type_class:
        mock_model_instance = Mock()
        mock_model_instance.value = "blip2"
        mock_model_type_class.return_value = mock_model_instance

        with patch("captiv.gui.sections.model.gr") as mock_gr:
            mock_gr.update.return_value = Mock()

            result = section.on_model_change("blip2")

    assert isinstance(result, tuple)
    assert len(result) == 5


def test_model_section_on_mode_change():
    """Test on_mode_change method."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_default_model = Mock()
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    with patch("captiv.gui.sections.model.gr") as mock_gr:
        mock_gr.update.return_value = {"visible": True}

        result = section.on_mode_change("custom")

        mock_gr.update.assert_called_once_with(visible=True)
        assert result == {"visible": True}


def test_model_section_get_variants_for_model():
    """Test get_variants_for_model method."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_model_manager.get_variants_for_model.return_value = ["base", "large"]
    mock_default_model = Mock()
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    mock_model = Mock()
    mock_model.value = "blip"

    result = section.get_variants_for_model(mock_model)

    assert result == ["base", "large"]
    mock_model_manager.get_variants_for_model.assert_called_once_with(mock_model)


def test_model_section_get_modes_for_model():
    """Test get_modes_for_model method."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_model_manager.get_modes_for_model.return_value = ["descriptive", "custom"]
    mock_default_model = Mock()
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    mock_model = Mock()
    mock_model.value = "blip"

    result = section.get_modes_for_model(mock_model)

    assert result == ["descriptive", "custom"]
    mock_model_manager.get_modes_for_model.assert_called_once_with(mock_model)


def test_model_section_get_prompt_options_for_model():
    """Test get_prompt_options_for_model method."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_model_manager.get_prompt_option_details.return_value = {
        "include_lighting": "Include lighting information",
        "character_name": "Character name: {name}",
    }
    mock_default_model = Mock()
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    mock_model = Mock()
    mock_model.value = "joycaption"

    result = section.get_prompt_options_for_model(mock_model)

    expected = {
        "include_lighting": "Include lighting information",
        "character_name": "Character name: {name}",
    }
    assert result == expected
    mock_model_manager.get_prompt_option_details.assert_called_once_with(mock_model)


def test_model_section_update_prompt_options_for_model():
    """Test update_prompt_options_for_model method."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_default_model = Mock()
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    section.get_prompt_options_for_model = Mock(
        return_value={
            "include_lighting": "Include lighting information",
            "character_name": "Character name: {name}",
        }
    )

    mock_model = Mock()
    mock_model.value = "joycaption"

    with patch("captiv.gui.sections.model.gr") as mock_gr:
        mock_gr.update.return_value = Mock()

        checkbox_update, character_update = section.update_prompt_options_for_model(
            mock_model
        )

    assert checkbox_update is not None
    assert character_update is not None


def test_model_section_get_current_model():
    """Test get_current_model method."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_default_model = Mock()
    mock_default_model.value = "blip"
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    result = section.get_current_model()

    assert result is mock_default_model


def test_model_section_imports_and_dependencies():
    """Test that all required imports and dependencies are available."""
    from captiv.gui.sections.model import (
        ModelManager,
        ModelSection,
        ModelType,
        gr,
        logger,
    )

    assert ModelSection is not None
    assert gr is not None
    assert logger is not None
    assert ModelManager is not None
    assert ModelType is not None


def test_model_section_method_signatures():
    """Test that methods have the expected signatures."""
    import inspect

    from captiv.gui.sections.model import ModelSection

    init_sig = inspect.signature(ModelSection.__init__)
    init_params = list(init_sig.parameters.keys())
    assert "self" in init_params
    assert "model_manager" in init_params

    create_sig = inspect.signature(ModelSection.create_section)
    create_params = list(create_sig.parameters.keys())
    assert "self" in create_params

    model_change_sig = inspect.signature(ModelSection.on_model_change)
    model_change_params = list(model_change_sig.parameters.keys())
    assert "self" in model_change_params
    assert "model_str" in model_change_params

    mode_change_sig = inspect.signature(ModelSection.on_mode_change)
    mode_change_params = list(mode_change_sig.parameters.keys())
    assert "self" in mode_change_params
    assert "mode" in mode_change_params


def test_model_section_exception_handling():
    """Test that ModelSection handles exceptions properly."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_default_model = Mock()
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    with (
        patch(
            "captiv.gui.sections.model.ModelType",
            side_effect=Exception("Invalid model"),
        ),
        patch("captiv.gui.sections.model.gr") as mock_gr,
    ):
        mock_gr.update.return_value = Mock()

        result = section.on_model_change("invalid_model")

    assert isinstance(result, tuple)
    assert len(result) == 5


@patch("captiv.gui.sections.model.ModelType")
def test_model_section_joycaption_fallback_options(mock_model_type):
    """Test that JoyCaption model gets fallback options when none are found."""
    from captiv.gui.sections.model import ModelSection

    mock_model_manager = Mock()
    mock_default_model = Mock()
    mock_model_manager.get_default_model.return_value = mock_default_model

    section = ModelSection(model_manager=mock_model_manager)

    section.get_prompt_options_for_model = Mock(return_value={})

    mock_joycaption_model = Mock()
    mock_joycaption_model.value = "joycaption"
    mock_model_type.JOYCAPTION = mock_joycaption_model

    with patch("captiv.gui.sections.model.gr") as mock_gr:
        mock_gr.update.return_value = Mock()

        checkbox_update, character_update = section.update_prompt_options_for_model(
            mock_joycaption_model
        )

    assert hasattr(section, "prompt_options_mapping")
    assert len(section.prompt_options_mapping) > 0
