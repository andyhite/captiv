"""Tests for JoyCaption model functionality."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from captiv.models.joycaption_model import JoyCaptionModel


@pytest.fixture
def mock_joycaption_model():
    """Create a mock JoyCaption model instance for testing."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch.object(JoyCaptionModel, "load_model") as mock_load_model,
        patch.object(JoyCaptionModel, "load_processor") as mock_load_processor,
        patch.object(JoyCaptionModel, "load_tokenizer") as mock_load_tokenizer,
    ):
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_load_model.return_value = mock_model_instance

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "formatted_chat"
        mock_processor.tokenizer.decode.return_value = "Generated caption"
        mock_load_processor.return_value = mock_processor

        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        model = JoyCaptionModel("joycaption-beta-one")
        yield model


class TestJoyCaptionModel:
    """Test JoyCaption model functionality."""

    def test_create_chat_with_prompt(self, mock_joycaption_model):
        """Test creating chat conversation format with prompt."""
        prompt = "Describe this image in detail"
        result = mock_joycaption_model.create_chat(prompt)

        mock_joycaption_model._processor.apply_chat_template.assert_called_once()
        call_args = mock_joycaption_model._processor.apply_chat_template.call_args

        convo = call_args[0][0]
        assert len(convo) == 2
        assert convo[0]["role"] == "system"
        assert convo[0]["content"] == "You are a helpful image captioner."
        assert convo[1]["role"] == "user"
        assert convo[1]["content"] == prompt

        assert call_args[1]["tokenize"] is False
        assert call_args[1]["add_generation_prompt"] is True

        assert result == "formatted_chat"

    def test_create_chat_with_none_prompt(self, mock_joycaption_model):
        """Test creating chat conversation format with None prompt."""
        mock_joycaption_model.create_chat(None)

        mock_joycaption_model._processor.apply_chat_template.assert_called_once()
        call_args = mock_joycaption_model._processor.apply_chat_template.call_args

        convo = call_args[0][0]
        assert convo[1]["content"] is None

    def test_process_inputs_calls_parent_with_chat(self, mock_joycaption_model):
        """Test that process_inputs calls parent method with chat formatting."""
        image = Image.new("RGB", (100, 100), color="red")
        prompt = "Test prompt"

        with patch(
            "captiv.models.base_model.BaseModel.process_inputs"
        ) as mock_parent_process:
            mock_pixel_values = MagicMock()
            mock_inputs = {"input_ids": MagicMock(), "pixel_values": mock_pixel_values}
            mock_parent_process.return_value = mock_inputs

            mock_joycaption_model._dtype = None

            result = mock_joycaption_model.process_inputs(image, prompt)

            mock_parent_process.assert_called_once_with(image, "formatted_chat")

            assert result == mock_inputs

    def test_generate_ids_excludes_input_tokens(self, mock_joycaption_model):
        """Test that generate_ids excludes input tokens from output."""
        mock_input_ids = MagicMock()
        mock_input_ids.shape = (1, 5)
        mock_inputs = {"input_ids": mock_input_ids}

        with patch(
            "captiv.models.base_model.BaseModel.generate_ids"
        ) as mock_parent_generate:
            mock_generated = MagicMock()
            mock_generated.__getitem__ = MagicMock(return_value="sliced_result")
            mock_parent_generate.return_value = mock_generated

            result = mock_joycaption_model.generate_ids(mock_inputs, max_new_tokens=50)

            mock_parent_generate.assert_called_once_with(mock_inputs, max_new_tokens=50)

            mock_generated.__getitem__.assert_called_once_with(slice(5, None))
            assert result == "sliced_result"

    def test_decode_caption_uses_processor_tokenizer(self, mock_joycaption_model):
        """Test that decode_caption uses processor's tokenizer with specific
        settings."""
        mock_generated_ids = MagicMock()

        result = mock_joycaption_model.decode_caption(mock_generated_ids)

        mock_joycaption_model._processor.tokenizer.decode.assert_called_once_with(
            mock_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        assert result == "Generated caption"

    def test_model_variants_configuration(self):
        """Test that JoyCaption model has correct variants configured."""
        variants = JoyCaptionModel.get_variants()

        assert "joycaption-alpha-two" in variants
        assert "joycaption-beta-one" in variants

        alpha_variant = variants["joycaption-alpha-two"]
        assert (
            alpha_variant["huggingface_id"]
            == "fancyfeast/llama-joycaption-alpha-two-hf-llava"
        )
        assert (
            alpha_variant.get("description")
            == "JoyCaption model (alpha two version) for image captioning"
        )
        assert alpha_variant.get("default_mode") == "default"

        beta_variant = variants["joycaption-beta-one"]
        assert (
            beta_variant["huggingface_id"]
            == "fancyfeast/llama-joycaption-beta-one-hf-llava"
        )
        assert (
            beta_variant.get("description")
            == "JoyCaption model (beta one version) for image captioning"
        )
        assert beta_variant.get("default_mode") == "default"

    def test_model_modes_configuration(self):
        """Test that JoyCaption model has correct modes configured."""
        modes = JoyCaptionModel.get_modes()

        assert "descriptive_formal" in modes
        assert "descriptive_casual" in modes
        assert "straightforward" in modes
        assert "stable_diffusion" in modes
        assert "danbooru" in modes
        assert "e621" in modes
        assert "default" in modes

        assert (
            modes["descriptive_formal"]
            == "Generate a formal, detailed description of this image"
        )
        assert modes["default"] == "Describe this image."
        stable_diffusion_mode = modes["stable_diffusion"]
        if stable_diffusion_mode:
            assert "stable diffusion prompt" in stable_diffusion_mode

    def test_model_prompt_options_configuration(self):
        """Test that JoyCaption model has correct prompt options configured."""
        options = JoyCaptionModel.get_prompt_options()

        assert "character_name" in options
        assert "exclude_immutable" in options
        assert "include_lighting" in options
        assert "keep_pg" in options
        assert "use_vulgar_language" in options

        assert "{character_name}" in options["character_name"]
        assert "Do NOT include anything sexual" in options["keep_pg"]
        assert "fucking" in options["use_vulgar_language"]

    def test_default_variant_is_beta_one(self):
        """Test that default variant is joycaption-beta-one."""
        assert JoyCaptionModel.DEFAULT_VARIANT == "joycaption-beta-one"

    def test_model_inheritance(self):
        """Test that JoyCaption model properly inherits from BaseModel."""
        from captiv.models.base_model import BaseModel

        assert issubclass(JoyCaptionModel, BaseModel)

    def test_model_uses_llava_components(self):
        """Test that JoyCaption model uses LLaVA components."""
        assert JoyCaptionModel.MODEL is not None
        assert JoyCaptionModel.PROCESSOR is not None

        assert "Llava" in str(JoyCaptionModel.MODEL)
        assert "Llava" in str(JoyCaptionModel.PROCESSOR)

    def test_comprehensive_mode_coverage(self):
        """Test that all expected modes are present and have reasonable content."""
        modes = JoyCaptionModel.get_modes()

        expected_modes = [
            "descriptive_formal",
            "descriptive_casual",
            "straightforward",
            "stable_diffusion",
            "midjourney",
            "danbooru",
            "e621",
            "rule34",
            "booru",
            "art_critic",
            "product_listing",
            "social_media",
            "creative",
            "technical",
            "poetic",
            "storytelling",
            "emotional",
            "humorous",
            "seo_friendly",
            "accessibility",
            "concise",
            "detailed",
            "default",
        ]

        for mode in expected_modes:
            assert mode in modes, f"Mode '{mode}' not found in JoyCaption modes"
            mode_desc = modes[mode]
            if mode_desc is not None:
                assert isinstance(mode_desc, str), (
                    f"Mode '{mode}' should have string description"
                )
                assert len(mode_desc) > 0, (
                    f"Mode '{mode}' should have non-empty description"
                )

    def test_comprehensive_prompt_options_coverage(self):
        """Test that all expected prompt options are present."""
        options = JoyCaptionModel.get_prompt_options()

        expected_options = [
            "character_name",
            "exclude_immutable",
            "include_lighting",
            "include_camera_angle",
            "include_watermark",
            "include_jpeg_artifacts",
            "include_camera_details",
            "keep_pg",
            "exclude_resolution",
            "include_quality",
            "include_composition",
            "exclude_text",
            "include_depth_of_field",
            "include_lighting_source",
            "exclude_ambiguity",
            "include_content_rating",
            "focus_important_elements",
            "exclude_artist_info",
            "include_orientation",
            "use_vulgar_language",
            "use_blunt_phrasing",
            "include_ages",
            "include_shot_type",
            "exclude_mood",
            "include_vantage_height",
            "mention_watermark",
            "avoid_meta_phrases",
        ]

        for option in expected_options:
            assert option in options, (
                f"Prompt option '{option}' not found in JoyCaption options"
            )
            assert isinstance(options[option], str), (
                f"Prompt option '{option}' should have string description"
            )
            assert len(options[option]) > 0, (
                f"Prompt option '{option}' should have non-empty description"
            )

    def test_mode_content_quality(self):
        """Test that mode content is appropriate and well-formed."""
        modes = JoyCaptionModel.get_modes()

        danbooru_mode = modes["danbooru"]
        if danbooru_mode:
            assert "comma-separated" in danbooru_mode
            assert "Danbooru tags" in danbooru_mode

        stable_diffusion_mode = modes["stable_diffusion"]
        if stable_diffusion_mode:
            assert "stable diffusion" in stable_diffusion_mode.lower()

        art_critic_mode = modes["art_critic"]
        if art_critic_mode:
            assert (
                "analyze" in art_critic_mode.lower()
                or "analysis" in art_critic_mode.lower()
            )

    def test_prompt_option_variable_support(self):
        """Test that character_name prompt option supports variables."""
        from captiv.models.base_model import BaseModel

        options = JoyCaptionModel.get_prompt_options()
        character_option = options["character_name"]
        assert "{character_name}" in character_option

        mock_model = BaseModel.__new__(BaseModel)
        variables = mock_model.extract_required_variables(character_option)
        assert "character_name" in variables
