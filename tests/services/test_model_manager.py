"""Tests for ModelManager service."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from captiv.services.config_manager import ConfigManager
from captiv.services.exceptions import (
    InvalidModelModeError,
    InvalidModelTypeError,
    InvalidModelVariantError,
    ModelConfigurationError,
)
from captiv.services.model_manager import ModelManager, ModelType
from captiv.utils.error_handling import EnhancedError, ErrorCategory


class TestModelManager:
    """Test cases for ModelManager."""

    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager."""
        mock_config = MagicMock(spec=ConfigManager)
        mock_config.get_config_value.return_value = "blip"
        return mock_config

    @pytest.fixture
    def model_manager(self, config_manager):
        """Create a ModelManager instance."""
        manager = ModelManager(config_manager)
        manager._instance_cache.clear()
        return manager

    @pytest.fixture
    def mock_model_class(self):
        """Create a mock model class."""
        mock_class = MagicMock()
        mock_class.get_variants.return_value = {
            "base": {"huggingface_id": "test/model", "description": "Test model"},
            "large": {
                "huggingface_id": "test/model-large",
                "description": "Large test model",
            },
        }
        mock_class.get_modes.return_value = {
            "default": None,
            "detailed": "Provide detailed description",
        }
        mock_class.get_prompt_options.return_value = {
            "short": "Generate short caption",
            "long": "Generate long caption",
        }
        mock_class.DEFAULT_VARIANT = "base"
        return mock_class

    def test_init_with_config_manager(self, config_manager):
        """Test ModelManager initialization with config manager."""
        manager = ModelManager(config_manager)
        assert manager.config_manager is config_manager

    def test_init_without_config_manager(self):
        """Test ModelManager initialization without config manager."""
        manager = ModelManager()
        assert manager.config_manager is not None
        assert isinstance(manager.config_manager, ConfigManager)

    def test_get_model_class_valid(self, model_manager):
        """Test getting model class for valid model type."""
        model_class = model_manager.get_model_class(ModelType.BLIP)
        assert model_class is not None

    def test_get_model_class_invalid(self, model_manager):
        """Test getting model class for invalid model type."""
        with pytest.raises(InvalidModelTypeError, match="Unknown model"):
            fake_model = "fake_model"
            model_manager.get_model_class(fake_model)

    def test_get_variants_for_model(self, model_manager, mock_model_class):
        """Test getting variants for a model."""
        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            variants = model_manager.get_variants_for_model(ModelType.BLIP)
            assert variants == ["base", "large"]

    def test_get_modes_for_model(self, model_manager, mock_model_class):
        """Test getting modes for a model."""
        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            modes = model_manager.get_modes_for_model(ModelType.BLIP)
            assert modes == ["default", "detailed"]

    def test_get_prompt_options_for_model(self, model_manager, mock_model_class):
        """Test getting prompt options for a model."""
        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            options = model_manager.get_prompt_options_for_model(ModelType.BLIP)
            assert options == ["short", "long"]

    def test_get_prompt_option_details(self, model_manager, mock_model_class):
        """Test getting prompt option details."""
        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            details = model_manager.get_prompt_option_details(ModelType.BLIP)
            expected = {
                "short": "Generate short caption",
                "long": "Generate long caption",
            }
            assert details == expected

    def test_get_variant_details(self, model_manager, mock_model_class):
        """Test getting variant details."""
        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            details = model_manager.get_variant_details(ModelType.BLIP)
            expected = {
                "base": {"huggingface_id": "test/model", "description": "Test model"},
                "large": {
                    "huggingface_id": "test/model-large",
                    "description": "Large test model",
                },
            }
            assert details == expected

    def test_get_mode_details(self, model_manager, mock_model_class):
        """Test getting mode details."""
        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            details = model_manager.get_mode_details(ModelType.BLIP)
            expected = {"default": None, "detailed": "Provide detailed description"}
            assert details == expected

    def test_validate_variant_valid(self, model_manager, mock_model_class):
        """Test validating a valid variant."""
        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            model_manager.validate_variant(ModelType.BLIP, "base")

    def test_validate_variant_invalid(self, model_manager, mock_model_class):
        """Test validating an invalid variant."""
        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            pytest.raises(
                InvalidModelVariantError, match="Invalid model variant 'invalid'"
            ),
        ):
            model_manager.validate_variant(ModelType.BLIP, "invalid")

    def test_validate_mode_valid(self, model_manager, mock_model_class):
        """Test validating a valid mode."""
        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            model_manager.validate_mode(ModelType.BLIP, "default")

    def test_validate_mode_invalid(self, model_manager, mock_model_class):
        """Test validating an invalid mode."""
        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            pytest.raises(InvalidModelModeError, match="Invalid mode 'invalid'"),
        ):
            model_manager.validate_mode(ModelType.BLIP, "invalid")

    def test_get_default_model_from_config(self, model_manager, config_manager):
        """Test getting default model from config."""
        config_manager.get_config_value.return_value = "blip2"
        default = model_manager.get_default_model()
        assert default == ModelType.BLIP2

    def test_get_default_model_fallback(self, model_manager, config_manager):
        """Test getting default model with fallback."""
        config_manager.get_config_value.return_value = "invalid_model"
        default = model_manager.get_default_model()
        assert default == ModelType.BLIP

    def test_get_default_model_none_fallback(self, model_manager, config_manager):
        """Test getting default model when config returns None."""
        config_manager.get_config_value.return_value = None
        default = model_manager.get_default_model()
        assert default == ModelType.BLIP

    def test_parse_torch_dtype_none(self, model_manager):
        """Test parsing None torch dtype."""
        result = model_manager.parse_torch_dtype(None)
        assert result is None

    def test_parse_torch_dtype_empty_string(self, model_manager):
        """Test parsing empty string torch dtype."""
        result = model_manager.parse_torch_dtype("")
        assert result is None

    def test_parse_torch_dtype_float16(self, model_manager):
        """Test parsing float16 torch dtype."""
        result = model_manager.parse_torch_dtype("float16")
        assert result == torch.float16

    def test_parse_torch_dtype_float32(self, model_manager):
        """Test parsing float32 torch dtype."""
        result = model_manager.parse_torch_dtype("float32")
        assert result == torch.float32

    def test_parse_torch_dtype_bfloat16(self, model_manager):
        """Test parsing bfloat16 torch dtype."""
        result = model_manager.parse_torch_dtype("bfloat16")
        assert result == torch.bfloat16

    def test_parse_torch_dtype_invalid(self, model_manager):
        """Test parsing invalid torch dtype."""
        with pytest.raises(ModelConfigurationError, match="Unsupported torch_dtype"):
            model_manager.parse_torch_dtype("invalid_dtype")

    def test_parse_prompt_options_none(self, model_manager):
        """Test parsing None prompt options."""
        result = model_manager.parse_prompt_options(None)
        assert result is None

    def test_parse_prompt_options_empty(self, model_manager):
        """Test parsing empty prompt options."""
        result = model_manager.parse_prompt_options("")
        assert result is None

    def test_parse_prompt_options_single(self, model_manager):
        """Test parsing single prompt option."""
        result = model_manager.parse_prompt_options("short")
        assert result == ["short"]

    def test_parse_prompt_options_multiple(self, model_manager):
        """Test parsing multiple prompt options."""
        result = model_manager.parse_prompt_options("short, long, detailed")
        assert result == ["short", "long", "detailed"]

    def test_parse_prompt_options_with_empty_values(self, model_manager):
        """Test parsing prompt options with empty values."""
        result = model_manager.parse_prompt_options("short,, long,")
        assert result == ["short", "long"]

    def test_parse_prompt_variables_none(self, model_manager):
        """Test parsing None prompt variables."""
        result = model_manager.parse_prompt_variables(None)
        assert result is None

    def test_parse_prompt_variables_empty(self, model_manager):
        """Test parsing empty prompt variables."""
        result = model_manager.parse_prompt_variables("")
        assert result is None

    def test_parse_prompt_variables_single(self, model_manager):
        """Test parsing single prompt variable."""
        result = model_manager.parse_prompt_variables("key=value")
        assert result == {"key": "value"}

    def test_parse_prompt_variables_multiple(self, model_manager):
        """Test parsing multiple prompt variables."""
        result = model_manager.parse_prompt_variables("key1=value1, key2=value2")
        assert result == {"key1": "value1", "key2": "value2"}

    def test_parse_prompt_variables_with_equals_in_value(self, model_manager):
        """Test parsing prompt variables with equals sign in value."""
        result = model_manager.parse_prompt_variables(
            "url=http://example.com/path?param=value"
        )
        assert result == {"url": "http://example.com/path?param=value"}

    def test_parse_prompt_variables_invalid_format(self, model_manager):
        """Test parsing prompt variables with invalid format."""
        with pytest.raises(ValueError, match="Invalid prompt variable format"):
            model_manager.parse_prompt_variables("invalid_format")

    def test_parse_prompt_variables_empty_key(self, model_manager):
        """Test parsing prompt variables with empty key."""
        with pytest.raises(ValueError, match="Empty key in prompt variable"):
            model_manager.parse_prompt_variables("=value")

    def test_parse_prompt_variables_empty_pairs(self, model_manager):
        """Test parsing prompt variables with empty pairs."""
        result = model_manager.parse_prompt_variables("key=value,,")
        assert result == {"key": "value"}

    def test_create_model_instance_with_variant(self, model_manager, mock_model_class):
        """Test creating model instance with specific variant."""
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            result = model_manager.create_model_instance(ModelType.BLIP, variant="base")
            assert result == mock_instance
            mock_model_class.assert_called_once_with("base", dtype=None)

    def test_create_model_instance_default_variant(
        self, model_manager, mock_model_class
    ):
        """Test creating model instance with default variant."""
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance
        mock_model_class.DEFAULT_VARIANT = "base"

        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            result = model_manager.create_model_instance(ModelType.BLIP)
            assert result == mock_instance
            mock_model_class.assert_called_once_with("base", dtype=None)

    def test_create_model_instance_no_default_variant(
        self, model_manager, mock_model_class
    ):
        """Test creating model instance when no default variant available."""
        mock_model_class.DEFAULT_VARIANT = None

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            pytest.raises(EnhancedError, match="No model variants available"),
        ):
            model_manager.create_model_instance(ModelType.BLIP)

    def test_create_model_instance_with_dtype(self, model_manager, mock_model_class):
        """Test creating model instance with specific dtype."""
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            result = model_manager.create_model_instance(
                ModelType.BLIP, variant="base", torch_dtype="float16"
            )
            assert result == mock_instance
            mock_model_class.assert_called_once_with("base", dtype=torch.float16)

    def test_create_model_instance_with_progress_callback(
        self, model_manager, mock_model_class
    ):
        """Test creating model instance with progress callback."""
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance
        progress_callback = MagicMock()

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            result = model_manager.create_model_instance(
                ModelType.BLIP, variant="base", progress_callback=progress_callback
            )
            assert result == mock_instance
            assert progress_callback.call_count == 3

    def test_create_model_instance_caching(self, model_manager, mock_model_class):
        """Test model instance caching."""
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            result1 = model_manager.create_model_instance(
                ModelType.BLIP, variant="base"
            )
            result2 = model_manager.create_model_instance(
                ModelType.BLIP, variant="base"
            )

            assert result1 == result2
            assert mock_model_class.call_count == 1

    def test_create_model_instance_import_error_joycaption(
        self, model_manager, mock_model_class
    ):
        """Test creating model instance with ImportError for JoyCaption."""
        mock_model_class.side_effect = ImportError("accelerate not found")

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            with pytest.raises(EnhancedError) as exc_info:
                model_manager.create_model_instance(
                    ModelType.JOYCAPTION, variant="base"
                )

            error = exc_info.value
            assert error.category == ErrorCategory.MODEL_LOADING
            assert "accelerate" in error.troubleshooting_tips[0]

    def test_create_model_instance_general_exception(
        self, model_manager, mock_model_class
    ):
        """Test creating model instance with general exception."""
        mock_model_class.side_effect = RuntimeError("General error")

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            with pytest.raises(EnhancedError) as exc_info:
                model_manager.create_model_instance(ModelType.BLIP, variant="base")

            error = exc_info.value
            assert error.category == ErrorCategory.MODEL_LOADING
            assert "Failed to create blip model variant instance" in str(error)

    def test_build_generation_params_all_none(self, model_manager):
        """Test building generation params with all None values."""
        result = model_manager.build_generation_params()
        assert result == {}

    def test_build_generation_params_all_values(self, model_manager):
        """Test building generation params with all values."""
        result = model_manager.build_generation_params(
            max_new_tokens=100,
            min_new_tokens=10,
            num_beams=5,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            prompt_options=["short"],
            prompt_variables={"key": "value"},
        )

        expected = {
            "max_new_tokens": 100,
            "min_new_tokens": 10,
            "num_beams": 5,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "prompt_options": ["short"],
            "prompt_variables": {"key": "value"},
        }
        assert result == expected

    def test_build_generation_params_partial(self, model_manager):
        """Test building generation params with partial values."""
        result = model_manager.build_generation_params(
            max_new_tokens=100, temperature=0.8, prompt_options=["short"]
        )

        expected = {
            "max_new_tokens": 100,
            "temperature": 0.8,
            "prompt_options": ["short"],
        }
        assert result == expected

    def test_generate_caption_with_prompt(self, model_manager):
        """Test generating caption with custom prompt."""
        mock_instance = MagicMock()
        mock_instance.caption_image.return_value = "Generated caption"

        result = model_manager.generate_caption(
            mock_instance, "image.jpg", prompt="Custom prompt"
        )

        assert result == "Generated caption"
        mock_instance.caption_image.assert_called_once_with(
            image_input="image.jpg", prompt_or_mode="Custom prompt"
        )

    def test_generate_caption_with_mode(self, model_manager):
        """Test generating caption with mode."""
        mock_instance = MagicMock()
        mock_instance.caption_image.return_value = "Generated caption"

        result = model_manager.generate_caption(
            mock_instance, "image.jpg", mode="detailed"
        )

        assert result == "Generated caption"
        mock_instance.caption_image.assert_called_once_with(
            image_input="image.jpg", prompt_or_mode="detailed"
        )

    def test_generate_caption_with_generation_params(self, model_manager):
        """Test generating caption with generation parameters."""
        mock_instance = MagicMock()
        mock_instance.caption_image.return_value = "Generated caption"

        generation_params = {"max_new_tokens": 100, "temperature": 0.8}

        result = model_manager.generate_caption(
            mock_instance,
            "image.jpg",
            mode="detailed",
            generation_params=generation_params,
        )

        assert result == "Generated caption"
        mock_instance.caption_image.assert_called_once_with(
            image_input="image.jpg",
            prompt_or_mode="detailed",
            max_new_tokens=100,
            temperature=0.8,
        )

    def test_generate_caption_prompt_overrides_mode(self, model_manager):
        """Test that custom prompt overrides mode."""
        mock_instance = MagicMock()
        mock_instance.caption_image.return_value = "Generated caption"

        result = model_manager.generate_caption(
            mock_instance, "image.jpg", mode="detailed", prompt="Custom prompt"
        )

        assert result == "Generated caption"
        mock_instance.caption_image.assert_called_once_with(
            image_input="image.jpg", prompt_or_mode="Custom prompt"
        )


class TestModelManagerEdgeCases:
    """Test edge cases and error scenarios for ModelManager."""

    @pytest.fixture
    def config_manager(self):
        """Create a mock config manager."""
        mock_config = MagicMock(spec=ConfigManager)
        mock_config.get_config_value.return_value = "blip"
        return mock_config

    @pytest.fixture
    def model_manager(self, config_manager):
        """Create a ModelManager instance."""
        manager = ModelManager(config_manager)
        manager._instance_cache.clear()
        return manager

    def test_create_model_instance_cache_key_generation(self, model_manager):
        """Test that cache keys are generated correctly with different dtypes."""
        mock_model_class = MagicMock()
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_model_class.side_effect = [mock_instance1, mock_instance2]
        mock_model_class.DEFAULT_VARIANT = "base"

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            result1 = model_manager.create_model_instance(
                ModelType.BLIP, variant="base", torch_dtype="float16"
            )

            result2 = model_manager.create_model_instance(
                ModelType.BLIP, variant="base", torch_dtype="float32"
            )

            assert result1 != result2
            assert mock_model_class.call_count == 2

    def test_create_model_instance_cache_with_none_dtype(self, model_manager):
        """Test caching with None dtype."""
        mock_model_class = MagicMock()
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance
        mock_model_class.DEFAULT_VARIANT = "base"

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            result1 = model_manager.create_model_instance(
                ModelType.BLIP, variant="base"
            )

            result2 = model_manager.create_model_instance(
                ModelType.BLIP, variant="base"
            )

            assert result1 == result2
            assert mock_model_class.call_count == 1

    def test_create_model_instance_import_error_non_joycaption(self, model_manager):
        """Test creating model instance with ImportError for non-JoyCaption model."""
        mock_model_class = MagicMock()
        mock_model_class.side_effect = ImportError("some other import error")
        mock_model_class.DEFAULT_VARIANT = "base"

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            with pytest.raises(EnhancedError) as exc_info:
                model_manager.create_model_instance(ModelType.BLIP, variant="base")

            error = exc_info.value
            assert error.category == ErrorCategory.MODEL_LOADING
            assert not any("accelerate" in tip for tip in error.troubleshooting_tips)

    def test_create_model_instance_runtime_error(self, model_manager):
        """Test creating model instance with RuntimeError."""
        mock_model_class = MagicMock()
        mock_model_class.side_effect = RuntimeError("Model loading failed")
        mock_model_class.DEFAULT_VARIANT = "base"

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            with pytest.raises(EnhancedError) as exc_info:
                model_manager.create_model_instance(ModelType.BLIP, variant="base")

            error = exc_info.value
            assert error.category == ErrorCategory.MODEL_LOADING
            assert "Failed to create blip model variant instance" in str(error)

    def test_create_model_instance_with_progress_callback_exception(
        self, model_manager
    ):
        """Test creating model instance when progress callback raises exception."""
        mock_model_class = MagicMock()
        mock_model_class.side_effect = Exception("Model creation failed")
        mock_model_class.DEFAULT_VARIANT = "base"

        progress_callback = MagicMock()

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            patch.object(model_manager, "validate_variant"),
        ):
            with pytest.raises(EnhancedError):
                model_manager.create_model_instance(
                    ModelType.BLIP,
                    variant="base",
                    progress_callback=progress_callback,
                )

            assert progress_callback.call_count >= 1

    def test_parse_prompt_options_whitespace_handling(self, model_manager):
        """Test parsing prompt options with various whitespace scenarios."""
        result = model_manager.parse_prompt_options("  short  ,  long  ,  detailed  ")
        assert result == ["short", "long", "detailed"]

        result = model_manager.parse_prompt_options("short\t,\nlong")
        assert result == ["short", "long"]

    def test_parse_prompt_options_only_empty_values(self, model_manager):
        """Test parsing prompt options with only empty values."""
        result = model_manager.parse_prompt_options("  ,  ,  ")
        assert result is None

    def test_parse_prompt_variables_complex_values(self, model_manager):
        """Test parsing prompt variables with complex values."""
        result = model_manager.parse_prompt_variables(
            "url=https://example.com/path?param=value&other=test, "
            'json={"key": "value"}, '
            "math=x=y+z"
        )
        expected = {
            "url": "https://example.com/path?param=value&other=test",
            "json": '{"key": "value"}',
            "math": "x=y+z",
        }
        assert result == expected

    def test_parse_prompt_variables_whitespace_in_keys_and_values(self, model_manager):
        """Test parsing prompt variables with whitespace in keys and values."""
        result = model_manager.parse_prompt_variables("  key1  =  value with spaces  ")
        assert result == {"key1": "value with spaces"}

    def test_parse_prompt_variables_empty_value(self, model_manager):
        """Test parsing prompt variables with empty value."""
        result = model_manager.parse_prompt_variables("key=")
        assert result == {"key": ""}

    def test_parse_prompt_variables_only_empty_pairs(self, model_manager):
        """Test parsing prompt variables with only empty pairs."""
        result = model_manager.parse_prompt_variables("  ,  ,  ")
        assert result is None

    def test_build_generation_params_zero_values(self, model_manager):
        """Test building generation params with zero values."""
        result = model_manager.build_generation_params(
            max_new_tokens=0,
            min_new_tokens=0,
            num_beams=0,
            temperature=0.0,
            top_k=0,
            top_p=0.0,
            repetition_penalty=0.0,
        )

        expected = {
            "max_new_tokens": 0,
            "min_new_tokens": 0,
            "num_beams": 0,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 0.0,
            "repetition_penalty": 0.0,
        }
        assert result == expected

    def test_build_generation_params_empty_lists_and_dicts(self, model_manager):
        """Test building generation params with empty lists and dicts."""
        result = model_manager.build_generation_params(
            prompt_options=[],
            prompt_variables={},
        )

        expected = {
            "prompt_options": [],
            "prompt_variables": {},
        }
        assert result == expected

    def test_generate_caption_with_none_mode_and_prompt(self, model_manager):
        """Test generating caption when both mode and prompt are None."""
        mock_instance = MagicMock()
        mock_instance.caption_image.return_value = "Generated caption"

        result = model_manager.generate_caption(
            mock_instance, "image.jpg", mode=None, prompt=None
        )

        assert result == "Generated caption"
        mock_instance.caption_image.assert_called_once_with(
            image_input="image.jpg", prompt_or_mode=None
        )

    def test_generate_caption_with_empty_generation_params(self, model_manager):
        """Test generating caption with empty generation params dict."""
        mock_instance = MagicMock()
        mock_instance.caption_image.return_value = "Generated caption"

        result = model_manager.generate_caption(
            mock_instance, "image.jpg", generation_params={}
        )

        assert result == "Generated caption"
        mock_instance.caption_image.assert_called_once_with(
            image_input="image.jpg", prompt_or_mode=None
        )

    def test_generate_caption_exception_propagation(self, model_manager):
        """Test that exceptions from model.caption_image are propagated."""
        mock_instance = MagicMock()
        mock_instance.caption_image.side_effect = ValueError("Image processing failed")

        with pytest.raises(ValueError, match="Image processing failed"):
            model_manager.generate_caption(mock_instance, "image.jpg")

    def test_get_default_model_with_type_error(self, model_manager, config_manager):
        """Test getting default model when config returns invalid type."""
        config_manager.get_config_value.side_effect = TypeError("Invalid type")
        default = model_manager.get_default_model()
        assert default == ModelType.BLIP

    def test_validate_variant_with_empty_variants_list(self, model_manager):
        """Test validating variant when model has no variants."""
        mock_model_class = MagicMock()
        mock_model_class.get_variants.return_value = {}

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            pytest.raises(InvalidModelVariantError, match="Available model variants: "),
        ):
            model_manager.validate_variant(ModelType.BLIP, "any_variant")

    def test_validate_mode_with_empty_modes_list(self, model_manager):
        """Test validating mode when model has no modes."""
        mock_model_class = MagicMock()
        mock_model_class.get_modes.return_value = {}

        with (
            patch.object(
                model_manager, "get_model_class", return_value=mock_model_class
            ),
            pytest.raises(InvalidModelModeError, match="Available modes: "),
        ):
            model_manager.validate_mode(ModelType.BLIP, "any_mode")

    def test_get_variants_for_model_empty_result(self, model_manager):
        """Test getting variants when model returns empty variants."""
        mock_model_class = MagicMock()
        mock_model_class.get_variants.return_value = {}

        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            variants = model_manager.get_variants_for_model(ModelType.BLIP)
            assert variants == []

    def test_get_modes_for_model_empty_result(self, model_manager):
        """Test getting modes when model returns empty modes."""
        mock_model_class = MagicMock()
        mock_model_class.get_modes.return_value = {}

        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            modes = model_manager.get_modes_for_model(ModelType.BLIP)
            assert modes == []

    def test_get_prompt_options_for_model_empty_result(self, model_manager):
        """Test getting prompt options when model returns empty options."""
        mock_model_class = MagicMock()
        mock_model_class.get_prompt_options.return_value = {}

        with patch.object(
            model_manager, "get_model_class", return_value=mock_model_class
        ):
            options = model_manager.get_prompt_options_for_model(ModelType.BLIP)
            assert options == []


class TestModelTypeEnum:
    """Test ModelType enum edge cases."""

    def test_model_type_string_values(self):
        """Test that ModelType enum has correct string values."""
        assert ModelType.BLIP == "blip"
        assert ModelType.BLIP2 == "blip2"
        assert ModelType.JOYCAPTION == "joycaption"
        assert ModelType.KOSMOS == "kosmos"
        assert ModelType.VIT_GPT2 == "vit-gpt2"

    def test_model_type_iteration(self):
        """Test iterating over ModelType enum."""
        model_types = list(ModelType)
        assert len(model_types) == 5
        assert ModelType.BLIP in model_types
        assert ModelType.BLIP2 in model_types

    def test_model_type_comparison(self):
        """Test ModelType enum comparison."""
        assert ModelType.BLIP != ModelType.BLIP2
        assert ModelType.BLIP == ModelType.BLIP
        assert ModelType.BLIP == "blip"
