"""Tests for BaseModel and related functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from captiv.models.base_model import BaseModel, ModelVariant, create_model


class TestBaseModel:
    """Test cases for BaseModel."""

    @pytest.fixture
    def mock_model_class(self):
        """Create a mock model class that inherits from BaseModel."""

        class MockModel(BaseModel):
            MODEL = MagicMock()
            TOKENIZER = MagicMock()
            PROCESSOR = MagicMock()

            MODES = {"default": None, "detailed": "Provide detailed description"}
            VARIANTS = {
                "base": {
                    "huggingface_id": "test/model-base",
                    "description": "Base model",
                },
                "large": {
                    "huggingface_id": "test/model-large",
                    "description": "Large model",
                    "default_mode": "detailed",
                },
            }
            PROMPT_OPTIONS = {
                "short": "Generate short caption",
                "long": "Generate long caption",
            }
            PROMPT_VARIABLES = {"style": "Caption style", "length": "Caption length"}
            DEFAULT_VARIANT = "base"

            def load_model(self):
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                return mock_model

            def load_tokenizer(self):
                return MagicMock()

            def load_processor(self):
                return MagicMock()

            def process_inputs(self, image, prompt):
                return {"inputs": "processed"}

            def generate_ids(self, inputs, **kwargs):
                return MagicMock()

            def decode_caption(self, generated_ids):
                return "Generated caption"

        return MockModel

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_init_cpu_device(self, mock_model_class):
        """Test initialization with CPU device."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            model = mock_model_class("base")
            assert model.device == "cpu"
            assert model.variant_key == "base"

    def test_init_cuda_device(self, mock_model_class):
        """Test initialization with CUDA device."""
        with patch("torch.cuda.is_available", return_value=True):
            model = mock_model_class("base")
            assert model.device == "cuda"

    def test_init_mps_device(self, mock_model_class):
        """Test initialization with MPS device."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            model = mock_model_class("base")
            assert model.device == "mps"

    def test_init_with_dtype(self, mock_model_class):
        """Test initialization with specific dtype."""
        model = mock_model_class("base", dtype=torch.float16)
        assert model._dtype == torch.float16

    def test_init_processor_tuple(self, mock_model_class):
        """Test initialization when processor returns tuple."""
        mock_model_class.load_processor = MagicMock(
            return_value=(MagicMock(), MagicMock())
        )
        model = mock_model_class("base")
        assert model._processor is not None

    def test_repr(self, mock_model_class):
        """Test string representation of model."""
        model = mock_model_class("base")
        repr_str = repr(model)
        assert "MockModel" in repr_str
        assert "test/model-base" in repr_str

    def test_variant_property(self, mock_model_class):
        """Test variant property."""
        model = mock_model_class("base")
        variant = model.variant
        assert variant["huggingface_id"] == "test/model-base"
        assert variant["description"] == "Base model"

    def test_variant_property_large(self, mock_model_class):
        """Test variant property for large model."""
        model = mock_model_class("large")
        variant = model.variant
        assert variant["huggingface_id"] == "test/model-large"
        assert variant["default_mode"] == "detailed"

    def test_default_mode_property(self, mock_model_class):
        """Test default_mode property."""
        model = mock_model_class("base")
        assert model.default_mode == "default"

        model_large = mock_model_class("large")
        assert model_large.default_mode == "detailed"

    def test_dtype_property_none(self, mock_model_class):
        """Test dtype property when None."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            model = mock_model_class("base")
            assert model.dtype is None

    def test_dtype_property_set(self, mock_model_class):
        """Test dtype property when set."""
        model = mock_model_class("base", dtype=torch.float16)
        assert model.dtype == torch.float16

    def test_get_variants_classmethod(self, mock_model_class):
        """Test get_variants class method."""
        variants = mock_model_class.get_variants()
        assert "base" in variants
        assert "large" in variants
        assert variants["base"]["huggingface_id"] == "test/model-base"

    def test_get_modes_classmethod(self, mock_model_class):
        """Test get_modes class method."""
        modes = mock_model_class.get_modes()
        assert "default" in modes
        assert "detailed" in modes
        assert modes["default"] is None
        assert modes["detailed"] == "Provide detailed description"

    def test_get_prompt_options_classmethod(self, mock_model_class):
        """Test get_prompt_options class method."""
        options = mock_model_class.get_prompt_options()
        assert "short" in options
        assert "long" in options
        assert options["short"] == "Generate short caption"

    def test_load_image_from_path(self, mock_model_class, temp_dir):
        """Test loading image from file path."""
        test_image = Image.new("RGB", (100, 100), color="red")
        image_path = temp_dir / "test.jpg"
        test_image.save(image_path)

        loaded_image = mock_model_class.load_image(str(image_path))
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.size == (100, 100)

    def test_load_image_from_pil(self, mock_model_class):
        """Test loading image from PIL Image object."""
        test_image = Image.new("RGB", (100, 100), color="blue")
        loaded_image = mock_model_class.load_image(test_image)
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.size == test_image.size
        assert loaded_image.mode == "RGB"

    def test_load_image_nonexistent_file(self, mock_model_class):
        """Test loading image from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            mock_model_class.load_image("/nonexistent/image.jpg")

    def test_load_image_invalid_file(self, mock_model_class, temp_dir):
        """Test loading image from invalid file."""
        text_file = temp_dir / "not_image.txt"
        text_file.write_text("This is not an image")

        with pytest.raises(ValueError, match=r"Could not open image file.*"):
            mock_model_class.load_image(str(text_file))

    def test_resolve_prompt_options_none(self, mock_model_class):
        """Test resolving prompt options when None."""
        model = mock_model_class("base")
        result = model.resolve_prompt_options("test prompt", None)
        assert result == "test prompt"

    def test_resolve_prompt_options_empty_list(self, mock_model_class):
        """Test resolving prompt options with empty list."""
        model = mock_model_class("base")
        result = model.resolve_prompt_options("test prompt", [])
        assert result == "test prompt"

    def test_resolve_prompt_options_valid(self, mock_model_class):
        """Test resolving valid prompt options."""
        model = mock_model_class("base")
        result = model.resolve_prompt_options("test prompt", ["short", "long"])
        expected = "test prompt\n\nGenerate short caption Generate long caption"
        assert result == expected

    def test_resolve_prompt_options_invalid(self, mock_model_class):
        """Test resolving invalid prompt options."""
        model = mock_model_class("base")
        result = model.resolve_prompt_options("test prompt", ["invalid_option"])
        assert result == "test prompt\n\n"

    def test_resolve_prompt_options_mixed(self, mock_model_class):
        """Test resolving mixed valid and invalid prompt options."""
        model = mock_model_class("base")
        result = model.resolve_prompt_options(
            "test prompt", ["short", "invalid_option"]
        )
        expected = "test prompt\n\nGenerate short caption"
        assert result == expected

    def test_extract_required_variables(self, mock_model_class):
        """Test extracting required variables from prompt."""
        model = mock_model_class("base")

        prompt = "Generate a {style} caption with {length} words about {subject}"
        variables = model.extract_required_variables(prompt)
        assert variables == {"style", "length", "subject"}

        prompt_no_vars = "Generate a simple caption"
        variables = model.extract_required_variables(prompt_no_vars)
        assert variables == set()

    def test_validate_prompt_variables_valid(self, mock_model_class):
        """Test validating valid prompt variables."""
        model = mock_model_class("base")
        prompt = "Generate a {style} caption"
        variables = {"style": "detailed"}

        model.validate_prompt_variables(prompt, variables)

    def test_validate_prompt_variables_missing(self, mock_model_class):
        """Test validating prompt variables with missing variables."""
        model = mock_model_class("base")
        prompt = "Generate a {style} caption with {length} words"
        variables = {"style": "detailed"}

        with pytest.raises(ValueError, match="Missing required prompt variables"):
            model.validate_prompt_variables(prompt, variables)

    def test_validate_prompt_variables_none(self, mock_model_class):
        """Test validating prompt variables when None provided."""
        model = mock_model_class("base")
        prompt = "Generate a {style} caption"

        with pytest.raises(ValueError, match="Missing required prompt variables"):
            model.validate_prompt_variables(prompt, None)

    def test_resolve_prompt_variables(self, mock_model_class):
        """Test resolving prompt variables."""
        model = mock_model_class("base")
        prompt = "Generate a {style} caption with {length} words"
        variables = {"style": "detailed", "length": "10"}

        resolved = model.resolve_prompt_variables(prompt, variables)
        assert resolved == "Generate a detailed caption with 10 words"

    def test_resolve_prompt_variables_with_invalid_key(self, mock_model_class):
        """Test resolving prompt variables with invalid key."""
        model = mock_model_class("base")
        prompt = "Generate a {style} caption"
        variables = {"style": "detailed", "invalid{key}": "value"}

        result = model.resolve_prompt_variables(prompt, variables)
        assert result == "Generate a detailed caption"

    def test_resolve_prompt_mode_none(self, mock_model_class):
        """Test resolving prompt mode when None."""
        model = mock_model_class("base")
        result = model.resolve_prompt_mode(None)
        assert result == "default"

    def test_resolve_prompt_mode_valid(self, mock_model_class):
        """Test resolving valid prompt mode."""
        model = mock_model_class("base")
        result = model.resolve_prompt_mode("detailed")
        assert result == "Provide detailed description"

    def test_resolve_prompt_mode_invalid(self, mock_model_class):
        """Test resolving invalid prompt mode."""
        model = mock_model_class("base")
        result = model.resolve_prompt_mode("invalid_mode")
        assert result == "invalid_mode"

    def test_resolve_prompt_with_mode(self, mock_model_class):
        """Test resolving prompt with mode."""
        model = mock_model_class("base")
        result = model.resolve_prompt("detailed", None, None)
        assert result == "Provide detailed description"

    def test_resolve_prompt_with_custom_prompt(self, mock_model_class):
        """Test resolving prompt with custom prompt."""
        model = mock_model_class("base")
        custom_prompt = "Custom prompt text"
        result = model.resolve_prompt(custom_prompt, None, None)
        assert result == custom_prompt

    def test_resolve_prompt_with_variables(self, mock_model_class):
        """Test resolving prompt with variables."""
        model = mock_model_class("base")
        prompt = "Generate a {style} caption"
        variables = {"style": "detailed"}
        result = model.resolve_prompt(prompt, None, variables)
        assert result == "Generate a detailed caption"

    def test_resolve_prompt_with_options(self, mock_model_class):
        """Test resolving prompt with options."""
        model = mock_model_class("base")
        result = model.resolve_prompt("detailed", ["short"], None)
        expected = "Provide detailed description\n\nGenerate short caption"
        assert result == expected

    def test_caption_image_basic(self, mock_model_class, temp_dir):
        """Test basic image captioning."""
        test_image = Image.new("RGB", (100, 100), color="red")
        image_path = temp_dir / "test.jpg"
        test_image.save(image_path)

        model = mock_model_class("base")

        with patch("captiv.models.base_model.torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock()
            with patch.object(model, "load_image", return_value=test_image):
                caption = model.caption_image(str(image_path), "default")
                assert caption == "Generated caption"

    def test_caption_image_with_options(self, mock_model_class, temp_dir):
        """Test image captioning with generation options."""
        test_image = Image.new("RGB", (100, 100), color="red")
        image_path = temp_dir / "test.jpg"
        test_image.save(image_path)

        model = mock_model_class("base")

        with patch("captiv.models.base_model.torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock()
            with patch.object(model, "load_image", return_value=test_image):
                caption = model.caption_image(
                    str(image_path),
                    "default",
                    max_new_tokens=100,
                    temperature=0.8,
                    prompt_options=["short"],
                )
                assert caption == "Generated caption"


class TestCreateModel:
    """Test cases for create_model function."""

    def test_create_model_with_class(self):
        """Test creating model with model class."""
        mock_model_class = MagicMock()
        mock_processor_class = MagicMock()
        variants: dict[str, ModelVariant] = {"base": {"huggingface_id": "test/model"}}

        result_class = create_model(
            model_class=mock_model_class,
            processor_class=mock_processor_class,
            default_variant="base",
            variants=variants,
        )

        assert issubclass(result_class, BaseModel)
        assert mock_model_class == result_class.MODEL
        assert mock_processor_class == result_class.PROCESSOR
        assert result_class.DEFAULT_VARIANT == "base"
        assert variants == result_class.VARIANTS

    def test_create_model_with_optional_params(self):
        """Test creating model with optional parameters."""
        mock_model_class = MagicMock()
        mock_processor_class = MagicMock()
        mock_tokenizer_class = MagicMock()
        variants: dict[str, ModelVariant] = {"base": {"huggingface_id": "test/model"}}
        modes = {"default": None, "detailed": "Detailed description"}
        prompt_options = {"short": "Short caption"}

        result_class = create_model(
            model_class=mock_model_class,
            processor_class=mock_processor_class,
            default_variant="base",
            variants=variants,
            tokenizer_class=mock_tokenizer_class,
            modes=modes,
            prompt_options=prompt_options,
        )

        assert mock_tokenizer_class == result_class.TOKENIZER
        assert modes == result_class.MODES
        assert prompt_options == result_class.PROMPT_OPTIONS


class TestBaseModelEdgeCases:
    """Test edge cases and error handling for BaseModel."""

    @pytest.fixture
    def mock_model_class(self):
        """Create a mock model class that inherits from BaseModel."""

        class MockModel(BaseModel):
            MODEL = MagicMock()
            TOKENIZER = MagicMock()
            PROCESSOR = MagicMock()

            MODES = {"default": None, "detailed": "Provide detailed description"}
            VARIANTS = {
                "base": {
                    "huggingface_id": "test/model-base",
                    "description": "Base model",
                },
                "large": {
                    "huggingface_id": "test/model-large",
                    "description": "Large model",
                    "default_mode": "detailed",
                },
            }
            PROMPT_OPTIONS = {
                "short": "Generate short caption",
                "long": "Generate long caption",
            }
            DEFAULT_VARIANT = "base"

            def load_model(self):
                if self.MODEL is None:
                    raise ValueError("MODEL class is not defined")
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                return mock_model

            def load_tokenizer(self):
                if self.TOKENIZER is None:
                    return None
                return MagicMock()

            def load_processor(self):
                if self.PROCESSOR is None:
                    return None
                return MagicMock()

            def generate_ids(self, inputs, **kwargs):
                return MagicMock()

        return MockModel

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_variant_property_invalid_variant(self, mock_model_class):
        """Test variant property with invalid variant key."""
        model = mock_model_class("invalid_variant")

        variant = model.variant
        assert variant["huggingface_id"] == "test/model-base"

    def test_variant_property_no_default_variant(self, mock_model_class):
        """Test variant property when no default variant exists."""
        original_default = mock_model_class.DEFAULT_VARIANT
        original_variants = mock_model_class.VARIANTS.copy()

        mock_model_class.DEFAULT_VARIANT = "nonexistent"
        mock_model_class.VARIANTS = {}

        try:
            model = mock_model_class("invalid_variant")
            with pytest.raises(ValueError, match="Variant 'invalid_variant' not found"):
                _ = model.variant
        finally:
            mock_model_class.DEFAULT_VARIANT = original_default
            mock_model_class.VARIANTS = original_variants

    def test_description_property_missing(self, mock_model_class):
        """Test description property when description is missing from variant."""
        mock_model_class.VARIANTS["no_desc"] = {"huggingface_id": "test/no-desc"}

        model = mock_model_class("no_desc")
        assert model.description is None

    def test_dtype_property_cuda_ampere(self, mock_model_class):
        """Test dtype property with CUDA Ampere GPU."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        ):
            model = mock_model_class("base")
            assert model.dtype == torch.bfloat16

    def test_dtype_property_cuda_older(self, mock_model_class):
        """Test dtype property with older CUDA GPU."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 0)),
        ):
            model = mock_model_class("base")
            assert model.dtype == torch.float16

    def test_dtype_property_non_cpu_device(self, mock_model_class):
        """Test dtype property with non-CPU device (MPS)."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            model = mock_model_class("base")
            assert model.dtype == torch.float16

    def test_load_image_invalid_type(self, mock_model_class):
        """Test loading image with invalid input type."""
        with pytest.raises(ValueError, match="Invalid image input"):
            mock_model_class.load_image(123)

    def test_load_image_corrupted_file(self, mock_model_class, temp_dir):
        """Test loading corrupted image file."""
        corrupted_file = temp_dir / "corrupted.jpg"
        corrupted_file.write_text("This is not an image")

        with pytest.raises(ValueError, match="Could not open image file"):
            mock_model_class.load_image(str(corrupted_file))

    def test_resolve_prompt_options_with_none_values(self, mock_model_class):
        """Test resolving prompt options that include None values."""
        model = mock_model_class("base")
        result = model.resolve_prompt_options(
            "test prompt", ["short", "none_option", "long"]
        )
        expected = "test prompt\n\nGenerate short caption Generate long caption"
        assert result == expected

    def test_extract_required_variables_empty_prompt(self, mock_model_class):
        """Test extracting variables from empty prompt."""
        model = mock_model_class("base")
        variables = model.extract_required_variables("")
        assert variables == set()

    def test_extract_required_variables_none_prompt(self, mock_model_class):
        """Test extracting variables from None prompt."""
        model = mock_model_class("base")
        variables = model.extract_required_variables(None)
        assert variables == set()

    def test_extract_required_variables_complex_patterns(self, mock_model_class):
        """Test extracting variables with complex patterns."""
        model = mock_model_class("base")

        prompt = "Generate {style} caption with {length:d} words and {precision:.2f} accuracy"  # noqa: E501
        variables = model.extract_required_variables(prompt)
        assert variables == {"style", "length", "precision"}

    def test_validate_prompt_variables_empty_prompt(self, mock_model_class):
        """Test validating variables for empty prompt."""
        model = mock_model_class("base")
        model.validate_prompt_variables("", {"key": "value"})

    def test_validate_prompt_variables_none_prompt(self, mock_model_class):
        """Test validating variables for None prompt."""
        model = mock_model_class("base")
        model.validate_prompt_variables(None, {"key": "value"})

    def test_resolve_prompt_variables_empty_prompt(self, mock_model_class):
        """Test resolving variables for empty prompt."""
        model = mock_model_class("base")
        result = model.resolve_prompt_variables("", {"key": "value"})
        assert result == ""

    def test_resolve_prompt_variables_none_prompt(self, mock_model_class):
        """Test resolving variables for None prompt."""
        model = mock_model_class("base")
        result = model.resolve_prompt_variables(None, {"key": "value"})
        assert result is None

    def test_resolve_prompt_variables_format_error(self, mock_model_class):
        """Test resolving variables with format error."""
        model = mock_model_class("base")
        prompt = "Generate {style} caption with {invalid:invalid_format}"
        variables = {"style": "detailed", "invalid": "value"}

        with pytest.raises(ValueError, match="Error formatting prompt"):
            model.resolve_prompt_variables(prompt, variables)

    def test_resolve_prompt_variables_no_variables_provided(self, mock_model_class):
        """Test resolving variables when no variables dict provided."""
        model = mock_model_class("base")
        prompt = "Generate caption"
        result = model.resolve_prompt_variables(prompt, None)
        assert result == prompt

    def test_decode_caption_no_tokenizer_or_processor(self, mock_model_class):
        """Test decoding caption when no tokenizer or processor available."""
        model = mock_model_class("base")
        model._tokenizer = None
        model._processor = None

        mock_tensor = MagicMock()
        with pytest.raises(ValueError, match="No tokenizer or processor available"):
            model.decode_caption(mock_tensor)

    def test_decode_caption_with_processor_fallback(self, mock_model_class):
        """Test decoding caption using processor when tokenizer is None."""
        model = mock_model_class("base")
        model._tokenizer = None
        mock_processor = MagicMock()
        mock_processor.decode.return_value = "Processor decoded caption"
        model._processor = mock_processor

        mock_tensor = MagicMock()
        result = model.decode_caption(mock_tensor)
        assert result == "Processor decoded caption"

    def test_load_model_no_model_class(self, mock_model_class):
        """Test loading model when MODEL class is None."""
        original_model = mock_model_class.MODEL
        mock_model_class.MODEL = None

        try:
            with pytest.raises(ValueError, match="MODEL class is not defined"):
                mock_model_class("base")
        finally:
            mock_model_class.MODEL = original_model

    def test_load_tokenizer_none(self, mock_model_class):
        """Test loading tokenizer when TOKENIZER is None."""
        original_tokenizer = mock_model_class.TOKENIZER
        mock_model_class.TOKENIZER = None

        try:
            model = mock_model_class("base")
            result = model.load_tokenizer()
            assert result is None
        finally:
            mock_model_class.TOKENIZER = original_tokenizer

    def test_load_processor_none(self, mock_model_class):
        """Test loading processor when PROCESSOR is None."""
        original_processor = mock_model_class.PROCESSOR
        mock_model_class.PROCESSOR = None

        try:
            model = mock_model_class("base")
            result = model.load_processor()
            assert result is None
        finally:
            mock_model_class.PROCESSOR = original_processor

    def test_process_inputs_tensor_conversion(self, mock_model_class):
        """Test process_inputs with tensor type conversion."""
        model = mock_model_class("base")

        mock_processor = MagicMock()

        mock_float_tensor = MagicMock()
        mock_float_tensor.to.return_value = mock_float_tensor
        mock_int_tensor = MagicMock()
        mock_int_tensor.to.return_value = mock_int_tensor

        mock_result = MagicMock()
        mock_result.to.return_value = {
            "pixel_values": mock_float_tensor,
            "input_ids": mock_int_tensor,
            "attention_mask": mock_int_tensor,
        }

        mock_processor.return_value = mock_result
        model._processor = mock_processor

        model.device = "cpu"

        test_image = Image.new("RGB", (100, 100), color="red")
        model.process_inputs(test_image, "test prompt")

        assert mock_processor.called

    def test_repr_with_special_characters(self, mock_model_class):
        """Test __repr__ method with special characters in huggingface_id."""
        mock_model_class.VARIANTS["special"] = {
            "huggingface_id": "test/model-with-special-chars_123",
            "description": "Model with special chars",
        }

        model = mock_model_class("special")
        repr_str = repr(model)
        assert "test/model-with-special-chars_123" in repr_str
        assert "MockModel" in repr_str


class TestCreateModelEdgeCases:
    """Test edge cases for create_model function."""

    def test_create_model_empty_variants(self):
        """Test creating model with empty variants dict."""
        mock_model_class = MagicMock()
        mock_processor_class = MagicMock()

        result_class = create_model(
            model_class=mock_model_class,
            processor_class=mock_processor_class,
            default_variant="base",
            variants={},
        )

        assert result_class.VARIANTS == {}

    def test_create_model_with_empty_modes(self):
        """Test creating model with empty modes dict."""
        mock_model_class = MagicMock()
        mock_processor_class = MagicMock()
        variants: dict[str, ModelVariant] = {"base": {"huggingface_id": "test/model"}}

        result_class = create_model(
            model_class=mock_model_class,
            processor_class=mock_processor_class,
            default_variant="base",
            variants=variants,
            modes={},
        )

        assert result_class.MODES == {}

    def test_create_model_with_empty_prompt_options(self):
        """Test creating model with empty prompt options dict."""
        mock_model_class = MagicMock()
        mock_processor_class = MagicMock()
        variants: dict[str, ModelVariant] = {"base": {"huggingface_id": "test/model"}}

        result_class = create_model(
            model_class=mock_model_class,
            processor_class=mock_processor_class,
            default_variant="base",
            variants=variants,
            prompt_options={},
        )

        assert result_class.PROMPT_OPTIONS == {}


class TestBaseModelMissingCoverage:
    """Test cases for missing coverage in BaseModel."""

    @pytest.fixture
    def mock_model_class(self):
        """Create a mock model class that inherits from BaseModel."""

        class MockModel(BaseModel):
            MODEL = MagicMock()
            TOKENIZER = MagicMock()
            PROCESSOR = MagicMock()

            MODES = {"default": None, "detailed": "Provide detailed description"}
            VARIANTS = {
                "base": {
                    "huggingface_id": "test/model-base",
                    "description": "Base model",
                },
            }
            PROMPT_OPTIONS = {
                "short": "Generate short caption",
                "long": "Generate long caption",
            }
            DEFAULT_VARIANT = "base"

            def load_model(self):
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                return mock_model

            def load_tokenizer(self):
                return MagicMock()

            def load_processor(self):
                return MagicMock()

        return MockModel

    def test_resolve_prompt_options_with_none_filtering(self, mock_model_class):
        """Test that None values in prompt options are filtered out."""
        model = mock_model_class("base")
        original_options = model.PROMPT_OPTIONS.copy()
        model.PROMPT_OPTIONS["none_option"] = None

        try:
            result = model.resolve_prompt_options(
                "test prompt", ["short", "none_option"]
            )
            expected = "test prompt\n\nGenerate short caption"
            assert result == expected
        finally:
            model.PROMPT_OPTIONS = original_options

    def test_extract_required_variables_with_format_specifiers(self, mock_model_class):
        """Test extracting variables with format specifiers."""
        model = mock_model_class("base")
        prompt = "Generate {style:>10} caption with {count:d} items"
        variables = model.extract_required_variables(prompt)
        assert variables == {"style", "count"}

    def test_resolve_prompt_variables_keyerror_fallback(self, mock_model_class):
        """Test KeyError fallback in resolve_prompt_variables."""
        model = mock_model_class("base")

        prompt = "Generate {style} caption with {missing_var}"
        variables = {"style": "detailed"}

        with pytest.raises(ValueError, match="Missing required prompt variables"):
            model.resolve_prompt_variables(prompt, variables)

    def test_process_inputs_exception_handling(self, mock_model_class):
        """Test exception handling in process_inputs tensor conversion."""
        model = mock_model_class("base")

        mock_processor = MagicMock()
        mock_obj = MagicMock()

        del mock_obj.to

        mock_result = MagicMock()
        mock_result.to.return_value = {"pixel_values": mock_obj}
        mock_result.items.return_value = [("pixel_values", mock_obj)]

        mock_processor.return_value = mock_result
        model._processor = mock_processor

        test_image = Image.new("RGB", (100, 100), color="red")
        result = model.process_inputs(test_image, "test prompt")
        assert result is not None

    def test_process_inputs_non_tensor_objects(self, mock_model_class):
        """Test process_inputs with non-tensor objects that have 'to' method."""
        model = mock_model_class("base")

        mock_processor = MagicMock()
        mock_obj = MagicMock()
        mock_obj.to.return_value = mock_obj
        del mock_obj.dtype

        mock_result = MagicMock()
        mock_result.to.return_value = {"some_key": mock_obj}
        mock_result.items.return_value = [("some_key", mock_obj)]

        mock_processor.return_value = mock_result
        model._processor = mock_processor

        test_image = Image.new("RGB", (100, 100), color="red")
        result = model.process_inputs(test_image, "test prompt")
        assert result is not None

    def test_huggingface_id_property(self, mock_model_class):
        """Test huggingface_id property."""
        model = mock_model_class("base")
        assert model.huggingface_id == "test/model-base"

    def test_load_image_pil_image_conversion(self, mock_model_class):
        """Test load_image with PIL Image that needs RGB conversion."""
        test_image = Image.new("L", (100, 100), color=128)
        result = mock_model_class.load_image(test_image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)
