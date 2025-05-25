"""Tests for the SystemDefaults configuration class."""

import pytest

from captiv.config.defaults.system_defaults import SystemDefaults


class TestSystemDefaults:
    """Test cases for the SystemDefaults class."""

    def test_init_creates_default_values(self):
        """Test that __init__ creates instance with default values."""
        config = SystemDefaults()

        assert config.supported_dtypes == ["float16", "float32", "bfloat16"]
        assert config.default_torch_dtype is None

    def test_validate_allows_none_dtype(self):
        """Test that validate allows None for default_torch_dtype."""
        config = SystemDefaults()
        config.default_torch_dtype = None

        config.validate()

        assert config.default_torch_dtype is None

    def test_validate_allows_valid_dtypes(self):
        """Test that validate allows valid torch dtypes."""
        config = SystemDefaults()

        for dtype in ["float16", "float32", "bfloat16"]:
            config.default_torch_dtype = dtype
            config.validate()
            assert config.default_torch_dtype == dtype

    def test_validate_raises_for_invalid_dtype(self):
        """Test that validate raises ValueError for invalid torch dtypes."""
        config = SystemDefaults()
        config.default_torch_dtype = "invalid_dtype"

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "Invalid torch dtype 'invalid_dtype'" in str(exc_info.value)
        assert "Supported dtypes: float16, float32, bfloat16" in str(exc_info.value)

    def test_validate_raises_for_unsupported_dtype(self):
        """Test that validate raises ValueError for unsupported but real torch
        dtypes."""
        config = SystemDefaults()
        config.default_torch_dtype = "float64"

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "Invalid torch dtype 'float64'" in str(exc_info.value)

    def test_from_dict_with_valid_dtype(self):
        """Test from_dict with valid torch dtype."""
        data = {"default_torch_dtype": "float16"}

        config = SystemDefaults.from_dict(data)

        assert config.default_torch_dtype == "float16"

    def test_from_dict_with_none_dtype(self):
        """Test from_dict with None torch dtype."""
        data = {"default_torch_dtype": None}

        config = SystemDefaults.from_dict(data)

        assert config.default_torch_dtype is None

    def test_to_dict_includes_all_attributes(self):
        """Test that to_dict includes all configuration attributes."""
        config = SystemDefaults()
        result = config.to_dict()

        expected_keys = {"supported_dtypes", "default_torch_dtype"}
        assert set(result.keys()) >= expected_keys

        assert result["supported_dtypes"] == ["float16", "float32", "bfloat16"]
        assert result["default_torch_dtype"] is None

    def test_to_dict_with_modified_values(self):
        """Test to_dict with modified configuration values."""
        config = SystemDefaults()
        config.default_torch_dtype = "bfloat16"

        result = config.to_dict()

        assert result["default_torch_dtype"] == "bfloat16"

    def test_roundtrip_from_dict_to_dict(self):
        """Test that from_dict -> to_dict preserves data correctly."""
        original_data = {"default_torch_dtype": "float32"}

        config = SystemDefaults.from_dict(original_data)
        result_data = config.to_dict()

        assert result_data["default_torch_dtype"] == "float32"
