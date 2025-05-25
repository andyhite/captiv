"""Tests for model defaults configuration."""

from captiv.config.defaults.model_defaults import ModelDefaults


class TestModelDefaults:
    """Test cases for ModelDefaults configuration."""

    def test_model_defaults_initialization(self):
        """Test ModelDefaults initialization with default values."""
        defaults = ModelDefaults()

        assert defaults.default_model == "blip"

        assert defaults.blip2_variant == "blip2-opt-2.7b"
        assert defaults.blip_variant == "blip-large"
        assert defaults.git_variant == "git-base"
        assert defaults.joycaption_variant == "joycaption-base"
        assert defaults.vit_gpt2_variant == "vit-gpt2"

        assert defaults.blip2_mode == "default"
        assert defaults.blip_mode is None
        assert defaults.git_mode is None
        assert defaults.joycaption_mode == "default"
        assert defaults.vit_gpt2_mode == "default"

    def test_validate_with_valid_default_model(self):
        """Test validate method with valid default model."""
        defaults = ModelDefaults()
        defaults.default_model = "blip2"

        defaults.validate()

        assert defaults.default_model == "blip2"

    def test_validate_with_invalid_default_model(self):
        """Test validate method with invalid default model."""
        defaults = ModelDefaults()
        defaults.default_model = "invalid_model"

        defaults.validate()

        assert defaults.default_model == "blip"

    def test_validate_with_all_valid_models(self):
        """Test validate method with all valid model types."""
        valid_models = ["blip", "blip2", "joycaption", "git", "vit-gpt2"]

        for model in valid_models:
            defaults = ModelDefaults()
            defaults.default_model = model
            defaults.validate()
            assert defaults.default_model == model

    def test_validate_preserves_other_attributes(self):
        """Test that validate only affects default_model and preserves other
        attributes."""
        defaults = ModelDefaults()

        original_blip_variant = "custom-blip-variant"
        original_blip2_mode = "custom-mode"

        defaults.blip_variant = original_blip_variant
        defaults.blip2_mode = original_blip2_mode
        defaults.default_model = "invalid_model"

        defaults.validate()

        assert defaults.default_model == "blip"
        assert defaults.blip_variant == original_blip_variant
        assert defaults.blip2_mode == original_blip2_mode

    def test_model_defaults_inheritance(self):
        """Test that ModelDefaults inherits from ConfigSection."""
        from captiv.config.config_section import ConfigSection

        defaults = ModelDefaults()
        assert isinstance(defaults, ConfigSection)

    def test_model_defaults_attributes_exist(self):
        """Test that all expected attributes exist on ModelDefaults."""
        defaults = ModelDefaults()

        expected_attributes = [
            "default_model",
            "blip2_variant",
            "blip_variant",
            "git_variant",
            "joycaption_variant",
            "vit_gpt2_variant",
            "blip2_mode",
            "blip_mode",
            "git_mode",
            "joycaption_mode",
            "vit_gpt2_mode",
        ]

        for attr in expected_attributes:
            assert hasattr(defaults, attr), f"Missing attribute: {attr}"
