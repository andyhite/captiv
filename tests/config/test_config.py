"""Tests for the main Config class."""

from captiv.config.config import Config
from captiv.config.defaults import (
    GenerationDefaults,
    GuiDefaults,
    ModelDefaults,
    SystemDefaults,
)


class TestConfig:
    """Test cases for the Config class."""

    def test_init_creates_default_sections(self):
        """Test that __init__ creates all default configuration sections."""
        config = Config()

        assert isinstance(config.model, ModelDefaults)
        assert isinstance(config.generation, GenerationDefaults)
        assert isinstance(config.system, SystemDefaults)
        assert isinstance(config.gui, GuiDefaults)

    def test_from_dict_empty_dict(self):
        """Test from_dict with an empty dictionary."""
        config = Config.from_dict({})

        assert isinstance(config.model, ModelDefaults)
        assert isinstance(config.generation, GenerationDefaults)
        assert isinstance(config.system, SystemDefaults)
        assert isinstance(config.gui, GuiDefaults)

    def test_from_dict_with_model_section(self):
        """Test from_dict with model section data."""
        data = {"model": {"default_model": "blip2", "blip2_variant": "blip2-opt-6.7b"}}

        config = Config.from_dict(data)

        assert config.model.default_model == "blip2"
        assert config.model.blip2_variant == "blip2-opt-6.7b"

    def test_from_dict_with_generation_section(self):
        """Test from_dict with generation section data."""
        data = {
            "generation": {"max_new_tokens": 64, "temperature": 0.8, "num_beams": 5}
        }

        config = Config.from_dict(data)

        assert config.generation.max_new_tokens == 64
        assert config.generation.temperature == 0.8
        assert config.generation.num_beams == 5

    def test_from_dict_with_system_section(self):
        """Test from_dict with system section data."""
        data = {"system": {"default_torch_dtype": "float16"}}

        config = Config.from_dict(data)

        assert config.system.default_torch_dtype == "float16"

    def test_from_dict_with_gui_section(self):
        """Test from_dict with GUI section data."""
        data = {"gui": {"host": "0.0.0.0", "port": 8080}}

        config = Config.from_dict(data)

        assert config.gui.host == "0.0.0.0"
        assert config.gui.port == 8080

    def test_from_dict_without_gui_section_creates_defaults(self):
        """Test that missing GUI section creates default GuiDefaults."""
        data = {"model": {"default_model": "blip"}}

        config = Config.from_dict(data)

        assert isinstance(config.gui, GuiDefaults)
        assert config.gui.host == "127.0.0.1"
        assert config.gui.port == 7860

    def test_from_dict_with_all_sections(self):
        """Test from_dict with all configuration sections."""
        data = {
            "model": {
                "default_model": "joycaption",
                "joycaption_variant": "joycaption-large",
            },
            "generation": {
                "max_new_tokens": 128,
                "min_new_tokens": 20,
                "temperature": 1.2,
            },
            "system": {"default_torch_dtype": "bfloat16"},
            "gui": {"host": "192.168.1.100", "port": 9000},
        }

        config = Config.from_dict(data)

        assert config.model.default_model == "joycaption"
        assert config.model.joycaption_variant == "joycaption-large"
        assert config.generation.max_new_tokens == 128
        assert config.generation.min_new_tokens == 20
        assert config.generation.temperature == 1.2
        assert config.system.default_torch_dtype == "bfloat16"
        assert config.gui.host == "192.168.1.100"
        assert config.gui.port == 9000

    def test_from_dict_with_non_dict_sections_ignored(self):
        """Test that non-dict section values are ignored."""
        data = {
            "model": "not_a_dict",
            "generation": ["not", "a", "dict"],
            "system": 42,
            "gui": None,
        }

        config = Config.from_dict(data)

        assert isinstance(config.model, ModelDefaults)
        assert config.model.default_model == "blip"
        assert isinstance(config.generation, GenerationDefaults)
        assert config.generation.max_new_tokens == 32
        assert isinstance(config.system, SystemDefaults)
        assert config.system.default_torch_dtype is None
        assert isinstance(config.gui, GuiDefaults)
        assert config.gui.host == "127.0.0.1"

    def test_to_dict_returns_all_sections(self):
        """Test that to_dict returns all configuration sections."""
        config = Config()
        result = config.to_dict()

        assert "model" in result
        assert "generation" in result
        assert "system" in result
        assert "gui" in result

        assert isinstance(result["model"], dict)
        assert isinstance(result["generation"], dict)
        assert isinstance(result["system"], dict)
        assert isinstance(result["gui"], dict)

    def test_to_dict_with_modified_values(self):
        """Test to_dict with modified configuration values."""
        config = Config()
        config.model.default_model = "blip2"
        config.generation.max_new_tokens = 64
        config.system.default_torch_dtype = "float16"
        config.gui.port = 8080

        result = config.to_dict()

        assert result["model"]["default_model"] == "blip2"
        assert result["generation"]["max_new_tokens"] == 64
        assert result["system"]["default_torch_dtype"] == "float16"
        assert result["gui"]["port"] == 8080

    def test_validate_calls_all_section_validations(self):
        """Test that validate calls validation on all sections."""
        config = Config()

        config.generation.max_new_tokens = -1
        config.generation.temperature = -0.5
        config.gui.port = 70000
        config.model.default_model = "invalid_model"

        config.validate()

        assert config.generation.max_new_tokens == 32
        assert config.generation.temperature == 1.0
        assert config.gui.port == 7860
        assert config.model.default_model == "blip"

    def test_from_dict_calls_validate(self):
        """Test that from_dict calls validate on the created config."""
        data = {
            "generation": {
                "max_new_tokens": -10,
                "temperature": 0,
            },
            "gui": {"port": 100000},
        }

        config = Config.from_dict(data)

        assert config.generation.max_new_tokens == 32
        assert config.generation.temperature == 1.0
        assert config.gui.port == 7860

    def test_roundtrip_from_dict_to_dict(self):
        """Test that from_dict -> to_dict preserves data."""
        original_data = {
            "model": {"default_model": "blip2", "blip2_variant": "blip2-opt-6.7b"},
            "generation": {"max_new_tokens": 64, "temperature": 0.8},
            "system": {"default_torch_dtype": "float16"},
            "gui": {"host": "0.0.0.0", "port": 8080},
        }

        config = Config.from_dict(original_data)
        result_data = config.to_dict()

        assert result_data["model"]["default_model"] == "blip2"
        assert result_data["model"]["blip2_variant"] == "blip2-opt-6.7b"
        assert result_data["generation"]["max_new_tokens"] == 64
        assert result_data["generation"]["temperature"] == 0.8
        assert result_data["system"]["default_torch_dtype"] == "float16"
        assert result_data["gui"]["host"] == "0.0.0.0"
        assert result_data["gui"]["port"] == 8080
        assert result_data["generation"]["temperature"] == 0.8
        assert result_data["system"]["default_torch_dtype"] == "float16"
        assert result_data["gui"]["host"] == "0.0.0.0"
        assert result_data["gui"]["port"] == 8080
