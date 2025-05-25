"""Tests for the GenerationDefaults configuration class."""

from captiv.config.defaults.generation_defaults import GenerationDefaults


class TestGenerationDefaults:
    """Test cases for the GenerationDefaults class."""

    def test_init_creates_default_values(self):
        """Test that __init__ creates instance with default values."""
        config = GenerationDefaults()

        assert config.max_new_tokens == 32
        assert config.min_new_tokens == 10
        assert config.num_beams == 3
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.0
        assert config.joycaption_guidance_scale == 7.5
        assert config.joycaption_quality_level == "standard"

    def test_from_dict_creates_instance_with_defaults(self):
        """Test that from_dict creates instance with default values when given empty
        dict."""
        config = GenerationDefaults.from_dict({})

        assert config.max_new_tokens == 32
        assert config.min_new_tokens == 10
        assert config.num_beams == 3
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.0
        assert config.joycaption_guidance_scale == 7.5
        assert config.joycaption_quality_level == "standard"

    def test_from_dict_updates_existing_attributes(self):
        """Test that from_dict updates attributes that exist on the class."""
        data = {
            "max_new_tokens": 64,
            "min_new_tokens": 5,
            "temperature": 0.8,
            "top_k": 40,
            "joycaption_quality_level": "high",
        }

        config = GenerationDefaults.from_dict(data)

        assert config.max_new_tokens == 64
        assert config.min_new_tokens == 5
        assert config.temperature == 0.8
        assert config.top_k == 40
        assert config.joycaption_quality_level == "high"
        assert config.num_beams == 3
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.0
        assert config.joycaption_guidance_scale == 7.5

    def test_from_dict_ignores_non_existing_attributes(self):
        """Test that from_dict ignores attributes that don't exist on the class."""
        data = {
            "max_new_tokens": 64,
            "non_existing_attr": "should_be_ignored",
            "another_fake_attr": 999,
        }

        config = GenerationDefaults.from_dict(data)

        assert config.max_new_tokens == 64
        assert not hasattr(config, "non_existing_attr")
        assert not hasattr(config, "another_fake_attr")

    def test_to_dict_includes_all_attributes(self):
        """Test that to_dict includes all configuration attributes."""
        config = GenerationDefaults()
        result = config.to_dict()

        expected_keys = {
            "max_new_tokens",
            "min_new_tokens",
            "num_beams",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "joycaption_guidance_scale",
            "joycaption_quality_level",
        }

        assert set(result.keys()) >= expected_keys
        assert result["max_new_tokens"] == 32
        assert result["min_new_tokens"] == 10
        assert result["num_beams"] == 3
        assert result["temperature"] == 1.0
        assert result["top_k"] == 50
        assert result["top_p"] == 0.9
        assert result["repetition_penalty"] == 1.0
        assert result["joycaption_guidance_scale"] == 7.5
        assert result["joycaption_quality_level"] == "standard"

    def test_to_dict_with_modified_values(self):
        """Test to_dict with modified configuration values."""
        config = GenerationDefaults()
        config.max_new_tokens = 128
        config.temperature = 0.7
        config.joycaption_quality_level = "draft"

        result = config.to_dict()

        assert result["max_new_tokens"] == 128
        assert result["temperature"] == 0.7
        assert result["joycaption_quality_level"] == "draft"

    def test_validate_corrects_invalid_max_new_tokens(self):
        """Test that validate corrects invalid max_new_tokens values."""
        config = GenerationDefaults()

        config.max_new_tokens = -5
        config.validate()
        assert config.max_new_tokens == 32

        config.max_new_tokens = 0
        config.validate()
        assert config.max_new_tokens == 32

    def test_validate_corrects_invalid_min_new_tokens(self):
        """Test that validate corrects invalid min_new_tokens values."""
        config = GenerationDefaults()

        config.min_new_tokens = -3
        config.validate()
        assert config.min_new_tokens == 10

        config.min_new_tokens = 0
        config.validate()
        assert config.min_new_tokens == 10

    def test_validate_corrects_min_new_tokens_greater_than_max_new_tokens(self):
        """Test that validate corrects min_new_tokens when > max_new_tokens."""
        config = GenerationDefaults()
        config.max_new_tokens = 20
        config.min_new_tokens = 30

        config.validate()

        assert config.min_new_tokens == 20

    def test_validate_corrects_invalid_num_beams(self):
        """Test that validate corrects invalid num_beams values."""
        config = GenerationDefaults()

        config.num_beams = -1
        config.validate()
        assert config.num_beams == 3

        config.num_beams = 0
        config.validate()
        assert config.num_beams == 3

    def test_validate_corrects_invalid_temperature(self):
        """Test that validate corrects invalid temperature values."""
        config = GenerationDefaults()

        config.temperature = -0.5
        config.validate()
        assert config.temperature == 1.0

        config.temperature = 0.0
        config.validate()
        assert config.temperature == 1.0

    def test_validate_corrects_invalid_top_k(self):
        """Test that validate corrects invalid top_k values."""
        config = GenerationDefaults()

        config.top_k = -10
        config.validate()
        assert config.top_k == 50

        config.top_k = 0
        config.validate()
        assert config.top_k == 50

    def test_validate_corrects_invalid_top_p(self):
        """Test that validate corrects invalid top_p values."""
        config = GenerationDefaults()

        config.top_p = -0.1
        config.validate()
        assert config.top_p == 0.9

        config.top_p = 0.0
        config.validate()
        assert config.top_p == 0.9

        config.top_p = 1.5
        config.validate()
        assert config.top_p == 0.9

    def test_validate_corrects_invalid_repetition_penalty(self):
        """Test that validate corrects invalid repetition_penalty values."""
        config = GenerationDefaults()

        config.repetition_penalty = 0.5
        config.validate()
        assert config.repetition_penalty == 1.0

    def test_validate_preserves_valid_values(self):
        """Test that validate preserves valid values."""
        config = GenerationDefaults()
        config.max_new_tokens = 64
        config.min_new_tokens = 15
        config.num_beams = 5
        config.temperature = 0.8
        config.top_k = 40
        config.top_p = 0.95
        config.repetition_penalty = 1.2

        config.validate()

        assert config.max_new_tokens == 64
        assert config.min_new_tokens == 15
        assert config.num_beams == 5
        assert config.temperature == 0.8
        assert config.top_k == 40
        assert config.top_p == 0.95
        assert config.repetition_penalty == 1.2

    def test_validate_edge_case_top_p_exactly_one(self):
        """Test that validate handles top_p exactly equal to 1.0."""
        config = GenerationDefaults()
        config.top_p = 1.0

        config.validate()

        assert config.top_p == 1.0

    def test_validate_edge_case_repetition_penalty_exactly_one(self):
        """Test that validate handles repetition_penalty exactly equal to 1.0."""
        config = GenerationDefaults()
        config.repetition_penalty = 1.0

        config.validate()

        assert config.repetition_penalty == 1.0

    def test_validate_multiple_invalid_values(self):
        """Test that validate corrects multiple invalid values at once."""
        config = GenerationDefaults()
        config.max_new_tokens = -10
        config.min_new_tokens = -5
        config.num_beams = 0
        config.temperature = -1.0
        config.top_k = -20
        config.top_p = 2.0
        config.repetition_penalty = 0.5

        config.validate()

        assert config.max_new_tokens == 32
        assert config.min_new_tokens == 10
        assert config.num_beams == 3
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.0

    def test_roundtrip_from_dict_to_dict(self):
        """Test that from_dict -> to_dict preserves data correctly."""
        original_data = {
            "max_new_tokens": 128,
            "min_new_tokens": 20,
            "num_beams": 4,
            "temperature": 0.7,
            "top_k": 30,
            "top_p": 0.85,
            "repetition_penalty": 1.1,
            "joycaption_guidance_scale": 8.0,
            "joycaption_quality_level": "high",
        }

        config = GenerationDefaults.from_dict(original_data)
        result_data = config.to_dict()

        assert result_data["max_new_tokens"] == 128
        assert result_data["min_new_tokens"] == 20
        assert result_data["num_beams"] == 4
        assert result_data["temperature"] == 0.7
        assert result_data["top_k"] == 30
        assert result_data["top_p"] == 0.85
        assert result_data["repetition_penalty"] == 1.1
        assert result_data["joycaption_guidance_scale"] == 8.0
        assert result_data["joycaption_quality_level"] == "high"

    def test_joycaption_specific_parameters(self):
        """Test JoyCaption-specific parameters are handled correctly."""
        config = GenerationDefaults()

        assert config.joycaption_guidance_scale == 7.5
        assert config.joycaption_quality_level == "standard"

        config.joycaption_guidance_scale = 10.0
        config.joycaption_quality_level = "draft"

        assert config.joycaption_guidance_scale == 10.0
        assert config.joycaption_quality_level == "draft"

    def test_from_dict_with_joycaption_parameters(self):
        """Test from_dict with JoyCaption-specific parameters."""
        data = {"joycaption_guidance_scale": 9.5, "joycaption_quality_level": "high"}

        config = GenerationDefaults.from_dict(data)

        assert config.joycaption_guidance_scale == 9.5
        assert config.joycaption_quality_level == "high"
