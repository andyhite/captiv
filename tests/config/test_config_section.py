"""Tests for the ConfigSection base class."""

import pytest

from captiv.config.config_section import ConfigSection


class MockConfigSection(ConfigSection):
    """Test implementation of ConfigSection for testing purposes."""

    test_string: str = "default_value"
    test_int: int = 42
    test_float: float = 3.14
    test_bool: bool = True

    def validate(self) -> None:
        """Test validation implementation."""
        if self.test_int < 0:
            self.test_int = 0


class MockConfigSectionWithMethods(ConfigSection):
    """Test implementation with methods to test filtering."""

    test_attr: str = "value"

    def validate(self) -> None:
        """Test validation."""
        pass

    def some_method(self) -> str:
        """A method that should be filtered out."""
        return "method_result"

    @property
    def some_property(self) -> str:
        """A property that should be filtered out."""
        return "property_result"


class MockConfigSectionWithPrivateAttrs(ConfigSection):
    """Test implementation with private attributes."""

    public_attr: str = "public"
    _private_attr: str = "private"
    __very_private_attr: str = "very_private"

    def validate(self) -> None:
        """Test validation."""
        pass


class TestConfigSection:
    """Test cases for the ConfigSection base class."""

    def test_from_dict_creates_instance_with_defaults(self):
        """Test that from_dict creates an instance with default values."""
        config = MockConfigSection.from_dict({})

        assert config.test_string == "default_value"
        assert config.test_int == 42
        assert config.test_float == 3.14
        assert config.test_bool is True

    def test_from_dict_updates_existing_attributes(self):
        """Test that from_dict updates attributes that exist on the class."""
        data = {"test_string": "updated_value", "test_int": 100, "test_float": 2.71}

        config = MockConfigSection.from_dict(data)

        assert config.test_string == "updated_value"
        assert config.test_int == 100
        assert config.test_float == 2.71
        assert config.test_bool is True

    def test_from_dict_ignores_non_existing_attributes(self):
        """Test that from_dict ignores attributes that don't exist on the class."""
        data = {
            "test_string": "updated_value",
            "non_existing_attr": "should_be_ignored",
            "another_fake_attr": 999,
        }

        config = MockConfigSection.from_dict(data)

        assert config.test_string == "updated_value"
        assert not hasattr(config, "non_existing_attr")
        assert not hasattr(config, "another_fake_attr")

    def test_from_dict_with_mixed_valid_invalid_data(self):
        """Test from_dict with a mix of valid and invalid attribute names."""
        data = {
            "test_string": "valid_update",
            "test_int": 200,
            "invalid_attr1": "ignored",
            "test_float": 1.41,
            "invalid_attr2": 123,
        }

        config = MockConfigSection.from_dict(data)

        assert config.test_string == "valid_update"
        assert config.test_int == 200
        assert config.test_float == 1.41
        assert config.test_bool is True
        assert not hasattr(config, "invalid_attr1")
        assert not hasattr(config, "invalid_attr2")

    def test_to_dict_includes_class_attributes(self):
        """Test that to_dict includes class attributes (defaults)."""
        config = MockConfigSection()
        result = config.to_dict()

        assert "test_string" in result
        assert "test_int" in result
        assert "test_float" in result
        assert "test_bool" in result

        assert result["test_string"] == "default_value"
        assert result["test_int"] == 42
        assert result["test_float"] == 3.14
        assert result["test_bool"] is True

    def test_to_dict_includes_instance_attributes(self):
        """Test that to_dict includes instance attributes and they override class
        attributes."""
        config = MockConfigSection()
        config.test_string = "instance_value"
        config.new_instance_attr = "new_value"

        result = config.to_dict()

        assert result["test_string"] == "instance_value"
        assert result["new_instance_attr"] == "new_value"
        assert result["test_int"] == 42

    def test_to_dict_excludes_methods_and_special_attributes(self):
        """Test that to_dict excludes methods and special attributes."""
        config = MockConfigSectionWithMethods()
        result = config.to_dict()

        assert "test_attr" in result
        assert "some_method" not in result
        assert "some_property" in result
        assert "validate" not in result
        assert "to_dict" not in result
        assert "from_dict" not in result

    def test_to_dict_excludes_private_attributes(self):
        """Test that to_dict excludes private attributes."""
        config = MockConfigSectionWithPrivateAttrs()
        result = config.to_dict()

        assert "public_attr" in result
        assert "_private_attr" not in result
        assert "__very_private_attr" not in result

    def test_to_dict_excludes_callable_instance_attributes(self):
        """Test that to_dict excludes callable instance attributes."""
        config = MockConfigSection()
        config.test_function = lambda x: x * 2

        result = config.to_dict()

        assert "test_function" not in result

    def test_to_dict_with_private_instance_attributes(self):
        """Test that to_dict excludes private instance attributes."""
        config = MockConfigSection()
        config._private_instance = "private"
        config.__very_private_instance = "very_private"
        config.public_instance = "public"

        result = config.to_dict()

        assert "public_instance" in result
        assert "_private_instance" not in result
        assert "__very_private_instance" not in result

    def test_roundtrip_from_dict_to_dict(self):
        """Test that from_dict -> to_dict preserves data correctly."""
        original_data = {
            "test_string": "roundtrip_value",
            "test_int": 999,
            "test_float": 2.718,
            "test_bool": False,
        }

        config = MockConfigSection.from_dict(original_data)
        result_data = config.to_dict()

        assert result_data["test_string"] == "roundtrip_value"
        assert result_data["test_int"] == 999
        assert result_data["test_float"] == 2.718
        assert result_data["test_bool"] is False

    def test_from_dict_returns_correct_type(self):
        """Test that from_dict returns an instance of the correct class."""
        config = MockConfigSection.from_dict({"test_string": "test"})

        assert isinstance(config, MockConfigSection)
        assert type(config) is MockConfigSection

    def test_from_dict_with_subclass_returns_subclass_instance(self):
        """Test that from_dict on a subclass returns the subclass type."""

        class SubMockConfigSection(MockConfigSection):
            sub_attr: str = "sub_default"

            def validate(self) -> None:
                super().validate()

        config = SubMockConfigSection.from_dict(
            {"test_string": "test", "sub_attr": "sub_test"}
        )

        assert isinstance(config, SubMockConfigSection)
        assert type(config) is SubMockConfigSection
        assert config.test_string == "test"
        assert config.sub_attr == "sub_test"

    def test_validate_is_abstract_method(self):
        """Test that ConfigSection.validate is abstract and cannot be instantiated
        directly."""
        with pytest.raises(TypeError):
            ConfigSection()

    def test_concrete_implementation_can_be_instantiated(self):
        """Test that concrete implementations can be instantiated."""
        config = MockConfigSection()
        assert isinstance(config, ConfigSection)
        assert isinstance(config, MockConfigSection)

    def test_validation_is_called_in_subclass(self):
        """Test that validation logic works in concrete implementations."""
        config = MockConfigSection()
        config.test_int = -5

        config.validate()

        assert config.test_int == 0

    def test_to_dict_handles_none_values(self):
        """Test that to_dict properly handles None values."""
        config = MockConfigSection()
        config.test_string = None

        result = config.to_dict()

        assert result["test_string"] is None

    def test_from_dict_handles_none_values(self):
        """Test that from_dict properly handles None values."""
        data = {"test_string": None, "test_int": None}

        config = MockConfigSection.from_dict(data)

        assert config.test_string is None
        assert config.test_int is None
