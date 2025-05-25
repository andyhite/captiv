"""Tests for ConfigManager service."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from captiv.services.config_manager import ConfigManager


class TestConfigManager:
    """Test the ConfigManager service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.toml"

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.config_file.exists():
            self.config_file.unlink()
        os.rmdir(self.temp_dir)

    def test_init_with_config_path(self):
        """Test ConfigManager initialization with a config path."""
        manager = ConfigManager(str(self.config_file))
        assert manager.config_path == self.config_file

    def test_init_without_config_path(self):
        """Test ConfigManager initialization without a config path."""
        manager = ConfigManager()
        expected_path = Path.home() / ".captiv" / "config.toml"
        assert manager.config_path == expected_path

    def test_config_dir_property(self):
        """Test the config_dir property."""
        manager = ConfigManager()
        expected_dir = Path.home() / ".captiv"
        assert manager.config_dir == expected_dir

    def test_config_path_property_with_custom_path(self):
        """Test the config_path property with custom path."""
        manager = ConfigManager(str(self.config_file))
        assert manager.config_path == self.config_file

    def test_config_path_property_default(self):
        """Test the config_path property with default path."""
        manager = ConfigManager()
        expected_path = Path.home() / ".captiv" / "config.toml"
        assert manager.config_path == expected_path

    @patch("captiv.services.config_manager.Config")
    def test_read_config_file_not_exists(self, mock_config_class):
        """Test reading config when file doesn't exist."""
        mock_default_config = MagicMock()
        mock_config_class.return_value = mock_default_config

        manager = ConfigManager(str(self.config_file))
        result = manager.read_config()

        assert result == mock_default_config

    @patch("captiv.services.config_manager.Config")
    @patch(
        "builtins.open", new_callable=mock_open, read_data='[section]\nkey = "value"'
    )
    @patch("captiv.services.config_manager.toml.load")
    def test_read_config_file_exists(
        self, mock_toml_load, mock_file, mock_config_class
    ):
        """Test reading config when file exists."""
        self.config_file.touch()

        mock_config_dict = {"section": {"key": "value"}}
        mock_toml_load.return_value = mock_config_dict

        mock_config = MagicMock()
        mock_config_class.from_dict.return_value = mock_config

        manager = ConfigManager(str(self.config_file))
        result = manager.read_config()

        assert result == mock_config
        mock_config_class.from_dict.assert_called_once_with(mock_config_dict)

    @patch("captiv.services.config_manager.Config")
    @patch("builtins.open", side_effect=OSError("File error"))
    def test_read_config_io_error(self, _mock_file, mock_config_class):
        """Test reading config when IO error occurs."""
        self.config_file.touch()

        mock_default_config = MagicMock()
        mock_config_class.return_value = mock_default_config

        manager = ConfigManager(str(self.config_file))
        result = manager.read_config()

        assert result == mock_default_config

    @patch("captiv.services.config_manager.toml.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_config(self, mock_file, mock_toml_dump):
        """Test writing config to file."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"section": {"key": "value"}}

        manager = ConfigManager(str(self.config_file))
        manager.write_config(mock_config)

        mock_toml_dump.assert_called_once_with(
            {"section": {"key": "value"}}, mock_file.return_value
        )

    @patch("captiv.services.config_manager.ConfigManager.read_config")
    def test_get_config(self, mock_read_config):
        """Test getting all config values."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"section": {"key": "value"}}
        mock_read_config.return_value = mock_config

        manager = ConfigManager(str(self.config_file))
        result = manager.get_config()

        assert result == {"section": {"key": "value"}}
        mock_config.to_dict.assert_called_once()

    @patch("captiv.services.config_manager.ConfigManager.read_config")
    def test_get_config_value_existing(self, mock_read_config):
        """Test getting an existing config value."""
        mock_config = MagicMock()
        mock_section = MagicMock()
        mock_section.key = "test_value"
        mock_config.section = mock_section
        mock_read_config.return_value = mock_config

        manager = ConfigManager(str(self.config_file))
        result = manager.get_config_value("section", "key")

        assert result == "test_value"

    @patch("captiv.services.config_manager.ConfigManager.read_config")
    def test_get_config_value_nonexistent_section(self, mock_read_config):
        """Test getting a config value from non-existent section."""
        mock_config = MagicMock()
        del mock_config.nonexistent
        mock_read_config.return_value = mock_config

        manager = ConfigManager(str(self.config_file))
        result = manager.get_config_value("nonexistent", "key")

        assert result is None

    @patch("captiv.services.config_manager.ConfigManager.read_config")
    @patch("captiv.services.config_manager.ConfigManager.write_config")
    def test_set_config_value_success(self, mock_write_config, mock_read_config):
        """Test setting a config value successfully."""
        mock_config = MagicMock()
        mock_section = MagicMock()
        mock_config.section = mock_section
        mock_read_config.return_value = mock_config

        manager = ConfigManager(str(self.config_file))
        manager.set_config_value("section", "key", "new_value")

        assert mock_section.key == "new_value"
        mock_config.validate.assert_called_once()
        mock_write_config.assert_called_once_with(mock_config)

    @patch("captiv.services.config_manager.ConfigManager.read_config")
    @patch("captiv.services.config_manager.ConfigManager.write_config")
    def test_clear_config_all(self, mock_write_config, mock_read_config):
        """Test clearing all config values."""
        self.config_file.touch()

        manager = ConfigManager(str(self.config_file))
        manager.clear_config()

        assert not self.config_file.exists()

    @patch("captiv.services.config_manager.ConfigManager.read_config")
    @patch("captiv.services.config_manager.ConfigManager.write_config")
    def test_clear_config_section(self, mock_write_config, mock_read_config):
        """Test clearing a specific config section."""
        mock_config = MagicMock()
        mock_section = MagicMock()

        mock_config.section = mock_section
        mock_read_config.return_value = mock_config

        manager = ConfigManager(str(self.config_file))
        manager.clear_config("section")

        mock_write_config.assert_called_once_with(mock_config)

    @patch("captiv.services.config_manager.ConfigManager.read_config")
    @patch("captiv.services.config_manager.ConfigManager.write_config")
    def test_unset_config_value_success(self, mock_write_config, mock_read_config):
        """Test unsetting a config value successfully."""
        mock_config = MagicMock()
        mock_section = MagicMock()
        mock_config.section = mock_section
        mock_read_config.return_value = mock_config

        mock_default_config = MagicMock()
        mock_default_section = MagicMock()
        mock_default_section.key = "default_value"
        mock_default_config.section = mock_default_section

        manager = ConfigManager(str(self.config_file))
        manager._default_config = mock_default_config
        manager.unset_config_value("section", "key")

        assert mock_section.key == "default_value"
        mock_write_config.assert_called_once_with(mock_config)


class TestConfigManagerEdgeCases:
    """Test edge cases and error scenarios for ConfigManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.toml"

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.config_file.exists():
            self.config_file.unlink()
        if os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_config_dir_creation(self):
        """Test that config directory is created if it doesn't exist."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path(self.temp_dir)

            manager = ConfigManager()
            config_dir = manager.config_dir

            assert config_dir.exists()
            assert config_dir.is_dir()

    def test_config_dir_already_exists(self):
        """Test config directory when it already exists."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path(self.temp_dir)

            captiv_dir = Path(self.temp_dir) / ".captiv"
            captiv_dir.mkdir()

            manager = ConfigManager()
            config_dir = manager.config_dir

            assert config_dir.exists()
            assert config_dir.is_dir()

    def test_read_config_toml_decode_error(self):
        """Test reading config when TOML decode error occurs."""
        self.config_file.write_text("invalid toml content [[[")

        manager = ConfigManager(str(self.config_file))
        result = manager.read_config()

        from captiv.config import Config

        assert isinstance(result, Config)

    def test_read_config_permission_error(self):
        """Test reading config when permission error occurs."""
        self.config_file.touch()

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            manager = ConfigManager(str(self.config_file))
            result = manager.read_config()

            from captiv.config import Config

            assert isinstance(result, Config)

    def test_write_config_creates_parent_directory(self):
        """Test that write_config creates parent directories."""
        nested_config_file = Path(self.temp_dir) / "nested" / "config.toml"

        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"section": {"key": "value"}}

        manager = ConfigManager(str(nested_config_file))
        manager.write_config(mock_config)

        assert nested_config_file.parent.exists()

    def test_write_config_permission_error(self):
        """Test write_config when permission error occurs."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"section": {"key": "value"}}

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(PermissionError):
                manager.write_config(mock_config)

    def test_get_config_value_nonexistent_key(self):
        """Test getting a config value for non-existent key."""
        mock_config = MagicMock()
        mock_section = MagicMock()
        del mock_section.nonexistent_key
        mock_config.section = mock_section

        with patch.object(ConfigManager, "read_config", return_value=mock_config):
            manager = ConfigManager(str(self.config_file))
            result = manager.get_config_value("section", "nonexistent_key")

            assert result is None

    def test_set_config_value_nonexistent_section(self):
        """Test setting a config value for non-existent section."""
        mock_config = MagicMock()
        del mock_config.nonexistent_section

        with patch.object(ConfigManager, "read_config", return_value=mock_config):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(
                ValueError, match="Section 'nonexistent_section' not found"
            ):
                manager.set_config_value("nonexistent_section", "key", "value")

    def test_set_config_value_nonexistent_key(self):
        """Test setting a config value for non-existent key."""
        mock_config = MagicMock()
        mock_section = MagicMock()
        del mock_section.nonexistent_key
        mock_config.section = mock_section

        with patch.object(ConfigManager, "read_config", return_value=mock_config):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(ValueError, match="Key 'nonexistent_key' not found"):
                manager.set_config_value("section", "nonexistent_key", "value")

    def test_set_config_value_validation_error(self):
        """Test setting a config value that fails validation."""
        mock_config = MagicMock()
        mock_section = MagicMock()
        mock_config.section = mock_section
        mock_config.validate.side_effect = ValueError("Validation failed")

        with (
            patch.object(ConfigManager, "read_config", return_value=mock_config),
            patch.object(ConfigManager, "write_config"),
        ):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(ValueError, match="Validation failed"):
                manager.set_config_value("section", "key", "invalid_value")

    def test_clear_config_nonexistent_section(self):
        """Test clearing a non-existent config section."""
        mock_config = MagicMock()
        del mock_config.nonexistent_section

        with patch.object(ConfigManager, "read_config", return_value=mock_config):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(
                ValueError, match="Section 'nonexistent_section' not found"
            ):
                manager.clear_config("nonexistent_section")

    def test_clear_config_all_file_not_exists(self):
        """Test clearing all config when file doesn't exist."""
        manager = ConfigManager(str(self.config_file))

        manager.clear_config()

        assert not self.config_file.exists()

    def test_clear_config_all_permission_error(self):
        """Test clearing all config when permission error occurs."""
        self.config_file.touch()

        with patch.object(
            Path, "unlink", side_effect=PermissionError("Permission denied")
        ):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(PermissionError):
                manager.clear_config()

    def test_unset_config_value_nonexistent_section(self):
        """Test unsetting a config value for non-existent section."""
        mock_config = MagicMock()
        del mock_config.nonexistent_section

        with patch.object(ConfigManager, "read_config", return_value=mock_config):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(
                ValueError, match="Section 'nonexistent_section' not found"
            ):
                manager.unset_config_value("nonexistent_section", "key")

    def test_unset_config_value_nonexistent_key(self):
        """Test unsetting a config value for non-existent key."""
        mock_config = MagicMock()
        mock_section = MagicMock()
        del mock_section.nonexistent_key
        mock_config.section = mock_section

        with patch.object(ConfigManager, "read_config", return_value=mock_config):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(ValueError, match="Key 'nonexistent_key' not found"):
                manager.unset_config_value("section", "nonexistent_key")

    def test_unset_config_value_no_default_key(self):
        """Test unsetting a config value when default doesn't have the key."""
        mock_config = MagicMock()
        mock_section = MagicMock()
        mock_config.section = mock_section

        mock_default_config = MagicMock()
        mock_default_section = MagicMock()
        del mock_default_section.key
        mock_default_config.section = mock_default_section

        with patch.object(ConfigManager, "read_config", return_value=mock_config):
            manager = ConfigManager(str(self.config_file))
            manager._default_config = mock_default_config

            with pytest.raises(ValueError, match="Could not determine default value"):
                manager.unset_config_value("section", "key")

    def test_config_path_with_relative_path(self):
        """Test config path with relative path."""
        manager = ConfigManager("./relative/config.toml")
        expected_path = Path("./relative/config.toml")
        assert manager.config_path == expected_path

    def test_config_path_with_absolute_path(self):
        """Test config path with absolute path."""
        abs_path = str(self.config_file.absolute())
        manager = ConfigManager(abs_path)
        assert manager.config_path == Path(abs_path)

    def test_read_config_with_complex_toml(self):
        """Test reading config with complex TOML structure."""
        complex_toml = """
        [section1]
        key1 = "value1"
        key2 = 42

        [section2]
        nested_key = "nested_value"

        [section3.subsection]
        deep_key = true
        """

        self.config_file.write_text(complex_toml)

        with patch("captiv.services.config_manager.Config") as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.from_dict.return_value = mock_config

            manager = ConfigManager(str(self.config_file))
            manager.read_config()

            mock_config_class.from_dict.assert_called_once()
            call_args = mock_config_class.from_dict.call_args[0][0]
            assert "section1" in call_args
            assert "section2" in call_args
            assert "section3" in call_args

    def test_write_config_with_complex_structure(self):
        """Test writing config with complex structure."""
        mock_config = MagicMock()
        complex_dict = {
            "section1": {"key1": "value1", "key2": 42},
            "section2": {"nested": {"deep": "value"}},
        }
        mock_config.to_dict.return_value = complex_dict

        manager = ConfigManager(str(self.config_file))
        manager.write_config(mock_config)

        assert self.config_file.exists()
        import toml

        written_content = toml.load(self.config_file)
        assert written_content == complex_dict

    def test_get_config_with_read_error(self):
        """Test get_config when read_config fails."""
        with patch.object(
            ConfigManager, "read_config", side_effect=OSError("Read failed")
        ):
            manager = ConfigManager(str(self.config_file))

            with pytest.raises(IOError, match="Read failed"):
                manager.get_config()

    def test_config_manager_with_none_path(self):
        """Test ConfigManager initialization with None path."""
        manager = ConfigManager(None)
        expected_path = Path.home() / ".captiv" / "config.toml"
        assert manager.config_path == expected_path
