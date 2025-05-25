from pathlib import Path
from typing import Any

import toml

from captiv.config import Config


class ConfigManager:
    def __init__(self, config_path: str | None = None):
        self._default_config = Config()
        self._config_path = config_path

    @property
    def config_dir(self) -> Path:
        """
        Get the directory where configuration files are stored.

        Returns:
            Path to the configuration directory
        """
        config_dir = Path.home() / ".captiv"
        config_dir.mkdir(exist_ok=True)
        return config_dir

    @property
    def config_path(self) -> Path:
        """
        Get the path to the configuration file.

        Returns:
            Path to the configuration file
        """
        if self._config_path:
            return Path(self._config_path)
        return self.config_dir / "config.toml"

    def read_config(self) -> Config:
        """
        Read the configuration from the configuration file.

        Returns:
            Config instance containing the configuration
        """
        path = self.config_path
        if not path.exists():
            return self._default_config

        try:
            with open(path) as f:
                config_dict = toml.load(f)
            return Config.from_dict(config_dict)
        except (OSError, toml.TomlDecodeError):
            return self._default_config

    def write_config(self, config: Config) -> None:
        """
        Write the configuration to the configuration file.

        Args:
            config: Config instance containing the configuration
        """
        path = self.config_path

        path.parent.mkdir(exist_ok=True)

        with open(path, "w") as f:
            toml.dump(config.to_dict(), f)

    def get_config(self) -> dict[str, dict[str, Any]]:
        """
        Get all configuration values.

        Returns:
            Dictionary containing the configuration
        """
        config = self.read_config()
        return config.to_dict()

    def get_config_value(self, section: str, key: str) -> Any:
        """
        Get a configuration value from a specific section.

        Args:
            section: Configuration section name
            key: Configuration key within the section

        Returns:
            Configuration value
        """
        config = self.read_config()
        if hasattr(config, section):
            section_obj = getattr(config, section)
            if hasattr(section_obj, key):
                return getattr(section_obj, key)

        return None

    def set_config_value(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value in a specific section.

        Args:
            section: Configuration section name
            key: Configuration key within the section
            value: Configuration value to set
        """
        config = self.read_config()

        if hasattr(config, section):
            section_obj = getattr(config, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                config.validate()
                self.write_config(config)
            else:
                raise ValueError(f"Key '{key}' not found in section '{section}'")
        else:
            raise ValueError(f"Section '{section}' not found in configuration")

    def clear_config(self, section: str | None = None) -> None:
        """
        Clear configuration values for a section or the entire configuration.

        Args:
            section: Configuration section name to clear. If None, clears entire config.
        """
        if section:
            config = self.read_config()
            if hasattr(config, section):
                section_class = getattr(config, section).__class__
                setattr(config, section, section_class())
                self.write_config(config)
            else:
                raise ValueError(f"Section '{section}' not found in configuration")
        else:
            path = self.config_path
            if path.exists():
                path.unlink()

    def unset_config_value(self, section: str, key: str) -> None:
        """
        Remove a configuration value, resetting it to the default.

        Args:
            section: Configuration section name
            key: Configuration key within the section
        """
        config = self.read_config()

        if not hasattr(config, section):
            raise ValueError(f"Section '{section}' not found in configuration")

        section_obj = getattr(config, section)
        if not hasattr(section_obj, key):
            raise ValueError(f"Key '{key}' not found in section '{section}'")

        default_section = getattr(self._default_config, section)
        if hasattr(default_section, key):
            default_value = getattr(default_section, key)
            setattr(section_obj, key, default_value)
            self.write_config(config)
        else:
            raise ValueError(f"Could not determine default value for {section}.{key}")
