"""Tests for the GuiDefaults configuration class."""

from captiv.config.defaults.gui_defaults import GuiDefaults


class TestGuiDefaults:
    """Test cases for the GuiDefaults class."""

    def test_init_creates_default_values(self) -> None:
        """Test that __init__ creates instance with default values."""
        config = GuiDefaults()

        assert config.host == "127.0.0.1"
        assert config.port == 7860
        assert config.ssl_keyfile is None
        assert config.ssl_certfile is None

    def test_validate_corrects_empty_host(self) -> None:
        """Test that validate corrects empty host values."""
        config = GuiDefaults()

        config.host = ""
        config.validate()
        assert config.host == "127.0.0.1"

        config.host = None
        config.validate()
        assert config.host == "127.0.0.1"

    def test_validate_corrects_invalid_port(self) -> None:
        """Test that validate corrects invalid port values."""
        config = GuiDefaults()

        config.port = None
        config.validate()
        assert config.port == 7860

        config.port = 0
        config.validate()
        assert config.port == 7860

        config.port = -1
        config.validate()
        assert config.port == 7860

        config.port = 70000
        config.validate()
        assert config.port == 7860

    def test_validate_corrects_empty_ssl_files(self) -> None:
        """Test that validate corrects empty SSL file paths."""
        config = GuiDefaults()

        config.ssl_keyfile = ""
        config.validate()
        assert config.ssl_keyfile is None

        config.ssl_certfile = ""
        config.validate()
        assert config.ssl_certfile is None

    def test_validate_preserves_valid_values(self) -> None:
        """Test that validate preserves valid values."""
        config = GuiDefaults()
        config.host = "0.0.0.0"
        config.port = 8080
        config.ssl_keyfile = "/path/to/key.pem"
        config.ssl_certfile = "/path/to/cert.pem"

        config.validate()

        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.ssl_keyfile == "/path/to/key.pem"
        assert config.ssl_certfile == "/path/to/cert.pem"

    def test_validate_preserves_valid_port_boundaries(self) -> None:
        """Test that validate preserves valid port boundary values."""
        config = GuiDefaults()

        config.port = 1
        config.validate()
        assert config.port == 1

        config.port = 65535
        config.validate()
        assert config.port == 65535

    def test_from_dict_with_all_values(self) -> None:
        """Test from_dict with all configuration values."""
        data = {
            "host": "192.168.1.100",
            "port": 9000,
            "ssl_keyfile": "/custom/key.pem",
            "ssl_certfile": "/custom/cert.pem",
        }

        config = GuiDefaults.from_dict(data)

        assert config.host == "192.168.1.100"
        assert config.port == 9000
        assert config.ssl_keyfile == "/custom/key.pem"
        assert config.ssl_certfile == "/custom/cert.pem"

    def test_to_dict_includes_all_attributes(self) -> None:
        """Test that to_dict includes all configuration attributes."""
        config = GuiDefaults()
        result = config.to_dict()

        expected_keys = {"host", "port", "ssl_keyfile", "ssl_certfile"}
        assert set(result.keys()) >= expected_keys

        assert result["host"] == "127.0.0.1"
        assert result["port"] == 7860
        assert result["ssl_keyfile"] is None
        assert result["ssl_certfile"] is None
