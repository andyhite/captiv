from ..config_section import ConfigSection


class GuiDefaults(ConfigSection):
    """GUI configuration defaults."""

    host: str = "127.0.0.1"
    port: int = 7860
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None

    def validate(self) -> None:
        """Validate GUI configuration."""
        if not self.host:
            self.host = "127.0.0.1"

        if self.port is None or self.port < 1 or self.port > 65535:
            self.port = 7860

        if self.ssl_keyfile == "":
            self.ssl_keyfile = None

        if self.ssl_certfile == "":
            self.ssl_certfile = None
