from ..config_section import ConfigSection


class SystemDefaults(ConfigSection):
    """System-level configuration defaults."""

    supported_dtypes: list[str] = ["float16", "float32", "bfloat16"]

    default_torch_dtype: str | None = None

    def validate(self) -> None:
        """Validate system configuration."""
        if (
            self.default_torch_dtype is not None
            and self.default_torch_dtype not in self.supported_dtypes
        ):
            raise ValueError(
                f"Invalid torch dtype '{self.default_torch_dtype}'. "
                f"Supported dtypes: {', '.join(self.supported_dtypes)}"
            )
