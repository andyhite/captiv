from ..config_section import ConfigSection


class ModelDefaults(ConfigSection):
    """Default configuration for models."""

    default_model: str = "blip"

    blip2_variant: str = "blip2-opt-2.7b"
    blip_variant: str = "blip-large"
    git_variant: str = "git-base"
    joycaption_variant: str = "joycaption-base"
    vit_gpt2_variant: str = "vit-gpt2"

    blip2_mode: str = "default"
    blip_mode: str | None = None
    git_mode: str | None = None
    joycaption_mode: str = "default"
    vit_gpt2_mode: str = "default"

    def validate(self) -> None:
        """Validate model defaults."""
        valid_models = ["blip", "blip2", "joycaption", "git", "vit-gpt2"]
        if self.default_model not in valid_models:
            self.default_model = "blip"
