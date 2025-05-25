from ..config_section import ConfigSection


class GenerationDefaults(ConfigSection):
    """Default configuration for text generation parameters."""

    max_new_tokens: int = 32
    min_new_tokens: int = 10
    num_beams: int = 3
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0

    joycaption_guidance_scale: float = 7.5
    joycaption_quality_level: str = "standard"

    def validate(self) -> None:
        """Validate generation parameters."""
        if self.max_new_tokens < 1:
            self.max_new_tokens = 32

        if self.min_new_tokens < 1:
            self.min_new_tokens = 10

        if self.min_new_tokens > self.max_new_tokens:
            self.min_new_tokens = self.max_new_tokens

        if self.num_beams < 1:
            self.num_beams = 3

        if self.temperature <= 0:
            self.temperature = 1.0

        if self.top_k < 1:
            self.top_k = 50

        if self.top_p <= 0 or self.top_p > 1:
            self.top_p = 0.9

        if self.repetition_penalty < 1:
            self.repetition_penalty = 1.0
