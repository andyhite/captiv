"""
RunPod configuration defaults.

This module provides default configuration values for RunPod integration, including API
settings, pod specifications, and deployment options.
"""

from ..config_section import ConfigSection


class RunPodDefaults(ConfigSection):
    """RunPod integration configuration defaults."""

    # API Configuration
    api_key: str | None = None
    template_id: str | None = None

    # Pod Configuration
    default_gpu_type: str = "NVIDIA RTX A4000"
    container_disk_size: int = 50  # GB
    volume_disk_size: int = 50  # GB
    ports: str = "7860/http,8888/http,22/tcp"

    # Pod Management
    auto_create: bool = True
    auto_terminate: bool = True
    max_idle_time: int = 300  # seconds
    startup_timeout: int = 600  # seconds

    # Model Configuration
    default_model_variant: str = "joycaption-beta-one"
    enable_model_caching: bool = True

    # Network Configuration
    request_timeout: int = 120  # seconds
    health_check_interval: int = 30  # seconds
    max_retries: int = 3

    def validate(self) -> None:
        """Validate RunPod configuration."""
        if self.container_disk_size < 20:
            raise ValueError("Container disk size must be at least 20 GB")

        if self.volume_disk_size < 20:
            raise ValueError("Volume disk size must be at least 20 GB")

        if self.startup_timeout < 60:
            raise ValueError("Startup timeout must be at least 60 seconds")

        if self.max_idle_time < 60:
            raise ValueError("Max idle time must be at least 60 seconds")

        if self.request_timeout < 30:
            raise ValueError("Request timeout must be at least 30 seconds")

        if self.health_check_interval < 10:
            raise ValueError("Health check interval must be at least 10 seconds")

        if self.max_retries < 1:
            raise ValueError("Max retries must be at least 1")
