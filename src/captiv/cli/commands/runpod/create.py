"""
Create command for RunPod management.

This module provides the command logic for creating new RunPod instances for remote
image captioning.
"""

from typing import Annotated

import typer
from loguru import logger

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services.config_manager import ConfigManager
from captiv.services.exceptions import RunPodError
from captiv.services.runpod_service import RunPodService

NameOption = Annotated[
    str,
    typer.Option(
        "--name",
        "-n",
        help="Name for the RunPod instance",
    ),
]

GpuTypeOption = Annotated[
    str,
    typer.Option(
        "--gpu-type",
        "-g",
        help="GPU type to use for the instance",
    ),
]

TemplateIdOption = Annotated[
    str | None,
    typer.Option(
        "--template-id",
        "-t",
        help="Template ID to use for pod creation",
    ),
]

WaitOption = Annotated[
    bool,
    typer.Option(
        "--wait/--no-wait",
        help="Wait for the pod to be ready before returning",
    ),
]


@handle_cli_errors
def command(
    name: NameOption = "captiv-joycaption",
    gpu_type: GpuTypeOption = "NVIDIA RTX A4000",
    template_id: TemplateIdOption = None,
    wait: WaitOption = True,
    config_file: ConfigFileOption = None,
) -> None:
    """Create a new RunPod instance for image captioning."""
    config_manager = ConfigManager(config_file)

    # Get API key from config
    api_key = config_manager.get_config_value("runpod", "api_key")
    if not api_key:
        typer.echo("Error: RunPod API key not configured.")
        typer.echo("Set it with: captiv config set runpod api_key YOUR_API_KEY")
        raise typer.Exit(1)

    # Use template from config if not provided
    if not template_id:
        template_id = config_manager.get_config_value("runpod", "template_id")

    try:
        runpod_service = RunPodService(api_key, template_id)

        typer.echo(f"Creating RunPod instance '{name}' with GPU type '{gpu_type}'...")

        pod_id = runpod_service.create_pod(name=name, gpu_type=gpu_type)

        typer.echo("✅ RunPod instance created successfully!")
        typer.echo(f"Pod ID: {pod_id}")

        if wait:
            typer.echo("⏳ Waiting for pod to be ready...")

            startup_timeout = config_manager.get_config_value(
                "runpod", "startup_timeout"
            )
            if startup_timeout is None:
                startup_timeout = 600

            runpod_service.wait_for_pod_ready(pod_id, timeout=startup_timeout)

            typer.echo("✅ Pod is ready!")
            if runpod_service.pod_endpoint:
                typer.echo(f"Endpoint: {runpod_service.pod_endpoint}")

            # Test health check
            if runpod_service.health_check():
                typer.echo("✅ Health check passed - service is ready for inference")
            else:
                typer.echo("⚠️  Health check failed - service may still be starting up")
        else:
            typer.echo("Use 'captiv runpod status' to check when the pod is ready")

    except RunPodError as e:
        typer.echo(f"❌ RunPod error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}")
        logger.exception("Failed to create RunPod instance")
        raise typer.Exit(1) from e
