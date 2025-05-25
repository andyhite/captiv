"""
List command for RunPod management.

This module provides the command logic for listing RunPod instances.
"""

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services.config_manager import ConfigManager
from captiv.services.exceptions import RunPodError
from captiv.services.runpod_service import RunPodService


@handle_cli_errors
def command(
    config_file: ConfigFileOption = None,
) -> None:
    """List RunPod instances."""
    config_manager = ConfigManager(config_file)

    # Get API key from config
    api_key = config_manager.get_config_value("runpod", "api_key")
    if not api_key:
        typer.echo("Error: RunPod API key not configured.")
        typer.echo("Set it with: captiv config set runpod api_key YOUR_API_KEY")
        raise typer.Exit(1)

    try:
        runpod_service = RunPodService(api_key)

        # Note: This is a simplified implementation
        # The actual RunPod API would need a list_pods method
        typer.echo("📋 RunPod instances:")
        typer.echo("(List functionality requires additional RunPod API implementation)")

        if runpod_service.active_pod_id:
            typer.echo(f"Active pod: {runpod_service.active_pod_id}")
            if runpod_service.pod_endpoint:
                typer.echo(f"Endpoint: {runpod_service.pod_endpoint}")
        else:
            typer.echo("No active pods found")

    except RunPodError as e:
        typer.echo(f"❌ RunPod error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}")
        raise typer.Exit(1) from e
