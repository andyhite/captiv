"""
Stop command for RunPod management.

This module provides the command logic for stopping RunPod instances.
"""

from typing import Annotated

import typer

from captiv.cli.error_handling import handle_cli_errors
from captiv.cli.options import ConfigFileOption
from captiv.services.config_manager import ConfigManager
from captiv.services.exceptions import RunPodError
from captiv.services.runpod_service import RunPodService

PodIdArgument = Annotated[
    str,
    typer.Argument(help="Pod ID to stop"),
]


@handle_cli_errors
def command(
    pod_id: PodIdArgument,
    config_file: ConfigFileOption = None,
) -> None:
    """Stop a RunPod instance (can be restarted later)."""
    config_manager = ConfigManager(config_file)

    # Get API key from config
    api_key = config_manager.get_config_value("runpod", "api_key")
    if not api_key:
        typer.echo("Error: RunPod API key not configured.")
        typer.echo("Set it with: captiv config set runpod api_key YOUR_API_KEY")
        raise typer.Exit(1)

    try:
        runpod_service = RunPodService(api_key)

        typer.echo(f"🛑 Stopping pod: {pod_id}")

        success = runpod_service.stop_pod(pod_id)

        if success:
            typer.echo("✅ Pod stopped successfully")
            typer.echo("💡 Use 'captiv runpod create' to start a new instance")
        else:
            typer.echo("❌ Failed to stop pod")
            raise typer.Exit(1)

    except RunPodError as e:
        typer.echo(f"❌ RunPod error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}")
        raise typer.Exit(1) from e
