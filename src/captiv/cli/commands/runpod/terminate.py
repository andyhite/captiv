"""
Terminate command for RunPod management.

This module provides the command logic for terminating RunPod instances.
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
    typer.Argument(help="Pod ID to terminate"),
]

ConfirmOption = Annotated[
    bool,
    typer.Option(
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
]


@handle_cli_errors
def command(
    pod_id: PodIdArgument,
    yes: ConfirmOption = False,
    config_file: ConfigFileOption = None,
) -> None:
    """Terminate a RunPod instance (permanent deletion)."""
    config_manager = ConfigManager(config_file)

    # Get API key from config
    api_key = config_manager.get_config_value("runpod", "api_key")
    if not api_key:
        typer.echo("Error: RunPod API key not configured.")
        typer.echo("Set it with: captiv config set runpod api_key YOUR_API_KEY")
        raise typer.Exit(1)

    # Confirmation prompt
    if not yes:
        confirm = typer.confirm(
            f"⚠️  Are you sure you want to terminate pod {pod_id}? "
            "This action cannot be undone."
        )
        if not confirm:
            typer.echo("Operation cancelled")
            return

    try:
        runpod_service = RunPodService(api_key)

        typer.echo(f"🗑️  Terminating pod: {pod_id}")

        success = runpod_service.terminate_pod(pod_id)

        if success:
            typer.echo("✅ Pod terminated successfully")
            typer.echo("💡 Use 'captiv runpod create' to start a new instance")
        else:
            typer.echo("❌ Failed to terminate pod")
            raise typer.Exit(1)

    except RunPodError as e:
        typer.echo(f"❌ RunPod error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}")
        raise typer.Exit(1) from e
