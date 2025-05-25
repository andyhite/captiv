"""
Status command for RunPod management.

This module provides the command logic for checking RunPod instance status.
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
    typer.Argument(help="Pod ID to check status for"),
]


@handle_cli_errors
def command(
    pod_id: PodIdArgument,
    config_file: ConfigFileOption = None,
) -> None:
    """Get status of a RunPod instance."""
    config_manager = ConfigManager(config_file)

    # Get API key from config
    api_key = config_manager.get_config_value("runpod", "api_key")
    if not api_key:
        typer.echo("Error: RunPod API key not configured.")
        typer.echo("Set it with: captiv config set runpod api_key YOUR_API_KEY")
        raise typer.Exit(1)

    try:
        runpod_service = RunPodService(api_key)

        typer.echo(f"📊 Checking status for pod: {pod_id}")

        pod_status = runpod_service.get_pod_status(pod_id)

        if not pod_status:
            typer.echo(f"❌ Pod {pod_id} not found")
            raise typer.Exit(1)

        typer.echo(f"Status: {pod_status.get('desiredStatus', 'Unknown')}")

        runtime = pod_status.get("runtime")
        if runtime:
            uptime = runtime.get("uptimeInSeconds", 0)
            typer.echo(f"Uptime: {uptime} seconds")

            ports = runtime.get("ports", [])
            for port in ports:
                if port.get("privatePort") == 7860:
                    if port.get("isIpPublic"):
                        endpoint = f"https://{pod_id}-7860.proxy.runpod.net"
                    else:
                        endpoint = f"http://{port.get('ip')}:{port.get('publicPort')}"
                    typer.echo(f"Endpoint: {endpoint}")
                    break

            gpus = runtime.get("gpus", [])
            for gpu in gpus:
                gpu_util = gpu.get("gpuUtilPercent", 0)
                mem_util = gpu.get("memoryUtilPercent", 0)
                typer.echo(f"GPU Utilization: {gpu_util}%")
                typer.echo(f"Memory Utilization: {mem_util}%")

        # Test health if pod is running
        if pod_status.get("desiredStatus") == "RUNNING":
            runpod_service.active_pod_id = pod_id
            if runtime and runtime.get("ports"):
                # Set endpoint for health check
                for port in runtime.get("ports", []):
                    if port.get("privatePort") == 7860:
                        if port.get("isIpPublic"):
                            runpod_service.pod_endpoint = (
                                f"https://{pod_id}-7860.proxy.runpod.net"
                            )
                        else:
                            runpod_service.pod_endpoint = (
                                f"http://{port.get('ip')}:{port.get('publicPort')}"
                            )
                        break

                if runpod_service.health_check():
                    typer.echo("✅ Health check: PASSED")
                else:
                    typer.echo("❌ Health check: FAILED")
            else:
                typer.echo("⚠️  No ports available for health check")

    except RunPodError as e:
        typer.echo(f"❌ RunPod error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}")
        raise typer.Exit(1) from e
