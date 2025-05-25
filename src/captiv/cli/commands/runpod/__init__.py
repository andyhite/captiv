"""RunPod management commands for the Captiv CLI."""

import typer

from .create import command as create_command
from .list import command as list_command
from .status import command as status_command
from .stop import command as stop_command
from .terminate import command as terminate_command

app = typer.Typer(help="Manage RunPod instances for remote inference")

app.command("create", help="Create a new RunPod instance")(create_command)
app.command("list", help="List RunPod instances")(list_command)
app.command("status", help="Get status of a RunPod instance")(status_command)
app.command("stop", help="Stop a RunPod instance")(stop_command)
app.command("terminate", help="Terminate a RunPod instance")(terminate_command)
