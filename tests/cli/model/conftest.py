"""Shared fixtures for model CLI command tests."""

import pytest
import typer
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def model_app():
    """Create a test app for model commands."""
    from captiv.cli.commands.model import list as list_cmd
    from captiv.cli.commands.model import show

    app = typer.Typer()
    app.command("list")(list_cmd.command)
    app.command("show")(show.command)
    return app
