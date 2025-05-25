"""Shared fixtures for GUI CLI command tests."""

import pytest
import typer
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def gui_app():
    """Create a test app for GUI commands."""
    from captiv.cli.commands.gui import launch

    app = typer.Typer()
    app.command()(launch.command)
    return app
