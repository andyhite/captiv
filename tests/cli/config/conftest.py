"""Shared fixtures for config CLI command tests."""

import pytest
import typer
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def config_app():
    """Create a test app for config commands."""
    from captiv.cli.commands.config import clear, get, unset
    from captiv.cli.commands.config import list as config_list
    from captiv.cli.commands.config import set as config_set

    app = typer.Typer()
    app.command("get")(get.command)
    app.command("set")(config_set.command)
    app.command("list")(config_list.command)
    app.command("clear")(clear.command)
    app.command("unset")(unset.command)
    return app
