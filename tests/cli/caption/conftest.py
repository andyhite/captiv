"""Shared fixtures for caption CLI command tests."""

import tempfile
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test image files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        test_image1 = temp_path / "test1.jpg"
        test_image2 = temp_path / "test2.png"
        test_image1.touch()
        test_image2.touch()

        caption1 = temp_path / "test1.txt"
        caption2 = temp_path / "test2.txt"
        caption1.write_text("Test caption 1")
        caption2.write_text("Test caption 2")

        yield temp_path


@pytest.fixture
def temp_single_image():
    """Create a temporary single image file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_image = temp_path / "single.jpg"
        test_image.touch()
        yield test_image


@pytest.fixture
def caption_app():
    """Create a test app for caption commands."""
    from captiv.cli.commands.caption import clear, generate, get, unset
    from captiv.cli.commands.caption import list as list_cmd
    from captiv.cli.commands.caption import set as caption_set

    app = typer.Typer()
    app.command("generate")(generate.command)
    app.command("list")(list_cmd.command)
    app.command("clear")(clear.command)
    app.command("get")(get.command)
    app.command("set")(caption_set.command)
    app.command("unset")(unset.command)
    return app
