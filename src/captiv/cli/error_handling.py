"""
Enhanced error handling utilities for the Captiv CLI.

This module provides decorators and context managers for standardized error handling in
CLI commands, with improved error messages and troubleshooting tips.
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import typer
from loguru import logger

from captiv.utils.error_handling import EnhancedError, create_enhanced_error

T = TypeVar("T")


def handle_cli_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for CLI commands that provides standardized error handling.

    This decorator wraps a CLI command function in a try-except block that catches
    exceptions, formats them appropriately for CLI output, and exits with a
    non-zero status code. It also provides troubleshooting tips for common errors.

    Args:
        func: The CLI command function to wrap.

    Returns:
        The wrapped function with error handling.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except EnhancedError as e:
            command_name = func.__name__
            if command_name == "command":
                command_name = func.__module__.split(".")[-1]

            typer.echo(f"Error in {command_name}: {e.message}", err=True)

            if e.context:
                typer.echo("\nContext:", err=True)
                for key, value in e.context.items():
                    typer.echo(f"  {key}: {value}", err=True)

            if e.troubleshooting_tips:
                typer.echo("\nTroubleshooting tips:", err=True)
                for i, tip in enumerate(e.troubleshooting_tips, 1):
                    typer.echo(f"  {i}. {tip}", err=True)

            logger.error(e.get_detailed_message(include_traceback=True))

            raise typer.Exit(1) from None

        except KeyboardInterrupt:
            typer.echo("\nOperation cancelled by user.", err=True)
            raise typer.Exit(1) from None

        except Exception as e:
            command_name = func.__name__
            if command_name == "command":
                command_name = func.__module__.split(".")[-1]

            enhanced_error = create_enhanced_error(e, context={"command": command_name})

            typer.echo(f"Error in {command_name}: {enhanced_error.message}", err=True)

            if enhanced_error.troubleshooting_tips:
                typer.echo("\nTroubleshooting tips:", err=True)
                for i, tip in enumerate(enhanced_error.troubleshooting_tips, 1):
                    typer.echo(f"  {i}. {tip}", err=True)

            logger.error(enhanced_error.get_detailed_message(include_traceback=True))

            raise typer.Exit(1) from None

    return wrapper


def format_cli_error(error: Exception, command_name: str) -> str:
    """
    Format an error message for CLI output.

    Args:
        error: The exception to format
        command_name: The name of the command that raised the exception

    Returns:
        A formatted error message
    """
    if isinstance(error, EnhancedError):
        return f"Error in {command_name}: {error.message}"
    else:
        return f"Error in {command_name}: {str(error)}"
