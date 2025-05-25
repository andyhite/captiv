"""Tests for CLI error handling module."""

from unittest.mock import call, patch

import pytest
from typer import Exit

from captiv.cli.error_handling import format_cli_error, handle_cli_errors
from captiv.utils.error_handling import EnhancedError


class TestHandleCliErrors:
    """Test the handle_cli_errors decorator."""

    def test_successful_function_execution(self):
        """Test that successful function execution passes through unchanged."""

        @handle_cli_errors
        def test_function(value):
            return value * 2

        result = test_function(5)
        assert result == 10

    @patch("captiv.cli.error_handling.typer.echo")
    @patch("captiv.cli.error_handling.logger")
    def test_enhanced_error_handling(self, mock_logger, mock_echo):
        """Test handling of EnhancedError exceptions."""
        enhanced_error = EnhancedError(
            message="Test error message",
            context={"file": "test.txt", "line": 42},
            troubleshooting_tips=["Check the file path", "Verify permissions"],
        )

        @handle_cli_errors
        def test_function():
            raise enhanced_error

        with pytest.raises(Exit) as exc_info:
            test_function()

        assert exc_info.value.exit_code == 1

        expected_calls = [
            call("Error in test_function: Test error message", err=True),
            call("\nContext:", err=True),
            call("  file: test.txt", err=True),
            call("  line: 42", err=True),
            call("\nTroubleshooting tips:", err=True),
            call("  1. Check the file path", err=True),
            call("  2. Verify permissions", err=True),
        ]
        mock_echo.assert_has_calls(expected_calls)

        mock_logger.error.assert_called_once()

    @patch("captiv.cli.error_handling.typer.echo")
    @patch("captiv.cli.error_handling.logger")
    def test_enhanced_error_without_context(self, mock_logger, mock_echo):
        """Test handling of EnhancedError without context."""
        enhanced_error = EnhancedError(
            message="Simple error", troubleshooting_tips=["Try again"]
        )

        @handle_cli_errors
        def test_function():
            raise enhanced_error

        with pytest.raises(Exit) as exc_info:
            test_function()

        assert exc_info.value.exit_code == 1

        context_calls = [
            call for call in mock_echo.call_args_list if "Context:" in str(call)
        ]
        assert len(context_calls) == 0

    @patch("captiv.cli.error_handling.typer.echo")
    @patch("captiv.cli.error_handling.logger")
    def test_enhanced_error_without_tips(self, mock_logger, mock_echo):
        """Test handling of EnhancedError without troubleshooting tips."""
        enhanced_error = EnhancedError(
            message="Error without tips", context={"key": "value"}
        )

        @handle_cli_errors
        def test_function():
            raise enhanced_error

        with pytest.raises(Exit) as exc_info:
            test_function()

        assert exc_info.value.exit_code == 1

        tip_calls = [
            call
            for call in mock_echo.call_args_list
            if "Troubleshooting tips:" in str(call)
        ]
        assert len(tip_calls) == 0

    @patch("captiv.cli.error_handling.create_enhanced_error")
    @patch("captiv.cli.error_handling.typer.echo")
    @patch("captiv.cli.error_handling.logger")
    def test_regular_exception_handling(
        self, mock_logger, mock_echo, mock_create_enhanced
    ):
        """Test handling of regular exceptions."""
        original_error = ValueError("Original error message")
        enhanced_error = EnhancedError(
            message="Enhanced error message", troubleshooting_tips=["Check your input"]
        )
        mock_create_enhanced.return_value = enhanced_error

        @handle_cli_errors
        def test_function():
            raise original_error

        with pytest.raises(Exit) as exc_info:
            test_function()

        assert exc_info.value.exit_code == 1

        mock_create_enhanced.assert_called_once_with(
            original_error, context={"command": "test_function"}
        )

        mock_echo.assert_any_call(
            "Error in test_function: Enhanced error message", err=True
        )

    @patch("captiv.cli.error_handling.create_enhanced_error")
    @patch("captiv.cli.error_handling.typer.echo")
    @patch("captiv.cli.error_handling.logger")
    def test_command_name_from_module(
        self, mock_logger, mock_echo, mock_create_enhanced
    ):
        """Test command name extraction from module when function name is 'command'."""
        original_error = ValueError("Test error")
        enhanced_error = EnhancedError(message="Enhanced error")
        mock_create_enhanced.return_value = enhanced_error

        def command():
            raise original_error

        command.__module__ = "captiv.cli.commands.config.set"
        decorated_command = handle_cli_errors(command)

        with pytest.raises(Exit):
            decorated_command()

        mock_create_enhanced.assert_called_once_with(
            original_error, context={"command": "set"}
        )

    @patch("captiv.cli.error_handling.create_enhanced_error")
    @patch("captiv.cli.error_handling.typer.echo")
    @patch("captiv.cli.error_handling.logger")
    def test_enhanced_error_command_name_from_module(
        self, mock_logger, mock_echo, mock_create_enhanced
    ):
        """Test command name extraction for EnhancedError when function name is
        'command'."""
        enhanced_error = EnhancedError(message="Test enhanced error")

        def command():
            raise enhanced_error

        command.__module__ = "captiv.cli.commands.model.list"
        decorated_command = handle_cli_errors(command)

        with pytest.raises(Exit):
            decorated_command()

        mock_echo.assert_any_call("Error in list: Test enhanced error", err=True)

    @patch("captiv.cli.error_handling.create_enhanced_error")
    @patch("captiv.cli.error_handling.typer.echo")
    @patch("captiv.cli.error_handling.logger")
    def test_regular_exception_without_tips(
        self, mock_logger, mock_echo, mock_create_enhanced
    ):
        """Test handling of regular exceptions without troubleshooting tips."""
        original_error = RuntimeError("Runtime error")
        enhanced_error = EnhancedError(message="Enhanced runtime error")
        mock_create_enhanced.return_value = enhanced_error

        @handle_cli_errors
        def test_function():
            raise original_error

        with pytest.raises(Exit):
            test_function()

        tip_calls = [
            call
            for call in mock_echo.call_args_list
            if "Troubleshooting tips:" in str(call)
        ]
        assert len(tip_calls) == 0


class TestFormatCliError:
    """Test the format_cli_error function."""

    def test_format_enhanced_error(self):
        """Test formatting of EnhancedError."""
        enhanced_error = EnhancedError(message="Enhanced error message")
        result = format_cli_error(enhanced_error, "test_command")
        assert result == "Error in test_command: Enhanced error message"

    def test_format_regular_error(self):
        """Test formatting of regular exceptions."""
        regular_error = ValueError("Regular error message")
        result = format_cli_error(regular_error, "test_command")
        assert result == "Error in test_command: Regular error message"

    def test_format_error_with_empty_message(self):
        """Test formatting of error with empty message."""
        empty_error = ValueError("")
        result = format_cli_error(empty_error, "test_command")
        assert result == "Error in test_command: "

    def test_format_error_with_none_message(self):
        """Test formatting of error with None message."""

        class CustomError(Exception):
            def __str__(self):
                return ""

        custom_error = CustomError()
        result = format_cli_error(custom_error, "test_command")
        assert result == "Error in test_command: "
