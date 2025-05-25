"""Tests for the captiv.gui.logging module."""

import logging
from unittest.mock import Mock, patch


def test_logging_module_imports():
    """Test that the logging module can be imported."""


def test_logging_module_exports():
    """Test that the logging module exports the expected items."""
    import captiv.gui.logging

    assert hasattr(captiv.gui.logging, "logger")
    assert hasattr(captiv.gui.logging, "setup_logging")
    assert hasattr(captiv.gui.logging, "InterceptHandler")

    assert hasattr(captiv.gui.logging, "__all__")
    assert "logger" in captiv.gui.logging.__all__
    assert "setup_logging" in captiv.gui.logging.__all__


def test_intercept_handler_class_exists():
    """Test that InterceptHandler class exists and can be instantiated."""
    from captiv.gui.logging import InterceptHandler

    assert isinstance(InterceptHandler, type)

    assert issubclass(InterceptHandler, logging.Handler)

    handler = InterceptHandler()
    assert isinstance(handler, InterceptHandler)
    assert isinstance(handler, logging.Handler)


def test_intercept_handler_emit_method():
    """Test that InterceptHandler has an emit method."""
    from captiv.gui.logging import InterceptHandler

    handler = InterceptHandler()
    assert hasattr(handler, "emit")
    assert callable(handler.emit)


@patch("captiv.gui.logging.logger")
def test_intercept_handler_emit_functionality(mock_logger):
    """Test that InterceptHandler.emit works correctly."""
    from captiv.gui.logging import InterceptHandler

    record = Mock()
    record.levelname = "INFO"
    record.levelno = 20
    record.exc_info = None
    record.getMessage.return_value = "Test message"

    mock_level = Mock()
    mock_level.name = "INFO"
    mock_logger.level.return_value = mock_level

    mock_opt = Mock()
    mock_logger.opt.return_value = mock_opt

    handler = InterceptHandler()
    handler.emit(record)

    mock_logger.level.assert_called_once_with("INFO")
    mock_logger.opt.assert_called_once_with(exception=None)
    mock_opt.log.assert_called_once_with("INFO", "Test message")


@patch("captiv.gui.logging.logger")
def test_intercept_handler_emit_with_exception(mock_logger):
    """Test that InterceptHandler.emit handles exceptions correctly."""
    from captiv.gui.logging import InterceptHandler

    record = Mock()
    record.levelname = "ERROR"
    record.levelno = 40
    record.exc_info = ("exc_type", "exc_value", "exc_traceback")
    record.getMessage.return_value = "Error message"

    mock_level = Mock()
    mock_level.name = "ERROR"
    mock_logger.level.return_value = mock_level

    mock_opt = Mock()
    mock_logger.opt.return_value = mock_opt

    handler = InterceptHandler()
    handler.emit(record)

    mock_logger.level.assert_called_once_with("ERROR")
    mock_logger.opt.assert_called_once_with(
        exception=("exc_type", "exc_value", "exc_traceback")
    )
    mock_opt.log.assert_called_once_with("ERROR", "Error message")


@patch("captiv.gui.logging.logger")
def test_intercept_handler_emit_with_invalid_level(mock_logger):
    """Test that InterceptHandler.emit handles invalid log levels."""
    from captiv.gui.logging import InterceptHandler

    record = Mock()
    record.levelname = "INVALID"
    record.levelno = 99
    record.exc_info = None
    record.getMessage.return_value = "Test message"

    mock_logger.level.side_effect = ValueError("Invalid level")

    mock_opt = Mock()
    mock_logger.opt.return_value = mock_opt

    handler = InterceptHandler()
    handler.emit(record)

    mock_logger.level.assert_called_once_with("INVALID")
    mock_logger.opt.assert_called_once_with(exception=None)
    mock_opt.log.assert_called_once_with(99, "Test message")


def test_setup_logging_function_exists():
    """Test that setup_logging function exists and can be called."""
    from captiv.gui.logging import setup_logging

    assert callable(setup_logging)


@patch("captiv.gui.logging.logger")
@patch("captiv.gui.logging.logging")
def test_setup_logging_functionality(mock_logging, mock_logger):
    """Test that setup_logging configures logging correctly."""
    from captiv.gui.logging import setup_logging

    setup_logging()

    mock_logger.configure.assert_called_once()

    mock_logging.basicConfig.assert_called_once()


@patch("captiv.gui.logging.logger")
@patch("captiv.gui.logging.logging")
def test_setup_logging_with_custom_level(mock_logging, mock_logger):
    """Test that setup_logging works with custom log level."""
    from captiv.gui.logging import setup_logging

    setup_logging(level="DEBUG")

    mock_logger.configure.assert_called_once()

    mock_logging.basicConfig.assert_called_once()


@patch("captiv.gui.logging.logger")
@patch("captiv.gui.logging.logging")
def test_setup_logging_with_intercept_libraries(mock_logging, mock_logger):
    """Test that setup_logging works with library interception."""
    from captiv.gui.logging import setup_logging

    mock_lib_logger = Mock()
    mock_logging.getLogger.return_value = mock_lib_logger

    setup_logging(intercept_libraries=["gradio", "transformers"])

    mock_logger.configure.assert_called_once()

    mock_logging.basicConfig.assert_called_once()

    assert mock_logging.getLogger.call_count == 2
    mock_logging.getLogger.assert_any_call("gradio")
    mock_logging.getLogger.assert_any_call("transformers")


def test_logger_import():
    """Test that logger can be imported from the module."""
    from captiv.gui.logging import logger

    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")
