"""
Enhanced error handling utilities for both CLI and GUI interfaces.
"""

import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from loguru import logger


class ErrorCategory(str, Enum):
    """Categories of errors for better error handling and reporting."""

    MODEL_LOADING = "model_loading"
    IMAGE_PROCESSING = "image_processing"
    CAPTION_GENERATION = "caption_generation"
    FILE_SYSTEM = "file_system"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    NETWORK = "network"
    UNKNOWN = "unknown"


class EnhancedError(Exception):
    """
    Enhanced error class with additional context and troubleshooting information.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        original_error: Optional[Exception] = None,
        troubleshooting_tips: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the enhanced error.

        Args:
            message: The error message
            category: The category of the error
            original_error: The original exception that caused this error
            troubleshooting_tips: List of troubleshooting tips
            context: Additional context information
        """
        self.message = message
        self.category = category
        self.original_error = original_error
        self.troubleshooting_tips = troubleshooting_tips or []
        self.context = context or {}

        # Build the full error message
        full_message = f"{message}"
        if original_error:
            full_message += f" (Original error: {str(original_error)})"

        super().__init__(full_message)

    def get_detailed_message(self, include_traceback: bool = False) -> str:
        """
        Get a detailed error message with context and troubleshooting tips.

        Args:
            include_traceback: Whether to include the traceback of the original error

        Returns:
            A detailed error message
        """
        lines = [f"Error: {self.message}"]

        if self.category != ErrorCategory.UNKNOWN:
            lines.append(f"Category: {self.category.value}")

        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  - {key}: {value}")

        if self.troubleshooting_tips:
            lines.append("Troubleshooting tips:")
            for i, tip in enumerate(self.troubleshooting_tips, 1):
                lines.append(f"  {i}. {tip}")

        if include_traceback and self.original_error:
            lines.append("Original error traceback:")
            tb = "".join(
                traceback.format_exception(
                    type(self.original_error),
                    self.original_error,
                    self.original_error.__traceback__,
                )
            )
            lines.append(tb)

        return "\n".join(lines)

    def log_error(self, level: str = "error") -> None:
        """
        Log the error with the appropriate level.

        Args:
            level: The logging level to use
        """
        log_func = getattr(logger, level)
        log_func(self.get_detailed_message(include_traceback=True))


# Dictionary mapping exception types to error categories and troubleshooting tips
ERROR_MAPPING: Dict[Type[Exception], Dict[str, Any]] = {
    # Model loading errors
    ImportError: {
        "category": ErrorCategory.MODEL_LOADING,
        "tips": [
            "Make sure all required dependencies are installed",
            "Check if the model requires additional packages",
            "Try reinstalling the package with 'pip install --force-reinstall'",
        ],
    },
    ModuleNotFoundError: {
        "category": ErrorCategory.MODEL_LOADING,
        "tips": [
            "Install the missing module with 'pip install <module_name>'",
            "Check if the module name is spelled correctly",
        ],
    },
    # File system errors
    FileNotFoundError: {
        "category": ErrorCategory.FILE_SYSTEM,
        "tips": [
            "Check if the file path is correct",
            "Verify that the file exists at the specified location",
            "Check file permissions",
        ],
    },
    PermissionError: {
        "category": ErrorCategory.FILE_SYSTEM,
        "tips": [
            "Check if you have the necessary permissions to access the file",
            "Try running the command with elevated privileges",
        ],
    },
    # Resource errors
    MemoryError: {
        "category": ErrorCategory.RESOURCE,
        "tips": [
            "Try using a smaller model or reducing batch size",
            "Close other applications to free up memory",
            "Consider using a machine with more RAM",
        ],
    },
    # Network errors
    ConnectionError: {
        "category": ErrorCategory.NETWORK,
        "tips": [
            "Check your internet connection",
            "Verify that the server is accessible",
            "Check if a firewall is blocking the connection",
        ],
    },
    TimeoutError: {
        "category": ErrorCategory.NETWORK,
        "tips": [
            "The operation timed out, try again later",
            "Check your internet connection speed",
            "Consider increasing the timeout value if possible",
        ],
    },
}


def categorize_error(error: Exception) -> Dict[str, Any]:
    """
    Categorize an error and provide troubleshooting tips.

    Args:
        error: The exception to categorize

    Returns:
        A dictionary with category and troubleshooting tips
    """
    # Check if the error type is directly in our mapping
    if type(error) in ERROR_MAPPING:
        return ERROR_MAPPING[type(error)]

    # Check if any parent class of the error is in our mapping
    for error_type, mapping in ERROR_MAPPING.items():
        if isinstance(error, error_type):
            return mapping

    # Default category and tips
    return {
        "category": ErrorCategory.UNKNOWN,
        "tips": [
            "Check the error message for specific details",
            "Review the application logs for more information",
            "If the issue persists, report it to the developers",
        ],
    }


def create_enhanced_error(
    error: Exception,
    message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> EnhancedError:
    """
    Create an enhanced error from a standard exception.

    Args:
        error: The original exception
        message: Optional custom message (if None, uses str(error))
        context: Additional context information

    Returns:
        An EnhancedError instance
    """
    error_info = categorize_error(error)

    return EnhancedError(
        message=message or str(error),
        category=error_info["category"],
        original_error=error,
        troubleshooting_tips=error_info["tips"],
        context=context,
    )


def handle_errors(func):
    """
    Decorator that catches exceptions and converts them to EnhancedErrors.

    This can be used by both CLI and GUI code to provide consistent error handling.

    Args:
        func: The function to wrap

    Returns:
        The wrapped function
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EnhancedError:
            # If it's already an EnhancedError, just re-raise it
            raise
        except Exception as e:
            # Get the function name for context
            func_name = getattr(func, "__name__", "unknown_function")

            # Create and log the enhanced error
            enhanced_error = create_enhanced_error(e, context={"function": func_name})
            enhanced_error.log_error()

            # Re-raise the enhanced error
            raise enhanced_error

    return wrapper
