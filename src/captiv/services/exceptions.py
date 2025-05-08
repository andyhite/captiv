"""
Custom exceptions for the captiv service layer.

This module defines a hierarchy of exceptions used throughout the application
to provide clear error messages and enable proper error handling.
"""


class CaptivServiceError(Exception):
    """Base exception for all service-related errors."""


class FileOperationError(CaptivServiceError):
    """Exception raised for file operation errors."""


class DirectoryNotFoundError(FileOperationError):
    """Exception raised when a directory is not found."""


class FileNotFoundError(FileOperationError):
    """Exception raised when a file is not found."""


class UnsupportedFileTypeError(FileOperationError):
    """Exception raised when a file type is not supported."""


class ModelError(CaptivServiceError):
    """Base exception for model-related errors."""


class ModelConfigurationError(ModelError):
    """Exception raised for model configuration errors."""


class InvalidModelTypeError(ModelError):
    """Exception raised when an invalid model type is specified."""


class InvalidModelVariantError(ModelError):
    """Exception raised when an invalid model variant is specified."""


class InvalidModelModeError(ModelError):
    """Exception raised when an invalid model mode is specified."""


class CaptionError(CaptivServiceError):
    """Exception raised for caption-related errors."""
