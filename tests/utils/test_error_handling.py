"""Tests for error handling utilities."""

from unittest.mock import PropertyMock, patch

import pytest

from captiv.utils.error_handling import (
    ERROR_MAPPING,
    EnhancedError,
    ErrorCategory,
    categorize_error,
    create_enhanced_error,
    handle_errors,
)


class TestEnhancedError:
    """Test cases for EnhancedError class."""

    def test_enhanced_error_basic(self):
        """Test basic EnhancedError creation."""
        error = EnhancedError("Test error message")

        assert error.message == "Test error message"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.original_error is None
        assert error.troubleshooting_tips == []
        assert error.context == {}
        assert str(error) == "Test error message"

    def test_enhanced_error_with_all_parameters(self):
        """Test EnhancedError with all parameters."""
        original_error = ValueError("Original error")
        tips = ["Tip 1", "Tip 2"]
        context = {"key": "value", "number": 42}

        error = EnhancedError(
            message="Enhanced error",
            category=ErrorCategory.MODEL_LOADING,
            original_error=original_error,
            troubleshooting_tips=tips,
            context=context,
        )

        assert error.message == "Enhanced error"
        assert error.category == ErrorCategory.MODEL_LOADING
        assert error.original_error == original_error
        assert error.troubleshooting_tips == tips
        assert error.context == context
        assert "Original error: Original error" in str(error)

    def test_enhanced_error_with_none_tips_and_context(self):
        """Test EnhancedError with None tips and context."""
        error = EnhancedError(
            message="Test error",
            troubleshooting_tips=None,
            context=None,
        )

        assert error.troubleshooting_tips == []
        assert error.context == {}

    def test_get_detailed_message_basic(self):
        """Test get_detailed_message with basic error."""
        error = EnhancedError("Test error")
        detailed = error.get_detailed_message()

        assert "Error: Test error" in detailed
        assert "Category:" not in detailed

    def test_get_detailed_message_with_category(self):
        """Test get_detailed_message with category."""
        error = EnhancedError("Test error", category=ErrorCategory.FILE_SYSTEM)
        detailed = error.get_detailed_message()

        assert "Error: Test error" in detailed
        assert "Category: file_system" in detailed

    def test_get_detailed_message_with_context(self):
        """Test get_detailed_message with context."""
        context = {"file": "test.txt", "line": 42}
        error = EnhancedError("Test error", context=context)
        detailed = error.get_detailed_message()

        assert "Context:" in detailed
        assert "- file: test.txt" in detailed
        assert "- line: 42" in detailed

    def test_get_detailed_message_with_tips(self):
        """Test get_detailed_message with troubleshooting tips."""
        tips = ["Check the file path", "Verify permissions"]
        error = EnhancedError("Test error", troubleshooting_tips=tips)
        detailed = error.get_detailed_message()

        assert "Troubleshooting tips:" in detailed
        assert "1. Check the file path" in detailed
        assert "2. Verify permissions" in detailed

    def test_get_detailed_message_with_traceback(self):
        """Test get_detailed_message with traceback."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = EnhancedError("Enhanced error", original_error=e)
            detailed = error.get_detailed_message(include_traceback=True)

            assert "Original error traceback:" in detailed
            assert "ValueError: Original error" in detailed

    def test_get_detailed_message_no_traceback_when_no_original_error(self):
        """Test get_detailed_message doesn't include traceback when no original
        error."""
        error = EnhancedError("Test error")
        detailed = error.get_detailed_message(include_traceback=True)

        assert "Original error traceback:" not in detailed

    def test_get_detailed_message_complete(self):
        """Test get_detailed_message with all components."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = EnhancedError(
                message="Complete error",
                category=ErrorCategory.MODEL_LOADING,
                original_error=e,
                troubleshooting_tips=["Tip 1", "Tip 2"],
                context={"key": "value"},
            )
            detailed = error.get_detailed_message(include_traceback=True)

            assert "Error: Complete error" in detailed
            assert "Category: model_loading" in detailed
            assert "Context:" in detailed
            assert "- key: value" in detailed
            assert "Troubleshooting tips:" in detailed
            assert "1. Tip 1" in detailed
            assert "2. Tip 2" in detailed
            assert "Original error traceback:" in detailed

    def test_log_error_default_level(self):
        """Test log_error with default level."""
        error = EnhancedError("Test error")

        with patch("captiv.utils.error_handling.logger") as mock_logger:
            error.log_error()
            mock_logger.error.assert_called_once()

    def test_log_error_custom_level(self):
        """Test log_error with custom level."""
        error = EnhancedError("Test error")

        with patch("captiv.utils.error_handling.logger") as mock_logger:
            error.log_error("warning")
            mock_logger.warning.assert_called_once()

    def test_log_error_invalid_level(self):
        """Test log_error with invalid level."""
        error = EnhancedError("Test error")

        with patch("captiv.utils.error_handling.logger") as mock_logger:
            mock_logger.invalid_level = PropertyMock(
                side_effect=AttributeError(
                    "'Logger' object has no attribute 'invalid_level'"
                )
            )
            with pytest.raises(AttributeError):
                error.log_error("invalid_level")


class TestErrorCategory:
    """Test cases for ErrorCategory enum."""

    def test_error_category_values(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.CAPTION_GENERATION == "caption_generation"
        assert ErrorCategory.CONFIGURATION == "configuration"
        assert ErrorCategory.FILE_SYSTEM == "file_system"
        assert ErrorCategory.IMAGE_PROCESSING == "image_processing"
        assert ErrorCategory.MODEL_LOADING == "model_loading"
        assert ErrorCategory.NETWORK == "network"
        assert ErrorCategory.RESOURCE == "resource"
        assert ErrorCategory.UNKNOWN == "unknown"

    def test_error_category_iteration(self):
        """Test iterating over ErrorCategory enum."""
        categories = list(ErrorCategory)
        assert len(categories) == 8
        assert ErrorCategory.UNKNOWN in categories


class TestCategorizeError:
    """Test cases for categorize_error function."""

    def test_categorize_error_direct_mapping(self):
        """Test categorizing error with direct type mapping."""
        error = FileNotFoundError("File not found")
        result = categorize_error(error)

        assert result["category"] == ErrorCategory.FILE_SYSTEM
        assert "Check if the file path is correct" in result["tips"]

    def test_categorize_error_inheritance_mapping(self):
        """Test categorizing error with inheritance mapping."""

        class CustomConnectionError(ConnectionError):
            pass

        error = CustomConnectionError("Connection failed")
        result = categorize_error(error)

        assert result["category"] == ErrorCategory.NETWORK
        assert "Check your internet connection" in result["tips"]

    def test_categorize_error_unknown_type(self):
        """Test categorizing unknown error type."""

        class UnknownError(Exception):
            pass

        error = UnknownError("Unknown error")
        result = categorize_error(error)

        assert result["category"] == ErrorCategory.UNKNOWN
        assert "Check the error message for specific details" in result["tips"]

    def test_categorize_error_all_mapped_types(self):
        """Test categorizing all mapped error types."""
        test_cases = [
            (ImportError("Import failed"), ErrorCategory.MODEL_LOADING),
            (ModuleNotFoundError("Module not found"), ErrorCategory.MODEL_LOADING),
            (FileNotFoundError("File not found"), ErrorCategory.FILE_SYSTEM),
            (PermissionError("Permission denied"), ErrorCategory.FILE_SYSTEM),
            (MemoryError("Out of memory"), ErrorCategory.RESOURCE),
            (ConnectionError("Connection failed"), ErrorCategory.NETWORK),
            (TimeoutError("Timeout"), ErrorCategory.NETWORK),
        ]

        for error, expected_category in test_cases:
            result = categorize_error(error)
            assert result["category"] == expected_category
            assert isinstance(result["tips"], list)
            assert len(result["tips"]) > 0


class TestCreateEnhancedError:
    """Test cases for create_enhanced_error function."""

    def test_create_enhanced_error_basic(self):
        """Test creating enhanced error from basic exception."""
        original_error = ValueError("Original error")
        enhanced = create_enhanced_error(original_error)

        assert enhanced.message == "Original error"
        assert enhanced.original_error == original_error
        assert enhanced.category == ErrorCategory.UNKNOWN
        assert len(enhanced.troubleshooting_tips) > 0

    def test_create_enhanced_error_with_custom_message(self):
        """Test creating enhanced error with custom message."""
        original_error = ValueError("Original error")
        enhanced = create_enhanced_error(original_error, message="Custom message")

        assert enhanced.message == "Custom message"
        assert enhanced.original_error == original_error

    def test_create_enhanced_error_with_context(self):
        """Test creating enhanced error with context."""
        original_error = FileNotFoundError("File not found")
        context = {"file_path": "/path/to/file.txt"}
        enhanced = create_enhanced_error(original_error, context=context)

        assert enhanced.context == context
        assert enhanced.category == ErrorCategory.FILE_SYSTEM

    def test_create_enhanced_error_mapped_type(self):
        """Test creating enhanced error from mapped error type."""
        original_error = ImportError("Module not found")
        enhanced = create_enhanced_error(original_error)

        assert enhanced.category == ErrorCategory.MODEL_LOADING
        assert (
            "Make sure all required dependencies are installed"
            in enhanced.troubleshooting_tips
        )


class TestHandleErrorsDecorator:
    """Test cases for handle_errors decorator."""

    def test_handle_errors_success(self):
        """Test handle_errors decorator with successful function."""

        @handle_errors
        def successful_function(x, y):
            return x + y

        result = successful_function(2, 3)
        assert result == 5

    def test_handle_errors_with_exception(self):
        """Test handle_errors decorator with function that raises exception."""

        @handle_errors
        def failing_function():
            raise ValueError("Function failed")

        with patch("captiv.utils.error_handling.logger") as mock_logger:
            with pytest.raises(EnhancedError) as exc_info:
                failing_function()

            error = exc_info.value
            assert isinstance(error, EnhancedError)
            assert "Function failed" in error.message
            assert error.context["function"] == "failing_function"
            mock_logger.error.assert_called_once()

    def test_handle_errors_with_enhanced_error(self):
        """Test handle_errors decorator with function that raises EnhancedError."""
        original_enhanced_error = EnhancedError("Already enhanced")

        @handle_errors
        def function_with_enhanced_error():
            raise original_enhanced_error

        with pytest.raises(EnhancedError) as exc_info:
            function_with_enhanced_error()

        assert exc_info.value is original_enhanced_error

    def test_handle_errors_with_lambda(self):
        """Test handle_errors decorator with lambda function."""
        failing_lambda = handle_errors(lambda: 1 / 0)

        with patch("captiv.utils.error_handling.logger"):
            with pytest.raises(EnhancedError) as exc_info:
                failing_lambda()

            error = exc_info.value
            assert "division by zero" in str(error).lower()
            assert error.context["function"] == "<lambda>"

    def test_handle_errors_preserves_function_metadata(self):
        """Test that handle_errors preserves function metadata."""

        @handle_errors
        def documented_function(x: int) -> int:
            """This function has documentation."""
            return x * 2

        result = documented_function(5)
        assert result == 10

    def test_handle_errors_with_function_args_and_kwargs(self):
        """Test handle_errors decorator with function that has args and kwargs."""

        @handle_errors
        def function_with_args(*args, **kwargs):
            if "fail" in kwargs:
                raise ValueError("Intentional failure")
            return sum(args) + sum(kwargs.values())

        result = function_with_args(1, 2, 3, extra=4)
        assert result == 10

        with patch("captiv.utils.error_handling.logger"), pytest.raises(EnhancedError):
            function_with_args(1, 2, fail=True)


class TestErrorMapping:
    """Test cases for ERROR_MAPPING constant."""

    def test_error_mapping_structure(self):
        """Test that ERROR_MAPPING has correct structure."""
        assert isinstance(ERROR_MAPPING, dict)

        for error_type, mapping in ERROR_MAPPING.items():
            assert issubclass(error_type, Exception)
            assert "category" in mapping
            assert "tips" in mapping
            assert isinstance(mapping["category"], ErrorCategory)
            assert isinstance(mapping["tips"], list)
            assert len(mapping["tips"]) > 0

    def test_error_mapping_completeness(self):
        """Test that ERROR_MAPPING covers expected error types."""
        expected_types = [
            ImportError,
            ModuleNotFoundError,
            FileNotFoundError,
            PermissionError,
            MemoryError,
            ConnectionError,
            TimeoutError,
        ]

        for error_type in expected_types:
            assert error_type in ERROR_MAPPING

    def test_error_mapping_tips_quality(self):
        """Test that error mapping tips are meaningful."""
        for _error_type, mapping in ERROR_MAPPING.items():
            tips = mapping["tips"]

            for tip in tips:
                assert isinstance(tip, str)
                assert len(tip.strip()) > 0
                assert any(
                    word in tip.lower()
                    for word in [
                        "check",
                        "verify",
                        "try",
                        "install",
                        "make sure",
                        "consider",
                    ]
                )


class TestErrorHandlingIntegration:
    """Integration tests for error handling components."""

    def test_full_error_handling_workflow(self):
        """Test complete error handling workflow."""

        @handle_errors
        def complex_function(file_path: str):
            if not file_path:
                raise ValueError("File path cannot be empty")
            if file_path == "not_found.txt":
                raise FileNotFoundError(f"File {file_path} not found")
            if file_path == "permission_denied.txt":
                raise PermissionError(f"Permission denied for {file_path}")
            return f"Success: {file_path}"

        result = complex_function("valid_file.txt")
        assert result == "Success: valid_file.txt"

        with patch("captiv.utils.error_handling.logger"):
            with pytest.raises(EnhancedError) as exc_info:
                complex_function("not_found.txt")

            error = exc_info.value
            assert error.category == ErrorCategory.FILE_SYSTEM
            assert "Check if the file path is correct" in error.troubleshooting_tips
            assert error.context["function"] == "complex_function"

    def test_nested_error_handling(self):
        """Test error handling with nested function calls."""

        @handle_errors
        def inner_function():
            raise ImportError("Missing dependency")

        @handle_errors
        def outer_function():
            return inner_function()

        with patch("captiv.utils.error_handling.logger"):
            with pytest.raises(EnhancedError) as exc_info:
                outer_function()

            error = exc_info.value
            assert error.category == ErrorCategory.MODEL_LOADING
            assert error.context["function"] == "inner_function"
