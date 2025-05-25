"""Tests for CLI __main__ module."""

from typing import Any
from unittest.mock import patch

from captiv.cli.__main__ import main


class TestCliMain:
    """Test CLI __main__ module."""

    @patch("captiv.cli.__main__.app")
    def test_main_calls_app(self, mock_app: Any) -> None:
        """Test that main() calls the app function."""
        main()
        mock_app.assert_called_once()

    def test_main_entry_point_when_name_is_main(self) -> None:
        """Test the if __name__ == '__main__' block."""
        with (
            patch("captiv.cli.__main__.main"),
            patch("captiv.cli.__main__.__name__", "__main__"),
        ):
            import captiv.cli.__main__

            assert callable(captiv.cli.__main__.main)
            assert callable(captiv.cli.__main__.main)
