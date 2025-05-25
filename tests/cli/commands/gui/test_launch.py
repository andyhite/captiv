"""Tests for GUI launch CLI command."""

from unittest.mock import patch


class TestGuiLaunchCommand:
    """Test the GUI launch command."""

    @patch("captiv.gui.main.main")
    def test_launch_gui_options(self, mock_main, runner, gui_app):
        """Test launching GUI with various option combinations."""
        result = runner.invoke(gui_app, [])
        assert result.exit_code == 0
        mock_main.assert_called_with(share=False, config_path=None, use_runpod=False)

        result = runner.invoke(gui_app, ["--share"])
        assert result.exit_code == 0
        mock_main.assert_called_with(share=True, config_path=None, use_runpod=False)

        result = runner.invoke(gui_app, ["--config-file", "/path/to/config.yaml"])
        assert result.exit_code == 0
        mock_main.assert_called_with(
            share=False, config_path="/path/to/config.yaml", use_runpod=False
        )

        result = runner.invoke(
            gui_app, ["--share", "--config-file", "/custom/config.yaml"]
        )
        assert result.exit_code == 0
        mock_main.assert_called_with(
            share=True, config_path="/custom/config.yaml", use_runpod=False
        )

        result = runner.invoke(gui_app, ["--config-file", ""])
        assert result.exit_code == 0
        mock_main.assert_called_with(share=False, config_path="", use_runpod=False)

        # Test RunPod flag
        result = runner.invoke(gui_app, ["--runpod"])
        assert result.exit_code == 0
        mock_main.assert_called_with(share=False, config_path=None, use_runpod=True)

        result = runner.invoke(gui_app, ["--share", "--runpod"])
        assert result.exit_code == 0
        mock_main.assert_called_with(share=True, config_path=None, use_runpod=True)

    @patch("captiv.gui.main.main")
    def test_config_file_path_variations(self, mock_main, runner, gui_app):
        """Test different config file path formats."""
        test_paths = [
            "/absolute/path/config.yaml",
            "relative/path/config.yaml",
            "./config.yaml",
            "../config.yaml",
            "config.yml",
            "config.json",
        ]

        for path in test_paths:
            mock_main.reset_mock()
            result = runner.invoke(gui_app, ["--config-file", path])
            assert result.exit_code == 0
            mock_main.assert_called_once_with(
                share=False, config_path=path, use_runpod=False
            )

    @patch("captiv.cli.commands.gui.launch.typer.echo")
    @patch("captiv.gui.main.main")
    def test_gradio_import_error_handling(self, mock_main, mock_echo, runner, gui_app):
        """Test handling of Gradio import errors with proper error messages."""
        gradio_errors = [
            "No module named 'gradio'",
            "cannot import name 'gradio'",
            "gradio module not found",
            "ImportError: gradio",
        ]

        for error_msg in gradio_errors:
            mock_main.side_effect = ImportError(error_msg)
            mock_echo.reset_mock()

            result = runner.invoke(gui_app, [])
            assert result.exit_code == 0
            mock_echo.assert_called_once()

            error_message = mock_echo.call_args[0][0]
            assert "Gradio 4.44.1 is not installed" in error_message
            assert "pip install gradio==4.44.1" in error_message
            assert "poetry add gradio==4.44.1" in error_message

    @patch("captiv.gui.main.main")
    def test_error_handling(self, mock_main, runner, gui_app):
        """Test handling of various errors during GUI launch."""
        mock_main.side_effect = ImportError("Some other import error")
        result = runner.invoke(gui_app, [])
        assert result.exit_code == 1

        mock_main.side_effect = RuntimeError("GUI failed to start")
        result = runner.invoke(gui_app, [])
        assert result.exit_code == 1

        mock_main.side_effect = KeyboardInterrupt()
        result = runner.invoke(gui_app, [])
        assert result.exit_code == 1
