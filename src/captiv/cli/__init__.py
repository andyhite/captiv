"""
Command-line interface for the Captiv image captioning library.

This package provides a CLI for generating image captions using the BLIP and BLIP-2 models.
"""

# Import the main function from cli.py
import importlib.util
import os

# Get the absolute path to the cli.py file
cli_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cli.py"))

# Load the cli.py module
spec = importlib.util.spec_from_file_location("cli_module", cli_path)
if spec is not None:
    cli_module = importlib.util.module_from_spec(spec)
    if spec.loader is not None:
        spec.loader.exec_module(cli_module)

        # Expose the main function
        main = cli_module.main
        run_app = cli_module.run_app
