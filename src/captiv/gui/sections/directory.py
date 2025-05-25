"""Directory navigation section for the Captiv GUI."""

import os
from pathlib import Path
from typing import Any

import gradio as gr
from loguru import logger

from captiv.gui.utils import get_subdirectories, normalize_path


class DirectorySection:
    """Directory navigation section for browsing and selecting directories."""

    def __init__(self, default_directory: str = ""):
        """
        Initialize the directory section.

        Args:
            default_directory: Optional default directory to start in
        """
        self.current_directory = default_directory or str(Path.home())
        logger.info(
            f"DirectorySection initialized with default directory: {self.current_directory}"  # noqa: E501
        )

        self.dir_dropdown = None

    def create_section(self) -> gr.Dropdown:
        """
        Create the directory navigation section UI components.

        Returns:
            The directory dropdown component
        """
        logger.debug(
            f"Creating directory section UI components. Initial directory: {self.current_directory}"  # noqa: E501
        )
        initial_options = self._get_dir_options(self.current_directory)
        logger.debug(f"Initial dropdown options: {initial_options}")

        self.dir_dropdown = gr.Dropdown(
            label="Select Image Directory",
            interactive=True,
            choices=initial_options,
            value=self.current_directory,
            container=False,
            scale=0,
            elem_classes="directory-selector",
        )

        return self.dir_dropdown

    def _get_dir_options(self, directory: str) -> list[str]:
        """
        Get directory options for the dropdown.

        Args:
            directory: The directory to get options for

        Returns:
            List of directory options
        """
        logger.debug(f"Getting directory options for: {directory}")
        directory = os.path.abspath(directory)
        parent = os.path.dirname(directory)
        subdirs = get_subdirectories(directory)
        options = []
        if parent and parent != directory:
            options.append(parent)
        options.append(directory)
        options.extend([os.path.join(directory, d) for d in subdirs])
        logger.debug(f"Directory options for '{directory}': {options}")
        return options

    def on_directory_change(self, directory: str) -> tuple[str, list[str]]:
        """
        Handle directory change event.

        Args:
            directory: The new directory path

        Returns:
            Tuple of (directory, subdirectories)
        """
        logger.debug(f"Directory change event triggered for: {directory}")
        try:
            if not os.path.isdir(directory):
                logger.error(f"Invalid directory selected: {directory}")
                raise ValueError(f"Invalid directory: {directory}")

            logger.info(f"Changing current directory to: {directory}")
            self.current_directory = directory

            subdirectories = get_subdirectories(directory)
            logger.debug(f"Subdirectories of '{directory}': {subdirectories}")

            return directory, subdirectories
        except Exception as e:
            logger.error(f"Error changing directory to '{directory}': {e}")
            return self.current_directory, get_subdirectories(self.current_directory)

    def on_parent_directory(self) -> tuple[str, list[str]]:
        """
        Handle parent directory button click.

        Returns:
            Tuple of (directory, subdirectories)
        """
        logger.debug(
            f"Parent directory event triggered. Current directory: {self.current_directory}"  # noqa: E501
        )
        parent_dir = str(Path(self.current_directory).parent)
        return self.on_directory_change(parent_dir)

    def on_subdirectory_select(self, subdirectory: str) -> tuple[str, list[str]]:
        """
        Handle subdirectory selection event.

        Args:
            subdirectory: The selected subdirectory name

        Returns:
            Tuple of (directory, subdirectories)
        """
        logger.debug(f"Subdirectory select event triggered for: {subdirectory}")
        if not subdirectory:
            logger.warning(
                "No subdirectory selected, returning current directory info."
            )
            return (
                self.current_directory,
                get_subdirectories(self.current_directory),
            )

        new_dir = os.path.join(self.current_directory, subdirectory)
        logger.debug(f"Constructed new directory path from subdirectory: {new_dir}")
        return self.on_directory_change(new_dir)

    def handle_dir_change(self, selected_dir) -> tuple[dict[str, Any], str]:
        """
        Handle directory dropdown change event.

        Args:
            selected_dir: The selected directory (string or dict)

        Returns:
            Tuple of (dropdown update, directory path)
        """
        logger.debug(f"Handling directory dropdown change. Selected: {selected_dir}")
        dir_path = normalize_path(selected_dir)
        logger.info(f"Normalized selected directory to: {dir_path}")
        self.current_directory = dir_path

        options = self._get_dir_options(dir_path)

        if (
            self.dir_dropdown is not None
            and hasattr(self.dir_dropdown, "value")
            and dir_path == self.dir_dropdown.value
        ):
            logger.debug(
                f"Dropdown value '{dir_path}' is same as current, only updating choices."  # noqa: E501
            )
            return gr.update(choices=options), dir_path

        logger.debug(f"Updating dropdown with new choices and value: {dir_path}")
        return gr.update(choices=options, value=dir_path), dir_path

    def get_current_directory(self) -> str:
        """
        Get the current directory.

        Returns:
            The current directory path
        """
        logger.debug(f"Getting current directory: {self.current_directory}")
        return self.current_directory
