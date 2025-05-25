"""Gallery section for the Captiv GUI."""

import os
import traceback
from pathlib import Path

import gradio as gr
from loguru import logger

from captiv.gui.utils import is_image_file
from captiv.services import CaptionFileManager


class GallerySection:
    """Gallery section for displaying and selecting images."""

    def __init__(self, caption_manager: CaptionFileManager):
        """
        Initialize the gallery section.

        Args:
            caption_manager: The caption file manager instance
        """
        self.caption_manager = caption_manager
        self.current_directory = str(os.path.expanduser("~"))
        logger.info(
            f"GallerySection initialized with default directory: {self.current_directory}"  # noqa: E501
        )
        self.current_image = None

        self.gallery = None
        self.selected_image = None

    def create_section(self) -> tuple[gr.Gallery, gr.State]:
        """
        Create the gallery section UI components.

        Returns:
            Tuple containing the gallery and selected image state
        """
        logger.debug("Creating gallery section UI components.")
        self.gallery = gr.Gallery(
            label="Images",
            show_label=True,
            columns=6,
            object_fit="cover",
            elem_classes="gallery",
            scale=1,
        )

        self.selected_image = gr.State(value=None)

        return self.gallery, self.selected_image

    def on_gallery_select(self, evt: gr.SelectData) -> str:
        logger.debug(f"Gallery selection event triggered: {evt}")
        """
        Handle gallery selection event.

        Args:
            evt: The selection event data

        Returns:
            The selected image path or empty string if invalid
        """
        if not evt:
            logger.warning("Gallery selection event is None.")
            return ""

        try:
            logger.debug(f"Processing gallery select event: {evt}")
            image_path = ""
            index = -1

            if hasattr(evt, "index") and isinstance(evt.index, int):
                index = evt.index
            elif hasattr(evt, "value") and isinstance(evt.value, int):
                index = evt.value

            if hasattr(evt, "value"):
                if isinstance(evt.value, str):
                    image_path = evt.value
                elif (
                    isinstance(evt.value, dict)
                    and "image" in evt.value
                    and "path" in evt.value["image"]
                ):
                    image_path = evt.value["image"]["path"]

            if not image_path and index >= 0:
                images_with_captions = self.caption_manager.list_images_and_captions(
                    Path(self.current_directory)
                )
                if images_with_captions and 0 <= index < len(images_with_captions):
                    image_name, _ = images_with_captions[index]
                    image_path = os.path.join(self.current_directory, image_name)

            if (
                not image_path
                or not os.path.exists(image_path)
                or os.path.isdir(image_path)
            ):
                logger.warning(
                    f"Invalid image path determined from gallery selection: '{image_path}'"  # noqa: E501
                )
                return ""

            if not is_image_file(image_path):
                logger.warning(f"Selected file is not a valid image: '{image_path}'")
                return ""

            self.current_image = os.path.abspath(image_path)
            logger.info(f"Image selected from gallery: {self.current_image}")
            return self.current_image

        except Exception as e:
            logger.error(f"Error in gallery selection: {e}")
            logger.debug(traceback.format_exc())
            return ""

    def get_gallery_images(self, directory: str) -> list[str]:
        """
        Get images for the gallery.

        Args:
            directory: The directory to get images from

        Returns:
            List of image paths
        """
        logger.debug(f"Attempting to get gallery images for directory: {directory}")
        try:
            images_with_captions = self.caption_manager.list_images_and_captions(
                Path(directory)
            )
            image_paths = [
                os.path.join(directory, image_name)
                for image_name, _ in images_with_captions
                if os.path.isfile(os.path.join(directory, image_name))
            ]
            logger.info(f"Found {len(image_paths)} images in directory '{directory}'.")
            logger.debug(f"Image paths: {image_paths}")
            return image_paths
        except Exception as e:
            logger.error(
                f"Error getting gallery images for directory '{directory}': {e}"
            )
            logger.debug(traceback.format_exc())
            return []

    def set_current_directory(self, directory: str) -> None:
        """
        Set the current directory.

        Args:
            directory: The new current directory
        """
        logger.info(f"Setting gallery current directory to: {directory}")
        self.current_directory = directory
