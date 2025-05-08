"""
Caption management section for the Captiv GUI.
"""

import os
import traceback
from typing import Tuple

import gradio as gr
from loguru import logger

from captiv.services.image_file_manager import ImageFileManager
from captiv.utils.error_handling import EnhancedError


class CaptionSection:
    """Caption management section for viewing and editing captions."""

    def __init__(self, file_manager: ImageFileManager):
        """Initialize the caption section.

        Args:
            file_manager: The image file manager instance
        """
        self.file_manager = file_manager
        logger.info("CaptionSection initialized.")

        # UI components
        self.caption_textbox = None
        self.save_status = None
        self.save_caption_btn = None
        self.generate_caption_btn = None
        self.caption_progress = None

    def create_section(
        self,
    ) -> Tuple[gr.Textbox, gr.Textbox, gr.Button, gr.Button, gr.Textbox]:
        """Create the caption management section UI components.

        Returns:
            Tuple containing the caption textbox, status textbox, save button, generate button, and progress indicator
        """
        logger.debug("Creating caption section UI components.")
        self.caption_textbox = gr.Textbox(
            label="Caption",
            placeholder="Select an image to view or edit its caption",
            lines=4,
            interactive=True,
        )

        # Keep a hidden status textbox for backward compatibility
        self.save_status = gr.Textbox(
            label="Status",
            interactive=False,
            visible=False,
            elem_id="save_status",
        )

        with gr.Row():
            self.save_caption_btn = gr.Button("Save caption", scale=1)
            self.generate_caption_btn = gr.Button("Generate caption", scale=1)

        # Add a progress indicator for caption generation
        self.caption_progress = gr.Textbox(
            label="Progress",
            value="Ready to generate caption",
            interactive=False,
            visible=True,
        )

        logger.debug("Caption section UI components created.")
        return (
            self.caption_textbox,
            self.save_status,
            self.save_caption_btn,
            self.generate_caption_btn,
            self.caption_progress,
        )

    def on_image_select(self, image_path: str) -> str:
        """Handle image selection event.

        Args:
            image_path: The selected image path

        Returns:
            The caption for the selected image
        """
        logger.debug(
            f"Image selection changed in CaptionSection. Selected image path: {image_path}"
        )
        if not image_path:
            logger.warning(
                "No image path provided to on_image_select in CaptionSection."
            )
            return ""

        # Check if the path is a directory and not a file
        if os.path.isdir(image_path):
            logger.warning(f"Selected path is a directory, not an image: {image_path}")
            return ""
        try:
            # Get the caption for the selected image
            logger.debug(f"Reading caption for image: {image_path}")
            caption_text = self.file_manager.read_caption(image_path)
            caption_to_display = caption_text or ""
            logger.info(f"Caption for '{image_path}': '{caption_to_display}'")
            return caption_to_display
        except FileNotFoundError:
            logger.info(f"No caption file found for image: {image_path}")
            return ""
        except Exception as e:
            logger.error(
                f"Error reading caption for '{image_path}' in CaptionSection: {e}"
            )
            logger.debug(traceback.format_exc())
            return ""

    def on_save_caption(self, image_path: str, caption: str) -> str:
        """Handle save caption button click.

        Args:
            image_path: The image path to save the caption for
            caption: The caption to save

        Returns:
            Status message (for backward compatibility)
        """
        logger.debug(
            f"Save caption triggered in CaptionSection. Image path: {image_path}, Caption: '{caption}'"
        )
        if not image_path:
            logger.warning(
                "Save caption called with no image selected in CaptionSection."
            )
            gr.Warning("No image selected")
            return "No image selected"

        try:
            logger.info(f"Saving caption for image: {image_path}")
            self.file_manager.write_caption(image_path, caption)
            success_msg = f"Caption saved for {os.path.basename(image_path)}"
            logger.info(success_msg)
            gr.Info(success_msg)
            return success_msg
        except FileNotFoundError:
            error_msg = f"Error saving caption: Image file not found at '{image_path}'."
            logger.error(error_msg)
            gr.Error("Error: Image file not found.")
            return "Error: Image file not found."
        except EnhancedError as e:
            # For enhanced errors, show the detailed message
            error_msg = f"Error saving caption: {e.message}"
            if e.troubleshooting_tips:
                error_details = "\n".join(f"- {tip}" for tip in e.troubleshooting_tips)
                error_msg = f"{error_msg}\n\nTroubleshooting tips:\n{error_details}"

            logger.error(error_msg)
            gr.Error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = (
                f"Error saving caption for '{image_path}' in CaptionSection: {e}"
            )
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            gr.Error(f"Error saving caption: {e}")
            return f"Error saving caption: {e}"
