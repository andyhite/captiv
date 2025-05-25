"""Caption management section for the Captiv GUI."""

import os
import traceback
from pathlib import Path

import gradio as gr
from loguru import logger

from captiv.services import CaptionFileManager
from captiv.utils.error_handling import EnhancedError


class CaptionSection:
    """Caption management section for viewing and editing captions."""

    def __init__(self, caption_manager: CaptionFileManager):
        """
        Initialize the caption section.

        Args:
            caption_manager: The caption file manager instance
        """
        self.caption_manager = caption_manager
        logger.info("CaptionSection initialized.")

        self.caption_textbox = None
        self.save_caption_btn = None
        self.generate_caption_btn = None

    def create_section(
        self,
    ) -> tuple[gr.Textbox, gr.Button, gr.Button]:
        """
        Create the caption management section UI components.

        Returns:
            Tuple containing the caption textbox, status textbox, save button,
            and generate button
        """
        logger.debug("Creating caption section UI components.")
        self.caption_textbox = gr.Textbox(
            label="Caption",
            placeholder="Select an image to view or edit its caption",
            lines=4,
            interactive=True,
        )

        with gr.Row():
            self.save_caption_btn = gr.Button("Save caption", scale=1)
            self.generate_caption_btn = gr.Button("Generate caption", scale=1)

        logger.debug("Caption section UI components created.")
        return (
            self.caption_textbox,
            self.save_caption_btn,
            self.generate_caption_btn,
        )

    def on_image_select(self, image_path: str) -> str:
        """
        Handle image selection event.

        Args:
            image_path: The selected image path

        Returns:
            The caption for the selected image
        """
        logger.debug(
            f"Image selection changed in CaptionSection. Selected image path: {image_path}"  # noqa: E501
        )
        if not image_path:
            logger.warning(
                "No image path provided to on_image_select in CaptionSection."
            )
            return ""

        if os.path.isdir(image_path):
            logger.warning(f"Selected path is a directory, not an image: {image_path}")
            return ""
        try:
            logger.debug(f"Reading caption for image: {image_path}")
            caption_text = self.caption_manager.read_caption(Path(image_path))
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
        """
        Handle save caption button click.

        Args:
            image_path: The image path to save the caption for
            caption: The caption to save

        Returns:
            Status message (for backward compatibility)
        """
        logger.debug(
            f"Save caption triggered in CaptionSection. Image path: {image_path}, Caption: '{caption}'"  # noqa: E501
        )
        if not image_path:
            logger.warning(
                "Save caption called with no image selected in CaptionSection."
            )
            gr.Warning("No image selected")
            return "No image selected"

        try:
            logger.info(f"Saving caption for image: {image_path}")
            self.caption_manager.write_caption(Path(image_path), caption)
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
