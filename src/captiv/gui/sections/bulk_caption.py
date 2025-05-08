"""
Bulk captioning section for the Captiv GUI.
"""

import os
import time
import traceback
from typing import Tuple

import gradio as gr
from loguru import logger

from captiv.services.caption_manager import CaptionManager
from captiv.services.image_file_manager import ImageFileManager
from captiv.services.model_manager import ModelType
from captiv.utils.error_handling import EnhancedError, handle_errors
from captiv.utils.progress import ProgressTracker


class BulkCaptionSection:
    """Bulk captioning section for captioning all images in a directory."""

    def __init__(self, caption_manager: CaptionManager, file_manager: ImageFileManager):
        """Initialize the bulk caption section.

        Args:
            caption_manager: The caption manager instance
            file_manager: The image file manager instance
        """
        self.caption_manager = caption_manager
        self.file_manager = file_manager
        logger.info("BulkCaptionSection initialized.")

        # UI components
        self.bulk_caption_btn = None
        self.bulk_caption_status = None

    def create_section(self) -> Tuple[gr.Button, gr.Textbox]:
        """Create the bulk captioning section UI components.

        Returns:
            Tuple containing the bulk caption button and status textbox
        """
        logger.debug("Creating bulk caption section UI components.")
        with gr.Row():
            self.bulk_caption_btn = gr.Button(
                "Generate captions for all images", scale=2
            )

        # Create a visible status textbox to show progress
        self.bulk_caption_status = gr.Textbox(
            label="Status",
            interactive=False,
            visible=True,
            value="Ready to generate captions for all images in the current directory.",
        )

        logger.debug("Bulk caption section UI components created.")
        return self.bulk_caption_btn, self.bulk_caption_status

    def gui_progress_callback(self, current: int, total: int, status_msg: str) -> str:
        """
        Progress callback for GUI updates.

        Args:
            current: Current progress
            total: Total items
            status_msg: Status message

        Returns:
            Formatted status message for the GUI
        """
        percent = min(100, int(100 * current / total))
        return f"{status_msg} ({percent}% complete)"

    @handle_errors
    def on_bulk_caption(
        self,
        directory: str,
        model_type_str: str,
        model: str,
        mode: str,
        prompt: str,
        max_length: int,
        min_length: int,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ):
        """Handle bulk caption button click.

        Args:
            directory: The directory containing images to caption
            model_type_str: The model type string
            model: The model name
            mode: The model mode
            prompt: The custom prompt
            max_length: The maximum length of the caption
            min_length: The minimum length of the caption
            num_beams: The number of beams for beam search
            temperature: The temperature for sampling
            top_k: The top-k value for sampling
            top_p: The top-p value for sampling
            repetition_penalty: The repetition penalty

        Returns:
            Status message
        """
        logger.info(
            f"Bulk captioning triggered for directory: {directory}, model: {model_type_str}/{model}, mode: {mode}"
        )
        logger.debug(
            f"Bulk caption parameters: prompt='{prompt}', max_length={max_length}, min_length={min_length}, num_beams={num_beams}, temperature={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}"
        )

        if not directory or not os.path.isdir(directory):
            logger.warning(
                f"Invalid directory selected for bulk captioning: {directory}"
            )
            gr.Warning("Invalid directory")
            return "Invalid directory"

        try:
            # Convert model type string to ModelType
            model_type = ModelType(model_type_str)
            logger.debug(
                f"Converted model_type_str '{model_type_str}' to ModelType: {model_type}"
            )

            # Initial status update
            initial_status = f"Scanning directory {directory} for images..."
            gr.Info(initial_status)
            # Return status updates through the status textbox
            return initial_status

            # Get all images in the directory
            logger.debug(f"Listing images with captions in directory: {directory}")
            images_with_captions = self.file_manager.list_images_with_captions(
                directory
            )
            total_images = len(images_with_captions)
            logger.info(f"Found {total_images} images in directory '{directory}'.")

            if total_images == 0:
                logger.warning(f"No images found in the directory: {directory}")
                gr.Warning("No images found in the directory")
                return "No images found in the directory"

            # Create a progress tracker for the bulk captioning process
            progress = ProgressTracker(
                total=total_images,
                description=f"Captioning images with {model_type_str}/{model}",
                callback=lambda current, total, msg: self.gui_progress_callback(
                    current, total, msg
                ),
            )

            # Update status with total images found
            status_msg = f"Found {total_images} images in {directory}. Starting captioning process..."
            gr.Info(status_msg)
            # We'll return the final status at the end, not yielding intermediate updates

            # Track statistics
            processed_count = 0
            skipped_count = 0
            error_count = 0
            start_time = time.time()

            # Generate captions for all images
            for i, (image_name, existing_caption) in enumerate(images_with_captions):
                image_path = os.path.join(directory, image_name)

                # Update progress
                current_status = (
                    f"Processing image {i + 1}/{total_images}: {image_name}"
                )
                progress.update(0, current_status)
                # Update the status but don't yield

                try:
                    # Skip if caption already exists
                    if existing_caption:
                        logger.info(
                            f"Skipping image '{image_path}', caption already exists: '{existing_caption}'"
                        )
                        skipped_count += 1
                        progress.update(
                            1, f"Skipped {image_name} (caption already exists)"
                        )
                        continue

                    # Get default variant if not provided
                    variant = model
                    if not variant:
                        model_class = (
                            self.caption_manager.model_manager.get_model_class(
                                model_type
                            )
                        )
                        default_variant = model_class.get_default_variant()
                        if default_variant:
                            variant = default_variant

                    # Generate caption
                    logger.debug(f"Generating caption for: {image_path}")
                    caption = self.caption_manager.generate_caption(
                        model_type=model_type,
                        image_path=image_path,
                        variant=variant,
                        mode=mode if mode else None,
                        prompt=prompt if prompt else None,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                    logger.info(f"Generated caption for '{image_path}': '{caption}'")

                    # Save caption
                    logger.debug(f"Saving caption for '{image_path}'")
                    self.file_manager.write_caption(image_path, caption)
                    processed_count += 1

                    # Update progress
                    progress.update(1, f"Captioned {image_name}")

                except EnhancedError as e:
                    # Handle enhanced errors with detailed information
                    error_msg = f"Error captioning {image_name}: {e.message}"
                    logger.error(error_msg)
                    error_count += 1
                    progress.update(1, f"Error captioning {image_name}")

                except Exception as e:
                    # Handle other exceptions
                    error_msg = f"Error captioning {image_name}: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    error_count += 1
                    progress.update(1, f"Error captioning {image_name}")

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Create final status message
            success_msg = (
                f"Captioning complete in {elapsed_time:.2f} seconds.\n"
                f"Generated captions for {processed_count} images.\n"
                f"Skipped {skipped_count} images (captions already existed).\n"
            )

            if error_count > 0:
                success_msg += f"Failed to caption {error_count} images due to errors."

            logger.info(success_msg)
            gr.Info(success_msg)
            progress.complete("Captioning complete")
            return success_msg

        except EnhancedError as e:
            # For enhanced errors, show the detailed message
            error_msg = f"Error during bulk captioning: {e.message}"
            if e.troubleshooting_tips:
                error_msg += "\n\nTroubleshooting tips:\n" + "\n".join(
                    f"- {tip}" for tip in e.troubleshooting_tips
                )
            logger.error(error_msg)
            gr.Error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Error during bulk captioning for directory '{directory}': {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            gr.Error(error_msg)
            return error_msg
