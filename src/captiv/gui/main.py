"""
Main GUI class for the Captiv image captioning system.
"""

import atexit
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import psutil
from loguru import logger

from captiv.config import read_config
from captiv.gui.sections.bulk_caption import BulkCaptionSection
from captiv.gui.sections.caption import CaptionSection
from captiv.gui.sections.directory import DirectorySection
from captiv.gui.sections.gallery import GallerySection
from captiv.gui.sections.model import ModelSection
from captiv.gui.styles import css
from captiv.services.caption_manager import CaptionManager
from captiv.services.image_file_manager import ImageFileManager
from captiv.services.model_manager import ModelManager, ModelType
from captiv.utils.error_handling import EnhancedError
from captiv.utils.progress import ProgressTracker


class CaptivGUI:
    """Gradio GUI for the Captiv image captioning system."""

    def __init__(self, share: bool = False, config_path: Optional[str] = None):
        """Initialize the CaptivGUI.

        Args:
            share: Whether to create a public URL for the GUI using Gradio's share feature.
                  Default is False (localhost only).
            config_path: Optional path to the configuration file.
        """
        # Initialize services
        self.caption_manager = CaptionManager()
        self.model_manager = ModelManager()
        self.file_manager = ImageFileManager()
        self.share = share
        self.config = read_config(config_path)
        logger.info(
            f"CaptivGUI initialized. Share: {self.share}, Config path: {config_path}"
        )
        logger.debug(f"Config loaded: {self.config}")

        # Initialize sections
        logger.info("Initializing GUI sections...")
        self.gallery_section = GallerySection(self.file_manager)
        self.directory_section = DirectorySection(str(Path.home()))
        self.caption_section = CaptionSection(self.file_manager)
        self.model_section = ModelSection(self.model_manager)
        self.bulk_caption_section = BulkCaptionSection(
            self.caption_manager, self.file_manager
        )
        logger.info("GUI sections initialized.")

        # Create the interface
        logger.info("Creating Gradio interface...")
        self.create_interface()
        logger.info("Gradio interface created.")

    def create_interface(self):
        """Create the Gradio interface."""
        try:
            # Create a blocks interface for more flexibility
            demo = gr.Blocks(
                title="Captiv - Image Captioning System",
                fill_height=True,
                fill_width=True,
                css=css,
            )

            with demo:
                with gr.Row(elem_classes="main", equal_height=True):
                    # Left column (wider) for the gallery
                    with gr.Column(scale=3, elem_classes="body"):
                        # Directory navigation section
                        self.dir_dropdown = self.directory_section.create_section()

                        # Gallery section
                        self.gallery, self.selected_image = (
                            self.gallery_section.create_section()
                        )

                    # Right column for all controls
                    with gr.Column(scale=1, elem_classes="sidebar"):
                        # Directory selector moved above the gallery

                        with gr.Tabs():
                            with gr.Tab("Individual"):
                                # 2. Caption management section
                                (
                                    self.caption_textbox,
                                    self.save_status,
                                    self.save_caption_btn,
                                    self.generate_caption_btn,
                                    self.caption_progress,
                                ) = self.caption_section.create_section()

                            with gr.Tab("Bulk"):
                                # 3. Bulk captioning section (at the bottom)
                                self.bulk_caption_btn, self.bulk_caption_status = (
                                    self.bulk_caption_section.create_section()
                                )

                        # 4. Model type configuration section
                        gr.Markdown("### Model")
                        (
                            self.model_type_dropdown,
                            self.model_dropdown,
                            self.mode_dropdown,
                            self.prompt_textbox,
                            self.advanced_options,
                        ) = self.model_section.create_section()

                # Set up all event handlers
                self.setup_event_handlers()

            # Store the interface
            self.interface = demo

            # Get GUI configuration
            host = self.config.gui.host
            port = self.config.gui.port
            ssl_keyfile = self.config.gui.ssl_keyfile
            ssl_certfile = self.config.gui.ssl_certfile

            # Launch the interface with Gradio 4.44.1 parameters
            # Patch for Gradio 4.44.1 compatibility issue with schema handling
            # Monkey patch the json_schema_to_python_type function to handle boolean schemas
            from gradio_client import utils as client_utils

            original_json_schema_to_python_type = (
                client_utils._json_schema_to_python_type
            )

            def patched_json_schema_to_python_type(schema, defs=None):
                # Handle case where schema is a boolean
                if isinstance(schema, bool):
                    return "bool"
                return original_json_schema_to_python_type(schema, defs)

            # Apply the monkey patch
            client_utils._json_schema_to_python_type = (
                patched_json_schema_to_python_type
            )

            # Fix for get_type function that's causing the TypeError
            original_get_type = client_utils.get_type

            def patched_get_type(schema):
                # Handle case where schema is a boolean
                if isinstance(schema, bool):
                    return "bool"
                return original_get_type(schema)

            # Apply the monkey patch
            client_utils.get_type = patched_get_type

            try:
                # Define cleanup function to be called on exit
                def cleanup():
                    logger.info("Cleaning up resources...")
                    # Get the current process
                    current_process = psutil.Process(os.getpid())

                    # Get all child processes
                    children = current_process.children(recursive=True)

                    # Terminate all child processes
                    for child in children:
                        try:
                            child.terminate()
                        except Exception:
                            pass

                    # Wait for them to terminate
                    gone, still_alive = psutil.wait_procs(children, timeout=3)

                    # Force kill any remaining processes
                    for p in still_alive:
                        try:
                            p.kill()
                        except Exception:
                            pass

                # Register the cleanup function to be called on exit
                atexit.register(cleanup)

                # Define signal handler for graceful shutdown
                def signal_handler(sig, frame):
                    logger.info("Shutting down Captiv GUI...")
                    sys.exit(0)

                # Register signal handlers
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                logger.info(f"Launching Gradio interface on {host}:{port}...")

                launch_kwargs = {
                    "share": self.share,
                    "server_name": host,
                    "server_port": port,
                    "ssl_keyfile": ssl_keyfile,
                    "ssl_certfile": ssl_certfile,
                    "show_api": False,  # Disable API info generation which can cause errors
                    "show_error": True,  # Show detailed errors for debugging
                    "quiet": True,  # Suppress Gradio output
                    "prevent_thread_lock": True,  # Critical parameter to prevent blocking
                }

                # Launch with the parameters
                app, local_url, share_url = self.interface.launch(**launch_kwargs)

                # Force flush the output buffer to ensure logs are visible
                sys.stdout.flush()

                # Store the server instance for proper shutdown
                self.server = app

                # Use our own logging to show what host and port the app is running on
                logger.info(f"Captiv GUI running on: {local_url}")
                if share_url:
                    logger.info(f"Public URL: {share_url}")

                # Keep the main thread running
                while True:
                    time.sleep(1)
            except OSError as e:
                if "Cannot find empty port" in str(e):
                    print(
                        f"Port {port} is already in use. Trying with a different port..."
                    )

                    logger.info("Trying with a different port...")
                    logger.info("Launching Gradio interface with auto-selected port...")

                    launch_kwargs = {
                        "share": self.share,  # Use the share parameter from the constructor
                        "server_name": host,
                        "server_port": None,  # Let Gradio find an available port
                        "ssl_keyfile": ssl_keyfile,
                        "ssl_certfile": ssl_certfile,
                        "show_api": False,
                        "show_error": True,
                        "quiet": True,  # Suppress Gradio output
                        "prevent_thread_lock": True,  # Critical parameter to prevent blocking
                    }

                    # Launch with the parameters
                    app, local_url, share_url = self.interface.launch(**launch_kwargs)

                    # Force flush the output buffer to ensure logs are visible
                    sys.stdout.flush()

                    # Store the server instance for proper shutdown
                    self.server = app

                    # Use our own logging to show what host and port the app is running on
                    logger.info(f"Captiv GUI running on: {local_url}")
                    if share_url:
                        logger.info(f"Public URL: {share_url}")

                    # Keep the main thread running
                    while True:
                        time.sleep(1)
                else:
                    raise
        except Exception as e:
            print(f"Error creating interface: {e}")
            print(traceback.format_exc())
            raise

    def setup_event_handlers(self):
        """Set up all event handlers after all components are defined."""
        logger.info("Setting up event handlers...")

        # Gallery section events
        logger.debug("Setting up gallery select event handler.")
        self.gallery.select(
            fn=self.gallery_section.on_gallery_select, outputs=[self.selected_image]
        )

        # Add event handler to update caption when selected_image changes
        logger.debug(
            "Setting up selected_image change event handler for caption update."
        )
        self.selected_image.change(
            fn=self.caption_section.on_image_select,
            inputs=[self.selected_image],
            outputs=[self.caption_textbox],
        )

        # Directory dropdown: update options and gallery on change
        logger.debug("Setting up directory dropdown change event handler.")
        self.dir_dropdown.change(
            fn=self.handle_dir_change,
            inputs=[self.dir_dropdown],
            outputs=[self.dir_dropdown, self.gallery],
        )

        # Caption section events
        logger.debug("Setting up save caption button click event handler.")
        self.save_caption_btn.click(
            fn=self.caption_section.on_save_caption,
            inputs=[self.selected_image, self.caption_textbox],
            outputs=[self.save_status],
        )

        logger.debug("Setting up generate caption button click event handler.")
        self.generate_caption_btn.click(
            fn=self.on_generate_caption,
            inputs=[
                self.selected_image,
                self.model_type_dropdown,
                self.model_dropdown,
                self.mode_dropdown,
                self.prompt_textbox,
                self.advanced_options["max_length"],
                self.advanced_options["min_length"],
                self.advanced_options["num_beams"],
                self.advanced_options["temperature"],
                self.advanced_options["top_k"],
                self.advanced_options["top_p"],
                self.advanced_options["repetition_penalty"],
            ],
            outputs=[self.caption_textbox, self.save_status, self.caption_progress],
        )

        # Model section events
        logger.debug("Setting up model type dropdown change event handler.")
        self.model_type_dropdown.change(
            fn=self.model_section.on_model_change,
            inputs=[self.model_type_dropdown],
            outputs=[
                self.model_dropdown,
                self.mode_dropdown,
            ],
        )

        # Mode dropdown event to show/hide custom prompt
        logger.debug("Setting up mode dropdown change event handler.")
        self.mode_dropdown.change(
            fn=self.model_section.on_mode_change,
            inputs=[self.mode_dropdown],
            outputs=[self.prompt_textbox],
        )

        # Bulk caption events
        logger.debug("Setting up bulk caption button click event handler.")
        self.bulk_caption_btn.click(
            fn=self.on_bulk_caption,
            inputs=[
                self.dir_dropdown,
                self.model_type_dropdown,
                self.model_dropdown,
                self.mode_dropdown,
                self.prompt_textbox,
                self.advanced_options["max_length"],
                self.advanced_options["min_length"],
                self.advanced_options["num_beams"],
                self.advanced_options["temperature"],
                self.advanced_options["top_k"],
                self.advanced_options["top_p"],
                self.advanced_options["repetition_penalty"],
            ],
            outputs=[self.bulk_caption_status],
        )
        logger.info("Event handlers set up.")

    def handle_dir_change(self, selected_dir) -> Tuple[Dict[str, Any], List[str]]:
        """Handle directory change event.

        Args:
            selected_dir: The selected directory

        Returns:
            Tuple of (dropdown update, gallery images)
        """
        logger.info(f"Handling directory change. Selected directory: {selected_dir}")
        # Update directory in the directory section
        dir_update, new_path = self.directory_section.handle_dir_change(selected_dir)
        logger.debug(
            f"Directory section updated. New path: {new_path}, Dropdown update: {dir_update}"
        )

        # Update directory in the gallery section
        self.gallery_section.set_current_directory(new_path)  # Use new_path directly
        logger.debug(
            f"Gallery section current directory set to: {self.gallery_section.current_directory}"
        )

        # Get images for the gallery
        # current_dir_for_gallery = self.directory_section.get_current_directory() # This was redundant
        logger.debug(f"Fetching gallery images for directory: {new_path}")
        gallery_images = self.gallery_section.get_gallery_images(
            new_path  # Use new_path directly
        )
        logger.info(
            f"handle_dir_change returning {len(gallery_images)} images for gallery."
        )
        logger.debug(f"Gallery images found: {gallery_images}")

        return dir_update, gallery_images

    def gui_progress_callback(self, current: int, total: int, status_msg: str) -> None:
        """
        Progress callback for GUI updates.

        Args:
            current: Current progress
            total: Total items
            status_msg: Status message
        """
        # Update the caption progress component
        # We'll handle this through the event handler return values

    def on_generate_caption(
        self,
        image_path: str,
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
    ) -> Tuple[str, str, str]:
        """Handle generate caption button click."""
        logger.info(
            f"Generate caption called for image: {image_path}, model: {model_type_str}/{model}, mode: {mode}"
        )
        logger.debug(
            f"Generate caption parameters: prompt='{prompt}', max_length={max_length}, min_length={min_length}, num_beams={num_beams}, temperature={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}"
        )
        if not image_path:
            logger.warning("Generate caption called with no image selected.")
            gr.Warning("No image selected")
            return "No image selected", "No image selected", "No image selected"

        try:
            # Update progress status
            self.gui_progress_callback(
                0, 1, f"Preparing to generate caption using {model_type_str} model..."
            )

            # Convert model type string to ModelType
            model_type = ModelType(model_type_str)
            logger.debug(
                f"Converted model_type_str '{model_type_str}' to ModelType: {model_type}"
            )

            # Create a progress tracker for model loading and caption generation
            progress = ProgressTracker(
                total=3,
                description=f"Generating caption with {model_type.value}/{model}",
                callback=self.gui_progress_callback,
            )

            # Update progress
            progress.update(1, f"Loading {model_type.value} model...")

            # Generate caption with progress tracking
            start_time = time.time()

            # No additional parameters needed
            additional_params = {}
            logger.debug("No additional parameters for caption generation")

            # Generate caption
            logger.debug("Calling caption_manager.generate_caption...")
            progress.update(1, "Analyzing image and generating caption...")

            caption = self.caption_manager.generate_caption(
                model_type=model_type,
                image_path=image_path,
                variant=model if model else None,
                mode=mode if mode else None,
                prompt=prompt if prompt else None,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **additional_params,
            )

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Update progress
            final_status = f"Caption generated in {elapsed_time:.2f} seconds"
            progress.complete(final_status)

            logger.info(f"Caption generated successfully for {image_path}: '{caption}'")
            return caption, "Caption generated successfully", final_status

        except EnhancedError as e:
            # For enhanced errors, show the detailed message
            error_msg = f"Error generating caption: {e.message}"
            if e.troubleshooting_tips:
                error_details = "\n".join(f"- {tip}" for tip in e.troubleshooting_tips)
                error_msg = f"{error_msg}\n\nTroubleshooting tips:\n{error_details}"

            logger.error(error_msg)
            gr.Error(error_msg)
            return error_msg, error_msg, f"Error: {e.message}"

        except Exception as e:
            error_msg = f"Error generating caption for {image_path}: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            gr.Error(error_msg)
            return error_msg, error_msg, f"Error: {str(e)}"

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
    ) -> str:
        logger.info(
            f"Bulk caption called for directory: {directory}, model: {model_type_str}/{model}, mode: {mode}"
        )
        logger.debug(
            f"Bulk caption parameters: prompt='{prompt}', max_length={max_length}, min_length={min_length}, num_beams={num_beams}, temperature={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}"
        )
        """Handle bulk caption button click."""
        try:
            logger.debug(
                f"Calling bulk_caption_section.on_bulk_caption for directory: {directory}"
            )
            status = self.bulk_caption_section.on_bulk_caption(
                directory,
                model_type_str,
                model,
                mode,
                prompt,
                max_length,
                min_length,
                num_beams,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            )
            logger.info(
                f"Bulk captioning for directory '{directory}' completed with status: {status}"
            )
            gr.Info(f"Bulk captioning completed: {status}")
            return status
        except Exception as e:
            error_msg = f"Error during bulk captioning for directory '{directory}': {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            gr.Error(error_msg)
            return error_msg


def main(share: bool = False, config_path: Optional[str] = None):
    logger.info(f"Main function called. Share: {share}, Config path: {config_path}")
    """Launch the Captiv GUI.

    Args:
        share: Whether to create a public URL for the GUI using Gradio's share feature.
              Default is False (localhost only).
        config_path: Optional path to the configuration file.
    """
    try:
        logger.info("Initializing and launching CaptivGUI...")
        # Initialize and launch the GUI
        CaptivGUI(share=share, config_path=config_path)
        logger.info("CaptivGUI launched successfully.")
    except (OSError, Exception) as e:
        logger.critical(f"Critical error launching GUI: {e}")
        logger.debug(traceback.format_exc())
        # Ensure traceback is imported if not already at the top level for this specific use
        import traceback as tb_print

        print(f"Error launching GUI: {e}")
        print(tb_print.format_exc())


if __name__ == "__main__":
    main()
