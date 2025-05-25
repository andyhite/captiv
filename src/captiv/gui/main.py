"""Main GUI class for the Captiv image captioning system."""

import atexit
import contextlib
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import psutil

from captiv.gui.logging import logger, setup_logging
from captiv.gui.sections.bulk_caption import BulkCaptionSection
from captiv.gui.sections.caption import CaptionSection
from captiv.gui.sections.directory import DirectorySection
from captiv.gui.sections.gallery import GallerySection
from captiv.gui.sections.model import ModelSection
from captiv.gui.styles import css
from captiv.services import (
    CaptionFileManager,
    ConfigManager,
    FileManager,
    ImageFileManager,
    ModelManager,
    ModelType,
)
from captiv.utils.error_handling import EnhancedError, handle_errors


class CaptivGUI:
    """Gradio GUI for the Captiv image captioning system."""

    def __init__(
        self,
        share: bool = False,
        config_path: str | None = None,
        use_runpod: bool = False,
    ):
        """
        Initialize the CaptivGUI.

        Args:
            share: Whether to create a public URL for the GUI.
                  Default is False (localhost only).
            config_path: Optional path to the configuration file.
            use_runpod: Whether to enable RunPod support for remote inference.
        """
        self.base_file_manager = FileManager()
        self.file_manager = ImageFileManager(self.base_file_manager)
        self.caption_manager = CaptionFileManager(
            self.base_file_manager, self.file_manager
        )
        self.config_manager = ConfigManager(config_path)
        self.model_manager = ModelManager(self.config_manager)
        self.use_runpod = use_runpod
        self.share = share
        self.config = self.config_manager.read_config()
        logger.info(
            f"CaptivGUI initialized. Share: {self.share}, Config path: {config_path}"
        )
        logger.debug(f"Config loaded: {self.config}")

        logger.info("Initializing GUI sections...")
        self.gallery_section = GallerySection(self.caption_manager)
        self.directory_section = DirectorySection(str(Path.home()))
        self.caption_section = CaptionSection(self.caption_manager)
        self.model_section = ModelSection(self.model_manager)
        self.bulk_caption_section = BulkCaptionSection(
            self.caption_manager, self.file_manager, self.model_manager, self.use_runpod
        )
        logger.info("GUI sections initialized.")

        logger.info("Creating Gradio interface...")
        self.create_interface()
        logger.info("Gradio interface created.")

    def create_interface(self):
        """Create the Gradio interface."""
        try:
            demo = gr.Blocks(
                title="Captiv - Image Captioning System",
                fill_height=True,
                fill_width=True,
                css=css,
            ).queue()

            with demo:
                with gr.Row(elem_classes="main", equal_height=True):
                    with gr.Column(scale=3, elem_classes="body"):
                        self.dir_dropdown = self.directory_section.create_section()

                        self.gallery, self.selected_image = (
                            self.gallery_section.create_section()
                        )

                    with gr.Column(scale=1, elem_classes="sidebar"):
                        with gr.Tabs():
                            with gr.Tab("Individual"):
                                (
                                    self.caption_textbox,
                                    self.save_caption_btn,
                                    self.generate_caption_btn,
                                ) = self.caption_section.create_section()

                            with gr.Tab("Bulk"):
                                self.bulk_caption_btn = (
                                    self.bulk_caption_section.create_section()
                                )

                        gr.Markdown("### Model")
                        (
                            self.model_dropdown,
                            self.model_variant_dropdown,
                            self.mode_dropdown,
                            self.prompt_textbox,
                            self.max_new_tokens_slider,
                            self.min_new_tokens_slider,
                            self.num_beams_slider,
                            self.temperature_slider,
                            self.top_k_slider,
                            self.top_p_slider,
                            self.repetition_penalty_slider,
                            self.prompt_options_accordion,
                            self.prompt_options_checkbox_group,
                            self.character_name_textbox,
                        ) = self.model_section.create_section()

                self.setup_event_handlers()

            self.interface = demo

            host = self.config.gui.host
            port = self.config.gui.port
            ssl_keyfile = self.config.gui.ssl_keyfile
            ssl_certfile = self.config.gui.ssl_certfile

            from gradio_client import utils as client_utils

            original_json_schema_to_python_type = (
                client_utils._json_schema_to_python_type
            )

            def patched_json_schema_to_python_type(schema, defs=None):
                if isinstance(schema, bool):
                    return "bool"
                return original_json_schema_to_python_type(schema, defs)

            client_utils._json_schema_to_python_type = (
                patched_json_schema_to_python_type
            )

            original_get_type = client_utils.get_type

            def patched_get_type(schema):
                if isinstance(schema, bool):
                    return "bool"
                return original_get_type(schema)

            client_utils.get_type = patched_get_type

            try:

                def cleanup():
                    logger.info("Cleaning up resources...")
                    current_process = psutil.Process(os.getpid())

                    children = current_process.children(recursive=True)

                    for child in children:
                        with contextlib.suppress(Exception):
                            child.terminate()

                    gone, still_alive = psutil.wait_procs(children, timeout=3)

                    for p in still_alive:
                        with contextlib.suppress(Exception):
                            p.kill()

                atexit.register(cleanup)

                def signal_handler(sig, frame):
                    logger.info("Shutting down Captiv GUI...")
                    sys.exit(0)

                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                logger.info(f"Launching Gradio interface on {host}:{port}...")

                launch_kwargs = {
                    "share": self.share,
                    "server_name": host,
                    "server_port": port,
                    "ssl_keyfile": ssl_keyfile,
                    "ssl_certfile": ssl_certfile,
                    "show_api": False,
                    "show_error": True,
                    "quiet": True,
                    "prevent_thread_lock": True,
                }

                app, local_url, share_url = self.interface.launch(**launch_kwargs)

                sys.stdout.flush()

                self.server = app

                logger.info(f"Captiv GUI running on: {local_url}")
                if share_url:
                    logger.info(f"Public URL: {share_url}")

                while True:
                    time.sleep(1)
            except OSError as e:
                if "Cannot find empty port" in str(e):
                    print(
                        f"Port {port} is already in use. Trying with a different port..."  # noqa: E501
                    )

                    logger.info("Trying with a different port...")
                    logger.info("Launching Gradio interface with auto-selected port...")

                    launch_kwargs = {
                        "share": self.share,
                        "server_name": host,
                        "server_port": None,
                        "ssl_keyfile": ssl_keyfile,
                        "ssl_certfile": ssl_certfile,
                        "show_api": False,
                        "show_error": True,
                        "quiet": True,
                        "prevent_thread_lock": True,
                    }

                    app, local_url, share_url = self.interface.launch(**launch_kwargs)

                    sys.stdout.flush()

                    self.server = app

                    logger.info(f"Captiv GUI running on: {local_url}")
                    if share_url:
                        logger.info(f"Public URL: {share_url}")

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
        logger.debug("Setting up event handlers...")
        self.gallery.select(
            fn=self.gallery_section.on_gallery_select, outputs=[self.selected_image]
        )
        self.selected_image.change(
            fn=self.caption_section.on_image_select,
            inputs=[self.selected_image],
            outputs=[self.caption_textbox],
        )
        self.dir_dropdown.change(
            fn=self.handle_dir_change,
            inputs=[self.dir_dropdown],
            outputs=[self.dir_dropdown, self.gallery],
        )
        self.save_caption_btn.click(
            fn=self.caption_section.on_save_caption,
            inputs=[self.selected_image, self.caption_textbox],
        )
        self.generate_caption_btn.click(
            fn=self.on_generate_caption,
            inputs=[
                self.selected_image,
                self.model_dropdown,
                self.model_variant_dropdown,
                self.mode_dropdown,
                self.prompt_textbox,
                self.max_new_tokens_slider,
                self.min_new_tokens_slider,
                self.num_beams_slider,
                self.temperature_slider,
                self.top_k_slider,
                self.top_p_slider,
                self.repetition_penalty_slider,
            ],
            outputs=[self.caption_textbox],
        )
        self.model_dropdown.change(
            fn=self.model_section.on_model_change,
            inputs=[self.model_dropdown],
            outputs=[
                self.model_variant_dropdown,
                self.mode_dropdown,
                self.prompt_options_accordion,
                self.prompt_options_checkbox_group,
                self.character_name_textbox,
            ],
        )
        self.mode_dropdown.change(
            fn=self.model_section.on_mode_change,
            inputs=[self.mode_dropdown],
            outputs=[self.prompt_textbox],
        )
        self.bulk_caption_btn.click(
            fn=self.on_bulk_caption,
            inputs=[
                self.dir_dropdown,
                self.model_dropdown,
                self.model_variant_dropdown,
                self.mode_dropdown,
                self.prompt_textbox,
                self.max_new_tokens_slider,
                self.min_new_tokens_slider,
                self.num_beams_slider,
                self.temperature_slider,
                self.top_k_slider,
                self.top_p_slider,
                self.repetition_penalty_slider,
            ],
        )

    def handle_dir_change(self, selected_dir) -> tuple[dict[str, Any], list[str]]:
        """
        Handle directory change event.

        Args:
            selected_dir: The selected directory

        Returns:
            Tuple of (dropdown update, gallery images)
        """
        logger.info(f"Handling directory change: {selected_dir}")
        dir_update, new_path = self.directory_section.handle_dir_change(selected_dir)
        self.gallery_section.set_current_directory(new_path)
        gallery_images = self.gallery_section.get_gallery_images(new_path)
        logger.debug(f"Found {len(gallery_images)} images in gallery")

        return dir_update, gallery_images

    def on_generate_caption(
        self,
        image_path: str,
        model_str: str,
        model_variant: str,
        mode: str,
        prompt: str,
        max_new_tokens: int,
        min_new_tokens: int,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        progress=gr.Progress(),
    ) -> str:
        """Handle generate caption button click."""
        logger.info(
            f"Generating caption for {image_path} with {model_str}/{model_variant}"
        )
        if not image_path:
            logger.warning("Generate caption called with no image selected.")
            gr.Warning("No image selected")
            return "No image selected"

        try:
            try:
                model = ModelType(model_str)
            except Exception:
                gr.Warning(f"Invalid model: {model_str}")
                return "Error: Invalid model"

            additional_params = {}
            selected_options = self.prompt_options_checkbox_group.value
            if hasattr(self.model_section, "prompt_options_mapping"):
                for display_name in selected_options:
                    if display_name in self.model_section.prompt_options_mapping:
                        option_name, _ = self.model_section.prompt_options_mapping[
                            display_name
                        ]
                        additional_params[option_name] = True
            character_name = self.character_name_textbox.value
            if character_name and self.character_name_textbox.visible:
                additional_params["character_name"] = character_name

            logger.debug(
                f"Calling caption_manager.generate_caption with params: {locals()}"
            )

            def progress_callback(step, total_steps, message):
                progress(step / total_steps, desc=message)

            try:
                model_instance = self.model_manager.create_model_instance(
                    model_type=model,
                    variant=model_variant if model_variant else None,
                    use_runpod=self.use_runpod,
                    progress_callback=progress_callback,
                )

                prompt_options_list = None
                if selected_options and hasattr(
                    self.model_section, "prompt_options_mapping"
                ):
                    prompt_options_list = []
                    for display_name in selected_options:
                        if display_name in self.model_section.prompt_options_mapping:
                            option_name, _ = self.model_section.prompt_options_mapping[
                                display_name
                            ]
                            prompt_options_list.append(option_name)

                prompt_variables_dict = None
                if character_name and self.character_name_textbox.visible:
                    prompt_variables_dict = {"character_name": character_name}

                gen_params = self.model_manager.build_generation_params(
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    prompt_options=prompt_options_list,
                    prompt_variables=prompt_variables_dict,
                )

                caption = model_instance.caption_image(
                    image_input=image_path,
                    prompt_or_mode=prompt if prompt else mode,
                    **gen_params,
                )
                logger.info(
                    f"Caption generated successfully for {image_path}: '{caption}'"
                )
                return caption
            except Exception as e:
                logger.error(f"Error generating caption for {image_path}: {e}")
                logger.debug(traceback.format_exc())
                gr.Error(str(e))
                return str(e)

        except EnhancedError as e:
            error_msg = f"Error generating caption: {e.message}"
            if e.troubleshooting_tips:
                error_details = "\n".join(f"- {tip}" for tip in e.troubleshooting_tips)
                error_msg = f"{error_msg}\n\nTroubleshooting tips:\n{error_details}"

            logger.error(error_msg)
            gr.Error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Error generating caption for {image_path}: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            gr.Error(error_msg)
            return error_msg

    def on_bulk_caption(
        self,
        directory: str,
        model_str: str,
        model_variant: str,
        mode: str,
        prompt: str,
        max_new_tokens: int,
        min_new_tokens: int,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        logger.info(
            f"Bulk caption called for directory: {directory}, model: {model_str}, variant: {model_variant}, mode: {mode}"  # noqa: E501
        )
        logger.debug(
            f"Bulk caption parameters: prompt='{prompt}', max_new_tokens={max_new_tokens}, min_new_tokens={min_new_tokens}, num_beams={num_beams}, temperature={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}"  # noqa: E501
        )
        """Handle bulk caption button click."""
        try:
            prompt_options_list = None
            selected_options = self.prompt_options_checkbox_group.value
            if selected_options and hasattr(
                self.model_section, "prompt_options_mapping"
            ):
                prompt_options_list = []
                for display_name in selected_options:
                    if display_name in self.model_section.prompt_options_mapping:
                        option_name, _ = self.model_section.prompt_options_mapping[
                            display_name
                        ]
                        prompt_options_list.append(option_name)

            prompt_variables_dict = None
            character_name = self.character_name_textbox.value
            if character_name and self.character_name_textbox.visible:
                prompt_variables_dict = {"character_name": character_name}

            logger.debug(
                f"Prompt options for bulk caption generation: {prompt_options_list}"
            )
            logger.debug(
                f"Prompt variables for bulk caption generation: {prompt_variables_dict}"
            )

            logger.debug(
                f"Calling bulk_caption_section.on_bulk_caption for directory: {directory}"  # noqa: E501
            )
            status = self.bulk_caption_section.on_bulk_caption(
                directory,
                model_str,
                model_variant,
                mode,
                prompt,
                max_new_tokens,
                min_new_tokens,
                num_beams,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                prompt_options=prompt_options_list,
                prompt_variables=prompt_variables_dict,
            )
            logger.info(
                f"Bulk captioning for directory '{directory}' completed with status: {status}"  # noqa: E501
            )
            gr.Info(f"Bulk captioning completed: {status}")
            return status
        except Exception as e:
            error_msg = f"Error during bulk captioning for directory '{directory}': {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            gr.Error(error_msg)
            return error_msg


@handle_errors
def main(share: bool = False, config_path: str | None = None, use_runpod: bool = False):
    """
    Launch the Captiv GUI.

    Args:
        share: Whether to create a public URL for the GUI using Gradio's share feature.
              Default is False (localhost only).
        config_path: Optional path to the configuration file.
        use_runpod: Whether to enable RunPod support for remote inference.
    """
    # Set up logging first
    setup_logging(level="INFO", intercept_libraries=["gradio"])

    logger.info(
        f"Main function called. Share: {share}, Config path: {config_path}, "
        f"RunPod: {use_runpod}"
    )
    logger.info("Initializing and launching CaptivGUI...")
    CaptivGUI(share=share, config_path=config_path, use_runpod=use_runpod)
    logger.info("CaptivGUI launched successfully.")


if __name__ == "__main__":
    main()
