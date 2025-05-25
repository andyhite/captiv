"""Bulk captioning section for the Captiv GUI."""

import os
import time
import traceback
from pathlib import Path

import gradio as gr
from loguru import logger

from captiv.services import CaptionFileManager, ImageFileManager, ModelManager
from captiv.services.model_manager import ModelType
from captiv.utils.error_handling import EnhancedError, handle_errors


class BulkCaptionSection:
    """Bulk captioning section for captioning all images in a directory."""

    def __init__(
        self,
        caption_manager: CaptionFileManager,
        file_manager: ImageFileManager,
        model_manager: ModelManager,
        use_runpod: bool = False,
    ):
        """
        Initialize the bulk caption section.

        Args:
            caption_manager: The caption manager instance
            file_manager: The image file manager instance
            model_manager: The model manager instance
            use_runpod: Whether to use RunPod for remote inference
        """
        self.caption_manager = caption_manager
        self.file_manager = file_manager
        self.model_manager = model_manager
        self.use_runpod = use_runpod

        self.bulk_caption_btn = None

    def create_section(self) -> gr.Button:
        """
        Create the bulk captioning section UI components.

        Returns:
            Tuple containing the bulk caption button and status textbox
        """
        with gr.Row():
            self.bulk_caption_btn = gr.Button(
                "Generate captions for all images", scale=2
            )

        return self.bulk_caption_btn

    @handle_errors
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
        progress=gr.Progress(),
        **additional_params,
    ):
        """
        Handle bulk caption button click.

        Args:
            directory: The directory containing images to caption
            model_str: The model string
            model_variant: The model variant name
            mode: The model mode
            prompt: The custom prompt
            max_new_tokens: The maximum number of tokens in the caption
            min_new_tokens: The minimum number of tokens in the caption
            num_beams: The number of beams for beam search
            temperature: The temperature for sampling
            top_k: The top-k value for sampling
            top_p: The top-p value for sampling
            repetition_penalty: The repetition penalty

        Returns:
            Status message
        """
        logger.info(
            f"Bulk captioning directory: {directory} with {model_str}/{model_variant}"
        )

        if not directory or not os.path.isdir(directory):
            logger.warning(
                f"Invalid directory selected for bulk captioning: {directory}"
            )
            gr.Warning("Invalid directory")
            return "Invalid directory"

        try:
            model = ModelType(model_str)

            initial_status = f"Scanning directory {directory} for images..."
            gr.Info(initial_status)

            images_with_captions = self.caption_manager.list_images_and_captions(
                Path(directory)
            )
            total_images = len(images_with_captions)
            logger.info(f"Found {total_images} images in directory '{directory}'.")

            if total_images == 0:
                logger.warning(f"No images found in the directory: {directory}")
                gr.Warning("No images found in the directory")
                return "No images found in the directory"

            progress(0, desc=f"Preparing to caption {total_images} images")

            status_msg = f"Found {total_images} images in {directory}. Starting captioning process..."  # noqa: E501
            gr.Info(status_msg)

            processed_count = 0
            skipped_count = 0
            error_count = 0
            start_time = time.time()

            for _i, (image_name, existing_caption) in enumerate(
                progress.tqdm(images_with_captions, desc="Captioning images")
            ):
                image_path = os.path.join(directory, image_name)

                try:
                    if existing_caption:
                        logger.debug(
                            f"Skipping image with existing caption: {image_path}"
                        )
                        skipped_count += 1
                        continue

                    variant = model_variant
                    if not variant:
                        model_class = self.model_manager.get_model_class(model)
                        default_variant = model_class.DEFAULT_VARIANT
                        if default_variant:
                            variant = default_variant
                        else:
                            variants = self.model_manager.get_variants_for_model(model)
                            if variants:
                                variant = variants[0]
                            else:
                                error_msg = (
                                    f"No variants available for {model.value} model"
                                )
                                logger.error(error_msg)
                                error_count += 1
                                continue

                    model_instance = self.model_manager.create_model_instance(
                        model_type=model,
                        variant=variant,
                        torch_dtype=None,
                        use_runpod=self.use_runpod,
                    )

                    prompt_options = additional_params.get("prompt_options")
                    prompt_variables = additional_params.get("prompt_variables")

                    generation_params = self.model_manager.build_generation_params(
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        prompt_options=prompt_options,
                        prompt_variables=prompt_variables,
                    )

                    caption = model_instance.caption_image(
                        image_input=image_path,
                        prompt_or_mode=prompt if prompt else mode,
                        **generation_params,
                    )
                    logger.info(f"Generated caption for '{image_path}'")
                    self.caption_manager.write_caption(Path(image_path), caption)
                    processed_count += 1

                except EnhancedError as e:
                    error_msg = f"Error captioning {image_name}: {e.message}"
                    logger.error(error_msg)
                    error_count += 1

                except Exception as e:
                    error_msg = f"Error captioning {image_name}: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    error_count += 1

            elapsed_time = time.time() - start_time

            success_msg = (
                f"Captioning complete in {elapsed_time:.2f} seconds.\n"
                f"Generated captions for {processed_count} images.\n"
                f"Skipped {skipped_count} images (captions already existed).\n"
            )

            if error_count > 0:
                success_msg += f"Failed to caption {error_count} images due to errors."

            gr.Info(success_msg)

            progress(1.0, desc="Captioning complete")
            return success_msg

        except EnhancedError as e:
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
