from pathlib import Path
from typing import List, Optional, Tuple

from captiv.services.exceptions import CaptionError
from captiv.services.image_file_manager import ImageFileManager
from captiv.services.model_manager import ModelManager, ModelType


class CaptionManager:
    """Manages caption operations for images."""

    def __init__(
        self,
        file_manager: Optional[ImageFileManager] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        """
        Initialize the CaptionManager.

        Args:
            file_manager: An ImageFileManager instance for file operations.
            model_manager: A ModelManager instance for model operations.
        """
        self.file_manager = file_manager or ImageFileManager()
        self.model_manager = model_manager or ModelManager()

    def list_images_with_captions(
        self, directory: str
    ) -> List[Tuple[str, Optional[str]]]:
        """
        List images in a directory with their captions.

        Args:
            directory: Path to the directory containing images.

        Returns:
            List of tuples containing (image_name, caption_text).

        Raises:
            DirectoryNotFoundError: If the directory does not exist.
        """
        return self.file_manager.list_images_with_captions(directory)

    def set_caption(self, image_path: str, caption: str) -> None:
        """
        Set the caption for an image.

        Args:
            image_path: Path to the image file.
            caption: Caption text to write.

        Raises:
            FileNotFoundError: If the image file does not exist.
            UnsupportedFileTypeError: If the image file has an unsupported extension.
        """
        self.file_manager.write_caption(image_path, caption)

    def generate_caption(
        self,
        model_type: ModelType,
        image_path: str,
        variant: Optional[str] = None,
        mode: Optional[str] = None,
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        torch_dtype: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a caption for an image using the specified model configuration.

        Args:
            model_type: The type of model to use.
            image_path: Path to the image file.
            variant: The variant of the model to use.
            mode: The mode to use for captioning.
            prompt: Custom prompt to use for captioning.
            max_length: Maximum length of the generated caption.
            min_length: Minimum length of the generated caption.
            num_beams: Number of beams for beam search.
            temperature: Temperature for sampling.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty parameter.
            torch_dtype: The torch dtype to use for the model.

        Returns:
            The generated caption.

        Raises:
            FileNotFoundError: If the image file does not exist.
            UnsupportedFileTypeError: If the image file has an unsupported extension.
            ModelError: If there's an error with the model.
            CaptionError: If there's an error generating the caption.
        """
        # Validate the image file
        self.file_manager.validate_image_file(image_path)

        try:
            # Create the model instance
            model = self.model_manager.create_model_instance(
                model_type, variant, torch_dtype
            )

            # Build generation parameters
            gen_params = self.model_manager.build_generation_params(
                max_length,
                min_length,
                num_beams,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            )

            # Generate the caption
            if prompt:
                caption = model.caption_image(
                    image_path, prompt=prompt, **gen_params, **kwargs
                )
            elif mode:
                # Validate the mode
                self.model_manager.validate_mode(model_type, mode)
                caption = model.caption_image(
                    image_path, prompt=mode, **gen_params, **kwargs
                )
            else:
                caption = model.caption_image(image_path, **gen_params, **kwargs)

            # Automatically save the generated caption
            # Skip validation since we already validated the image file
            img_path = Path(image_path)
            caption_file = img_path.with_suffix(".txt")
            caption_file.write_text(caption, encoding="utf-8")

            return caption

        except Exception as e:
            # Wrap any exceptions in a CaptionError
            raise CaptionError(f"Failed to generate caption: {str(e)}") from e

    def get_supported_extensions(self):
        """
        Return supported image file extensions.

        Returns:
            Set of supported file extensions.
        """
        return self.file_manager.get_supported_extensions()
