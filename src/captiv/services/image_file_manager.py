"""
Image file management service.

This module provides utilities for managing image files and their associated captions.
"""

from pathlib import Path
from typing import List, Optional, Set, Tuple

from captiv.logging import logger
from captiv.services.exceptions import (
    DirectoryNotFoundError,
    FileNotFoundError,
    FileOperationError,
    UnsupportedFileTypeError,
)


class ImageFileManager:
    """Manages file system operations for image files and their captions."""

    # Set of supported image file extensions
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

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
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise DirectoryNotFoundError(f"{directory} is not a directory.")

        images = sorted(
            [
                f
                for f in dir_path.iterdir()
                if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        )

        result = []
        for img in images:
            caption_file = img.with_suffix(".txt")
            caption = None
            if caption_file.exists() and caption_file.is_file():
                try:
                    text = caption_file.read_text(encoding="utf-8").strip()
                    if text:
                        caption = text
                except Exception as e:
                    logger.warning(f"Error reading caption file {caption_file}: {e}")
            result.append((img.name, caption))

        return result

    def write_caption(self, image_path: str, caption: str) -> None:
        """
        Write a caption to a text file associated with an image.

        Args:
            image_path: Path to the image file.
            caption: Caption text to write.

        Raises:
            FileNotFoundError: If the image file does not exist.
            UnsupportedFileTypeError: If the image file has an unsupported extension.
        """
        self.validate_image_file(image_path)
        img_path = Path(image_path)
        caption_file = img_path.with_suffix(".txt")
        try:
            caption_file.write_text(caption, encoding="utf-8")
        except Exception as e:
            logger.error(f"Error writing caption to {caption_file}: {e}")
            raise FileOperationError(f"Failed to write caption: {e}")

    def read_caption(self, image_path: str) -> Optional[str]:
        """
        Read the caption for an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Caption text if it exists, None otherwise.

        Raises:
            FileNotFoundError: If the image file does not exist.
            UnsupportedFileTypeError: If the image file has an unsupported extension.
        """
        self.validate_image_file(image_path)
        img_path = Path(image_path)
        caption_file = img_path.with_suffix(".txt")

        if caption_file.exists() and caption_file.is_file():
            try:
                return caption_file.read_text(encoding="utf-8").strip() or None
            except Exception as e:
                logger.warning(f"Error reading caption file {caption_file}: {e}")
                return None
        return None

    def get_supported_extensions(self) -> Set[str]:
        """Return supported image file extensions."""
        return self.SUPPORTED_EXTENSIONS

    def validate_image_file(self, image_path: str) -> None:
        """
        Validate that the image file is supported and exists.

        Args:
            image_path: Path to the image file.

        Raises:
            FileNotFoundError: If the image file does not exist.
            UnsupportedFileTypeError: If the image file has an unsupported extension.
        """
        img_path = Path(image_path)
        # Check extension first, then existence
        if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported image extension '{img_path.suffix}'."
            )
        if not img_path.is_file():
            raise FileNotFoundError(f"{image_path} is not a file.")
