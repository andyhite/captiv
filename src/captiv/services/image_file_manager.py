from pathlib import Path

from captiv.services.file_manager import FileManager


class ImageFileManager:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    def list_image_files(self, directory: Path) -> list[Path]:
        """
        List all image files in a directory.

        Args:
            directory: Path to the directory containing image files.

        Returns:
            List of image file paths.
        """
        return self.file_manager.list_files(directory, self.SUPPORTED_EXTENSIONS)
