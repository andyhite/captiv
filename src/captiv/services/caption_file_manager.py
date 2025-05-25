from pathlib import Path

from captiv.services.exceptions import FileOperationError
from captiv.services.file_manager import FileManager
from captiv.services.image_file_manager import ImageFileManager


class CaptionFileManager:
    def __init__(self, file_manager: FileManager, image_file_manager: ImageFileManager):
        self.file_manager = file_manager or FileManager()
        self.image_file_manager = image_file_manager or ImageFileManager(file_manager)

    def list_images_and_captions(
        self, directory: Path
    ) -> list[tuple[Path, str | None]]:
        """
        List all image files in a directory along with their captions.

        Args:
            directory: Path to the directory containing image files.

        Returns:
            List of tuples containing image file paths and their corresponding captions.
        """
        images = self.image_file_manager.list_image_files(directory)
        return [(image, self.read_caption(image)) for image in images]

    def get_caption_file_path(self, image_file: Path) -> Path:
        """
        Get the caption file path corresponding to an image file.

        Args:
            image_file: Path to the image file.

        Returns:
            Path to the caption file for the image file
        """
        return Path(image_file).with_suffix(".txt")

    def read_caption(self, image_file: Path) -> str:
        """
        Read the caption from a file corresponding to the image file.

        Args:
            image_file: Path to the image file.

        Returns:
            Caption text.
        """
        caption_file = self.get_caption_file_path(image_file)

        try:
            return self.file_manager.read_file(caption_file)
        except Exception:
            return ""

    def write_caption(self, image_file: Path, caption: str) -> None:
        """
        Write a caption to a file corresponding to the image file.

        Args:
            image_file: Path to the image file.
            caption: Caption text to write.

        Raises:
            FileOperationError: If writing the caption fails.
        """
        caption_file = self.get_caption_file_path(image_file)

        try:
            self.file_manager.write_file(caption_file, caption)
        except Exception as e:
            raise FileOperationError(f"Failed to write caption: {e}") from None

    def delete_caption(self, image_file: Path) -> None:
        """
        Delete the caption file corresponding to the image file.

        Args:
            image_file: Path to the image file.

        Raises:
            FileOperationError: If deleting the caption fails.
        """
        caption_file = self.get_caption_file_path(image_file)

        try:
            self.file_manager.delete_file(caption_file)
        except Exception as e:
            raise FileOperationError(f"Failed to delete caption: {e}") from None

    def clear_captions(self, directory: Path) -> None:
        """
        Clear all caption files in the specified directory.

        Args:
            directory: Path to the directory containing caption files.

        Raises:
            FileOperationError: If clearing captions fails.
        """
        try:
            for image_file in self.image_file_manager.list_image_files(directory):
                self.delete_caption(image_file)
        except Exception as e:
            raise FileOperationError(f"Failed to clear captions: {e}") from None
