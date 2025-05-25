from pathlib import Path

from captiv.services.exceptions import DirectoryNotFoundError, FileOperationError


class FileManager:
    def list_files(self, directory: Path, extensions: set[str] | None) -> list[Path]:
        """
        List files in a directory with specified extensions.

        Args:
            directory: Path to the directory containing files.
            extensions: Set of file extensions to filter.

        Returns:
            List of file paths matching the specified extensions.

        Raises:
            DirectoryNotFoundError: If the directory does not exist.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise DirectoryNotFoundError(f"{directory} is not a directory.")

        files = sorted(
            [
                f
                for f in dir_path.iterdir()
                if f.is_file()
                and (extensions is None or f.suffix.lower() in extensions)
            ]
        )
        return files

    def read_file(self, file_path: Path) -> str:
        """
        Read the content of a file.

        Args:
            file_path: Path to the file.

        Returns:
            Content of the file as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file = Path(file_path)
        if not file.is_file():
            raise FileNotFoundError(f"{file_path} is not a file.")

        try:
            return file.read_text(encoding="utf-8").strip()
        except Exception as e:
            raise FileOperationError(f"Failed to read file: {e}") from None

    def write_file(self, file_path: Path, content: str) -> None:
        """
        Write content to a file.

        Args:
            file_path: Path to the file.
            content: Content to write to the file.

        Raises:
            FileOperationError: If there was an error writing to the file.
        """
        file = Path(file_path)
        try:
            file.write_text(content, encoding="utf-8")
        except Exception as e:
            raise FileOperationError(f"Failed to write file: {e}") from None

    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file was deleted, False if it didn't exist.

        Raises:
            FileOperationError: If there was an error deleting the file.
        """
        file = Path(file_path)
        if file.exists():
            if file.is_file():
                try:
                    file.unlink()
                    return True
                except Exception as e:
                    raise FileOperationError(f"Failed to delete file: {e}") from None
            else:
                return False
        return False
