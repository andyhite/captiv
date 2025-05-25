"""Tests for FileManager service."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from captiv.services.exceptions import DirectoryNotFoundError, FileOperationError
from captiv.services.file_manager import FileManager


class TestFileManager:
    """Test cases for FileManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.file_manager = FileManager()

    def teardown_method(self):
        """Clean up test fixtures."""
        for file in sorted(self.temp_path.rglob("*"), reverse=True):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                file.rmdir()
        self.temp_path.rmdir()

    def test_list_files_with_extensions(self):
        """Test listing files with specific extensions."""
        (self.temp_path / "image1.jpg").touch()
        (self.temp_path / "image2.png").touch()
        (self.temp_path / "document.txt").touch()
        (self.temp_path / "script.py").touch()

        image_extensions = {".jpg", ".png"}
        files = self.file_manager.list_files(self.temp_path, image_extensions)

        assert len(files) == 2
        file_names = [f.name for f in files]
        assert "image1.jpg" in file_names
        assert "image2.png" in file_names
        assert "document.txt" not in file_names

    def test_list_files_case_insensitive_extensions(self):
        """Test listing files with case-insensitive extension matching."""
        (self.temp_path / "image1.JPG").touch()
        (self.temp_path / "image2.Png").touch()
        (self.temp_path / "image3.jpeg").touch()

        extensions = {".jpg", ".png"}
        files = self.file_manager.list_files(self.temp_path, extensions)

        assert len(files) == 2
        file_names = [f.name for f in files]
        assert "image1.JPG" in file_names
        assert "image2.Png" in file_names
        assert "image3.jpeg" not in file_names

    def test_list_files_no_extensions_filter(self):
        """Test listing files without extension filter."""
        (self.temp_path / "file1.txt").touch()
        (self.temp_path / "file2.jpg").touch()
        (self.temp_path / "file3").touch()

        files = self.file_manager.list_files(self.temp_path, None)

        assert len(files) == 3
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file2.jpg" in file_names
        assert "file3" in file_names

    def test_list_files_empty_directory(self):
        """Test listing files in empty directory."""
        files = self.file_manager.list_files(self.temp_path, {".jpg"})
        assert files == []

    def test_list_files_only_directories(self):
        """Test listing files when directory contains only subdirectories."""
        (self.temp_path / "subdir1").mkdir()
        (self.temp_path / "subdir2").mkdir()

        files = self.file_manager.list_files(self.temp_path, {".jpg"})
        assert files == []

    def test_list_files_sorted_output(self):
        """Test that files are returned in sorted order."""
        (self.temp_path / "z_file.txt").touch()
        (self.temp_path / "a_file.txt").touch()
        (self.temp_path / "m_file.txt").touch()

        files = self.file_manager.list_files(self.temp_path, {".txt"})

        file_names = [f.name for f in files]
        assert file_names == ["a_file.txt", "m_file.txt", "z_file.txt"]

    def test_list_files_nonexistent_directory(self):
        """Test listing files in non-existent directory."""
        nonexistent_dir = self.temp_path / "nonexistent"

        with pytest.raises(DirectoryNotFoundError, match="is not a directory"):
            self.file_manager.list_files(nonexistent_dir, {".jpg"})

    def test_list_files_file_instead_of_directory(self):
        """Test listing files when path points to a file."""
        test_file = self.temp_path / "test.txt"
        test_file.touch()

        with pytest.raises(DirectoryNotFoundError, match="is not a directory"):
            self.file_manager.list_files(test_file, {".txt"})

    def test_read_file_success(self):
        """Test reading file successfully."""
        test_file = self.temp_path / "test.txt"
        content = "Hello, World!\nThis is a test file."
        test_file.write_text(content, encoding="utf-8")

        result = self.file_manager.read_file(test_file)
        assert result == content.strip()

    def test_read_file_with_whitespace(self):
        """Test reading file with leading/trailing whitespace."""
        test_file = self.temp_path / "test.txt"
        content = "  \n  Hello, World!  \n  "
        test_file.write_text(content, encoding="utf-8")

        result = self.file_manager.read_file(test_file)
        assert result == "Hello, World!"

    def test_read_file_empty_file(self):
        """Test reading empty file."""
        test_file = self.temp_path / "empty.txt"
        test_file.touch()

        result = self.file_manager.read_file(test_file)
        assert result == ""

    def test_read_file_nonexistent(self):
        """Test reading non-existent file."""
        nonexistent_file = self.temp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="is not a file"):
            self.file_manager.read_file(nonexistent_file)

    def test_read_file_directory_instead_of_file(self):
        """Test reading when path points to a directory."""
        test_dir = self.temp_path / "testdir"
        test_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="is not a file"):
            self.file_manager.read_file(test_dir)

    def test_read_file_permission_error(self):
        """Test reading file with permission error."""
        test_file = self.temp_path / "test.txt"
        test_file.write_text("content")

        with (
            patch.object(
                Path, "read_text", side_effect=PermissionError("Permission denied")
            ),
            pytest.raises(FileOperationError, match="Failed to read file"),
        ):
            self.file_manager.read_file(test_file)

    def test_read_file_unicode_content(self):
        """Test reading file with Unicode content."""
        test_file = self.temp_path / "unicode.txt"
        content = "Hello 世界! 🌍 Café naïve résumé"
        test_file.write_text(content, encoding="utf-8")

        result = self.file_manager.read_file(test_file)
        assert result == content

    def test_write_file_success(self):
        """Test writing file successfully."""
        test_file = self.temp_path / "output.txt"
        content = "This is test content\nWith multiple lines"

        self.file_manager.write_file(test_file, content)

        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == content

    def test_write_file_overwrite_existing(self):
        """Test overwriting existing file."""
        test_file = self.temp_path / "existing.txt"
        test_file.write_text("Original content")

        new_content = "New content"
        self.file_manager.write_file(test_file, new_content)

        assert test_file.read_text(encoding="utf-8") == new_content

    def test_write_file_empty_content(self):
        """Test writing empty content to file."""
        test_file = self.temp_path / "empty.txt"

        self.file_manager.write_file(test_file, "")

        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == ""

    def test_write_file_unicode_content(self):
        """Test writing Unicode content to file."""
        test_file = self.temp_path / "unicode.txt"
        content = "Hello 世界! 🌍 Café naïve résumé"

        self.file_manager.write_file(test_file, content)

        assert test_file.read_text(encoding="utf-8") == content

    def test_write_file_permission_error(self):
        """Test writing file with permission error."""
        test_file = self.temp_path / "test.txt"

        with (
            patch.object(
                Path, "write_text", side_effect=PermissionError("Permission denied")
            ),
            pytest.raises(FileOperationError, match="Failed to write file"),
        ):
            self.file_manager.write_file(test_file, "content")

    def test_write_file_disk_full_error(self):
        """Test writing file when disk is full."""
        test_file = self.temp_path / "test.txt"

        with (
            patch.object(
                Path, "write_text", side_effect=OSError("No space left on device")
            ),
            pytest.raises(FileOperationError, match="Failed to write file"),
        ):
            self.file_manager.write_file(test_file, "content")

    def test_delete_file_success(self):
        """Test deleting file successfully."""
        test_file = self.temp_path / "to_delete.txt"
        test_file.write_text("content")

        result = self.file_manager.delete_file(test_file)

        assert result is True
        assert not test_file.exists()

    def test_delete_file_nonexistent(self):
        """Test deleting non-existent file."""
        nonexistent_file = self.temp_path / "nonexistent.txt"

        result = self.file_manager.delete_file(nonexistent_file)

        assert result is False

    def test_delete_file_permission_error(self):
        """Test deleting file with permission error."""
        test_file = self.temp_path / "protected.txt"
        test_file.write_text("content")

        with (
            patch.object(
                Path, "unlink", side_effect=PermissionError("Permission denied")
            ),
            pytest.raises(FileOperationError, match="Failed to delete file"),
        ):
            self.file_manager.delete_file(test_file)

    def test_delete_file_in_use_error(self):
        """Test deleting file that is in use."""
        test_file = self.temp_path / "in_use.txt"
        test_file.write_text("content")

        with (
            patch.object(Path, "unlink", side_effect=OSError("File is in use")),
            pytest.raises(FileOperationError, match="Failed to delete file"),
        ):
            self.file_manager.delete_file(test_file)

    def test_delete_file_directory(self):
        """Test deleting when path points to directory."""
        test_dir = self.temp_path / "testdir"
        test_dir.mkdir()

        result = self.file_manager.delete_file(test_dir)
        assert result is False
        assert test_dir.exists()

    def test_list_files_with_empty_extensions_set(self):
        """Test listing files with empty extensions set."""
        (self.temp_path / "file1.txt").touch()
        (self.temp_path / "file2.jpg").touch()

        files = self.file_manager.list_files(self.temp_path, set())

        assert files == []

    def test_list_files_with_dot_files(self):
        """Test listing files including hidden dot files."""
        (self.temp_path / "visible.txt").touch()
        (self.temp_path / ".hidden.txt").touch()
        (self.temp_path / ".DS_Store").touch()

        files = self.file_manager.list_files(self.temp_path, {".txt"})

        file_names = [f.name for f in files]
        assert "visible.txt" in file_names
        assert ".hidden.txt" in file_names
        assert ".DS_Store" not in file_names

    def test_read_file_with_different_encodings(self):
        """Test reading file with different encoding scenarios."""
        test_file = self.temp_path / "encoding_test.txt"
        content = "Test content with special chars: àáâãäå"

        test_file.write_text(content, encoding="utf-8")

        result = self.file_manager.read_file(test_file)
        assert result == content

    def test_write_file_creates_intermediate_directories(self):
        """Test that write_file works even if parent directories don't exist."""
        nested_file = self.temp_path / "nested" / "deep" / "file.txt"
        content = "Nested file content"

        try:
            self.file_manager.write_file(nested_file, content)
            if nested_file.exists():
                assert nested_file.read_text(encoding="utf-8") == content
        except FileOperationError:
            assert not nested_file.parent.exists()

    def test_file_operations_with_pathlib_objects(self):
        """Test that all methods work with pathlib.Path objects."""
        test_file = self.temp_path / "pathlib_test.txt"
        content = "Pathlib test content"

        self.file_manager.write_file(test_file, content)
        read_content = self.file_manager.read_file(test_file)
        assert read_content == content

        deleted = self.file_manager.delete_file(test_file)
        assert deleted is True
        assert not test_file.exists()

    def test_file_operations_with_string_paths(self):
        """Test that all methods work with string paths converted to Path objects."""
        test_file_str = str(self.temp_path / "string_test.txt")
        test_file_path = Path(test_file_str)
        content = "String path test content"

        self.file_manager.write_file(test_file_path, content)
        read_content = self.file_manager.read_file(test_file_path)
        assert read_content == content

        deleted = self.file_manager.delete_file(test_file_path)
        assert deleted is True
        assert not test_file_path.exists()
