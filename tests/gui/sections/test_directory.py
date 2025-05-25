"""Tests for the captiv.gui.sections.directory module."""

from pathlib import Path
from unittest.mock import Mock, patch


def test_directory_module_imports():
    """Test that the directory module can be imported."""


def test_directory_section_class_exists():
    """Test that DirectorySection class exists and can be imported."""
    from captiv.gui.sections.directory import DirectorySection

    assert isinstance(DirectorySection, type)


@patch("captiv.gui.sections.directory.Path.home")
def test_directory_section_init_default(mock_home):
    """Test DirectorySection initialization with default directory."""
    from captiv.gui.sections.directory import DirectorySection

    mock_home.return_value = Path("/home/user")

    section = DirectorySection()

    assert section.current_directory == "/home/user"
    assert section.dir_dropdown is None


def test_directory_section_init_custom():
    """Test DirectorySection initialization with custom directory."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection(default_directory="/custom/path")

    assert section.current_directory == "/custom/path"
    assert section.dir_dropdown is None


@patch("captiv.gui.sections.directory.gr")
def test_directory_section_create_section(mock_gr):
    """Test DirectorySection create_section method."""
    from captiv.gui.sections.directory import DirectorySection

    mock_dropdown = Mock()
    mock_gr.Dropdown.return_value = mock_dropdown

    section = DirectorySection(default_directory="/test/path")

    with patch.object(
        section, "_get_dir_options", return_value=["/test/path", "/test"]
    ):
        result = section.create_section()

    assert result is mock_dropdown
    assert section.dir_dropdown is mock_dropdown

    mock_gr.Dropdown.assert_called_once_with(
        label="Select Image Directory",
        interactive=True,
        choices=["/test/path", "/test"],
        value="/test/path",
        container=False,
        scale=0,
        elem_classes="directory-selector",
    )


@patch("captiv.gui.sections.directory.get_subdirectories")
@patch("captiv.gui.sections.directory.os.path.dirname")
@patch("captiv.gui.sections.directory.os.path.abspath")
def test_directory_section_get_dir_options(
    mock_abspath, mock_dirname, mock_get_subdirs
):
    """Test _get_dir_options method."""
    from captiv.gui.sections.directory import DirectorySection

    mock_abspath.return_value = "/absolute/test/path"
    mock_dirname.return_value = "/absolute/test"
    mock_get_subdirs.return_value = ["subdir1", "subdir2"]

    with patch(
        "captiv.gui.sections.directory.os.path.join",
        side_effect=lambda *args: "/".join(args),
    ):
        section = DirectorySection()

        result = section._get_dir_options("/test/path")

    expected = [
        "/absolute/test",
        "/absolute/test/path",
        "/absolute/test/path/subdir1",
        "/absolute/test/path/subdir2",
    ]
    assert result == expected


def test_directory_section_on_directory_change_valid():
    """Test on_directory_change with valid directory."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection()

    with (
        patch("captiv.gui.sections.directory.os.path.isdir", return_value=True),
        patch(
            "captiv.gui.sections.directory.get_subdirectories",
            return_value=["sub1", "sub2"],
        ),
    ):
        directory, subdirs = section.on_directory_change("/valid/directory")

    assert directory == "/valid/directory"
    assert subdirs == ["sub1", "sub2"]
    assert section.current_directory == "/valid/directory"


def test_directory_section_on_directory_change_invalid():
    """Test on_directory_change with invalid directory."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection(default_directory="/default/path")

    with (
        patch("captiv.gui.sections.directory.os.path.isdir", return_value=False),
        patch(
            "captiv.gui.sections.directory.get_subdirectories",
            return_value=["default_sub"],
        ),
    ):
        directory, subdirs = section.on_directory_change("/invalid/directory")

    assert directory == "/default/path"
    assert subdirs == ["default_sub"]


def test_directory_section_on_parent_directory():
    """Test on_parent_directory method."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection(default_directory="/test/current/path")

    with (
        patch("captiv.gui.sections.directory.Path") as mock_path,
        patch.object(
            section, "on_directory_change", return_value=("/test/current", ["sub"])
        ) as mock_change,
    ):
        mock_path.return_value.parent = "/test/current"

        result = section.on_parent_directory()

    assert result == ("/test/current", ["sub"])
    mock_change.assert_called_once_with("/test/current")


def test_directory_section_on_subdirectory_select_valid():
    """Test on_subdirectory_select with valid subdirectory."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection(default_directory="/test/current")

    with (
        patch(
            "captiv.gui.sections.directory.os.path.join",
            return_value="/test/current/subdir",
        ),
        patch.object(
            section, "on_directory_change", return_value=("/test/current/subdir", [])
        ) as mock_change,
    ):
        result = section.on_subdirectory_select("subdir")

    assert result == ("/test/current/subdir", [])
    mock_change.assert_called_once_with("/test/current/subdir")


def test_directory_section_on_subdirectory_select_empty():
    """Test on_subdirectory_select with empty subdirectory."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection(default_directory="/test/current")

    with patch(
        "captiv.gui.sections.directory.get_subdirectories", return_value=["sub1"]
    ):
        result = section.on_subdirectory_select("")

    assert result == ("/test/current", ["sub1"])


@patch("captiv.gui.sections.directory.gr")
def test_directory_section_handle_dir_change(mock_gr):
    """Test handle_dir_change method."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection()

    with (
        patch(
            "captiv.gui.sections.directory.normalize_path",
            return_value="/normalized/path",
        ),
        patch.object(
            section, "_get_dir_options", return_value=["/normalized/path", "/parent"]
        ),
    ):
        update, path = section.handle_dir_change("/some/path")

    assert path == "/normalized/path"
    assert section.current_directory == "/normalized/path"

    mock_gr.update.assert_called_once_with(
        choices=["/normalized/path", "/parent"], value="/normalized/path"
    )


@patch("captiv.gui.sections.directory.gr")
def test_directory_section_handle_dir_change_same_value(mock_gr):
    """Test handle_dir_change when value hasn't changed."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection()

    section.dir_dropdown = Mock()
    section.dir_dropdown.value = "/same/path"

    with (
        patch(
            "captiv.gui.sections.directory.normalize_path", return_value="/same/path"
        ),
        patch.object(
            section, "_get_dir_options", return_value=["/same/path", "/parent"]
        ),
    ):
        update, path = section.handle_dir_change("/same/path")

    assert path == "/same/path"

    mock_gr.update.assert_called_once_with(choices=["/same/path", "/parent"])


def test_directory_section_get_current_directory():
    """Test get_current_directory method."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection(default_directory="/test/directory")

    result = section.get_current_directory()

    assert result == "/test/directory"


def test_directory_section_imports_and_dependencies():
    """Test that all required imports and dependencies are available."""
    from captiv.gui.sections.directory import (
        DirectorySection,
        get_subdirectories,
        gr,
        logger,
        normalize_path,
    )

    assert DirectorySection is not None
    assert gr is not None
    assert logger is not None
    assert get_subdirectories is not None
    assert normalize_path is not None


def test_directory_section_method_signatures():
    """Test that methods have the expected signatures."""
    import inspect

    from captiv.gui.sections.directory import DirectorySection

    init_sig = inspect.signature(DirectorySection.__init__)
    init_params = list(init_sig.parameters.keys())
    assert "self" in init_params
    assert "default_directory" in init_params

    create_sig = inspect.signature(DirectorySection.create_section)
    create_params = list(create_sig.parameters.keys())
    assert "self" in create_params

    handle_sig = inspect.signature(DirectorySection.handle_dir_change)
    handle_params = list(handle_sig.parameters.keys())
    assert "self" in handle_params
    assert "selected_dir" in handle_params


def test_directory_section_exception_handling():
    """Test that DirectorySection handles exceptions properly."""
    from captiv.gui.sections.directory import DirectorySection

    section = DirectorySection(default_directory="/test/current")

    with (
        patch(
            "captiv.gui.sections.directory.os.path.isdir",
            side_effect=Exception("Unexpected error"),
        ),
        patch(
            "captiv.gui.sections.directory.get_subdirectories", return_value=["sub1"]
        ),
    ):
        directory, subdirs = section.on_directory_change("/problematic/directory")

    assert directory == "/test/current"
    assert subdirs == ["sub1"]
