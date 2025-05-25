"""Utility functions for the Captiv GUI."""

import os


def is_image_file(path: str) -> bool:
    """Check if a file is an image based on its extension."""
    valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    return any(path.lower().endswith(ext) for ext in valid_extensions)


def get_subdirectories(directory: str) -> list[str]:
    """Get subdirectories of the given directory."""
    try:
        return sorted(
            [
                d
                for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d)) and not d.startswith(".")
            ]
        )
    except Exception:
        return []


def normalize_path(path_or_dict) -> str:
    """Normalize a path that might be a string or a dict with a 'value' key."""
    if isinstance(path_or_dict, dict) and "value" in path_or_dict:
        return str(path_or_dict["value"])
    return str(path_or_dict)
