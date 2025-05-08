import io
import os

import numpy as np
from PIL import Image


def create_test_image(width=100, height=100, color=(255, 0, 0)):
    """
    Create a simple PIL Image for testing purposes.

    Args:
        width: Width of the test image
        height: Height of the test image
        color: RGB color tuple for the image

    Returns:
        A PIL Image object
    """
    # Create a solid color image
    img_array = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(img_array)


def get_test_image_path():
    """
    Returns the path to a test image file that exists in the test directory.

    Returns:
        str: Path to a test image file
    """
    # Create a test directory if it doesn't exist
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_images_dir = os.path.join(test_dir, "test_images")
    os.makedirs(test_images_dir, exist_ok=True)

    # Create a test image file if it doesn't exist
    test_image_path = os.path.join(test_images_dir, "test_image.jpg")
    if not os.path.exists(test_image_path):
        img = create_test_image()
        img.save(test_image_path)

    return test_image_path


def get_test_image_file():
    """
    Returns a file-like object containing a test image.

    Returns:
        io.BytesIO: A file-like object containing a JPEG image
    """
    img = create_test_image()
    img_file = io.BytesIO()
    img.save(img_file, format="JPEG")
    img_file.seek(0)
    return img_file
