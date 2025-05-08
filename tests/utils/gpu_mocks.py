"""
GPU mocking utilities for testing.

This module provides utilities for mocking GPU-related functionality
to ensure tests can run on platforms without GPUs.
"""

import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


def is_package_installed(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


@contextmanager
def mock_torch_cuda():
    """
    Mock torch.cuda to simulate a CUDA-enabled environment.

    This context manager mocks torch.cuda to make it appear as if CUDA is available,
    even on platforms without GPUs.
    """
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_name", return_value="Test GPU"),
        patch("torch.cuda.empty_cache"),
        patch("torch.cuda.memory_allocated", return_value=0),
        patch("torch.cuda.memory_reserved", return_value=0),
    ):
        yield


@contextmanager
def mock_torch_mps():
    """
    Mock torch.mps to simulate an MPS-enabled environment (Apple Silicon).

    This context manager mocks torch.mps to make it appear as if MPS is available,
    even on platforms without Apple Silicon.
    """
    # Check if torch.mps exists (only in newer PyTorch versions)
    if hasattr(importlib.import_module("torch"), "mps"):
        # Only mock is_available, which is the only attribute we need
        with patch("torch.mps.is_available", return_value=True):
            yield
    else:
        # If torch.mps doesn't exist, just yield
        yield


@contextmanager
def mock_accelerate():
    """
    Mock the accelerate package for testing.

    This context manager mocks the accelerate package to make it appear as if it's installed,
    even if it's not.
    """
    # Check if accelerate is already installed
    if is_package_installed("accelerate"):
        # If it's installed, no need to mock
        yield
    else:
        # If it's not installed, mock it
        mock_accelerate_module = MagicMock()
        mock_accelerate_module.__version__ = "0.20.0"

        # Mock the infer_auto_device_map function
        mock_accelerate_module.infer_auto_device_map = MagicMock(
            return_value={"model": "cpu"}
        )

        # Mock the init_empty_weights context manager
        mock_init_empty_weights = MagicMock()
        mock_init_empty_weights.__enter__ = MagicMock(return_value=None)
        mock_init_empty_weights.__exit__ = MagicMock(return_value=None)
        mock_accelerate_module.init_empty_weights = MagicMock(
            return_value=mock_init_empty_weights
        )

        # Mock the load_checkpoint_and_dispatch function
        mock_accelerate_module.load_checkpoint_and_dispatch = MagicMock(
            return_value=MagicMock()
        )

        # Add the mock to sys.modules
        with patch.dict(sys.modules, {"accelerate": mock_accelerate_module}):
            yield


@contextmanager
def mock_gpu_environment():
    """
    Mock the entire GPU environment for testing.

    This context manager mocks torch.cuda, torch.mps, and the accelerate package
    to simulate a GPU-enabled environment, even on platforms without GPUs.
    """
    with mock_torch_cuda(), mock_torch_mps(), mock_accelerate():
        yield


class GPUTestSkip:
    """
    Decorator class for skipping tests that require a GPU.

    This class provides decorators for skipping tests that require a GPU
    when running on platforms without GPUs.
    """

    @staticmethod
    def skip_if_no_gpu(func):
        """
        Skip a test if no GPU is available.

        Args:
            func: The test function to decorate.

        Returns:
            The decorated function.
        """
        import torch

        # Check if CUDA or MPS is available
        has_gpu = torch.cuda.is_available()
        if hasattr(torch, "mps"):
            has_gpu = has_gpu or torch.mps.is_available()

        # Skip the test if no GPU is available
        return pytest.mark.skipif(not has_gpu, reason="Test requires a GPU")(func)

    @staticmethod
    def skip_if_no_cuda(func):
        """
        Skip a test if CUDA is not available.

        Args:
            func: The test function to decorate.

        Returns:
            The decorated function.
        """
        import torch

        # Skip the test if CUDA is not available
        return pytest.mark.skipif(
            not torch.cuda.is_available(), reason="Test requires CUDA"
        )(func)

    @staticmethod
    def skip_if_no_mps(func):
        """
        Skip a test if MPS is not available.

        Args:
            func: The test function to decorate.

        Returns:
            The decorated function.
        """
        import torch

        # Check if MPS is available
        has_mps = False
        if hasattr(torch, "mps"):
            has_mps = torch.mps.is_available()

        # Skip the test if MPS is not available
        return pytest.mark.skipif(
            not has_mps, reason="Test requires MPS (Apple Silicon)"
        )(func)

    @staticmethod
    def skip_if_no_accelerate(func):
        """
        Skip a test if the accelerate package is not installed.

        Args:
            func: The test function to decorate.

        Returns:
            The decorated function.
        """
        # Skip the test if accelerate is not installed
        return pytest.mark.skipif(
            not is_package_installed("accelerate"),
            reason="Test requires the accelerate package",
        )(func)


# Create instances of the decorators for easier use
skip_if_no_gpu = GPUTestSkip.skip_if_no_gpu
skip_if_no_cuda = GPUTestSkip.skip_if_no_cuda
skip_if_no_mps = GPUTestSkip.skip_if_no_mps
skip_if_no_accelerate = GPUTestSkip.skip_if_no_accelerate


# Fixture for mocking the GPU environment
@pytest.fixture
def mock_gpu():
    """Fixture for mocking the GPU environment."""
    with mock_gpu_environment():
        yield


# Fixture for mocking torch.cuda
@pytest.fixture
def mock_cuda():
    """Fixture for mocking torch.cuda."""
    with mock_torch_cuda():
        yield


# Fixture for mocking torch.mps
@pytest.fixture
def mock_mps():
    """Fixture for mocking torch.mps."""
    with mock_torch_mps():
        yield


# Fixture for mocking the accelerate package
@pytest.fixture
def mock_accelerate_package():
    """Fixture for mocking the accelerate package."""
    with mock_accelerate():
        yield
