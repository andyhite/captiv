"""
Pytest configuration file for the Captiv tests.

This file contains fixtures that are available to all tests.
"""

import pytest

from tests.utils.gpu_mocks import (
    mock_accelerate,
    mock_gpu_environment,
    mock_torch_cuda,
    mock_torch_mps,
)

# Re-export the fixtures from gpu_mocks.py
# This makes them available to all tests without having to import them explicitly


# Fixture for mocking the entire GPU environment
@pytest.fixture
def mock_gpu_env():
    """Fixture for mocking the entire GPU environment."""
    with mock_gpu_environment():
        yield


# Fixture for mocking the GPU
@pytest.fixture
def mock_gpu():
    """Fixture for mocking the GPU environment."""
    with mock_gpu_environment():
        yield


# Fixture for mocking CUDA
@pytest.fixture
def mock_cuda():
    """Fixture for mocking torch.cuda."""
    with mock_torch_cuda():
        yield


# Fixture for mocking MPS
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
