"""
Progress indicator utilities for both CLI and GUI interfaces.
"""

import sys
import time
from typing import Any, Callable, Iterator, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


class ProgressTracker:
    """
    A progress tracker that can be used by both CLI and GUI interfaces.

    This class provides a consistent way to track and report progress
    for long-running operations across different interfaces.
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        callback: Optional[Callable[[int, int, str], Any]] = None,
    ):
        """
        Initialize the progress tracker.

        Args:
            total: Total number of items to process
            description: Description of the operation
            callback: Optional callback function to call with progress updates
                     Function signature: callback(current, total, status_message)
        """
        self.total = total
        self.current = 0
        self.description = description
        self.callback = callback
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval = 0.1  # seconds

    def update(self, increment: int = 1, status: Optional[str] = None) -> None:
        """
        Update the progress.

        Args:
            increment: Number of items to increment by
            status: Optional status message
        """
        self.current += increment
        current_time = time.time()

        # Only update if enough time has passed since the last update
        # This prevents too frequent updates that could slow down processing
        if (current_time - self.last_update_time) >= self.update_interval:
            self.last_update_time = current_time

            # Calculate percentage and elapsed time
            percent = min(100, int(100 * self.current / self.total))
            current_time - self.start_time

            # Create status message
            if status:
                status_msg = f"{self.description}: {percent}% complete - {status}"
            else:
                status_msg = f"{self.description}: {percent}% complete"

            # Call the callback if provided
            if self.callback:
                self.callback(self.current, self.total, status_msg)

            logger.debug(
                f"Progress update: {self.current}/{self.total} ({percent}%) - {status_msg}"
            )

    def complete(self, status: str = "Complete") -> None:
        """
        Mark the progress as complete.

        Args:
            status: Status message for completion
        """
        self.current = self.total
        elapsed = time.time() - self.start_time

        status_msg = f"{self.description}: 100% complete - {status} (in {elapsed:.2f}s)"

        if self.callback:
            self.callback(self.total, self.total, status_msg)

        logger.info(status_msg)


def cli_progress_callback(current: int, total: int, status_msg: str) -> None:
    """
    Default progress callback for CLI interface.

    Args:
        current: Current progress
        total: Total items
        status_msg: Status message
    """
    percent = min(100, int(100 * current / total))
    bar_length = 30
    filled_length = int(bar_length * current / total)

    bar = "█" * filled_length + "░" * (bar_length - filled_length)

    # Use carriage return to update the same line
    sys.stdout.write(f"\r{status_msg} [{bar}] {percent}%")
    sys.stdout.flush()

    # Print a newline when complete
    if current >= total:
        sys.stdout.write("\n")


def track_progress(
    iterable: Iterator[T],
    total: int,
    description: str = "Processing",
    callback: Optional[Callable[[int, int, str], Any]] = None,
) -> Iterator[T]:
    """
    Wrap an iterable with progress tracking.

    Args:
        iterable: The iterable to track progress for
        total: Total number of items
        description: Description of the operation
        callback: Optional callback function for progress updates

    Yields:
        Items from the original iterable
    """
    tracker = ProgressTracker(total, description, callback)

    for item in iterable:
        yield item
        tracker.update()

    tracker.complete()
