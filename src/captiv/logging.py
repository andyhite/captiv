"""
Logging configuration for the Captiv application.

This module sets up Loguru as the logging solution for the application and
provides a way to intercept standard logging messages and route them through Loguru.
"""

import logging
import sys
from typing import List, Optional

from loguru import logger

# Remove default handler
logger.remove()

# Add a handler that writes to stderr with a specific format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",  # Changed from INFO to DEBUG
    colorize=True,
)


class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging messages and routes them through Loguru.

    This handler intercepts standard logging messages from libraries like Gradio
    and routes them through Loguru for consistent logging.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercept logging messages and pass them to Loguru.

        Args:
            record: The logging record to intercept.
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Simply log the message without calculating depth
        # This prevents the "logging.callHandlers:1762" format from appearing
        logger.opt(exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    level: str = "INFO", intercept_libraries: Optional[List[str]] = None
) -> None:
    """
    Set up logging for the application.

    Args:
        level: The logging level to use.
        intercept_libraries: A list of library names to intercept logging from.
    """
    # Set Loguru level
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": (
                    "DEBUG" if level == "INFO" else level
                ),  # Default to DEBUG if INFO is passed, otherwise use specified level
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
                "colorize": True,
            }
        ]
    )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Intercept specific libraries if requested
    if intercept_libraries:
        for lib in intercept_libraries:
            lib_logger = logging.getLogger(lib)
            lib_logger.handlers = [InterceptHandler()]
            lib_logger.propagate = False
            lib_logger.level = 0


# Export the logger for use in other modules
__all__ = ["logger", "setup_logging"]
