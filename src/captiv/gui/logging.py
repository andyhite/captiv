"""
Logging configuration for the Captiv application.

This module sets up Loguru as the logging solution for the application and provides a
way to intercept standard logging messages and route them through Loguru.
"""

import logging
import sys

from loguru import logger

logger.remove()

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",  # noqa: E501
    level="INFO",
    colorize=True,
)


class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging messages and routes them through Loguru.

    This handler intercepts standard logging messages from libraries like Gradio and
    routes them through Loguru for consistent logging.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercept logging messages and pass them to Loguru.

        Args:
            record: The logging record to intercept.
        """
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    level: str = "INFO", intercept_libraries: list[str] | None = None
) -> None:
    """
    Set up logging for the application.

    Args:
        level: The logging level to use.
        intercept_libraries: A list of library names to intercept logging from.
    """
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": level,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",  # noqa: E501
                "colorize": True,
            }
        ]
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    if intercept_libraries:
        for lib in intercept_libraries:
            lib_logger = logging.getLogger(lib)
            lib_logger.handlers = [InterceptHandler()]
            lib_logger.propagate = False
            lib_logger.level = 0


__all__ = ["logger", "setup_logging"]
