"""
Loguru-based structured logger factory.

Usage:
    from src.common.logging import setup_logger, get_logger

    setup_logger(level="INFO", log_file=Path("logs/app.log"))
    log = get_logger(__name__)
    log.info("Starting pipeline")
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger


def setup_logger(
    level: str = "INFO",
    log_file: Path | None = None,
    serialize: bool = False,
) -> None:
    """
    Configure Loguru globally.

    Args:
        level:      Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file:   Optional path for rotating file handler.
        serialize:  If True, file output is structured JSON.
    """
    _logger.remove()  # Remove Loguru's default stderr handler

    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    _logger.add(
        sys.stderr,
        format=console_fmt,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(log_file),
            format=console_fmt if not serialize else "{message}",
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            serialize=serialize,
            enqueue=True,  # Non-blocking, thread-safe
        )


def get_logger(name: str):  # type: ignore[return]
    """Return a logger bound to the given module name."""
    return _logger.bind(module=name)
