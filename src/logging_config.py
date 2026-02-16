"""Logging configuration using Loguru.

This module configures structured logging with rotation and appropriate formatting.
"""

import sys
from pathlib import Path

from loguru import logger

from src.config import settings


def setup_logger():
    """Configure Loguru logger with rotation and formatting.

    Sets up console and file logging with appropriate formats and rotation policies.
    """
    # Remove default handler
    logger.remove()

    # Console handler with colorized output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # File handler with rotation (10 MB max, 10 files retained)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="10 MB",
        retention=10,
        compression="zip",
    )

    logger.info(f"Logging configured with level: {settings.log_level}")


# Configure logging on module import
setup_logger()
