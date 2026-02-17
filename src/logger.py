"""
Centralized logging configuration.
"""

import logging
import os


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (typically __name__).
        log_file: Optional path to log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
