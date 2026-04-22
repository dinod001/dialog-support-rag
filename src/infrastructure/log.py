import logging
import os
from typing import Optional

from .config import settings


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger that writes logs to the project logs folder.

    Example:
        logger = get_logger(__name__)
        logger.info("Application started")
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when called multiple times.
    if logger.handlers:
        return logger

    level_name = str(settings.logging.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger.setLevel(level)
    logger.propagate = False

    logs_dir_name = settings.paths.get("logs_dir", "logs")
    logs_dir = os.path.join(settings.project_root, logs_dir_name)
    os.makedirs(logs_dir, exist_ok=True)

    default_file_name = f"{name.replace('.', '_')}.log"
    target_log_file = log_file or default_file_name
    target_log_path = os.path.join(logs_dir, target_log_file)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(target_log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if settings.logging.get("enabled", True):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
