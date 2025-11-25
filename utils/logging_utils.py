"""Simple logging utilities."""
from __future__ import annotations
import logging
from typing import Optional


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
