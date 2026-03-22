"""
Logging & utility helpers for Polymarket Weather Bot V6.
"""

import logging
import sys
from datetime import datetime, timezone


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the bot logger."""
    logger = logging.getLogger("weather_bot")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


log = setup_logging()
