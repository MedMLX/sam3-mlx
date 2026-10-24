# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Logging helpers ported from the official SAM3 surface."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TextIO


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """A command-line formatter with different colors for each level."""

    def __init__(self) -> None:
        super().__init__()
        reset = "\033[0m"
        colors = {
            logging.DEBUG: f"{reset}\033[36m",
            logging.INFO: f"{reset}\033[32m",
            logging.WARNING: f"{reset}\033[33m",
            logging.ERROR: f"{reset}\033[31m",
            logging.CRITICAL: f"{reset}\033[35m",
        }
        fmt_str = (
            "{color}%(levelname)s %(asctime)s %(process)d "
            "%(filename)s:%(lineno)4d:{reset} %(message)s"
        )
        self.formatters = {
            level: logging.Formatter(fmt_str.format(color=color, reset=reset))
            for level, color in colors.items()
        }
        self.default_formatter = self.formatters[logging.INFO]

    def format(self, record: logging.LogRecord) -> str:
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        return formatter.format(record)


class _ColoredStreamHandler(logging.StreamHandler[TextIO]):
    """Identify handlers installed by this module without dynamic attributes."""


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured command-line logger."""
    if "LOG_LEVEL" in os.environ:
        level_name = os.environ["LOG_LEVEL"].upper()
        if level_name not in LOG_LEVELS:
            warnings.warn(
                f"Ignoring invalid LOG_LEVEL {os.environ['LOG_LEVEL']!r}; "
                f"expected one of {sorted(LOG_LEVELS)}.",
                stacklevel=2,
            )
        else:
            level = LOG_LEVELS[level_name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    existing_handler = next(
        (
            handler
            for handler in logger.handlers
            if isinstance(handler, _ColoredStreamHandler)
        ),
        None,
    )
    if existing_handler is None:
        handler = _ColoredStreamHandler()
        handler.setLevel(level)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
    else:
        existing_handler.setLevel(level)
    return logger
