from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


class InterceptHandler(logging.Handler):
    """Route standard logging records through loguru so third-party logs share formatting."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=2, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(log_level: str, logger_name: str, log_file: Optional[Path] = None) -> None:
    """Configure loguru sinks for console (and optional file) with consistent formatting."""
    logger.remove()
    logger.configure(extra={"logger_name": logger_name})

    console_format = "<level>{level: <8}</level> | {file.name}:{line} | {extra[logger_name]} | {message}"
    logger.add(sys.stdout, format=console_format, level=log_level.upper(), colorize=True)

    if log_file is not None:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {file.name}:{line} | {extra[logger_name]} | {message}"
        )
        logger.add(Path(log_file), format=file_format, level=log_level.upper())

    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=getattr(logging, log_level.upper(), logging.INFO),
        force=True,
    )
