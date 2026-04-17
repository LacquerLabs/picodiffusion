"""Structured JSON logging for picodiffusion.

All log output is newline-delimited JSON on stdout.
Import ``get_logger`` from this module wherever you need to log.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Formats every log record as a single JSON line on stdout."""

    def format(self, record: logging.LogRecord) -> str:
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        obj: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra: dict[str, Any] = getattr(record, "fields", {})
        obj.update(extra)
        if record.exc_info and record.exc_info[1] is not None:
            obj["traceback"] = self.formatException(record.exc_info)
        return json.dumps(obj)


_formatter = _JsonFormatter()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that writes JSON to stdout.

    Args:
        name: The logger name, e.g. "picodiffusion.pipeline".

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


def setup_uvicorn_logging() -> None:
    """Redirect uvicorn's access and error loggers to JSON output."""
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_formatter)
        uv_logger.addHandler(handler)
        uv_logger.propagate = False
