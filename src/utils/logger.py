"""
Logger factory
==============
Creates a coloured console + rotating-file logger.
Use ``get_logger(__name__)`` in every module.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

_COLOURS = {
    "DEBUG":    "\033[36m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[35m",
    "RESET":    "\033[0m",
}


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        c = _COLOURS.get(record.levelname, "")
        r = _COLOURS["RESET"]
        record.levelname = f"{c}{record.levelname:<8}{r}"
        return super().format(record)


def setup_root_logger(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
) -> None:
    """Call once from main entry-points (train.py, api/main.py, etc.)."""
    root = logging.getLogger("shm")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if root.handlers:
        return

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_ColourFormatter(_FMT, datefmt=_DATE))
    root.addHandler(ch)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(p, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
        root.addHandler(fh)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'shm' root namespace."""
    return logging.getLogger(f"shm.{name}")
