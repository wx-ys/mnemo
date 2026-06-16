"""Mnemo logging system.

Three-tier logging:
    mnemo.log       — main log (INFO+)
    mnemo.debug.log — debug log (DEBUG+, weekly rotation)
    mnemo.error.log — error log (ERROR+)
"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logging(data_dir: Path, level: str = "INFO") -> logging.Logger:
    """Initialize the Mnemo logging system.

    Parameters
    ----------
    data_dir : Path
        Data directory (logs are written to ``data_dir/.mnemo/logs/``).
    level : str, optional
        Log level name. Default is 'INFO'.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir = data_dir / ".mnemo" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mnemo")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # Main log (file)
    # NOTE: Console output is handled exclusively through the on_progress
    # callback → ProgressDisplay Rich spinner, never through logging handlers.
    main_handler = logging.FileHandler(log_dir / "mnemo.log", encoding="utf-8")
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(formatter)
    logger.addHandler(main_handler)

    # Debug log (weekly rotation, keep 4 weeks)
    debug_handler = TimedRotatingFileHandler(
        log_dir / "mnemo.debug.log",
        when="W0",
        backupCount=4,
        encoding="utf-8",
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    # Error log
    error_handler = logging.FileHandler(log_dir / "mnemo.error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the Mnemo logger instance.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger("mnemo")
