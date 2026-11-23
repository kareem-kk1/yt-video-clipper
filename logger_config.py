
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import LOG_DIR


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_FILE = LOG_DIR / "shorts_generator.log"

LOGGER_NAME = "ShortsGenerator"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5


# ============================================================
# LOGGER SETUP
# ============================================================

def setup_logger() -> logging.Logger:
    """Configure and return the application logger."""

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    app_logger = logging.getLogger(LOGGER_NAME)

    app_logger.setLevel(logging.DEBUG)

    # Avoid duplicate output through the root logger.
    app_logger.propagate = False

    # Prevent duplicate handlers during Streamlit reloads.
    if app_logger.handlers:
        return app_logger

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # --------------------------------------------------------
    # CONSOLE HANDLER
    # --------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(
        logging.INFO
    )

    console_handler.setFormatter(
        formatter
    )

    # --------------------------------------------------------
    # ROTATING FILE HANDLER
    # --------------------------------------------------------

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.DEBUG
    )

    file_handler.setFormatter(
        formatter
    )

    # --------------------------------------------------------
    # REGISTER HANDLERS
    # --------------------------------------------------------

    app_logger.addHandler(
        console_handler
    )

    app_logger.addHandler(
        file_handler
    )

    app_logger.info(
        "Logging system initialized."
    )

    return app_logger


# ============================================================
# GLOBAL LOGGER
# ============================================================

logger = setup_logger()


# ============================================================
# LOGGING HELPER
# ============================================================

def log_step(
    step: str,
    message: str,
) -> None:
    """Log a clearly separated application processing step."""

    separator = "=" * 65

    logger.info(separator)
    logger.info("[%s] %s", step, message)
    logger.info(separator)
