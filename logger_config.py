import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "shorts_generator.log"


def setup_logger():
    logger = logging.getLogger("ShortsGenerator")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Rotating file logging
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.info("Logging system initialized.")

    return logger


# Initialize the logger instance
logger = setup_logger()


def log_step(step, message):
    logger.info("=" * 65)
    logger.info("[%s] %s", step, message)
    logger.info("=" * 65)

