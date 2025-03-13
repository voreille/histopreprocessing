import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(log_file=None):
    """Configure logging to print to stdout and optionally save to a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers (avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream handler (logs to stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Optional: Log to file if specified
    if log_file:
        log_file_path = Path(log_file).resolve()
        log_file_path.parent.mkdir(parents=True,
                                   exist_ok=True)  # Ensure directory exists
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.info("Logging initialized.")
