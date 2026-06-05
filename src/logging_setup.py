import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO, *, is_main: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level if is_main else logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
