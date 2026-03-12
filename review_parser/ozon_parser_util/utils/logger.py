from __future__ import annotations
import logging
import sys
from ozon_parser_util.config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else getattr(logging, LOG_LEVEL, logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)