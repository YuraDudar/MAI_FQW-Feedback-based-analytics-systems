"""Logging setup for the WB parser."""
from __future__ import annotations

import logging
import sys

from wb_parser_util.config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL


def setup_logging(verbose: bool = False) -> None:
    """
    Configure the root logger once.

    :param verbose: if True, override LOG_LEVEL with DEBUG
    """
    level_name = "DEBUG" if verbose else LOG_LEVEL
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)