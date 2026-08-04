"""Consistent human-readable logging for direct Python scripts and batch jobs."""

from __future__ import annotations

import logging
import sys


def configure_logging(verbose: int = 0) -> None:
    level = logging.WARNING if verbose <= 0 else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
