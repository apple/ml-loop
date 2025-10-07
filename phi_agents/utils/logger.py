#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import logging
from typing import Any


def get_phi_logger() -> logging.Logger:
    return logging.getLogger(__name__)


def setup_phi_logger() -> None:
    """Sets up a logger in this process that logs process ID and thread name."""
    logger = get_phi_logger()
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"PHI: %(asctime)s - %(process)s - %(threadName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


setup_phi_logger()


class NullLogger(logging.Logger):
    def __init__(self) -> None:
        pass

    def __getattr__(self, name: Any) -> Any:
        def no_op(*args: Any, **kwargs: Any) -> Any:
            pass

        return no_op
