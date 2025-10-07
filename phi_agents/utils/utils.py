#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import contextlib
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from phi_agents.utils.logger import NullLogger


@contextmanager
def timeit(
    name: str,
    logger: logging.Logger | NullLogger | None,
    enable: bool = True,
) -> Iterator[None]:
    if not enable:
        return

    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        if logger is not None:
            logger.info(f"{name} took: {end_time - start_time:.2f} s")


def torch_dist_barrier() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


@contextlib.contextmanager
def barrier_guard(before: bool = True, after: bool = True):  # type: ignore
    import torch.distributed as dist

    if not dist.is_initialized():
        yield
        return

    if before:
        dist.barrier()

    yield

    if after:
        dist.barrier()
