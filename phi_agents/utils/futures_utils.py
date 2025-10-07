#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from collections.abc import Sequence
from concurrent.futures import Future, as_completed
from typing import Any

import wandb
from tqdm import tqdm


def wait_check_exceptions(
    futures: Sequence[Future[Any]], pbar: tqdm | None = None, log_to_wandb: bool = False
) -> None:
    for future in as_completed(futures):
        if pbar is not None:
            pbar.update(1)
            if log_to_wandb:
                wandb.log({"progress": pbar.n / pbar.total})

        try:
            future.result()
        except Exception as exc:
            # Cancel unfinished futures, otherwise we have to wait for all of them to complete
            for future in futures:
                if not future.done():
                    future.cancel()
            raise exc
