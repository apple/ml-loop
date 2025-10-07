#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import multiprocessing
import re
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import phi_agents.utils.file_utils as fu

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess


PREFIX_CHECKPOINT_DIR = "checkpoint"
re_checkpoint = re.compile(PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

DONE_FILENAME = "done.txt"


def checkpoint_name(iterations_completed: int) -> str:
    return f"{PREFIX_CHECKPOINT_DIR}-{iterations_completed}"


def commit_checkpoint(checkpoint_path: Path) -> None:
    with fu.uri_open(checkpoint_path / DONE_FILENAME, "w") as f:
        f.write("Done")


def get_all_checkpoints(folder: Path, *, _done_only: bool = True) -> list[Path]:
    if not fu.exists(folder):
        return []

    content = fu.listdir_path(folder)
    checkpoints = [
        path
        for path in content
        if re_checkpoint.search(fu.path_to_str(path)) is not None and fu.is_dir(path)
    ]
    if _done_only:
        checkpoints = [path for path in checkpoints if fu.exists(path / DONE_FILENAME)]

    return sorted(checkpoints, key=lambda x: int(re_checkpoint.findall(fu.path_to_str(x))[0]))


def get_last_checkpoint(folder: Path, *, _done_only: bool = True) -> Path | None:
    checkpoints = get_all_checkpoints(folder, _done_only=_done_only)

    if len(checkpoints) == 0:
        return None
    else:
        return checkpoints[-1]


class CloudCheckpointer:
    def __init__(self, cloud_path: Path, local_path: Path, local_rank: int, max_ckpts: int | None):
        self.cloud_path = cloud_path
        self._local_path = local_path
        self._local_rank = local_rank
        self._proc: BaseProcess | None = None
        self._start_event: Event | None = None
        self._done_event: Event | None = None
        self._exit_event: Event | None = None
        self._upload_running: bool = False
        self._max_ckpts: int | None = max_ckpts
        if self._max_ckpts is not None and self._max_ckpts <= 0:
            raise ValueError(f"{max_ckpts=} must be greater than 0.")

    @staticmethod
    def _uploader_thread(
        start_event: Event,
        done_event: Event,
        exit_event: Event,
        local_path: Path,
        cloud_path: Path,
        max_ckpts: int | None,
    ) -> None:
        while True:
            start_event.wait()
            start_event.clear()

            if exit_event.is_set():
                break

            if (
                last_checkpoint_local := get_last_checkpoint(local_path, _done_only=False)
            ) is not None:
                cloud_checkpoint = cloud_path / last_checkpoint_local.name
                fu.copy(last_checkpoint_local, cloud_checkpoint)

                print(f"Going to write {cloud_checkpoint}")
                commit_checkpoint(cloud_path / last_checkpoint_local.name)

            if max_ckpts:
                # Delete all checkpoints except latest n (defined via max_ckpts).
                # This adds an extra layer of safety in case any checkpoints are corrupted, and
                # makes it very unlikely that a checkpoint is deleted before eval jobs finish
                # reading it.
                assert max_ckpts > 0
                for ckpt in get_all_checkpoints(cloud_path, _done_only=False)[:-max_ckpts]:
                    fu.delete(ckpt, recursive=True)

            done_event.set()

    def _init_proc(self, local_path: Path) -> None:
        if self._proc is not None and self._local_path == local_path:
            return

        self.stop()

        mp_ctx = multiprocessing.get_context("spawn")
        self._local_path = local_path
        self._done_event = mp_ctx.Event()
        self._exit_event = mp_ctx.Event()
        self._start_event = mp_ctx.Event()
        self._start_event.clear()
        self._proc = mp_ctx.Process(
            target=self._uploader_thread,
            args=(
                self._start_event,
                self._done_event,
                self._exit_event,
                self._local_path,
                self.cloud_path,
                self._max_ckpts,
            ),
        )
        self._proc.start()

    def stop(self) -> None:
        if self._proc is not None:
            assert self._done_event is not None
            self.sync()
            p = self._proc
            self._proc = None

            assert self._exit_event is not None
            self._exit_event.set()
            assert self._start_event is not None
            self._start_event.set()

            p.join()

    def sync(self) -> None:
        if self._upload_running and self._done_event is not None:
            self._done_event.wait()
            self._done_event.clear()
            self._upload_running = False

    def on_save(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._local_rank == 0:
            self._init_proc(self._local_path)
            assert self._start_event is not None
            assert self._done_event is not None
            self.sync()

            self._upload_running = True
            self._start_event.set()

    def on_step_end(self) -> None:
        self.sync()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


if __name__ == "__main__":
    print(get_all_checkpoints(Path("my_dir")))
    print(get_last_checkpoint(Path("my_dir")))
