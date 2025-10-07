#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from argparse_dataclass import ArgumentParser

import phi_agents.utils.file_utils as file_utils
from phi_agents.appworld.interface import DEFAULT_SPLITS_DIR, get_appworld_root
from phi_agents.utils.file_utils import (
    filesystem_for_scheme,
    get_scheme_and_path,
    path_to_str,
    project_source_root,
)


@dataclass
class Options:
    mode: str
    """Install or uninstall."""

    dataset_name: str
    """A label for this set of tasks. A text file containing the task IDs will be create at $APPWORLD_ROOT/data/datasets/{dataset_name}.txt"""

    tasks_uri: Optional[str] = None  # noqa: UP007
    """A URI (local filesystem path or S3 path) that contains subdirectories that are AppWorld task directories. Must end in /"""

    split: bool = True
    """Whether to create a 'split' listing that is identical to the dataset listing and with the same dataset_name label."""

    splits_dir: str = DEFAULT_SPLITS_DIR
    """The split directory where a listing ot the task IDs should be written (at {splits_dir}/{datset_name}.txt)"""

    verbose: bool = False

    def __post_init__(self) -> None:
        if self.dataset_name.endswith(".txt"):
            raise ValueError("Dataset name should not end in .txt. This will be added later.")
        if self.tasks_uri is not None and not self.tasks_uri.endswith("/"):
            raise ValueError("Tasks URI should end with /")


def get_dataset_path(dataset_name: str) -> Path:
    return Path(get_appworld_root()) / "data" / "datasets" / f"{dataset_name}.txt"


def write_task_ids(path: Path, task_ids: list[str]) -> None:
    with open(path, "w") as f:
        for task_id in task_ids:
            f.write(task_id + "\n")


def get_task_ids(tasks_uri: str) -> list[str]:
    tasks_uri = path_to_str(options.tasks_uri)
    scheme, _ = get_scheme_and_path(options.tasks_uri)
    fs = filesystem_for_scheme(scheme)
    ls_infos = (info for info in fs.ls(tasks_uri, detail=True) if info["type"] == "directory")
    if scheme == "s3":
        return [Path(info["Key"]).name for info in ls_infos]
    else:
        return [Path(info["name"]).name for info in ls_infos]


def install(options: Options) -> None:
    assert options.mode == "install"
    start = time.time()
    print(f"installing dataset {options.dataset_name} into appworld root at {get_appworld_root()}")

    if options.tasks_uri is None:
        raise ValueError("tasks_uri is required to install.")

    dataset_path = get_dataset_path(options.dataset_name)
    if dataset_path.exists():
        raise ValueError(f"Dataset {options.dataset_name} already exists. Uninstall it first.")

    task_ids = get_task_ids(options.tasks_uri)
    local_task_dir = Path(get_appworld_root()) / "data" / "tasks"

    # check that none of the task directories do not already exist (UUIDs should be unique)
    for task_id in task_ids:
        local_path = local_task_dir / task_id
        if local_path.exists():
            raise ValueError(f"Attempted to overwrite task {task_id} at {local_path}")

    # write the dataset listing to $APPWORLD_ROOT/data/datasets/
    if options.verbose:
        print(f"Writing task IDs to {dataset_path}")
    write_task_ids(dataset_path, task_ids)

    # optionally write split listing
    if options.split:
        split_path = project_source_root() / options.splits_dir / f"{options.dataset_name}.txt"
        if options.verbose:
            print(f"Writing task IDs to {split_path}")
        write_task_ids(split_path, task_ids)

    # actually do the copy
    for task_id in task_ids:
        assert options.tasks_uri.endswith("/")
        src, dst = f"{options.tasks_uri}{task_id}", local_task_dir / task_id
        if options.verbose:
            print(f"Copying {src} to {dst}")
        file_utils.copy(src, local_task_dir / task_id)

    # NOTE: it turns out that copying each directory one at a time from S3 is actually not slower than all at once with this:
    # 94 seconds one at a time vs 128 seconds all at once
    # src, dst = f"{options.tasks_uri}", f"{local_task_dir.as_posix()}/"
    # if options.verbose:
    #     print(f"Copying {src} to {dst}")
    # file_utils.copy(src, dst)

    # NOTE: it would be faster if all tasks were zipped together into a single file (or individually), but we can save that for later
    # (it is convenient to be able to view scenarios on S3 without unzipping)

    print(f"done ({time.time() - start} seconds)")


def uninstall(options: Options) -> None:
    assert options.mode == "uninstall"
    start = time.time()
    print(
        f"uninstalling dataset {options.dataset_name} from appworld root at {get_appworld_root()}"
    )

    if options.tasks_uri is not None:
        raise ValueError("tasks_uri not allowed with uninstall.")

    dataset_path = get_dataset_path(options.dataset_name)
    if not dataset_path.exists():
        raise ValueError(f"Dataset {options.dataset_name} is not installed.")

    # get the task IDs
    with open(dataset_path) as f:
        task_ids = f.read().splitlines()

    # remove the task directories
    tasks_dir = Path(get_appworld_root()) / "data" / "tasks"
    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        if not task_dir.exists() and task_dir.is_dir():
            raise ValueError(f"Expected {task_dir} to exist and be a directory")
        if options.verbose:
            print(f"Removing task directory {task_dir}")
        shutil.rmtree(task_dir)

    # remove the dataset task ID listing
    if options.verbose:
        print(f"Removing task IDs listing at {dataset_path}")
    dataset_path.unlink()

    # remove the split task ID listing
    if options.split:
        split_path = project_source_root() / options.splits_dir / f"{options.dataset_name}.txt"
        if options.verbose:
            print(f"Removing task IDs listing at {split_path}")
        split_path.unlink()

    print(f"done ({time.time() - start} seconds)")


if __name__ == "__main__":
    parser = ArgumentParser(Options)
    options = parser.parse_args(sys.argv[1:])
    if options.mode == "install":
        install(options)
    elif options.mode == "uninstall":
        uninstall(options)
    else:
        raise ValueError(f"Unknown mode: {options.mode}")
