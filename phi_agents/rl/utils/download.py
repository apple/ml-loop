#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import errors, repo_exists

import phi_agents.utils.file_utils as fu
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


def locate_hf_cli() -> str | None:
    bin_dir = Path(sys.executable).resolve().parent  # .../env/bin
    cli_binary = "huggingface-cli"
    cli = bin_dir / cli_binary

    if cli.exists() and cli.is_file():
        logger.debug(f"Found {cli_binary} at {cli}")
        return str(cli)

    logger.warning(f"{cli_binary} not found at {cli}! Is `huggingface-hub` installed?")
    logger.info(f"Fallback to {cli_binary} in PATH ({shutil.which('huggingface-cli')=})")
    return cli_binary


def download_model(
    name_or_path: str, hf_args: list[str] | None = None, base_dir: Path | None = None
) -> str:
    name_or_path_parts = Path(name_or_path).parts
    base_dir = base_dir or Path.cwd()
    dst_name = (
        base_dir / ".model_cache" / name_or_path_parts[-2] / name_or_path_parts[-1]
    ).as_posix()

    if not fu.exists(name_or_path) and safe_hf_repo_exists(name_or_path):
        cmd = [
            locate_hf_cli(),
            "download",
            name_or_path,
            "--local-dir",
            dst_name,
            "--exclude=*consolidated*",  # Unnecessary consoldated files contain duplicate weights to the safetensor files used by hf
        ]
        if hf_args:
            cmd += hf_args

        print(" ".join(cmd))
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        subprocess.check_call(cmd)
        name_or_path = dst_name

    elif fu.get_scheme(name_or_path) != "file":
        print(f"Downloading {name_or_path} to {dst_name}")
        if not fu.exists(dst_name):
            fu.copy(name_or_path, dst_name)
        name_or_path = dst_name
    else:
        assert fu.exists(name_or_path), f"Could not figure out how to download model {name_or_path}"

    return name_or_path


def distributed_download_models(model_name_or_path: str, local_rank: int) -> str:
    paths: list[str | None] = []
    if local_rank == 0:
        paths.append(download_model(model_name_or_path))
    else:
        paths = [None]

    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(paths, src=0)

    assert isinstance(paths[0], str)
    return paths[0]


def download_adapter(adapter_path: Path) -> Path:
    if fu.get_scheme(adapter_path) == "s3":
        temp_dir = tempfile.mkdtemp()
        fu.copy(adapter_path / "*", temp_dir)
        adapter_path = Path(temp_dir)
    else:
        assert fu.exists(adapter_path)

    return adapter_path


def safe_hf_repo_exists(repo_id: str) -> bool:
    try:
        return repo_exists(repo_id)  # type: ignore # (hf's lib doesn't have type stubs)
    except errors.HFValidationError:
        pass
    return False
