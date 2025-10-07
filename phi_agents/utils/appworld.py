#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import re
from pathlib import Path

import phi_agents.utils.file_utils as fu
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


def download_dir(src_dir: Path, dest_dir: Path, exists_ok: bool) -> Path:
    """Download a directory.

    Args:
        src_dir: Source directory.
        dest_dir: Destination directory, which will be created.
        exists_ok: If True and the dest_dir already exists, use it.
    """
    if src_dir.exists():
        raise ValueError(f"{src_dir} already exists locally.")

    # Check it's a directory and not a file
    if not fu.is_dir(src_dir):
        raise ValueError(f"Expected a dir at {src_dir}")

    if fu.exists(dest_dir):
        if exists_ok:
            print("Dest dir already exists! Using it.")
        else:
            raise ValueError(f"Destination directory {dest_dir} already exists!")
    else:
        try:
            fu.copy(src_dir, dest_dir)
        except Exception as e:
            if fu.exists(dest_dir):
                fu.delete(dest_dir)
            raise e

    return dest_dir


def get_experiment_dir(experiment_name: str) -> Path:
    return Path(os.environ["APPWORLD_ROOT"]) / "experiments" / "outputs" / experiment_name


def get_task_dir(experiment_name: str, task_id: str, root: Path | None = None) -> Path:
    """Get the AppWorld task directory.

    Args:
        experiment_name: AppWorld experiment name.
        task_id: .
        root: Root experiment directory (can be local or S3). Defaults to get_experiment_dir().
    """
    root_experiment_dir = get_experiment_dir(experiment_name) if root is None else root
    return root_experiment_dir / "tasks" / task_id


def get_log_dir(experiment_name: str, task_id: str, root: Path | None = None) -> Path:
    """Get the log directory for the AppWorld task."""
    return get_task_dir(experiment_name, task_id, root=root) / "logs"


def get_episode_path(experiment_name: str, task_id: str, root: Path | None = None) -> Path:
    return get_log_dir(experiment_name, task_id, root=root) / "episode.json"


def extract_code_format_output(msg_content: str) -> str:
    output_code: str = ""
    # Allow both ```python and ```py (Qwen seems to like ```py quite a bit)
    partial_code_regex = r".*```(python|py)\n?(.*)"
    full_code_regex = r"```(python|py)\n?(.*?)```"
    ignore_multiple_calls = False

    match_end = 0
    # Handle multiple calls
    for re_match in re.finditer(full_code_regex, msg_content, flags=re.DOTALL):
        code = re_match.group(2).strip()
        if ignore_multiple_calls:
            return code
        output_code += code + "\n"
        match_end = re_match.end()

    # check for partial code match at end (no terminating ```)  following the last match
    partial_m = re.match(partial_code_regex, msg_content[match_end:], flags=re.DOTALL)
    if partial_m:
        output_code += partial_m.group(2).strip()
        # terminated due to stop condition. Add stop condition to output.
    if len(output_code) == 0:
        logger.info(f"No code found in action: {msg_content}")
        return ""
    else:
        return output_code
