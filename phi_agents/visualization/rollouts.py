#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from argparse_dataclass import dataclass as ap_dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from phi_agents.evals.appworld_evals import Episode
from phi_agents.evals.appworld_rollout_data import AppWorldTrainingRollout
from phi_agents.rl.appworld_scenario_runner import AppWorldScenario
from phi_agents.rl.type_defs import Message


@ap_dataclass
class Options:
    task_id: str
    download_dir: str = "/tmp"
    # Need union for argparse_dataclass
    step: Union[int, None] = None  # noqa: UP007


@dataclass
class RolloutInfo:
    rollout_dir: Path
    num_ranks: int
    num_scenarios: int
    num_rollouts: int
    current_rank: int = 0
    current_scenario: int = 0
    current_rollout: int = 0

    @property
    def current_index(self) -> int:
        shape = [self.num_ranks, self.num_scenarios, self.num_rollouts]
        coords = [self.current_rank, self.current_scenario, self.current_rollout]
        return int(np.ravel_multi_index(coords, shape))

    @current_index.setter
    def current_index(self, index: int) -> None:
        shape = [self.num_ranks, self.num_scenarios, self.num_rollouts]
        coords = [int(c) for c in np.unravel_index(index, shape)]
        self.current_rank, self.current_scenario, self.current_rollout = coords

    def __len__(self) -> int:
        return self.num_ranks * self.num_rollouts * self.num_scenarios


def get_rollout_dir_info(rollout_dir: Path) -> RolloutInfo:
    """Get info about the rollout dir.

    Get rank, num scenarios, and num rollouts based on the directory files.
    """
    rollout_zips = list(rollout_dir.glob("*"))
    ranks = []
    for rz in rollout_zips:
        match = re.search(r"r(\d+).zip", rz.as_posix())
        assert match is not None
        ranks.append(int(match.group(1)))
    num_ranks = max(ranks) + 1

    zf = zipfile.ZipFile(rollout_zips[0], "r")
    scenario_idxs = []
    rollout_idxs = []
    for name in zf.namelist():
        if "rollout" in name:
            match = re.search(r"rollout.r\d+\.(\d+)\.(\d+)\.yaml", name)
            assert match
            scenario_idxs.append(int(match.group(1)))
            rollout_idxs.append(int(match.group(2)))

    num_scenarios = max(scenario_idxs) + 1
    num_rollouts = max(rollout_idxs) + 1
    return RolloutInfo(
        rollout_dir=rollout_dir,
        num_ranks=num_ranks,
        num_scenarios=num_scenarios,
        num_rollouts=num_rollouts,
    )


def load_training_rollout(rollout_info: RolloutInfo) -> AppWorldTrainingRollout:
    rollout_zip = rollout_info.rollout_dir / f"rollouts_r{rollout_info.current_rank}.zip"
    zf = zipfile.ZipFile(rollout_zip, "r")
    with zf.open(
        f"rollout.r{rollout_info.current_rank}.{rollout_info.current_scenario}.{rollout_info.current_rollout}.yaml"
    ) as fh:
        r_bytes = fh.read()
        if isinstance(r_bytes, bytes):
            r_str = r_bytes.decode("utf-8")
        else:
            r_str = r_bytes
        assert isinstance(r_str, str)

    return AppWorldTrainingRollout.from_yaml(r_str)


def load_scenario(rollout_info: RolloutInfo) -> AppWorldScenario:
    rollout_zip = rollout_info.rollout_dir / f"rollouts_r{rollout_info.current_rank}.zip"
    zf = zipfile.ZipFile(rollout_zip, "r")
    with zf.open(
        f"scenario.r{rollout_info.current_rank}.{rollout_info.current_scenario}.yaml"
    ) as fh:
        r_bytes = fh.read()
        if isinstance(r_bytes, bytes):
            r_str = r_bytes.decode("utf-8")
        else:
            r_str = r_bytes
        assert isinstance(r_str, str)

    scenario = AppWorldScenario.from_yaml(r_str)
    assert isinstance(scenario, AppWorldScenario)
    return scenario


def convert_rollout_to_episode(
    rollout: AppWorldTrainingRollout, experiment_name: str = "n/a"
) -> Episode:
    return Episode(
        experiment_name=experiment_name,
        task=rollout.appworld_rollout_data.task,
        chat_history=rollout.messages,
        eval_result=rollout.appworld_rollout_data.eval_result,
        num_prompt_messages=rollout.appworld_rollout_data.num_prompt_messages,
        n_execution_failed=rollout.appworld_rollout_data.n_execution_failed,
        n_no_code_found=rollout.appworld_rollout_data.n_no_code_found,
        cancelled=False,
    )


def get_approx_num_tokens_in_message(tokenizer: PreTrainedTokenizerBase, msg: Message) -> int:
    """Get approximate number of tokens.

    different models will use header's and special tokens, so this won't be exact.
    """
    num_special_tokens = 2  # for the start and end header
    num_message_tokens = (
        len(tokenizer.encode(msg.content))
        + len(tokenizer.encode(msg.role, add_special_tokens=False))
        + len(tokenizer.encode("\n\n", add_special_tokens=False))
        + num_special_tokens
    )

    return num_message_tokens
