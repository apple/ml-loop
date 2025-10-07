#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
from typing import Any, Literal

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from phi_agents.inference.config import AppWorldConfig, LLMConfig, MainInferenceConfig
from phi_agents.vllm.vllm_server import VLLMServer

Mode = Literal[
    "train",
    "eval",
    "create_remote_dev_machine",
]


cs = ConfigStore.instance()
cs.store(group="top_level", name="base", node=MainInferenceConfig)
cs.store(group="appworld", name="base", node=AppWorldConfig)
cs.store(group="llm", name="base", node=LLMConfig)
cs.store(group="llm/vllm_server", name="base", node=VLLMServer.Conf)


def install_extra_datasets_command(extra_datasets: dict[str, str]) -> str:
    if len(extra_datasets) == 0:
        return 'echo "No extra datasets to install."'
    install_commands: list[str] = []
    for dataset_name, tasks_uri in extra_datasets.items():
        install_commands.append(
            f"python scripts/appworld_phi_data.py --mode install --dataset-name {dataset_name} --tasks-uri {tasks_uri}"
        )
    return " && ".join(install_commands)


OmegaConf.register_new_resolver(
    "install_extra_datasets_command", install_extra_datasets_command, replace=True
)


def safe_min(a: int | float | None, b: int | float | None) -> int | float | None:
    if a is not None and b is not None:
        return min(a, b)
    return a if b is None else b


def appworld_split_to_num_scenarios(split_name: str) -> int:
    with open(os.environ["APPWORLD_ROOT"] + f"/data/datasets/{split_name}.txt") as fh:
        return len(fh.readlines())


def register_hydra_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "appworld_split_to_num_scenarios",
        appworld_split_to_num_scenarios,
        replace=True,
        use_cache=True,
    )

    OmegaConf.register_new_resolver("safe_min", safe_min, replace=True)


def get_config(mode: Mode, overrides: list[str] | None = None) -> DictConfig:
    register_hydra_resolvers()

    match mode:
        case "train":
            with hydra.initialize("conf", version_base="1.3"):
                cfg = hydra.compose("config.yaml", overrides=overrides)
        case "eval":
            with hydra.initialize("conf", version_base="1.3"):
                cfg = hydra.compose("appworld_eval.yaml", overrides=overrides)
        case _:
            raise ValueError(
                f"Invalid {mode=}, see config.py for a list of valid configuration modes"
            )
    return cfg


def convert_to_str(arg: Any) -> Any:
    arg = str(arg)
    return arg.replace(" ", "")


OmegaConf.register_new_resolver("convert_to_str", convert_to_str, replace=True)


def add(v1: Any, v2: Any) -> Any:
    return v1 + v2


OmegaConf.register_new_resolver("add", add, replace=True)
