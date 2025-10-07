#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING

from phi_agents.vllm.vllm_server import VLLMServer


@dataclass
class AppWorldEnvConfig:
    max_interactions: int
    raise_on_unsafe_syntax: bool
    sparse_reward: bool
    no_code_found_penalty: float
    execution_failed_penalty: float

    def __post_init__(self) -> None:
        assert self.no_code_found_penalty >= 0.0
        assert self.execution_failed_penalty >= 0.0


@dataclass
class AppWorldConfig:
    env: AppWorldEnvConfig

    # field for use with hydra.utils.instantiate
    # see https://hydra.cc/docs/advanced/instantiate_objects/overview/
    agent: dict[str, Any]
    experiment_name: str | None


@dataclass
class LLMConfig:
    parameter_count: str  # Used only for cluster job attribution
    inference_server_class: str
    vllm_server: VLLMServer.Conf
    base_model_path: str
    max_gpu_mem_utilization: float | None
    vllm_class: dict[str, Any]
    temperature: float
    adapter_path: Path | None
    compile_torch_model: bool

    lora_target_modules: list[str]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float


@dataclass
class MainInferenceConfig:
    experiment_name: str
    llm: LLMConfig
    eval_seed: int
    scenario_sampler: Any
    scenario_runner: Any
    num_scenario_runners: int
    hydra: Any = MISSING
    log_dir: str = "logs/default"
    wandb_project: str = "default_project"
    rl: Any = field(default_factory=dict)
