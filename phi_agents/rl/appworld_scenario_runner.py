#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import cattrs
import numpy as np
import yaml

from phi_agents.appworld.interface import AppWorldInterface, load_task_ids
from phi_agents.evals.appworld_evals import run_vllm_inference_single_server_single_task
from phi_agents.evals.appworld_rollout_data import AppWorldRolloutData, AppWorldTrainingRollout
from phi_agents.inference.config import AppWorldConfig
from phi_agents.rl.type_defs import (
    PolicyMessage,
    PolicyTokenInfo,
    Scenario,
    ScenarioRunner,
    TrainingRollout,
)
from phi_agents.utils.logger import get_phi_logger

if TYPE_CHECKING:
    from phi_agents.rl.llm import TrainableLLM


logger = get_phi_logger()


@dataclass
class AppWorldScenario(Scenario):
    task_id: str
    dataset_name: str

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Scenario:
        return cast(AppWorldScenario, cattrs.structure(yaml.safe_load(yaml_str), cls))


class AppWorldScenarioSampler(Iterator[AppWorldScenario]):
    """Sampler for AppWorld scenarios (tasks)."""

    def __init__(
        self,
        dataset_name: str,
        seed: int | None = None,
        task_id: str | None = None,
    ):
        if task_id:
            self.all_task_ids = [task_id]
        else:
            self.all_task_ids = load_task_ids(dataset_name)
        self.rng = np.random.default_rng(seed)
        self.task_ids: list[str] = []
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[AppWorldScenario]:
        return self

    def __next__(self) -> AppWorldScenario:
        if len(self.task_ids) == 0:
            self.task_ids = copy.copy(self.all_task_ids)
            self.rng.shuffle(self.task_ids)
        task_id = str(self.task_ids.pop())
        return AppWorldScenario(task_id=task_id, dataset_name=self.dataset_name)


class MixtureAppWorldScenarioSampler(Iterator[AppWorldScenario]):
    """Mixture of two samplers for AppWorld scenarios (tasks)."""

    def __init__(self, samplers: Sequence[Iterator[AppWorldScenario]], probs: Sequence[float]):
        self.samplers = samplers
        self.probs = probs
        if not np.isclose(sum(probs), 1):
            raise ValueError(f"Probabilities do not sum to 1: {probs}")
        if len(samplers) != len(probs):
            raise ValueError(
                f"Number of samplers does not match number of probabilites {len(samplers)} != {len(probs)}"
            )
        self.rng = np.random.default_rng()

    def __iter__(self) -> Iterator[AppWorldScenario]:
        return self

    def __next__(self) -> AppWorldScenario:
        sampler = cast(Iterator[AppWorldScenario], self.rng.choice(self.samplers, p=self.probs))
        return next(sampler)


class AppWorldScenarioRunner(ScenarioRunner):
    """Runs AppWorld using a VLLM server for LLM generation."""

    def __init__(self, *, appworld_config: AppWorldConfig | dict[str, Any]):
        self.appworld_config = (
            cast(AppWorldConfig, cattrs.structure(appworld_config, AppWorldConfig))
            if isinstance(appworld_config, dict)
            else appworld_config
        )

        self.world = AppWorldInterface(stdout_to_devnull=True)
        self.lock = threading.Lock()

    def run(self, scenario: Scenario, llm: TrainableLLM) -> TrainingRollout:
        assert isinstance(scenario, AppWorldScenario)
        task_id = scenario.task_id

        assert self.world.server is not None

        experiment_name = self.appworld_config.experiment_name or f"runner_{self.world.port:06}"
        if self.appworld_config.experiment_name is not None:
            assert (
                self.appworld_config.agent["mode"] == "eval"
            ), "Because multiple runners could be writing to the same directory, we want to make sure this is only used during eval"

        with self.lock:
            logger.info(f"Acquired lock for AppWorld at {self.world.remote_environment_url}")
            logger.info(f"Generating episode; experiment_name={experiment_name}, task_id={task_id}")
            start = time.perf_counter()
            episode = run_vllm_inference_single_server_single_task(
                world=self.world,
                task_id=task_id,
                experiment_name=experiment_name,
                appworld_config=self.appworld_config,
                llm=llm,
                with_evaluation=True,
            )

        assert episode.eval_result is not None
        eval_result = episode.eval_result

        if self.appworld_config.env.sparse_reward:
            ret = float(eval_result.success)
        else:
            no_code_found_penalty = self.appworld_config.env.no_code_found_penalty
            execution_failed_penalty = self.appworld_config.env.execution_failed_penalty
            ret = (
                len(eval_result.passes) / eval_result.num_tests  # Percentage of unit tests passed
                - execution_failed_penalty * episode.n_execution_failed  # Execution code failures
                - no_code_found_penalty * episode.n_no_code_found  # Missing code blocks
            )
            ret = float(np.clip(ret, a_min=0.0, a_max=None))  # Return must be >= 0

        messages = episode.chat_history

        elapsed = time.perf_counter() - start
        n_policy_messages = sum(isinstance(msg, PolicyMessage) for msg in messages)
        logger.info(
            f"Episode completed (elapsed={elapsed:.2f} seconds, ret={ret:.3f}, {n_policy_messages=})"
        )

        if sum(msg.ipython for msg in messages if isinstance(msg, PolicyMessage)) > 0:
            logger.warning(f"got <|python_tag|> in rollout: {messages}")

        appworld_rollout_data = AppWorldRolloutData.from_episode(episode, scenario.dataset_name)

        return AppWorldTrainingRollout(
            messages,
            ret,
            elapsed,
            episode.cancelled,
            PolicyTokenInfo() if episode.cancelled else llm.get_policy_token_info(messages),
            appworld_rollout_data=appworld_rollout_data,
        )

    def cleanup(self) -> None:
        self.world.close_server()
