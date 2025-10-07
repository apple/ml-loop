#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, TypedDict, cast

import cattrs
import hydra.utils
import requests

from phi_agents.appworld.interface import (
    AppWorldExecutionError,
    AppWorldInterface,
    AppWorldTaskEvalResult,
    Failure,
    Pass,
    Task,
)
from phi_agents.inference.config import AppWorldConfig
from phi_agents.rl.llm import TrainableLLM
from phi_agents.rl.type_defs import Message, message_from_dict
from phi_agents.rl.vllm_client import MaxSeqLenExceeded
from phi_agents.utils.cattrs_conversion import get_converter
from phi_agents.utils.logger import get_phi_logger

converter = get_converter()
logger = get_phi_logger()

type TaskId = str


@dataclass(frozen=True)
class AggregateEvalResult:
    task_goal_completion: float
    scenario_goal_completion: float


@dataclass(frozen=True)
class TaskEvalResult:
    success: bool
    difficulty: int  # Literal[1, 2, 3]
    num_tests: int
    passes: list[Pass]
    failures: list[Failure]
    num_interactions: int

    @classmethod
    def create(
        cls,
        appworld_task_eval_result: AppWorldTaskEvalResult,
        num_interactions: int,
    ) -> Self:
        return cls(
            success=appworld_task_eval_result.success,
            difficulty=appworld_task_eval_result.difficulty,
            num_tests=appworld_task_eval_result.num_tests,
            passes=appworld_task_eval_result.passes,
            failures=appworld_task_eval_result.failures,
            num_interactions=num_interactions,
        )


class TaskMetricsCollection(dict[TaskId, TaskEvalResult]):
    """A collection of tasks and their metrics as a mapping from task ID to eval result."""

    @property
    def n_successes(self) -> int:
        return sum(eval_result.success for eval_result in self.values())

    @property
    def n_tasks(self) -> int:
        return len(self)

    @property
    def task_success_rate(self) -> float:
        """Percentage of successful tasks ie. TGC."""
        return self.n_successes / self.n_tasks

    @property
    def avg_pass_rate(self) -> float:
        """Average percentage of unit tests passed per task."""
        return (
            sum(len(eval_result.passes) / eval_result.num_tests for eval_result in self.values())
            / self.n_tasks
        )

    @property
    def collective_metrics(self) -> dict[str, int | float]:
        return {
            "n_successes": self.n_successes,
            "n_tasks": self.n_tasks,
            "task_success_rate": self.task_success_rate,
            "avg_pass_rate": self.avg_pass_rate,
        }


class EpisodeDict(TypedDict):
    experiment_name: str
    task: dict[str, Any]
    chat_history: list[dict[str, Any]]
    eval_result: dict[str, Any] | None
    num_prompt_messages: int | None
    n_execution_failed: int  # Number of turns where the code block execution failed
    n_no_code_found: int  # Number of turns where the executed code block was empty
    cancelled: bool


@dataclass(frozen=True)
class Episode:
    experiment_name: str
    task: Task
    chat_history: list[Message]
    eval_result: TaskEvalResult | None
    num_prompt_messages: int | None
    n_execution_failed: int  # Number of turns where the code block execution failed
    n_no_code_found: int  # Number of turns where the executed code block was empty
    cancelled: bool

    def asdict(self) -> EpisodeDict:
        return {
            "experiment_name": self.experiment_name,
            "task": converter.unstructure(self.task),
            "chat_history": [m.asdict() for m in self.chat_history],
            "eval_result": asdict(self.eval_result) if self.eval_result else None,
            "cancelled": self.cancelled,
            "num_prompt_messages": self.num_prompt_messages,
            "n_execution_failed": self.n_execution_failed,
            "n_no_code_found": self.n_no_code_found,
        }

    def save(self, json_path: Path) -> None:
        """Save the episode to a json file."""
        with open(json_path, "w") as f:
            json.dump(self.asdict(), f)

    @classmethod
    def load(cls, json_path: Path) -> Self:
        """Load the episode from a json file."""
        with open(json_path) as f:
            json_data = json.load(f)
        return cls(
            experiment_name=json_data["experiment_name"],
            task=converter.structure(json_data["task"], Task),
            chat_history=[message_from_dict(m) for m in json_data["chat_history"]],
            eval_result=cast(
                TaskEvalResult,
                converter.structure(json_data["eval_result"], TaskEvalResult),
            )
            if json_data["eval_result"]
            else None,
            num_prompt_messages=json_data.get("num_prompt_messages"),
            n_execution_failed=json_data.get("n_execution_failed", -1),
            n_no_code_found=json_data.get("n_no_code_found", -1),
            cancelled=json_data.get("cancelled", False),
        )


def filter_by_difficulty(
    task_metrics_collection: TaskMetricsCollection, difficulty: int
) -> TaskMetricsCollection:
    """Filter TaskMetricsCollection by difficulty."""
    task_metrics_collection_dict = {
        k: v for k, v in task_metrics_collection.items() if v.difficulty == difficulty
    }
    return TaskMetricsCollection(task_metrics_collection_dict)


@dataclass(frozen=True)
class ExperimentEvalResult:
    """All metrics (dataset-level and task-level)."""

    aggregate: AggregateEvalResult
    individual: TaskMetricsCollection


def load_eval_result(json_path: Path) -> ExperimentEvalResult:
    with open(json_path) as f:
        json_data = json.load(f)
    json_data["individual"] = TaskMetricsCollection(
        cattrs.structure(json_data["individual"], dict[TaskId, TaskEvalResult])
    )
    return cast(ExperimentEvalResult, cattrs.structure(json_data, ExperimentEvalResult))


class RolloutCancelled(RuntimeError):
    """Raised when the rollout generation is externally cancelled, e.g. by the RL system or on termination."""

    pass


class AppworldAgent(ABC):
    @abstractmethod
    def next_code_block(self, output: str | None, world: AppWorldInterface) -> str:
        pass

    @abstractmethod
    def history_as_messages(self, last_execution_output: str) -> list[Message]:
        pass


def execution_failed(appworld_output: str) -> bool:
    """Check if the execution failed e.g. due to syntax errors.

    Args:
        appworld_output: .

    Returns:
        True if the execution failed, else False.
    """
    return "Execution failed." in appworld_output


def no_code_found(code: str) -> bool:
    """Check if there is no code block found.

    This function exists solely for readability.
    """
    return code == ""


def _run_vllm_inference_single_server_single_task(
    world: AppWorldInterface,
    task_id: str,
    experiment_name: str,
    appworld_config: AppWorldConfig,
    llm: TrainableLLM,
    with_evaluation: bool,
) -> Episode:
    """Run inference on a single task for a vLLM-based agent."""
    task = world.initialize(
        task_id=task_id,
        experiment_name=experiment_name,
        raise_on_unsafe_syntax=appworld_config.env.raise_on_unsafe_syntax,
    )
    # TODO: Load lora adapter at agent construction

    agent = cast(AppworldAgent, hydra.utils.instantiate(appworld_config.agent, llm=llm, task=task))
    # Get how many messages we prompt the agent with before the task begins
    num_prompt_messages = len(agent.history_as_messages("dummy message")) - 1

    output: str | None = None
    cancelled = False
    # Until the task is completed or max_interactions is reached
    logger.info(f"-------------------------- {task_id} ------------------------------")

    n_execution_failed = 0
    n_no_code_found = 0
    for interaction in range(appworld_config.env.max_interactions):
        # ask the agent to generate the code block based on the history.
        code = None
        try:
            code = agent.next_code_block(output, world)
        except MaxSeqLenExceeded:
            logger.warning(f"Early stopping {task_id} after {interaction + 1} interactions")
            output = "Terminating episode: exceeded max sequence length"
            break
        except RolloutCancelled:
            logger.warning(f"Rollout for {task_id=} cancelled after {interaction + 1} interactions")
            output = "Terminating episode: cancelled"
            cancelled = True
            break

        # execute the code in the world environment
        try:
            output = world.execute(code)
        except AppWorldExecutionError:
            logger.exception(f"Found AppWorldExecutionError when running code:\n{code}")
            raise

        if no_code_found(code):
            n_no_code_found += 1
        if execution_failed(output):
            n_execution_failed += 1

        # stop if agent has committed the task to be complete.
        if world.task_completed():
            break
        elif interaction >= appworld_config.env.max_interactions - 1:
            logger.info(
                f"Agent ran out of turns: {task_id=}, interaction {interaction + 1}/{appworld_config.env.max_interactions}"
            )

    if with_evaluation:
        appworld_eval_result: AppWorldTaskEvalResult = world.evaluate()
        eval_result = TaskEvalResult.create(
            appworld_eval_result,
            num_interactions=interaction + 1,
        )
    else:
        eval_result = None

    try:
        world.close_world()
    except requests.ConnectionError:
        logger.exception("world.close_world failed, AppWorld process likely crashed")

    assert output is not None
    return Episode(
        experiment_name=experiment_name,
        task=task,
        chat_history=agent.history_as_messages(output),
        eval_result=eval_result,
        num_prompt_messages=num_prompt_messages,
        cancelled=cancelled,
        n_execution_failed=n_execution_failed,
        n_no_code_found=n_no_code_found,
    )


def run_vllm_inference_single_server_single_task(
    world: AppWorldInterface,
    task_id: str,
    experiment_name: str,
    appworld_config: AppWorldConfig,
    llm: TrainableLLM,
    with_evaluation: bool,
    max_retries: int = 5,
) -> Episode:
    """Run inference on a single task for a vLLM-based agent.

    This includes an outer loop to catch spontaneous AppWorld crashes.
    """
    for _ in range(max_retries):
        try:
            episode = _run_vllm_inference_single_server_single_task(
                world=world,
                task_id=task_id,
                experiment_name=experiment_name,
                appworld_config=appworld_config,
                llm=llm,
                with_evaluation=with_evaluation,
            )
        except (AppWorldExecutionError, requests.HTTPError, requests.ConnectionError):
            logger.exception("Caught exception while running vllm inference")
            world.restart()
            continue
        break
    else:  # Ran the loop for max_retries but never hit 'break'
        raise ValueError("Exceeded max retries on inference restarts.")

    logger.info(f"Returning episode from {task_id}")
    return episode
