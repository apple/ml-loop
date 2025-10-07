#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import itertools
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import plotly.graph_objects as go
import wandb
import wandb.plot
import yaml
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from phi_agents.appworld.interface import Task
from phi_agents.evals.appworld_evals import Episode, TaskEvalResult
from phi_agents.rl.type_defs import PolicyTokenInfo, RolloutLoader, TrainingRollout
from phi_agents.utils.cattrs_conversion import get_converter
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()
converter = get_converter()


@dataclass(frozen=True)
class AppWorldRolloutData:
    """Metadata attached to rollouts e.g. for wandb visualization."""

    task: Task
    eval_result: TaskEvalResult
    dataset_name: str
    num_prompt_messages: int | None
    n_execution_failed: int
    n_no_code_found: int

    @classmethod
    def from_episode(cls, episode: Episode, dataset_name: str) -> Self:
        if episode.eval_result is None:
            raise ValueError("Cannot construct AppWorldRolloutData if episode.eval_result is None.")

        return cls(
            task=episode.task,
            eval_result=episode.eval_result,
            dataset_name=dataset_name,
            num_prompt_messages=episode.num_prompt_messages,
            n_execution_failed=episode.n_execution_failed,
            n_no_code_found=episode.n_no_code_found,
        )


@dataclass
class AppWorldTrainingRollout(TrainingRollout):
    appworld_rollout_data: AppWorldRolloutData

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "AppWorldTrainingRollout":
        rollout_yaml = yaml.load(yaml_str, Loader=RolloutLoader)
        if "cancelled" not in rollout_yaml:
            rollout_yaml["cancelled"] = False
        if "num_prompt_messages" not in rollout_yaml["appworld_rollout_data"]:
            rollout_yaml["appworld_rollout_data"]["num_prompt_messages"] = 0
        if "n_execution_failed" not in rollout_yaml["appworld_rollout_data"]:
            rollout_yaml["appworld_rollout_data"]["n_execution_failed"] = -1
        if "n_no_code_found" not in rollout_yaml["appworld_rollout_data"]:
            rollout_yaml["appworld_rollout_data"]["n_no_code_found"] = -1
        rollout = cls(**rollout_yaml)
        rollout.appworld_rollout_data = converter.structure(
            rollout.appworld_rollout_data, AppWorldRolloutData
        )
        rollout.policy_token_info = converter.structure(rollout.policy_token_info, PolicyTokenInfo)

        return rollout


def _success_rate(rollouts: Sequence[AppWorldTrainingRollout]) -> float:
    return sum(r.appworld_rollout_data.eval_result.success for r in rollouts) / len(rollouts)


def _avg_return(rollouts: Sequence[AppWorldTrainingRollout]) -> float:
    return sum(r.ret for r in rollouts) / len(rollouts)


def make_histogram(values: Sequence[int | float]) -> go.Figure:
    """Make a histogram as a plotly figure.

    Assumes the data is in range [0, 1] and divides the histogram into 10 bins.

    Args:
        values: Values for histogram.

    Returns:
        plotly figure.
    """
    fig = go.Figure(data=[go.Histogram(x=values, xbins=dict(start=0, end=1, size=0.1))])

    # Add labels and title
    fig.update_layout(
        xaxis=dict(range=[0, 1], dtick=0.1),
        template="plotly_white",
    )

    return fig


def _count_return_1(returns: list[float]) -> int:
    """Count how many returns have the value 1."""
    return sum(r == 1 for r in returns)


def log_appworld_rollouts_to_wandb(
    global_step: int,
    rollouts: Sequence[AppWorldTrainingRollout],
    prefix: str,
    log_highest_return: bool,
    iterations_completed: int,
) -> None:
    """Compute and log appworld rollout metrics to wandb."""
    _prefix = prefix + "/"
    metrics: dict[str, float] = {}

    ############################
    # by source (dataset_name) #
    ############################

    rollouts_by_source: dict[str, list[AppWorldTrainingRollout]] = defaultdict(list)
    for rollout in rollouts:
        rollouts_by_source[rollout.appworld_rollout_data.dataset_name].append(rollout)

    avg_return_by_source: dict[str, float] = {
        str(source): _avg_return(cur_r) for source, cur_r in rollouts_by_source.items()
    }

    success_rate_by_source: dict[str, float] = {
        str(source): _success_rate(cur_r) for source, cur_r in rollouts_by_source.items()
    }

    metrics = {
        **metrics,
        **{_prefix + f"source_success_rate__{k}": v for k, v in success_rate_by_source.items()},
        **{_prefix + f"source_avg_return__{k}": v for k, v in avg_return_by_source.items()},
    }

    #################
    # by difficulty #
    #################

    rollouts_by_difficulty: dict[int, list[AppWorldTrainingRollout]] = defaultdict(list)
    for rollout in rollouts:
        rollouts_by_difficulty[rollout.appworld_rollout_data.eval_result.difficulty].append(rollout)

    # Success rate by difficulty
    success_rate_by_difficulty: dict[str, float] = {
        str(dfcty): _success_rate(cur_r) for dfcty, cur_r in rollouts_by_difficulty.items()
    }
    success_rate_by_difficulty["avg_over_dfcty"] = sum(success_rate_by_difficulty.values()) / len(
        success_rate_by_difficulty  # Requires dict to only have difficulty keys until now
    )
    metrics = {  # Add to metrics dict
        **metrics,
        **{_prefix + f"success_rate__{k}": v for k, v in success_rate_by_difficulty.items()},
    }

    # Num rollouts by difficulty
    n_rollouts_by_difficulty: dict[str, int] = {
        str(dfcty): len(cur_r) for dfcty, cur_r in rollouts_by_difficulty.items()
    }
    n_rollouts_by_difficulty["total"] = sum(
        n_rollouts_by_difficulty.values()
    )  # Requires dict to only have difficulty keys until now
    metrics = {  # Add to metrics dict
        **metrics,
        **{_prefix + f"n_rollouts__{k}": v for k, v in n_rollouts_by_difficulty.items()},
    }

    # Histogram of return
    returns_by_difficulty: dict[str, list[float]] = {
        str(dfcty): [r.ret for r in cur_rollouts]
        for dfcty, cur_rollouts in rollouts_by_difficulty.items()
    }
    returns_by_difficulty["total"] = list(
        itertools.chain(*returns_by_difficulty.values())
    )  # Requires dict to only have difficulty keys until now
    plots: dict[str, go.Figure] = {
        _prefix + f"returns_difficulty_hist_{k}": make_histogram(v)
        for k, v in returns_by_difficulty.items()
    }

    if log_highest_return:
        # Determine the highest return achieved on a per task basis
        per_task_return_tracker: dict[tuple[Task, int], float] = defaultdict(float)
        for r in rollouts:
            cur_task = r.appworld_rollout_data.task
            cur_dfcty = r.appworld_rollout_data.eval_result.difficulty
            cur_return = r.ret
            per_task_return_tracker[(cur_task, cur_dfcty)] = max(
                cur_return, per_task_return_tracker[(cur_task, cur_dfcty)]
            )

        # Log to wandb
        highest_returns_by_difficulty: dict[str, list[float]] = defaultdict(list)
        for (_, cur_dfcty), highest_ret in per_task_return_tracker.items():
            highest_returns_by_difficulty[str(cur_dfcty)].append(highest_ret)
        highest_returns_by_difficulty["total"] = list(per_task_return_tracker.values())

        attained_return_1: dict[str, float] = {
            k: _count_return_1(v) / len(v) for k, v in highest_returns_by_difficulty.items()
        }
        metrics = {  # Add to metrics dict
            **metrics,
            **{_prefix + f"attained_return_1__{k}": v for k, v in attained_return_1.items()},
        }

        # Add to plots dict
        for k, v in highest_returns_by_difficulty.items():
            plots[_prefix + f"highest_returns_difficulty_hist_{k}"] = make_histogram(v)

    wandb.log(
        {
            "global_step": global_step,
            "iterations_completed": iterations_completed,
            **metrics,
            **plots,
        },
    )


def log_appworld_rollouts_html(
    rollouts: list[AppWorldTrainingRollout],
    global_step: int,
    iterations_completed: int,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    successful_rollouts = [r for r in rollouts if r.appworld_rollout_data.eval_result.success]
    failed_rollouts = [r for r in rollouts if not r.appworld_rollout_data.eval_result.success]
    log_n_successful = min(len(successful_rollouts), 2)
    log_n_failed = min(len(failed_rollouts), 2)

    # randomly select some failed and successful rollouts to log
    failed_rollouts = random.sample(failed_rollouts, log_n_failed)
    successful_rollouts = random.sample(successful_rollouts, log_n_successful)

    log_dict: dict[str, Any] = {
        "global_step": global_step,
        "iterations_completed": iterations_completed,
    }

    from phi_agents.evals.html_utils import rollout_html_wandb

    for i, r in enumerate(successful_rollouts):
        log_dict[f"html/successful_rollout_{i}"] = rollout_html_wandb(r, tokenizer)
    for i, r in enumerate(failed_rollouts):
        log_dict[f"html/failed_rollout_{i}"] = rollout_html_wandb(r, tokenizer)
    wandb.log(log_dict)
