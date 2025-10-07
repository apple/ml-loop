#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import wandb
import wandb.util
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ConstantLR, LRScheduler
from torch.optim.optimizer import Optimizer

from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


class LossType(StrEnum):
    PG_PER_TRAJECTORY = "pg_per_trajectory"  # pg stands for Policy Gradient
    PG_PER_TOKEN = "pg_per_token"


class Baseline(StrEnum):
    LNO = "leave_none_out"
    LOO = "leave_one_out"


def is_policy_gradient_loss(loss_type: LossType) -> bool:
    return loss_type in (LossType.PG_PER_TRAJECTORY, LossType.PG_PER_TOKEN)


def inference_and_learning_share_gpus(inference_gpus: list[int], learning_gpus: list[int]) -> bool:
    share_gpus = len(set(inference_gpus).intersection(learning_gpus)) > 0
    return share_gpus


def wandb_init(full_cfg: DictConfig, run_name_suffix: str = "") -> str:
    """Initialize wandb. get existing run ID from cloud if we are resuming."""
    experiment_name: str = full_cfg.experiment_name
    wandb_cfg = full_cfg.wandb

    wandb_run_name = wandb_cfg.run
    if not wandb_run_name:
        datetime_str = datetime.now().strftime("%y-%m-%d_%H-%M")
        wandb_run_name = (
            f"{experiment_name}_{datetime_str}_{wandb.util.generate_id()}{run_name_suffix}"
        )

    if not wandb_cfg.group:
        wandb_cfg.group = wandb_run_name

    wandb_config = OmegaConf.to_container(full_cfg, resolve=True)
    assert isinstance(wandb_config, dict)

    wandb_log_dir: str = os.environ.get("WANDB_LOG_DIR", "_wandb_logs")
    Path(wandb_log_dir).mkdir(exist_ok=True)

    wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        name=wandb_run_name,
        id=wandb_run_name,  # should be unique because of generate_id()
        dir=wandb_log_dir,
        resume="allow",
        config=wandb_config,
    )
    assert wandb.run is not None

    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step", step_sync=True)

    return str(wandb_run_name)


def get_const_lr(
    optimizer: Optimizer,
    factor: float = 1.0,  # keeps the learning rate constant (no decay factor)
    total_iters: int = 0,  # no warmup or decay steps
    num_training_steps: int = 100,  # this is not used by the ConstantLR scheduler
    last_epoch: int = -1,
) -> LRScheduler:
    """
    Simply an adapter to make the interface more similar to `get_cosine_schedule_with_warmup()`.
    See `ConstantLR` docstring for parameter descriptions.
    """
    return ConstantLR(optimizer, factor, total_iters, last_epoch)


class GradientRMS:
    def __init__(self) -> None:
        self.mean: float = 0.0
        self.var: float = 0.0
        self.count: int = 0

    def update(self, value: float) -> tuple[float, float]:
        """Updates running mean and variance with a new scalar value."""
        if value > 0.0:  # update statistics only on non-zero gradients
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.var += delta * delta2
        return self.mean, self._std_dev()

    def _std_dev(self) -> float:
        if self.count == 0:
            return 0.0
        std_dev: float = (self.var / self.count) ** 0.5
        return std_dev

    def should_update(
        self, value: float, max_norm: float, threshold_std: float = 5.0, min_count: int = 2
    ) -> bool:
        """
        Checks if the value is within the specified number of standard deviations.
        Also do not update stats if the gradient norm exceeds the threshold.
        """
        if value > max_norm:
            return False
        if self.count <= min_count:
            return True
        return value <= self.mean + threshold_std * self._std_dev()
