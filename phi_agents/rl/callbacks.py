#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from pathlib import Path
from typing import Any

import ray
from ray.exceptions import RayActorError

from phi_agents.rl.eval import EvalWorker
from phi_agents.rl.train import RLOOTrainer
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


class Callback:
    def __init__(self, algo: RLOOTrainer) -> None:
        self._algo = algo

    def before_iteration(self, iteration: int, last_checkpoint_local_path: Path | None) -> None:
        pass

    def before_new_rollouts(self) -> None:
        """
        Called right before new RL rollouts are requested,
        and thus right before the LoRA adapter is updated on the inference server.
        """
        pass

    def after_iteration(self, iteration: int) -> None:
        pass

    def shutdown(self) -> None:
        pass


class CallbackList:
    def __init__(self, callbacks: list[Callback]) -> None:
        self._callbacks = callbacks

    def __getattr__(self, name: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for cb in self._callbacks:
                getattr(cb, name)(*args, **kwargs)

        return wrapper


class EvalCallback(Callback):
    def __init__(
        self,
        algo: RLOOTrainer,
        wandb_project: str,
        wandb_group: str,
        wandb_run: str | None,
    ) -> None:
        super().__init__(algo)
        self._wandb_project = wandb_project
        self._wandb_group = wandb_group
        self._wandb_run = wandb_run

        self._eval_actor: ray.actor.ActorHandle[Any] | None = None
        self._eval_future: ray.ObjectRef | None = None

    def _start_actor(self) -> None:
        logger.info("Creating new EvalWorker actor...")
        self._eval_actor = EvalWorker.remote(self._algo._full_cfg)  # type: ignore

    def _ensure_actor(self) -> None:
        if self._eval_actor is None:
            self._start_actor()
            return

        try:
            ray.get(self._eval_actor.ping.remote())
        except RayActorError:
            logger.warning("EvalWorker actor is dead -> restarting...")
            self._start_actor()
        except Exception as e:
            logger.exception(f"Unexpected error pinging EvalWorker: {e}")
            self._start_actor()

    def _await_completion(self) -> None:
        if self._eval_future is None:
            return

        try:
            logger.info("Waiting for Ray eval task to complete...")
            ray.get(self._eval_future)
            logger.info("Eval task completed.")
        except RayActorError:
            logger.warning("EvalWorker crashed during eval. Restarting actor...")
            self._start_actor()
        except Exception as e:
            logger.exception(f"Eval failed: {e}")
        finally:
            self._eval_future = None

    def before_iteration(self, iteration: int, last_checkpoint_local_path: Path | None) -> None:
        cfg = self._algo._cfg
        if not cfg.eval.enable or not self._wandb_run:
            logger.debug("Eval callback is disabled...")
            return

        if iteration % cfg.eval.eval_every_n_iterations != 0:
            logger.debug(f"Skipping eval due to {cfg.eval.eval_every_n_iterations=}")
            return

        self._ensure_actor()

        assert self._eval_future is None, f"Previous eval still running: {self._eval_future=}"
        assert self._eval_actor is not None, f"{self._eval_actor=}"
        try:
            self._eval_future = self._eval_actor.eval.remote(last_checkpoint_local_path)
        except RayActorError:
            logger.exception(f"Could not start eval for {last_checkpoint_local_path=}")

    def before_new_rollouts(self) -> None:
        # Make sure eval doesn't overlap LoRA swapping
        self._await_completion()

    def shutdown(self) -> None:
        if self._eval_actor is not None:
            try:
                logger.info("Shutting down EvalWorker actor...")
                ray.get(self._eval_actor.shutdown.remote())
            except RayActorError:
                logger.warning("EvalWorker actor already dead during shutdown.")
