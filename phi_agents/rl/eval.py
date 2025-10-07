#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import itertools
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import hydra
import hydra.utils
import ray
import wandb
from omegaconf import DictConfig, OmegaConf
from torchtune.training.checkpointing._utils import safe_torch_load

import phi_agents.utils.file_utils as fu
from phi_agents.rl.config import register_hydra_resolvers
from phi_agents.rl.parallel_scenario_sampler import ParallelScenarioSampler
from phi_agents.rl.rl_utils import wandb_init
from phi_agents.rl.type_defs import Scenario, TrainingRollout
from phi_agents.rl.vllm_rollout_worker import VLLMRolloutWorker
from phi_agents.utils.logger import get_phi_logger
from phi_agents.utils.utils import torch_dist_barrier

logger = get_phi_logger()


class RolloutSummaries(ABC):
    @abstractmethod
    def summary(
        self, scenarios: list[Scenario], rollouts: list[TrainingRollout]
    ) -> dict[str, float]: ...


@ray.remote
class EvalWorker:
    def __init__(self, cfg: DictConfig) -> None:
        threading.current_thread().name = "EvalLoopMain"
        register_hydra_resolvers()

        if "scenario_sampler" in cfg.rl.eval.overrides:
            raise ValueError(
                "rl.eval.overrides.scenario_sampler is not allowed. "
                "The scenario sampler used by eval is independently configured at rl.eval.scenario_sampler."
            )

        self._cfg = OmegaConf.merge(cfg, cfg.rl.eval.overrides)
        del cfg
        assert isinstance(self._cfg, DictConfig)
        logger.info(f"Eval cfg: {OmegaConf.to_yaml(self._cfg)}")
        logger.info(f"{self._cfg.rl.params.rollouts_per_scenario=}")

        self._scenario_sampler = ParallelScenarioSampler(
            create_sampler_func=lambda: hydra.utils.instantiate(self._cfg.rl.eval.scenario_sampler),
            num_threads=1,
        )

        assert (
            self._cfg.llm.vllm_server.allow_connect_to_existing is True
        ), "Local RL eval loop relies on allow_connect_to_existing==True"

        self._rollout_worker = VLLMRolloutWorker(
            scenario_sampler=self._scenario_sampler,
            rollouts_per_scenario=self._cfg.rl.params.rollouts_per_scenario,
            runner_cfg=self._cfg.rl.scenario_runner,
            rank=0,
            local_rank=0,
            barrier=torch_dist_barrier,
            inference_gpus=self._cfg.rl.gpu_allocation.inference_gpus,  # local eval loop shares inference GPUs with the training process
            exclusive_inference_and_learning=False,
            max_gpu_mem_utilization=None,
            num_runners=self._cfg.rl.num_scenario_runners,
            llm_cfg=self._cfg.llm,
        )

        if self._cfg.wandb.enable:
            wandb_init(self._cfg, run_name_suffix="_EVAL")

        self._summaries: RolloutSummaries = hydra.utils.instantiate(self._cfg.rl.eval.summaries)

    def ping(self) -> str:
        return "OK"

    def eval(self, last_checkpoint_local_path: Path | None) -> None:
        try:
            logger.debug(f"eval() {last_checkpoint_local_path=}")

            start = time.perf_counter()

            if last_checkpoint_local_path is not None:
                last_checkpoint_local_path = Path(last_checkpoint_local_path)

                try:
                    trainer_state = safe_torch_load(
                        last_checkpoint_local_path / "trainer_state.pt", mmap=False
                    )
                    training_iteration = trainer_state["iterations_completed"]
                    global_step = trainer_state["global_step"]
                except ValueError as exc:
                    # corrupted checkpoint
                    logger.exception(f"Could not load checkpoint: {exc}")
                    training_iteration = global_step = None

            else:
                training_iteration = global_step = 0

            lora_path = fu.lora_path(last_checkpoint_local_path)

            # do not reload LoRA, local eval loop relies on the right adapter already being loaded
            # we will fail if a different adapter is loaded
            logger.info(
                f"request_rollout_generation {lora_path=} {self._cfg.rl.params.scenarios_per_iteration=}"
            )
            self._rollout_worker.request_rollout_generation(
                self._cfg.rl.params.scenarios_per_iteration, lora_path, reload_lora=False
            )
            scenarios, rollouts = self._rollout_worker.get_rollouts(
                self._cfg.rl.params.scenarios_per_iteration
            )
            rollouts_flat = list(itertools.chain(*rollouts))  # flatten the list of lists

            elapsed = time.perf_counter() - start
            logger.debug(f"Finished eval iteration, took {elapsed:.1f} s")
            n_output_tokens = sum([sum(r.policy_token_info.is_output) for r in rollouts_flat])
            perf_summary = {
                "eval_perf/output_tokens_per_iteration": n_output_tokens,
                "eval_perf/output_tokens_per_sec": n_output_tokens / elapsed,
                "eval_perf/rollouts_per_sec": len(rollouts_flat) / elapsed,
            }
            logger.debug(f"{n_output_tokens=} {perf_summary=}")

            if self._cfg.wandb.enable:
                wandb.log(
                    dict(
                        global_step=global_step,
                        iterations_completed=training_iteration,
                        **perf_summary,
                        **self._summaries.summary(scenarios, rollouts_flat),
                    )
                )

        except KeyboardInterrupt:
            logger.warning(f"Interrupted")
        except Exception as exc:
            logger.exception(f"Unhandled exception in eval loop: {exc}")

    def shutdown(self) -> None:
        logger.info(f"Stopping components...")
        self._scenario_sampler.stop()
        self._rollout_worker.stop()
