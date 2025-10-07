#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Run inference on AppWorld tasks."""

import itertools
import shutil
import threading
from pathlib import Path
from typing import cast

import hydra
import hydra.utils
import torch
from omegaconf import OmegaConf

# needed to get the config cs.store registered
import phi_agents.rl.config as rl_config
from phi_agents.evals.appworld_rollout_data import AppWorldTrainingRollout
from phi_agents.inference.config import MainInferenceConfig
from phi_agents.rl.parallel_scenario_sampler import ParallelScenarioSampler
from phi_agents.rl.utils.download import download_adapter, download_model
from phi_agents.rl.vllm_rollout_worker import VLLMRolloutWorker
from phi_agents.utils.appworld import get_episode_path
from phi_agents.utils.file_utils import lora_path
from phi_agents.utils.logger import get_phi_logger
from phi_agents.utils.utils import torch_dist_barrier
from phi_agents.visualization.rollouts import convert_rollout_to_episode

logger = get_phi_logger()


@hydra.main(version_base=None, config_path="../phi_agents/rl/conf", config_name="appworld_eval")  # type: ignore[misc]
def main(cfg: MainInferenceConfig) -> None:
    """Run inference."""
    rl_config.register_hydra_resolvers()

    appworld_cfg = cfg.scenario_runner.appworld_config
    assert appworld_cfg.experiment_name
    print(OmegaConf.to_yaml(cfg))
    print("Experiment name:", appworld_cfg.experiment_name)
    print("Dataset:", cfg.scenario_sampler.dataset_name)

    # Determine local version of base_model_path if necessary
    local_base_model_path = download_model(str(cfg.llm.base_model_path))
    cfg.llm.base_model_path = local_base_model_path

    if cfg.llm.adapter_path is not None:
        local_adapter_path = download_adapter(cfg.llm.adapter_path)
        downloaded_locally = local_adapter_path != cfg.llm.adapter_path
        cfg.llm.adapter_path = Path(local_adapter_path)

    cancellation_event = threading.Event()

    try:
        sampler = ParallelScenarioSampler(
            create_sampler_func=lambda: hydra.utils.instantiate(cfg.scenario_sampler)
        )

        rollout_worker = VLLMRolloutWorker(
            scenario_sampler=sampler,
            rollouts_per_scenario=1,
            runner_cfg=cfg.scenario_runner,
            rank=0,
            local_rank=0,
            barrier=torch_dist_barrier,
            inference_gpus=list(range(torch.cuda.device_count())),
            exclusive_inference_and_learning=False,
            max_gpu_mem_utilization=None,
            num_runners=cfg.num_scenario_runners,
            llm_cfg=cfg.llm,
        )

        n_scenarios = rl_config.appworld_split_to_num_scenarios(cfg.scenario_sampler.dataset_name)
        rollout_worker.request_rollout_generation(
            n_scenarios=n_scenarios,
            adapter_path=lora_path(cfg.llm.adapter_path),
        )

        _scenarios, rollouts = rollout_worker.get_rollouts(
            n_scenarios=n_scenarios,
            show_progress=True,
            rollouts_fraction=1,
        )

        rollouts_flat = list(itertools.chain(*rollouts))

        for rollout in rollouts_flat:
            episode = convert_rollout_to_episode(
                cast(AppWorldTrainingRollout, rollout), experiment_name=appworld_cfg.experiment_name
            )
            json_path = get_episode_path(appworld_cfg.experiment_name, episode.task.task_id)
            episode.save(json_path)
            logger.debug(f"Episode {episode.task.task_id=} saved to {json_path=}")

        logger.debug("Done")

    except KeyboardInterrupt:
        logger.warning("Interrupted! Cancelling the rollouts!")
        cancellation_event.set()
        raise
    finally:
        if cfg.llm.adapter_path is not None and downloaded_locally:
            shutil.rmtree(cfg.llm.adapter_path)
            logger.debug(f"Deleted local adapter copy {cfg.llm.adapter_path}...")


if __name__ == "__main__":
    main()
