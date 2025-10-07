#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import math
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, cast

import hydra
import hydra.utils
import psutil
import ray
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from phi_agents.rl.type_defs import Scenario, ScenarioRunner, TrainingRollout
from phi_agents.rl.utils.download import download_model
from phi_agents.rl.utils.ray_utils import (
    VLLM_RESOURCE,
    BarrierFn,
    connect_ray_cluster,
    copy_dir_to_remote,
    run_on_nodes_with_resource,
)
from phi_agents.utils.file_utils import read_directory_files
from phi_agents.utils.logger import get_phi_logger
from phi_agents.utils.utils import barrier_guard, timeit
from phi_agents.vllm.vllm_server import VLLMServer, get_lora_model_id

if TYPE_CHECKING:
    from collections.abc import Generator

    from omegaconf import DictConfig

    from phi_agents.rl.llm import TrainableLLM
    from phi_agents.rl.parallel_scenario_sampler import ParallelScenarioSampler

START_PORT = 5555

logger = get_phi_logger()

ScenarioIdx = int
ScenariosAndRollouts = tuple[list[Scenario], list[list[TrainingRollout]]]

RolloutGenerationTask = tuple[ScenarioIdx, Scenario, int]


class VLLMRolloutWorker:
    def __init__(
        self,
        *,
        llm_cfg: DictConfig,
        scenario_sampler: ParallelScenarioSampler,
        rollouts_per_scenario: int,
        runner_cfg: DictConfig,
        rank: int,
        local_rank: int,
        barrier: BarrierFn,
        inference_gpus: list[int],
        exclusive_inference_and_learning: bool,
        max_gpu_mem_utilization: float | None,
        start_port: int = START_PORT,
        num_runners: int | None = None,
    ):
        if not ray.is_initialized():
            connect_ray_cluster(rank, barrier)
        self._scenario_sampler = scenario_sampler
        self._rollouts_per_scenario = rollouts_per_scenario
        self._vllm_server_cfg = llm_cfg.vllm_server
        self._llm_cfg = llm_cfg
        self._runner_cfg = runner_cfg
        self._local_rank = local_rank
        self._exclusive_inference_and_learning = exclusive_inference_and_learning
        self._max_gpu_mem_utilization: float | None = max_gpu_mem_utilization
        self._inference_gpus_per_vllm_server: list[list[int]] = [
            l.tolist()
            for l in torch.tensor(inference_gpus).split(
                self._llm_cfg.vllm_server.gpus_per_vllm_server
            )
        ]

        self._vllm_server_ports: list[int] = [
            start_port + i * 2 for i in range(len(self._inference_gpus_per_vllm_server))
        ]
        self._servers: list[ray.remote.ActorHandle[VLLMServer]] | None = None
        self._hosts: list[str] | None = None

        self._loaded_adapter_model_id: str | None = None

        self._queue: Queue[tuple[Scenario, ScenarioIdx, TrainingRollout] | Exception] = Queue()
        self._mutex = threading.RLock()
        self._generation_requested = threading.Condition(self._mutex)
        self._stop_event = threading.Event()

        # set this event to cancel currently active completions
        self._cancellation_event = threading.Event()

        self._n_scenarios_requested: int = 0

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        if num_runners is None:
            # default to one ScenarioRunner (within this process) per inference server
            num_runners = len(self._vllm_server_ports)

        self._scenario_runners: list[ScenarioRunner] = [None] * num_runners  # type: ignore
        self._runner_idx_to_vllm_server_port: list[int] = [None] * num_runners  # type: ignore

        def _create_runner(runner_idx_: int) -> tuple[int, int, ScenarioRunner]:
            global_runner_idx = local_rank * num_runners + runner_idx_
            port = self._vllm_server_ports[global_runner_idx % len(self._vllm_server_ports)]
            runner = cast(
                ScenarioRunner, hydra.utils.instantiate(self._runner_cfg, _recursive_=False)
            )
            return runner_idx_, port, runner

        # parallel creation of scenario runners significantly speeds up startup, especially with AppWorld
        with ThreadPoolExecutor() as executor:
            for i, port, runner in executor.map(_create_runner, range(num_runners)):
                logger.info(f"Scenario runner {i} {port=} is ready!")
                self._runner_idx_to_vllm_server_port[i] = port
                self._scenario_runners[i] = runner

        self._local_base_model_path: Path | None = None
        self._is_base_model_path_distributed = False

    def _start_servers(self) -> tuple[list[ray.actor.ActorHandle[VLLMServer]], Path]:
        assert self._local_rank == 0, "only one process should start the servers"

        server_cls: type = hydra.utils.get_class(self._llm_cfg.inference_server_class)
        servers: list[ray.actor.ActorHandle[Any]] = []

        InferenceServerRemote = ray.remote(server_cls)
        assert len(self._vllm_server_ports) == len(self._inference_gpus_per_vllm_server)
        for i, inference_gpus in enumerate(self._inference_gpus_per_vllm_server):
            vllm_port = self._vllm_server_ports[i]
            vllm_rpc_port = vllm_port + 1
            logger.info(
                f"Starting inference server with GPUs {inference_gpus=}, {vllm_port=} {vllm_rpc_port=}"
            )

            # Require that we are on a node marked with VLLM_RESOURCE
            vllm_server = InferenceServerRemote.options(
                num_gpus=len(inference_gpus),
                name=self._get_actor_server_name(vllm_port),
                resources={VLLM_RESOURCE: 0.01},
            ).remote(
                conf=self._vllm_server_cfg,
                port=vllm_port,
                # https://github.com/vllm-project/vllm/issues/7196
                # NOTE: after https://github.com/vllm-project/vllm/pull/7222/files is released, no longer necessary
                rpc_port=vllm_rpc_port,
                cuda_visible_devices=[str(inference_gpu) for inference_gpu in inference_gpus],
                max_gpu_mem_utilization=self._max_gpu_mem_utilization,
            )
            servers.append(vllm_server)

        # Call any function to ensure the ray actor is scheduled
        ray.get([server.get_node_ip_address.remote() for server in servers])
        logger.info("Downloading for remote servers.")
        # Download the base model to all nodes with vllm_server resource. This has to come after
        # creating the vllm_server actor so that ray can schedule the nodes required w/ autoscaling.
        paths: list[str] = run_on_nodes_with_resource(
            remote_fn=ray.remote(download_model),
            resource_name=VLLM_RESOURCE,
            run_on_current_node=True,
            name_or_path=self._llm_cfg.base_model_path,
            base_dir=Path.cwd(),
        )
        # Ensure all the paths are the same
        assert len(paths) > 0 and paths == [paths[0] * len(paths)]
        logger.info(f"Done downloading for remote servers. got \n{paths=}")
        local_base_model_path = Path(paths[0])

        for i, server in enumerate(servers):
            cpu_affinity_info = (i, len(servers))
            ray.get(
                server.start_server.remote(
                    base_model_path=local_base_model_path, cpu_affinity_info=cpu_affinity_info
                )
            )
        return servers, local_base_model_path

    def await_servers(self) -> None:
        if self._local_rank == 0:
            assert self._servers, f"{self._servers=} {self._local_rank=}"

            start = time.perf_counter()
            logger.info(f"Awaiting {len(self._servers)} servers...")
            ray.get([server.await_startup.remote() for server in self._servers])
            elapsed = time.perf_counter() - start
            logger.info(f"Awaiting {len(self._servers)} servers...Done in {elapsed:.2f} s")

    def _get_actor_server_name(self, port: int) -> str:
        return f"{threading.current_thread().name}_vllm_server:{port}"

    def _reload_lora_adapters(
        self,
        adapter_path: Path | None,
        adapter_model_id: str | None,
        previous_adapter_model_id: str | None,
    ) -> None:
        assert self._servers is not None

        if previous_adapter_model_id is not None:
            logger.info(f"Unloading adapter model ID: {previous_adapter_model_id}...")

            refs = [
                server.unload_lora_adapter.remote(previous_adapter_model_id)
                for server in self._servers
            ]
            ray.get(refs)

        if adapter_path is not None and adapter_model_id is not None:
            logger.info("Resetting prefix cache...")
            # Should not be strictly necessary because we change the LORA adapter between iterations
            # anyway and the prefix cache is adapter-specific.
            # But should not hurt to explicitly clear the old prefix cache since we won't need it again.
            # If we ever have a version of full fine tune that does not require vllm restart,
            # this will definitely be necessary.
            refs = [server.reset_prefix_cache.remote() for server in self._servers]
            ray.get(refs)

            model_name = self._llm_cfg.vllm_class._target_.lower().split(".")[-1]
            logger.info(
                f"Loading adapter model ID: {adapter_model_id} path {adapter_path} {model_name=}..."
            )

            files_data = read_directory_files(adapter_path)
            target_dir = adapter_path.parent / f"{adapter_path.stem}_{model_name}"

            copy_results = run_on_nodes_with_resource(
                copy_dir_to_remote,
                resource_name=VLLM_RESOURCE,
                run_on_current_node=True,
                source_files=files_data,
                target_dir=str(target_dir),
            )

            for i, result in enumerate(copy_results):
                logger.info(f"Node {i}: Copied {result} files to {target_dir}")

            refs = [
                server.load_lora_adapter.remote(target_dir, adapter_model_id)
                for server in self._servers
            ]
            ray.get(refs)

    def _submit_scenarios_and_get_rollouts(
        self, n_scenarios: int
    ) -> Generator[tuple[Scenario, ScenarioIdx, TrainingRollout], None, None]:
        # submit an asynchronous task for reach rollout
        assert self._hosts is not None

        # construct an LLM for each server
        llms: list[TrainableLLM] = []
        for host, port in zip(self._hosts, self._vllm_server_ports, strict=True):
            llms.append(
                hydra.utils.instantiate(
                    self._llm_cfg.vllm_class,
                    host=host,
                    port=port,
                    base_model_path=self._local_base_model_path,
                    model_id=self._loaded_adapter_model_id,
                    temperature=self._llm_cfg.temperature,
                    max_model_len=self._vllm_server_cfg.max_model_len,
                    cancellation_event=self._cancellation_event,
                )
            )

        # mapping from rollout generation tasks to tuples (scenario_idx, scenario, runner_idx)
        futures: dict[Future[TrainingRollout], RolloutGenerationTask] = {}

        task_queue: Queue[tuple[ScenarioIdx, Scenario, TrainableLLM]] = Queue()
        n_tasks = 0
        scenarios = [next(self._scenario_sampler) for _ in range(n_scenarios)]
        for _rollout_idx in range(self._rollouts_per_scenario):
            for sc_idx, scenario in enumerate(scenarios):
                llm = llms[(n_tasks + self._local_rank) % len(llms)]
                task_queue.put((sc_idx, scenario, llm))
                n_tasks += 1

        with ThreadPoolExecutor(max_workers=len(self._scenario_runners)) as executor:

            def _maybe_submit_task(runner_idx_: int) -> bool:
                try:
                    next_task = task_queue.get_nowait()
                    sc_idx, scenario, llm = next_task
                    future = executor.submit(self._scenario_runners[runner_idx_].run, scenario, llm)
                    futures[future] = (sc_idx, scenario, runner_idx_)
                    return True
                except Empty:
                    return False

            # submit one task to each scenario runner
            # for blocking scenario runners, this is the maximum amount of work we can do in parallel
            for runner_idx in range(len(self._scenario_runners)):
                if not _maybe_submit_task(runner_idx):
                    break

            while futures:
                completed_future = next(as_completed(futures.keys()))
                this_sc_idx, this_scenario, this_runner_idx = futures[completed_future]
                rollout = completed_future.result()
                yield this_scenario, this_sc_idx, rollout

                futures.pop(completed_future)

                # scenario runner `runner_idx` is now free, submit more tasks if we have any
                _maybe_submit_task(this_runner_idx)

        # Request scenario generation for the next iteration early, so we can
        # start working on them in the background while learning is
        # happening. This assumes that n_scenarios does not change between training iterations.
        logger.info(f"rank{self._local_rank}: Request scenario generation...")
        self._scenario_sampler.ensure_scenarios_requested(n_scenarios)

    def _stop_servers(self, shutdown: bool = False) -> None:
        assert self._servers is not None
        ray.get(
            [vllm_server.stop_server.remote(shutdown=shutdown) for vllm_server in self._servers]
        )

        for vllm_server in self._servers:
            try:
                ray.get(vllm_server.await_termination.remote(shutdown=shutdown))
            except ray.exceptions.RayTaskError as e:
                if isinstance(e.cause, subprocess.TimeoutExpired | psutil.TimeoutExpired):
                    ray.get(vllm_server.terminate_forcefully.remote())
                else:
                    raise e.cause from e
            finally:
                ray.kill(vllm_server)

        self._servers = None
        # no LORA adapters are loaded since we destroyed the servers
        self._loaded_adapter_model_id = None

    def start_servers(self) -> None:
        if self._local_rank == 0 and self._servers is None:
            with timeit("start_inference_servers", logger):
                self._servers, self._local_base_model_path = self._start_servers()

    def request_rollout_generation(
        self, n_scenarios: int, adapter_path: Path | None, reload_lora: bool = True
    ) -> None:
        """Initialize the rollout worker: start the server, load the correct LORA weights, etc."""
        # make sure all processes finished all other work before we load new weights
        # prevents potential issue where a straggler VLLM process still processes requests
        # with the previous version of the weights
        # also: sync after all VLLM servers are up and have the latest copy of weights

        with barrier_guard(before=True, after=True):
            if self._local_rank == 0:
                logger.info(f"Requesting rollout generation with: {adapter_path=}")

            _adapter_model_id = (
                None if adapter_path is None else get_lora_model_id(adapter_path.parent.name)
            )
            logger.info(f"request_rollout_generation() {_adapter_model_id=}")

            if self._local_rank == 0:
                if self._servers is None:
                    self.start_servers()
                    self.await_servers()

                if reload_lora:
                    with timeit("reload_lora_adapters", logger):
                        self._reload_lora_adapters(
                            adapter_path, _adapter_model_id, self._loaded_adapter_model_id
                        )

                local_base_model_path = [self._local_base_model_path]
            else:
                local_base_model_path = [None]

            self._loaded_adapter_model_id = _adapter_model_id

        if not self._is_base_model_path_distributed:
            if torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(local_base_model_path, src=0)

            assert isinstance(local_base_model_path[0], Path)
            self._local_base_model_path = local_base_model_path[0]
            self._is_base_model_path_distributed = True

        _local_ip = ray.util.get_node_ip_address()

        if self._hosts is None:
            self._hosts = []
            for port in self._vllm_server_ports:
                server = ray.get_actor(self._get_actor_server_name(port))
                node_ip = ray.get(server.get_node_ip_address.remote())
                if node_ip == _local_ip:
                    logger.info("Inference server is on the same node, use loopback")
                    self._hosts.append("127.0.0.1")  # same node -- use loopback for simplicity
                else:
                    self._hosts.append(node_ip)

        logger.info(f"{self._hosts=} {_local_ip=}")

        # actually ask the worker thread to start working on new rollouts
        with self._mutex:
            self._n_scenarios_requested = n_scenarios
            self._generation_requested.notify_all()

    def get_rollouts(
        self,
        n_scenarios: int,
        rollouts_fraction: float = 1.0,
        rollouts_per_scenario_fraction: float = 1.0,
        world_size: int = 1,
        device: torch.device | None = None,
        show_progress: bool = False,
    ) -> ScenariosAndRollouts:
        scenarios: list[Scenario | None] = [None] * n_scenarios
        rollouts: list[list[TrainingRollout]] = [[] for _ in range(n_scenarios)]

        n_rollouts_collected = n_rollouts_cancelled = 0
        n_rollouts_total = n_scenarios * self._rollouts_per_scenario

        # unequal size batches across workers are not supported, so we require this to be a multiple of world_size
        world_rollouts_required = max(
            world_size, math.ceil(n_rollouts_total * rollouts_fraction) * world_size
        )
        assert world_rollouts_required > 0 and world_rollouts_required % world_size == 0
        rollouts_per_scenario_required = max(
            1, round(self._rollouts_per_scenario * rollouts_per_scenario_fraction)
        )

        sufficient_rollouts_collected = False

        progress = tqdm(total=n_rollouts_total, unit="rollout", disable=not show_progress)
        while True:
            try:
                msg = self._queue.get(block=True, timeout=1.0)
                if isinstance(msg, Exception):
                    exception = msg
                    logger.warning(f"rank{self._local_rank}: {exception=}")
                    # if we do not propagate an exception like this, the main thread will hang
                    # forever waiting for more items from the queue
                    raise exception

                scenario, scenario_idx, rollout = msg
                if rollout.cancelled:
                    n_rollouts_cancelled += 1
                else:
                    scenarios[scenario_idx] = scenario
                    rollouts[scenario_idx].append(rollout)
                    n_rollouts_collected += 1
                    logger.info(
                        f"rank{self._local_rank}: {scenario_idx=} {n_rollouts_collected=} {n_rollouts_cancelled=} {n_rollouts_total=}"
                    )

                progress.update(1)

                continue  # spin until we exhaust completed rollouts in the queue
            except Empty:
                pass

            min_rollouts_per_scenario = min(
                len(rollouts[scenario_idx]) for scenario_idx in range(n_scenarios)
            )

            if sufficient_rollouts_collected:
                stats_msg = f"{sufficient_rollouts_collected=} {n_rollouts_collected=} {n_rollouts_cancelled=} {min_rollouts_per_scenario=} {n_rollouts_total=}"
                if not self._cancellation_event.is_set():
                    logger.info(f"Cancelling rollout collection: {stats_msg}")
                    self._cancellation_event.set()

                if n_rollouts_collected + n_rollouts_cancelled >= n_rollouts_total:
                    logger.info(f"Rollout collection completed: {stats_msg}")
                    # important: do not break the loop until a sufficient number of rollouts is collected
                    # across the entire system, even if this particular worker is 100% done
                    break

            else:
                assert n_rollouts_cancelled == 0

                if dist.is_initialized():
                    # it is important that all workers enter this section the same number of times
                    assert device is not None

                    n_collected_ = torch.tensor(
                        n_rollouts_collected, dtype=torch.int64, device=device
                    )
                    min_rollouts_ = torch.tensor(
                        min_rollouts_per_scenario, dtype=torch.int64, device=device
                    )

                    dist.all_reduce(n_collected_, op=dist.ReduceOp.SUM)
                    dist.all_reduce(min_rollouts_, op=dist.ReduceOp.MIN)
                    world_rollouts_collected = n_collected_.item()
                    world_min_per_scenario = min_rollouts_.item()
                else:
                    assert world_size == 1
                    world_rollouts_collected = n_rollouts_collected
                    world_min_per_scenario = min_rollouts_per_scenario

                if (
                    world_rollouts_collected >= world_rollouts_required
                    and world_min_per_scenario >= rollouts_per_scenario_required
                ):
                    sufficient_rollouts_collected = True

        progress.close()

        assert len(scenarios) == len(rollouts) == n_scenarios, "Incorrect number of scenarios"
        assert all(s is not None for s in scenarios), f"Found None in {scenarios=}"

        assert all(
            len(r) >= min_rollouts_per_scenario for r in rollouts
        ), "Incorrect number of rollouts generated per scenario"

        if self._exclusive_inference_and_learning:
            # we need to wait for all workers to get their rollouts before we're ready to terminate
            # the server (in sync version)
            # also: after this section all learner GPU workers should wait for the VLLM servers to stop to avoid OOM issues
            with barrier_guard(before=True, after=True):
                if self._local_rank == 0:
                    with timeit("stop_servers", logger):
                        self._stop_servers()

        return cast(list[Scenario], scenarios), rollouts

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            with self._mutex:
                if self._n_scenarios_requested <= 0:
                    self._generation_requested.wait(timeout=1.0)
                    continue

                n_requested = self._n_scenarios_requested
                self._n_scenarios_requested = 0

            self._cancellation_event.clear()

            try:
                for scenario, sc_idx, rollout in self._submit_scenarios_and_get_rollouts(
                    n_requested
                ):
                    self._queue.put((scenario, sc_idx, rollout))
            except Exception as exc:
                logger.warning(f"Exception {exc} in the worker thread")
                self._queue.put(exc)
                break

        logger.info(f"RolloutWorker thread exits...")

    def stop(self) -> None:
        logger.info("Cancelling rollout generation...")
        self._cancellation_event.set()

        with self._mutex:
            self._generation_requested.notify_all()

        self._stop_event.set()
        self._worker_thread.join()
        with ThreadPoolExecutor(max_workers=len(self._scenario_runners)) as executor:
            futures = [executor.submit(runner.cleanup) for runner in self._scenario_runners]
            logger.info(f"Waiting for {len(self._scenario_runners)} scenario runners to finish...")
            wait(futures)

        if self._servers is not None:
            self._stop_servers(shutdown=True)
