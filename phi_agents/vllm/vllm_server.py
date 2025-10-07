#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import os
import subprocess
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import psutil
import ray
import requests

from phi_agents.utils.cuda import cuda_arc_to_major, get_cuda_architecture
from phi_agents.utils.file_utils import get_path_to_python_env_bin
from phi_agents.utils.logger import get_phi_logger

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

logger = get_phi_logger()

LORA_MODEL_ID = "phi_lora_model"


@dataclass(frozen=True)
class VLLMServerInfo:
    """Server info e.g. used by client."""

    host: str
    port: int


def get_lora_model_id(checkpoint_name: str) -> str:
    return f"{LORA_MODEL_ID}_{checkpoint_name}"


def find_existing_vllm_process(port: int) -> psutil.Process | None:
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            if not proc.info["cmdline"]:
                continue
            cmd = proc.info["cmdline"]
            try:
                serve_idx = cmd.index("serve")
            except ValueError:
                continue

            if serve_idx > 0 and "vllm" in cmd[serve_idx - 1]:
                logger.info(f"{proc=} {proc.info['cmdline']=}")

                # Search for `--port` followed by our target port
                for i in range(len(cmd) - 1):
                    if cmd[i] == "--port" and cmd[i + 1] == str(port):
                        return proc
                for i in range(len(cmd)):
                    if cmd[i] == f"--port={str(port)}":
                        return proc

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def processes_using_port(port: str) -> list[psutil.Process]:
    processes = []
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            process = psutil.Process(conn.pid)
            processes.append(process)
    return processes


def kill_process_by_port(port: str) -> bool:
    for process in processes_using_port(port):
        try:
            logger.info(f"Killing process {process.name()} (PID: {process.pid}) using port {port}")
            process.kill()
            return True
        except psutil.AccessDenied:
            logger.info(f"Permission denied when trying to kill process (PID: {process.pid})")
        except psutil.NoSuchProcess:
            logger.info(f"No such process (PID: {process.pid}) exists anymore")
    logger.info(f"No process found using port {port}")
    return False


def suppress_train_envvars(env: dict[str, str]) -> dict[str, str]:
    # remove some training process envvars from vllm's envvars (e.g. torch.distributed/HF Accelerate)
    # this is required for newer versions of vllm >= 0.6.5 as they interfere with vllm's own parallelism
    for var in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
        "LOCAL_WORLD_SIZE",
        # this is just to make sure this envvar does not propagate to vllm if it's set for the trainer
        # if needed, this can be set separately for vllm together with other envvars below
        "TORCHDYNAMO_DISABLE",
        # vLLM breaks with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        "PYTORCH_CUDA_ALLOC_CONF",
    ]:
        env.pop(var, None)

    for var in list(env.keys()):
        if var.startswith("ACCELERATE") or var.startswith("FSDP"):
            env.pop(var, None)

    return env


class VLLMServer:
    _port: int
    _rpc_port: int
    _base_model_path: Path | None
    _process: psutil.Process | None = None
    _child_processes_to_stop: list[psutil.Process] = []
    _max_gpu_mem_utilization: float | None = None
    _connected_to_existing: bool = False

    @dataclass
    class Conf:
        """Configuration for VLLMServer."""

        executable: str = "vllm"
        max_model_len: int | None = None
        disable_log_stats: bool = False
        uvicorn_log_level: str = (  # Literal["debug", "info", "warning", "error", "critical", "trace"] = (
            "warning"
        )
        max_tries: int = 240
        wait_seconds: float = 1.0
        enable_lora: bool = True
        enable_prefix_caching: bool = True
        seed: int | None = None  # will use a random seed if None
        gpus_per_vllm_server: int = 1
        max_lora_rank: int = 16
        torch_dtype: str = "bfloat16"
        eager_mode: bool = False
        allow_connect_to_existing: bool = True

    def __init__(
        self,
        # NOTE: static errors are not created on accessing undefined DictConfig attributes
        conf: VLLMServer.Conf | DictConfig,
        port: int = 8000,
        rpc_port: int = 5570,
        cuda_visible_devices: list[str] | None = None,
        max_gpu_mem_utilization: float | None = None,
    ):
        self._port = port
        self._rpc_port = rpc_port
        self.host = "localhost"
        self._url = f"http://{self.host}:{self._port}"
        self._base_model_path = None
        self._cuda_visible_devices = cuda_visible_devices
        self._conf = conf
        self._max_gpu_mem_utilization = max_gpu_mem_utilization

    @property
    def port(self) -> int:
        return self._port

    @property
    def info(self) -> VLLMServerInfo:
        return VLLMServerInfo(host=self.host, port=self.port)

    def get_node_ip_address(self) -> str:
        return cast(str, ray.util.get_node_ip_address())

    def start_server(
        self, base_model_path: Path, cpu_affinity_info: tuple[int, int] | None = None
    ) -> None:
        assert self._process is None
        self._base_model_path = base_model_path

        env = os.environ.copy()
        env = suppress_train_envvars(env)

        # turn off telemetry
        # https://docs.vllm.ai/en/latest/serving/usage_stats.html
        env["VLLM_NO_USAGE_STATS"] = "0"

        env["VLLM_LOGGING_LEVEL"] = "INFO"

        # needed for reset_prefix_cache https://docs.vllm.ai/en/stable/serving/env_vars.html
        env["VLLM_SERVER_DEV_MODE"] = "1"

        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # limit to certain GPUs
        if self._cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self._cuda_visible_devices)

        # we use our own entrypoint instead of the vllm executable as a workaround for
        # https://github.com/vllm-project/vllm/issues/7196
        env["VLLM_RPC_PORT"] = str(self._rpc_port)
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = str(True)

        executable = get_path_to_python_env_bin("vllm", override_path=self._conf.executable)

        args = [
            str(executable),
            "serve",
            "--disable-log-requests",
            "--port",
            str(self._port),
            self._base_model_path.as_posix(),
            "--return-tokens-as-token-ids",
        ]

        if get_cuda_architecture() <= cuda_arc_to_major["volta"]:
            args.append("--dtype=half")
        else:
            args.append(f"--dtype={self._conf.torch_dtype}")

        if self._conf.enable_lora:
            # https://docs.vllm.ai/en/latest/models/lora.html#serving-lora-adapters
            args.extend(["--enable-lora"])

        max_capture_length = 32768
        if self._conf.max_model_len is not None:
            args.extend(["--max-model-len", str(self._conf.max_model_len)])
            max_capture_length = min(max_capture_length, self._conf.max_model_len)

        if self._conf.disable_log_stats:
            args.append("--disable-log-stats")

        if self._conf.enable_prefix_caching:
            if get_cuda_architecture() <= cuda_arc_to_major["volta"]:
                # https://github.com/vllm-project/vllm/issues/6723#issuecomment-2251907301
                warnings.warn("Prefix caching not supported by Volta — leaving it off.")  # noqa: B028
            else:
                args.append("--enable-prefix-caching")

        if self._max_gpu_mem_utilization is not None:
            args.append(f"--gpu-memory-utilization={self._max_gpu_mem_utilization:0.2f}")

        if hasattr(self._conf, "max_lora_rank") and self._conf.max_lora_rank is not None:
            args.append(f"--max-lora-rank={self._conf.max_lora_rank}")

        # vLLM defaults to seed 0, we either use a random seed or specific one if requested.
        if self._conf.seed is None:
            seed = uuid.uuid4().int % (2**32)
        else:
            seed = self._conf.seed
        args.append(f"--seed={seed}")

        args.append(f"--uvicorn-log-level={self._conf.uvicorn_log_level}")

        num_gpus = 1 if self._cuda_visible_devices is None else len(self._cuda_visible_devices)
        tensor_parallel = num_gpus
        pipeline_parallel = num_gpus // tensor_parallel
        args.extend(
            [
                "--tensor-parallel-size",
                str(tensor_parallel),
                "--pipeline-parallel-size",
                str(pipeline_parallel),
            ]
        )

        args.extend(
            [
                "--swap-space=16",
                "--disable-sliding-window",
                "--disable-cascade-attn",  # this apparently causes some issues on p4de if not disabled?
                f"--max-seq-len-to-capture={max_capture_length}",  # by @jackson
            ]
        )

        args.append("--generation-config=vllm")
        # --override-generation-config should not be necessary, --generation-config=vllm already wipes out model-specific settings in generation_config.json

        # only require logprob for the SAMPLED token
        args.append("--max-logprobs=1")

        if self._conf.eager_mode:
            args.append("--enforce-eager")

        logger.info(
            f"Just in case, checking if we still need to kill the process using port {str(self.port)}..."
        )

        if self._conf.allow_connect_to_existing:
            existing_vllm_proc = find_existing_vllm_process(self.port)
            if existing_vllm_proc is not None:
                logger.info(f"Will attempt to connect to existing vLLM process on port {self.port}")
                self._process = existing_vllm_proc
                self._connected_to_existing = True
                return
            else:
                logger.info(f"Could not find {existing_vllm_proc=} {self.port=}")

        kill_process_by_port(str(self.port))
        logger.info(f"Launching vllm subprocess {self._url}...")
        logger.info(args)
        logger.info(f"{env=}")
        _popen = subprocess.Popen(args, env=env)
        self._process = psutil.Process(_popen.pid) if _popen else None
        logger.info(f"{_popen=} {self._process=}")

    def await_startup(self) -> None:
        assert self._base_model_path is not None
        assert self._process is not None

        start_wait = time.perf_counter()
        logger.info(f"Waiting for {self._url}/v1/models response..")
        try:
            response = None
            for _ in range(self._conf.max_tries):
                try:
                    response = requests.get(f"{self._url}/v1/models", timeout=5)
                except requests.exceptions.ConnectionError:
                    logger.info(
                        f"Still waiting for inference server init, {time.perf_counter() - start_wait:.1f}s..."
                    )
                    pass
                if response is not None:
                    break
                time.sleep(self._conf.wait_seconds)
            if response is None:
                raise RuntimeError(f"Could not connect to {self._url}")

            # sanity check on response
            models = response.json()["data"]
            model_ids = [m["id"] for m in models]
            assert (
                self._base_model_path.as_posix() in model_ids
            ), f"{model_ids=} {self._base_model_path.as_posix()=}"

        except Exception:
            self.stop_server()
            raise

    def stop_server(self, shutdown: bool = False) -> None:
        if shutdown and self._conf.allow_connect_to_existing and self._connected_to_existing:
            logger.info(f"Keep the vLLM process running, we don't want to terminate...")
            return

        assert self._process is not None
        logger.info(f"Sending SIGTERM command to the subprocess {self._url}...")
        self._child_processes_to_stop = self._process.children(recursive=True)
        self._process.terminate()
        time.sleep(0.5)

        for child in self._child_processes_to_stop:
            if child.is_running():
                logger.info(f"Sending SIGTERM to child process {child.pid}...")
                child.terminate()

    def await_termination(self, timeout: float | None = 10.0, shutdown: bool = False) -> None:
        """
        TimeoutException is raised if this hits the timeout. Using timeout=None risks freezing
        the entire training job indefinitely.
        """
        if shutdown and self._conf.allow_connect_to_existing and self._connected_to_existing:
            logger.info(f"Keep the vLLM process running, we don't want to terminate...")
            return

        assert self._process is not None
        logger.info(f"Waiting for subprocess {self._url} and children to terminate...")
        try:
            # Wait for both main and child processes together
            psutil.wait_procs([self._process] + self._child_processes_to_stop, timeout=timeout)

        except psutil.TimeoutExpired:
            logger.info("Timeout expired while waiting for processes to terminate")
            raise

        # processes successfully terminated
        self._process = None
        self._child_processes_to_stop = []

    def terminate_forcefully(self) -> None:
        assert self._process is not None
        logger.info(f"Sending SIGKILL command to the subprocess {self._url}...")
        self._process.kill()
        for child in self._child_processes_to_stop:
            if child.is_running():
                logger.info(f"Sending SIGKILL to child process {child.pid}...")
                child.kill()

        self._process = None
        self._child_processes_to_stop = []

    def reset_prefix_cache(self) -> None:
        try:
            response = None
            for attempt in range(self._conf.max_tries):
                logger.info(f"Resetting prefix cache... attempt={attempt}")
                try:
                    response = requests.post(f"{self._url}/reset_prefix_cache")
                except requests.exceptions.ConnectionError:
                    logger.warning("Connection error during reset_prefix_cache request")
                if response is not None:
                    break
                time.sleep(self._conf.wait_seconds)

            if response is None:
                raise RuntimeError("Could not reset prefix cache")

            if response.status_code == 404:
                logger.info("Reset prefix cache not supported by this version of VLLM")
                return

            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to reset prefix cache: {response.text} ({response.status_code=})"
                )
        except Exception as e:
            logger.error(f"Error resetting prefix cache: {str(e)}")
            raise

    def unload_lora_adapter(self, model_id: str) -> None:
        try:
            response = None
            for attempt in range(self._conf.max_tries):
                logger.info(f"Unloading LORA adapter... {attempt=}")
                try:
                    response = requests.post(
                        f"{self._url}/v1/unload_lora_adapter",
                        json={"lora_name": model_id},
                    )
                except requests.exceptions.ConnectionError:
                    pass
                if response is not None:
                    break
                time.sleep(self._conf.wait_seconds)

            if response is None:
                raise RuntimeError(f"Could not unload LORA adapter {model_id}")

            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to unload LORA adapter: {response.text} {response.status_code=}"
                )

        except Exception as e:
            logger.info(f"Error loading LORA adapter: {str(e)}")
            raise

    def load_lora_adapter(self, adapter_path: Path, model_id: str) -> None:
        try:
            response = None
            for attempt in range(self._conf.max_tries):
                is_dir = adapter_path.is_dir()
                logger.info(
                    f"Loading LORA adapter {adapter_path=} {is_dir=} {model_id=}... {attempt=}"
                )
                try:
                    response = requests.post(
                        f"{self._url}/v1/load_lora_adapter",
                        json={"lora_name": model_id, "lora_path": adapter_path.as_posix()},
                    )
                except requests.exceptions.ConnectionError:
                    pass
                if response is not None:
                    break
                time.sleep(self._conf.wait_seconds)

            if response is None:
                raise RuntimeError(f"Could not load LORA adapter from {adapter_path}")

            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to load LORA adapter: {response.text} {response.status_code=}"
                )

        except Exception as e:
            logger.info(f"Error loading LORA adapter: {str(e)}")
            raise
