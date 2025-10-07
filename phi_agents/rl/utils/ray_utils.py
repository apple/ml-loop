#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ray

from phi_agents.utils.logger import get_phi_logger

VLLM_RESOURCE = "vllm_server"

BarrierFn = Callable[[], None]

logger = get_phi_logger()


def connect_ray_cluster(rank: int, barrier: BarrierFn) -> None:
    """Connect to or create a ray cluster."""
    namespace = "train"
    log_to_driver = True

    local_mode = False
    if rank == 0:
        logger.info(f"{rank=} ray init {local_mode=}")

        resources = {VLLM_RESOURCE: 1}
        # to avoid confusing ray.init clear visible devices first
        cvd = "CUDA_VISIBLE_DEVICES"
        orig = None
        if cvd in os.environ:
            orig = os.environ[cvd]
            del os.environ[cvd]
        ray.init(
            log_to_driver=log_to_driver,
            namespace=namespace,
            resources=resources,
            local_mode=local_mode,
        )
        if orig is not None:
            os.environ[cvd] = orig

    barrier()

    if rank != 0:
        time.sleep(0.5)
        logger.info(f"{rank=} ray init...")
        ray.init(
            address="auto",
            log_to_driver=log_to_driver,
            namespace=namespace,
        )


def run_on_nodes_with_resource(
    remote_fn: Any, resource_name: str, run_on_current_node: bool, *args: Any, **kwargs: Any
) -> Any:
    """Run remote_fn on all ray nodes that contain resource_name."""
    refs = []
    for node in ray.nodes():
        # For the vllm servers that are not on this host
        if resource_name in node["Resources"] and (
            run_on_current_node or ray.util.get_node_ip_address() != node["NodeManagerAddress"]
        ):
            # This ensures the download happens on the requested node
            node_resource = {f"node:{node['NodeName']}": 0.01}
            refs.append(remote_fn.options(resources=node_resource).remote(*args, **kwargs))

    return ray.get(refs)


@ray.remote  # type: ignore
def copy_dir_to_remote(source_files: list[tuple[bytes, str]], target_dir: str | Path) -> int:
    target_path = Path(target_dir)

    target_path.mkdir(parents=True, exist_ok=True)

    for content, rel_path in source_files:
        dst_path = target_path / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(content)

    return len(source_files)
