#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
A simple script to check if distributed training works correctly.
This should succeed if the NCCL installation is healthy.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank: int, world_size: int) -> None:
    print(f"Running on rank {rank}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 1) Gather device indices on rank 0
    device_idx = torch.tensor([rank], device=f"cuda:{rank}", dtype=torch.int64)
    gathered_devices = [
        torch.zeros(1, device=f"cuda:{rank}", dtype=torch.int64) for _ in range(world_size)
    ]
    dist.all_gather(gathered_devices, device_idx)
    if rank == 0:
        device_indices = [t.item() for t in gathered_devices]
        print("Gathered device indices on rank 0:", device_indices)

    # 2) Gather process IDs on all ranks
    pid = os.getpid()
    pid_tensor = torch.tensor([pid], device=f"cuda:{rank}", dtype=torch.int64)
    gathered_pids = [
        torch.zeros(1, device=f"cuda:{rank}", dtype=torch.int64) for _ in range(world_size)
    ]
    dist.all_gather(gathered_pids, pid_tensor)
    pids = [t.item() for t in gathered_pids]
    print(f"Rank {rank} gathered process IDs:", pids)
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running on {world_size} GPUs")
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    print("Healthcheck complete!")
