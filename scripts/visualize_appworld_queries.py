#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Visualize task queries."""

from phi_agents.appworld.interface import AppWorldInterface, load_task_ids


def main() -> None:
    dataset_name: str = "train"

    task_ids = load_task_ids(dataset_name)
    print(f"Found {len(task_ids)} tasks in {dataset_name} split.")

    try:
        world = AppWorldInterface(stdout_to_devnull=True)
        for idx, task_id in enumerate(task_ids):
            # Load the appworld environment for the task
            task = world.initialize(
                task_id=task_id, experiment_name="dummy_experiment", raise_on_unsafe_syntax=True
            )
            print(f"{idx} / {len(task_ids)}: {task.instruction}")
            world.close_world()
    finally:
        world.close_server()


if __name__ == "__main__":
    main()
