#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Write task IDs for a given split to a txt file, line by line."""

# Requires appworld installed
from appworld import load_task_ids

dataset_name = "train"  # Or dev, test_normal, test_challenge

# For each task in the dataset split
task_ids = load_task_ids(dataset_name)
write_path = f"{dataset_name}.txt"
with open(write_path, "w") as f:
    for task_id in task_ids:
        f.write(f"{task_id}\n")
print(f"Wrote to {write_path}")
