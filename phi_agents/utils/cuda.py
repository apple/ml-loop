#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from typing import cast

import torch

cuda_arc_to_major = {"volta": 7, "ampere": 8, "hopper": 9, "blackwell": 10}


def get_cuda_architecture() -> int:
    if torch.cuda.is_available():
        version = torch.cuda.get_device_capability(0)[0]
        return cast(int, version)
    else:
        raise RuntimeError("No cuda available.")
