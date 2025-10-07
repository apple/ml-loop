#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import random

import psutil


def get_free_port(
    n_subsequent_ports: int = 1,
    verbose: bool = True,
    min_port: int = 15_000,
    max_port: int = 64_000,
) -> int:
    """Start a server on a random port."""
    base_port = random.randint(min_port, max_port - n_subsequent_ports + 1)

    is_used = [is_port_used(base_port + i) for i in range(n_subsequent_ports)]

    while any(is_used):
        base_port += n_subsequent_ports
        if verbose:
            print(
                f"The following ports are already used: {[base_port + i for i in range(n_subsequent_ports) if is_used[i]]}"
            )
        is_used = [is_port_used(base_port + i) for i in range(n_subsequent_ports)]
    return base_port


def is_port_used(port: int) -> bool:
    """Check whether or not a port is used."""
    return port in [conn.laddr.port for conn in psutil.net_connections()]
