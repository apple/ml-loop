#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import signal
import subprocess
from pathlib import Path

from phi_agents.utils.logger import get_phi_logger

DEFAULT_APPWORLD_BIN = "appworld-env/bin/appworld"


logger = get_phi_logger()


def launch_appworld_environment_server(
    port: int | None, docker: bool, appworld_root: str | None, stdout_to_devnull: bool
) -> subprocess.Popen[bytes]:
    """Launch an AppWorld server.

    Run `appworld serve --help` for more details.

    Args:
        port: .
        docker: Whether to run it in a docker container.
        appworld_root: .
        stdout_to_devnull: Whether to redirect stdout to devnull.
    """
    appworld_cmd = DEFAULT_APPWORLD_BIN if Path(DEFAULT_APPWORLD_BIN).exists() else "appworld"
    args = [appworld_cmd, "serve", "environment", "--no-show-usage"]
    if port:
        args.extend(["--port", str(port)])
    if docker:
        args.append("--docker")
    if appworld_root:
        args.extend(["--root", appworld_root])
    return subprocess.Popen(args, stdout=subprocess.DEVNULL if stdout_to_devnull else None)


def stop_appworld_environment_server(popen: subprocess.Popen[bytes]) -> None:
    logger.info("sending SIGINT command to the appworld subprocess...")
    popen.send_signal(signal.SIGINT)
    popen.wait(timeout=20)
    logger.info("appworld server terminated")


def force_stop_appworld_environment_server(popen: subprocess.Popen[bytes]) -> None:
    logger.info("sending SIGKILL command to the appworld subprocess...")
    popen.send_signal(signal.SIGKILL)
    popen.wait(timeout=10)
    logger.info("appworld server forcefully terminated")
