#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Self, cast

import cattrs
import requests
from dateutil import parser

from phi_agents.utils.logger import get_phi_logger
from phi_agents.utils.network_utils import get_free_port

from .server import (
    force_stop_appworld_environment_server,
    launch_appworld_environment_server,
    stop_appworld_environment_server,
)

logger = get_phi_logger()

DUMMY_TASK_ID = "82e2fac_1"
DEFAULT_SPLITS_DIR = "data/appworld_splits"


class AppWorldInitializeTaskError(Exception):
    pass


class AppWorldExecutionError(Exception):
    pass


@dataclass(frozen=True)
class Supervisor:
    first_name: str
    last_name: str
    phone_number: str
    email: str


@dataclass(frozen=True)
class Task:
    task_id: str
    datetime: datetime
    instruction: str
    supervisor: Supervisor


# Source: https://github.com/StonyBrookNLP/appworld/blob/main/src/appworld/evaluator.py
@dataclass(frozen=True)
class Pass:
    requirement: str
    label: str | None


# Source: # Source: https://github.com/StonyBrookNLP/appworld/blob/main/src/appworld/evaluator.py
@dataclass(frozen=True)
class Failure:
    requirement: str
    trace: str
    label: str | None


@dataclass(frozen=True)
class AppWorldTaskEvalResult:
    success: bool
    difficulty: int  # Literal[1, 2, 3]
    num_tests: int
    passes: list[Pass]
    failures: list[Failure]

    def __post_init__(self) -> None:
        assert self.difficulty in (1, 2, 3)


def load_task_ids(
    split: str,  # "train", "dev", "test_normal", "test_challenge", "train_difficulty_1", etc.
    txt_dir: str = DEFAULT_SPLITS_DIR,
) -> list[str]:
    """Load task IDs from a txt file.

    Returns:
        List of task IDs e.g. ['82e2fac_1', '82e2fac_2', ...]
    """
    from phi_agents.utils.file_utils import project_source_root

    lines = []
    for _cur_data_split in split.split("_and_"):
        with open(project_source_root() / txt_dir / f"{_cur_data_split}.txt") as f:
            lines.extend([line.rstrip() for line in f])
    lines_set = set(lines)

    with open(project_source_root() / txt_dir / f"denylist.txt") as f:
        denylist = [line.rstrip() for line in f]
    for _denylist_task_id in denylist:
        lines_set.discard(_denylist_task_id)
    return list(lines)


def get_appworld_root() -> str:
    return os.environ["APPWORLD_ROOT"]


class AppWorldInterface:
    """Run AppWorld without importing appworld."""

    def __init__(
        self, stdout_to_devnull: bool, timeout_seconds: int = 100, max_restarts_on_error: int = 2
    ) -> None:
        """Construct AppWorldInterface.

        Args:
            stdout_to_devnull: Hides the output from the server.
            timeout_seconds: Timeout on any call to the server.
            max_restarts_on_error: If we fail on waiting to connect to server, retry this many times. This can help with port collisions at scale.
        """
        self._server: subprocess.Popen[bytes] | None = None
        self._docker: bool = False  # avoiding docker within docker
        self._max_wait_tries: int = 5
        self._wait_seconds: float = 1.0
        self._stdout_to_devnull = stdout_to_devnull
        self._init_server()
        self.timeout_seconds = timeout_seconds
        self._task_id: str | None = None
        self._last_closed_task_id: str | None = None
        self._max_restarts_on_error = max_restarts_on_error

        self._wait_for_server_ready()

    def _init_server(self) -> None:
        self._port = get_free_port()
        appworld_root = get_appworld_root()
        logger.info(f"Launching AppWorld server with port: {self._port}")
        self._server = launch_appworld_environment_server(
            port=self._port,
            docker=self._docker,
            appworld_root=appworld_root,
            stdout_to_devnull=self._stdout_to_devnull,
        )
        self._remote_environment_url = f"http://localhost:{self._port}"

    def raise_if_server_closed(self) -> None:
        if self.clean:
            raise ValueError("Server is closed.")

    def _wait_for_server_ready(self) -> None:
        logger.info(f"Waiting for AppWorld server at {self._remote_environment_url} to be ready.")
        try:
            response = None
            for attempt_restart in range(self._max_restarts_on_error + 1):
                for attempt_wait in range(self._max_wait_tries):
                    try:
                        response = requests.get(
                            f"{self._remote_environment_url}/tasks/{DUMMY_TASK_ID}"
                        )
                    except requests.exceptions.ConnectionError:
                        pass
                    if response is not None:
                        # sanity check on response
                        output = response.json()["output"]
                        assert output["task_id"] == DUMMY_TASK_ID
                        logger.info(f"AppWorld server at {self._remote_environment_url} is ready.")
                        return

                    # try and reduce noise from this print
                    if attempt_restart >= self._max_restarts_on_error:
                        logger.warning(
                            f"Failed to connect {attempt_wait=}/{self._max_wait_tries}"
                            f" {attempt_restart=}/{self._max_restarts_on_error}"
                        )
                    time.sleep(self._wait_seconds)

                if attempt_restart < self._max_restarts_on_error:
                    logger.warning("Restarting server after client connection failure.")
                    self.server.terminate()
                    self._init_server()

            if response is None:
                raise RuntimeError(f"Could not connect to {self._remote_environment_url}")
        except Exception:
            self.server.terminate()
            logger.exception("Error occured when waiting for AppWorld server to be ready.")
            raise

    @property
    def clean(self) -> bool:
        return self._server is None

    @property
    def task_id(self) -> str:
        if self._task_id is None:
            raise ValueError("Task ID has not been set.")
        return self._task_id

    @property
    def port(self) -> int:
        return self._port

    @property
    def server(self) -> subprocess.Popen[bytes]:
        if self._server is None:
            raise ValueError("Server has been closed already.")
        return self._server

    @property
    def remote_environment_url(self) -> str:
        return self._remote_environment_url

    def initialize(
        self, task_id: str, experiment_name: str, raise_on_unsafe_syntax: bool, verbose: bool = True
    ) -> Task:
        """Initialize an AppWorld with task ID.

        Returns:
            Dictionary of format:
            {
                'datetime': '2023-05-18T12:00:00',
                'instruction': 'What is the title of the most-liked song in my Spotify playlists.'
                 'supervisor': {'email': 'joyce-weav@gmail.com',
                                'first_name': 'Joyce',
                                'last_name': 'Weaver',
                                'phone_number': '3155673041'},
                 'task_id': '82e2fac_1'}
        """
        self.experiment_name = experiment_name
        self.executed_code: list[str] = []
        if verbose:
            logger.info(
                f"Initializing AppWorld world with task_id={task_id}, experiment_name={experiment_name}"
            )
        if self._task_id is not None:
            raise ValueError(
                f"Called initialize on task {task_id=} but task {self.task_id=} has already been started."
            )
        self._task_id = task_id
        try:
            output = self._remote_environment_call(
                "initialize",
                task_id=task_id,
                experiment_name=experiment_name,
                raise_on_unsafe_syntax=raise_on_unsafe_syntax,
                remote_docker=self._docker,
            )
        except Exception as e:
            self._task_id = None
            raise AppWorldInitializeTaskError(f"Failed to initialize task {task_id}") from e
        supervisor = cattrs.structure(output["supervisor"], Supervisor)
        task = Task(  # TODO: Use cattrs for this
            task_id=output["task_id"],
            datetime=parser.isoparse(output["datetime"]),
            instruction=output["instruction"],
            supervisor=supervisor,
        )
        return task

    def _remote_environment_call(self, method_name: str, **kwargs: Any) -> Any:
        """Communicate with remote AppWorld environment server.

        Largely borrows from:
        https://github.com/StonyBrookNLP/appworld/blob/main/src/appworld/environment.py#L328
        """
        kwargs["task_id"] = self.task_id

        try:
            response = requests.post(
                f"{self.remote_environment_url}/{method_name}",
                json=kwargs,
                timeout=self.timeout_seconds,
            )
        except requests.exceptions.ConnectionError as exception:
            logger.exception(
                "AppWorld environment server is not reachable "
                f"at the URL: {self.remote_environment_url}."
            )
            raise exception

        try:
            response.raise_for_status()
        except requests.HTTPError as exception:
            logger.exception(
                f"AppWorld remote environment call to method '{method_name}' failed: {response.text}."
            )
            raise exception

        return response.json()["output"]

    def execute(self, code: str) -> str:
        self.raise_if_server_closed()
        try:
            message = self._remote_environment_call("execute", code=code)
        except Exception as e:
            raise AppWorldExecutionError from e
        if "(sqlite3.ProgrammingError) Cannot operate on a closed database." in message:
            raise AppWorldExecutionError(
                "Looks like you are operating on a closed task world object."
            )
        # self.environment_io.append({"input": code, "output": message})
        self.executed_code.append(code)
        return cast(str, message)

    def get_state(self) -> tuple[str, str, Sequence[str]]:
        assert self._task_id is not None
        return (self._task_id, self.experiment_name, tuple(self.executed_code))

    def set_state(self, task_id: str, experiment_name: str, executed_code: Sequence[str]) -> None:
        self.close_world()
        self.initialize(
            task_id, experiment_name, raise_on_unsafe_syntax=False, verbose=False
        )  # Hacky
        for code in executed_code:
            self.execute(code)

    def evaluate(self, suppress_errors: bool = True) -> AppWorldTaskEvalResult:
        """Evaluate.

        Returns:
            appworld.evaluator.TestTracker as dict.
        """
        self.raise_if_server_closed()
        return cast(
            AppWorldTaskEvalResult,
            cattrs.structure(
                self._remote_environment_call("evaluate", suppress_errors=suppress_errors),
                AppWorldTaskEvalResult,
            ),
        )

    # Not sure what this does yet
    # def save_logs(self) -> None:
    #     # Check that it returns None
    #     import pdb; pdb.set_trace()  # Alternatively return None
    #     self.raise_if_server_closed()
    #     output = self._remote_environment_call("save_logs")
    #     assert output is None

    # Not sure what these do yet, and how to handle self._task_id
    # def save_state(self, state_id: str | None = None) -> str:
    #     output = self._remote_environment_call("save_state", state_id=state_id)
    #     assert isinstance(output, str)
    #     import pdb; pdb.set_trace()
    #     return output

    # Not sure what these do yet, and how to handle self._task_id
    # def load_state(self, state_id: str) -> None:
    #     output = self._remote_environment_call("load_state", state_id=state_id)
    #     import pdb; pdb.set_trace()
    #     assert output is None

    def task_completed(self) -> bool:
        self.raise_if_server_closed()
        output = self._remote_environment_call("task_completed")
        return cast(bool, output)

    def __enter__(self) -> Self:
        if self.clean:
            self._init_server()
            self._wait_for_server_ready()
        return self

    def __exit__(
        self,
        exe_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close_server()

    def restart(self) -> None:
        logger.info(f"Attempting to restart AppWorld server.")
        if self.clean:
            raise ValueError("Server was not already started.")
        self.close_server()
        assert self.clean
        self._init_server()
        self._wait_for_server_ready()
        self._task_id = None
        self._last_closed_task_id = None

    def close_world(self) -> None:
        self.raise_if_server_closed()
        output = self._remote_environment_call("close")
        self._last_closed_task_id = self.task_id
        self._task_id = None
        assert output is None

    def _close_all(self) -> None:  # Not entirely sure what this does
        output = self._remote_environment_call("close_all")
        self._task_id = None
        assert output is None

    def close_server(self) -> None:
        logger.info(f"Trying to close AppWorld server at {self.remote_environment_url}")
        self.raise_if_server_closed()
        try:
            # try to close the server cleanly
            # When terminating server, try to call close_all whenever possible, otherwise process hangs
            if self._task_id is not None:
                self._close_all()
            elif self._last_closed_task_id is not None:
                # We called close_world() but we still need to call close_all()
                self._task_id = self._last_closed_task_id
                self._close_all()
        except Exception:
            logger.exception(
                "close_all failed, but we will continue to terminate the process anyways; moving on."
            )
        try:
            stop_appworld_environment_server(self.server)
        except Exception:
            logger.exception(
                f"SIGINT for AppWorld server process {self.remote_environment_url} failed to terminate the process."
            )
            force_stop_appworld_environment_server(self.server)
        self._server = None
        self._task_id = None
