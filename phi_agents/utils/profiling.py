#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import gc
import logging
import os
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import ContextDecorator
from types import TracebackType
from typing import Any, Self

import numpy as np
import pynvml
import torch

from phi_agents.utils.logger import NullLogger, get_phi_logger

global_logger = get_phi_logger()


PHI_AGENTS_PROFILE_ENVVAR = "PHI_AGENTS_PROFILE"
PHI_AGENTS_PROFILE_MEMORY_ENVVAR = "PHI_AGENTS_PROFILE_MEMORY"


class NVMLPeakMemProfiler:
    def __init__(
        self,
        key: str,
        poll_interval: float = 0.01,
        logger: logging.Logger | NullLogger | None = None,
    ) -> None:
        self.logger = logger
        self.key = key
        self.poll_interval = poll_interval

        self._peak_bytes: int = 0
        self._start: float = 0.0
        self._stop_evt: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

        def poll() -> None:
            while True:
                used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                self._peak_bytes = max(self._peak_bytes, used)
                if self._stop_evt.wait(self.poll_interval):
                    break

        self._thread = threading.Thread(target=poll, daemon=True)
        torch.cuda.synchronize()
        self._thread.start()
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start

        self._stop_evt.set()
        self._thread.join()  # type: ignore[union-attr]

        if self.logger is not None:
            self.logger.info(
                "Peak memory in %s: %.2f GB, took %.4f s",
                self.key,
                self._peak_bytes / (1024**3),
                elapsed,
            )

        ProfilingMemoryResultsInstance.add_result(self.key, self._peak_bytes, 1, 0)


def format_delimiter(s: str, width: int = 50, character: str = "=") -> str:
    if len(s) >= width:
        return s
    padding = (width - len(s)) // 2
    return f"{'=' * padding}{s}{'=' * (width - len(s) - padding)}"


class TimedWindowThroughput:
    """Approximately compute throughput in a timed window, e.g. 5 minutes.

    Assumes multiple calls to update() happen within the window, otherwise it's advised to increase
    the averaging period.
    """

    _avg_period_seconds: float
    _values: deque[tuple[float, float]]

    def __init__(self, avg_period_seconds: float = 300.0) -> None:
        self._avg_period_seconds = avg_period_seconds
        self._values = deque()

    def update(self, value: float) -> None:
        """Maintains _values deque such that the first and the last element cover approx. the desired time period.

        Value is assumed to be monotonically increasing.
        """
        now = time.perf_counter()
        self._values.append((now, value))

        while (
            len(self._values) > 2
        ):  # keep at least 2 datapoints so we can always produce measurements
            t0, _ = self._values[0]
            if now - t0 <= self._avg_period_seconds:
                break

            self._values.popleft()

    def compute(self) -> float | None:
        """Returns None if not enough data."""
        if len(self._values) <= 1:
            return None
        t0, v0 = self._values[0]
        tLast, vLast = self._values[-1]
        time_passed = max(tLast - t0, 1e-8)
        return (vLast - v0) / time_passed


class Speedometer:
    _avg_period_seconds: float
    _throughputs: dict[str, TimedWindowThroughput]

    def __init__(self, avg_period_seconds: float = 300.0) -> None:
        self._avg_period_seconds = avg_period_seconds
        self._throughputs = {}

    def track(self, **values: float) -> None:
        for k, v in values.items():
            if k not in self._throughputs:
                self._throughputs[k] = TimedWindowThroughput(self._avg_period_seconds)

            self._throughputs[k].update(v)

    def summary(self, key_prefix: str = "") -> dict[str, float]:
        res = dict()
        for k, t in self._throughputs.items():
            throughput = t.compute()
            if throughput is not None:
                res[f"{key_prefix}{k}_per_second"] = throughput

        return res

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info("\n")
        logger.info(format_delimiter("Throughput stats:"))
        for k, v in self.summary().items():
            logger.info(f"{k}: {v:.2f}")
        logger.info(format_delimiter("End throughput stats"))
        logger.info("\n")


class WindowedRunningMean:
    _buffer: np.ndarray
    _ptr: int
    _sum: float
    _count: int
    _burn_remaining: int

    def __init__(self, window_size: int, burn: int = 0) -> None:
        self._burn_remaining = burn
        self._buffer = np.zeros((window_size,), dtype=np.float32)

        self._ptr = 0
        self._count = 0
        self._sum = 0

    def add(self, x: float) -> None:
        self._burn_remaining = max(self._burn_remaining - 1, 0)
        if self._burn_remaining > 0:
            return

        if self._count == self._buffer.shape[0]:
            self._sum -= self._buffer[self._ptr]

        self._sum += x
        self._buffer[self._ptr] = x

        self._count = min(self._count + 1, self._buffer.shape[0])
        self._ptr = (self._ptr + 1) % self._buffer.shape[0]

    @property
    def mean(self) -> float:
        return self._sum / self._count

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def buffer(self) -> np.ndarray:
        return self._buffer[: self._count]


EventType = float | tuple[Any, Any]


class _ProfilingResults:
    _events_in_flight: list[tuple[str, EventType, int, int]]
    _total_time_result: dict[str, float]
    _inclusive_time_result: dict[str, WindowedRunningMean]
    _exclusive_time_result: dict[str, WindowedRunningMean]
    _elements_results: dict[str, WindowedRunningMean]
    num_elements: int
    _depth_times: list[float]
    enabled: bool

    def __init__(self) -> None:
        self._total_time_result = {}
        self._inclusive_time_result = {}
        self._exclusive_time_result = {}
        self._events_in_flight = []
        self._elements_results = {}
        self.num_elements = 0
        self._depth_times = []
        self.enabled: bool = int(os.getenv(PHI_AGENTS_PROFILE_ENVVAR, 1)) == 1

    def enqueue_event(
        self,
        key: str,
        event: EventType,
        stack_depth: int,
        window_size: int = 100,
        burn: int = 3,
        count: int | None = None,
    ) -> None:
        if key not in self._total_time_result:
            self._total_time_result[key] = 0.0

        if key not in self._inclusive_time_result:
            self._inclusive_time_result[key] = WindowedRunningMean(window_size, burn)

        if key not in self._elements_results:
            self._elements_results[key] = WindowedRunningMean(window_size, burn)

        if count is None:
            count = self.num_elements

        self._events_in_flight.append((key, event, count, stack_depth))

    def _process_event(self, key: str, event: EventType, count: int, stack_depth: int) -> None:
        if isinstance(event, float):
            self.add_result(key, event, stack_depth, count=count)
        else:
            assert isinstance(event, tuple)
            start, end = event

            end.synchronize()

            self.add_result(key, start.elapsed_time(end), stack_depth, count=count)

    def state_dict(self) -> dict[str, Any]:
        return {"profiler_total_time_result": self._total_time_result}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        # this seems to be the only thing we want to preserve between restarts
        self._total_time_result = d["profiler_total_time_result"]

    def flush_events(self) -> None:
        for event in self._events_in_flight:
            self._process_event(*event)

        self._events_in_flight = []

    def add_result(
        self,
        key: str,
        x: float,
        stack_depth: int,
        window_size: int = 100,
        burn: int = 3,
        count: int | None = None,
    ) -> None:
        if key not in self._total_time_result:
            self._total_time_result[key] = 0.0

        self._total_time_result[key] += x

        if key not in self._inclusive_time_result:
            self._inclusive_time_result[key] = WindowedRunningMean(window_size, burn)

        self._inclusive_time_result[key].add(x)

        if key not in self._exclusive_time_result:
            self._exclusive_time_result[key] = WindowedRunningMean(window_size, burn)

        while stack_depth >= len(self._depth_times):
            self._depth_times.append(0.0)

        self._exclusive_time_result[key].add(x - self._depth_times[stack_depth])

        self._depth_times[stack_depth] = 0.0
        if stack_depth > 0:
            self._depth_times[stack_depth - 1] += x

        if key not in self._elements_results:
            self._elements_results[key] = WindowedRunningMean(window_size, burn)

        if count is None:
            count = self.num_elements

        self._elements_results[key].add(count)

    def log_summary(self, logger: logging.Logger) -> None:
        if all(v.count < 1 for v in self._inclusive_time_result.values()):
            return

        logger.info("\n")
        logger.info(format_delimiter("Timing"))

        for k, v in self._inclusive_time_result.items():
            if v.count < 1:
                continue

            logger.info(f"    {k}: {v.mean:.2f}ms")

        logger.info(format_delimiter("End"))
        logger.info("\n")

    def dict_summary(self) -> dict[str, float]:
        res = dict()

        for k, v in self._inclusive_time_result.items():
            if v.count < 1:
                continue

            assert k in self._elements_results

            res[f"time/inclusive_time/{k}"] = v.mean / 1e3
            if self._exclusive_time_result[k].count > 0:
                res[f"time/exclusive_time/{k}"] = self._exclusive_time_result[k].mean / 1e3
            res[f"time/throughput/{k}"] = self._elements_results[k].sum / v.sum * 1e3
            res[f"total_time_seconds/{k}"] = self._total_time_result[k] / 1e3

        return res


# TODO: if we start using multithreading, should this also be threading.local?
# or just avoid having this global instance altogether
ProfilingResultsInstance = _ProfilingResults()


def profiler_state_dict() -> dict[str, Any]:
    if not ProfilingResultsInstance.enabled:
        return dict()
    d = ProfilingResultsInstance.state_dict()
    # print(f"Saving profiler state: {d}")
    return d


def profiler_load_state_dict(d: dict[str, Any], logger: logging.Logger) -> None:
    if not ProfilingResultsInstance.enabled:
        return
    logger.info(f"Loading profiler state: {d}")
    return ProfilingResultsInstance.load_state_dict(d)


def log_profiling_summary(logger: logging.Logger) -> None:
    if not ProfilingResultsInstance.enabled:
        return

    ProfilingResultsInstance.flush_events()
    ProfilingResultsInstance.log_summary(logger)


def profiling_summary() -> dict[str, float]:
    if not ProfilingResultsInstance.enabled:
        return dict()

    ProfilingResultsInstance.flush_events()
    return ProfilingResultsInstance.dict_summary()


def profiling_memory_summary() -> dict[str, float]:
    if not ProfilingMemoryResultsInstance.enabled:
        return dict()

    return ProfilingMemoryResultsInstance.dict_summary()


_stack_depth = threading.local()
_stack_depth.val = 0


class profile(ContextDecorator):
    __slots__ = (
        "key",
        "window_size",
        "burn",
        "cuda_device",
        "start_event",
        "stream",
        "t_start",
    )

    def __init__(
        self,
        key: str,
        *,
        window_size: int = 100,
        burn: int = 3,
        use_cuda_event: bool = True,
        cuda_device: torch.device | None = None,
    ):
        if not ProfilingResultsInstance.enabled:
            return

        self.key = key
        self.window_size = window_size
        self.burn = burn
        self.use_cuda_event = use_cuda_event and torch.cuda.is_available()
        self.cuda_device = cuda_device

        self.stream: torch.cuda.Stream | None = None

        self.t_start: float | None = None

    def __enter__(self) -> None:
        global _stack_depth

        if not ProfilingResultsInstance.enabled:
            return

        _stack_depth.val += 1

        if self.use_cuda_event:
            self.start_event = torch.cuda.Event(enable_timing=True)

            self.stream = torch.cuda.current_stream(self.cuda_device)

            self.start_event.record(self.stream)
        else:
            self.t_start = time.perf_counter()

    def __exit__(self, *exc: Any) -> None:
        global _stack_depth
        if not ProfilingResultsInstance.enabled:
            return

        _stack_depth.val -= 1

        event: EventType
        if self.use_cuda_event:
            assert self.stream is not None
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(self.stream)
            event = (self.start_event, end_event)
        else:
            assert self.t_start is not None

            event = (time.perf_counter() - self.t_start) * 1e3

        ProfilingResultsInstance.enqueue_event(
            self.key, event, _stack_depth.val, self.window_size, self.burn
        )


class set_profiling_num_elements(ContextDecorator):
    __slots__ = ("val", "old_val")
    val: int
    old_val: int | None

    def __init__(self, val_or_fn: int | Callable[[], float | int | torch.Tensor]) -> None:
        if not ProfilingResultsInstance.enabled:
            return

        if isinstance(val_or_fn, int):
            self.val = val_or_fn
        else:
            self.val = int(val_or_fn())

        self.old_val = None

    def __enter__(self) -> None:
        if not ProfilingResultsInstance.enabled:
            return

        self.old_val = ProfilingResultsInstance.num_elements
        ProfilingResultsInstance.num_elements = self.val

    def __exit__(self, *exc: Any) -> None:
        if not ProfilingResultsInstance.enabled:
            return

        assert self.old_val is not None
        ProfilingResultsInstance.num_elements = self.old_val


class no_profile(ContextDecorator):
    __slots__ = ("old",)

    def __enter__(self) -> None:
        self._old = ProfilingResultsInstance.enabled
        ProfilingResultsInstance.enabled = False

    def __exit__(self, *exc: Any) -> None:
        ProfilingResultsInstance.enabled = self._old


class _ProfilingMemoryResults:
    _memory_result: dict[str, WindowedRunningMean]
    _elements_results: dict[str, WindowedRunningMean]
    enabled: bool

    def __init__(self) -> None:
        self._memory_result = {}
        self._elements_results = {}
        self.enabled: bool = int(os.getenv(PHI_AGENTS_PROFILE_MEMORY_ENVVAR, 1)) == 1

    def add_result(
        self, key: str, memory_bytes: int, window_size: int, burn: int = 3, count: int | None = None
    ) -> None:
        if key not in self._memory_result:
            self._memory_result[key] = WindowedRunningMean(window_size, burn)

        self._memory_result[key].add(memory_bytes)

        if key not in self._elements_results:
            self._elements_results[key] = WindowedRunningMean(window_size, burn)

        if count is None:
            count = ProfilingResultsInstance.num_elements

        self._elements_results[key].add(count)

    def dict_summary(self) -> dict[str, float]:
        res = dict()
        for k, v in self._memory_result.items():
            if v.count < 1:
                continue

            assert k in self._elements_results

            res[f"memory/mb_peak/{k}"] = v.mean / 1024**2
            if not any(self._elements_results[k].buffer == 0):
                res[f"memory/mb_per_item/{k}"] = (
                    v.buffer / self._elements_results[k].buffer
                ).mean() / 1024**2

        return res


ProfilingMemoryResultsInstance = _ProfilingMemoryResults()
_stack_depth_memory = threading.local()
_stack_depth_memory.val = 0


class profile_memory(ContextDecorator):
    __slots__ = (
        "key",
        "window_size",
        "burn",
        "cuda_device",
        "stats0",
        "stats1",
        "enabled",
    )

    def __init__(
        self,
        key: str,
        *,
        window_size: int = 10,
        burn: int = 3,
        cuda_device: torch.device | None = None,
        sample_rate: float = 1.0,
    ):
        """
        Because we have to synchronize and clear caches to measure memory, you may not want to run
        on every iteration, so a sample_rate can be used to decrease how often we measure peak
        memory usage.
        """
        self.key = key
        self.window_size = window_size
        self.burn = burn
        self.cuda_device = cuda_device
        self.stats0 = None
        self.stats1 = None
        self.enabled = ProfilingMemoryResultsInstance.enabled and torch.cuda.is_available()
        if random.random() > sample_rate:
            self.enabled = False

    def _mem_delta(self) -> int:
        assert self.stats0 is not None and self.stats1 is not None
        return self.stats1["allocated_bytes.all.peak"] - self.stats0["allocated_bytes.all.peak"]

    def __enter__(self) -> None:
        global _stack_depth_memory
        if not self.enabled:
            return

        assert _stack_depth_memory.val == 0, "Can not use nested memory profiler"
        _stack_depth_memory.val += 1

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.stats0 = torch.cuda.memory_stats()

    def __exit__(self, *exc: Any) -> None:
        global _stack_depth_memory
        if not self.enabled:
            return

        _stack_depth_memory.val -= 1

        torch.cuda.synchronize()
        self.stats1 = torch.cuda.memory_stats()

        ProfilingMemoryResultsInstance.add_result(
            self.key, self._mem_delta(), self.window_size, self.burn
        )
