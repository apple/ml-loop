#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import queue
import threading
from collections.abc import Callable, Iterator
from queue import Queue

from phi_agents.rl.type_defs import SimpleScenario

SamplerFactory = Callable[[], Iterator[SimpleScenario]]


class ParallelScenarioSampler(Iterator[SimpleScenario]):
    """Wraps the Iterator[Scenario]."""

    def __init__(self, create_sampler_func: SamplerFactory, num_threads: int = 1) -> None:
        self._n_scenarios_requested = 0
        self._n_scenarios_in_progress = 0

        self._queue: Queue[SimpleScenario] = Queue()
        self._create_sampler_func = create_sampler_func

        self._mutex = threading.RLock()  # reentrant lock for simplicity
        self._generation_requested = threading.Condition(self._mutex)
        self._stop_event = threading.Event()

        self._verbose = False

        self._threads = [
            threading.Thread(target=self._worker, args=(i,), daemon=True)
            for i in range(num_threads)
        ]
        for thread in self._threads:
            thread.start()

    def __iter__(self) -> Iterator[SimpleScenario]:
        return self

    def ensure_scenarios_requested(self, num_scenarios: int) -> None:
        """Non-blocking; make sure at least num_scenarios are available or requested."""
        with self._mutex:  # same as "with self._generation_requested"
            # Note that qsize is not a 100% reliable measure of the number of available requests
            # since we can't call blocking queue.get() under a mutex. This is not a big deal
            # because in the worst case scenario we just request extra scenarios later.
            # Should be very rare anyway.
            currently_requested_or_available = (
                self._n_scenarios_requested + self._n_scenarios_in_progress + self._queue.qsize()
            )
            n_to_request = max(0, num_scenarios - currently_requested_or_available)
            if n_to_request > 0:
                self._n_scenarios_requested += n_to_request
                self._generation_requested.notify_all()

    def __next__(self) -> SimpleScenario:
        """
        Blocks until a scenario is available. Submits an additional request if there's nothing
        enqueued.
        """
        self.ensure_scenarios_requested(1)
        return self._queue.get(block=True)

    def _worker(self, thread_idx: int) -> None:
        sampler = self._create_sampler_func()
        while not self._stop_event.is_set():
            with self._mutex:
                if self._n_scenarios_requested > 0:
                    self._n_scenarios_requested -= 1
                    self._n_scenarios_in_progress += 1
                    if self._verbose:
                        print(
                            f"Thread {thread_idx} working... {self._n_scenarios_requested} {self._n_scenarios_in_progress}"
                        )
                else:
                    if self._verbose:
                        print(f"Thread {thread_idx} waiting for request")
                    self._generation_requested.wait(timeout=1.0)
                    continue

            try:
                scenario = next(sampler)
            except StopIteration:
                assert "Could not query the next scenario from the iterator!"  # should not happen

            with self._mutex:
                try:
                    self._n_scenarios_in_progress -= 1
                    # don't want to block under mutex
                    self._queue.put(scenario, block=False)

                    if self._verbose:
                        print(
                            f"Thread {thread_idx} Done!... {self._n_scenarios_requested} {self._n_scenarios_in_progress}"
                        )
                except queue.Full:
                    print(
                        f"Scenario queue full! This should not happen! {self._n_scenarios_requested=} {self._n_scenarios_in_progress=} {self._queue.qsize()}"
                    )
                    # let's just lose this scenario for simplicity, but again, this should never happen

    def stop(self) -> None:
        with self._mutex:
            self._generation_requested.notify_all()

        self._stop_event.set()
        for thread in self._threads:
            thread.join()
