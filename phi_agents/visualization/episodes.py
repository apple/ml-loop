#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import random
import sys
import termios
import tty
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Self

from rich import get_console, print
from rich.columns import Columns
from rich.syntax import Syntax
from rich.text import Text

import phi_agents.utils.file_utils as fu
from phi_agents.evals.appworld_evals import (
    Episode,
    TaskEvalResult,
    TaskMetricsCollection,
    load_eval_result,
)
from phi_agents.utils.appworld import (
    download_dir,
    get_episode_path,
    get_experiment_dir,
)

if TYPE_CHECKING:
    from rich.console import RenderableType

MAX_ROLE_WIDTH = 10


ROLE_STYLE = {
    "system": "bold dark_blue",
    "user": "bold dark_orange",
    "assistant": "bold dark_magenta",
    "ipython": "bold dark_green",
}


def get_keypress() -> str:
    """Get single keypress event."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def pad_role(role: str) -> str:
    return role.ljust(MAX_ROLE_WIDTH + 1)


def display(episode: Episode, eval_result: TaskEvalResult | None) -> None:
    # Print task info
    author = Text(pad_role("Task"))
    author.stylize(style="bold yellow")
    content: RenderableType
    content = Text(episode.task.instruction)
    print(Columns([author, content], align="left"))

    for message in episode.chat_history:
        author = Text(pad_role(message.role))
        author.stylize(style=ROLE_STYLE[message.role])
        if message.role == "assistant":
            content = Syntax(message.content, "python")
        else:
            content = Text(message.content)
        print(Columns([author, content], align="left"))

    if eval_result:
        print(f"\nTask ID: {episode.task.task_id}")
        print(f"Task instruction: {episode.task.instruction}")
        print(f"Num interactions: {eval_result.num_interactions}")
        print("\n--------------- EVALUATION RESULTS ---------------")
        print("success:", eval_result.success)
        print("difficulty:", eval_result.difficulty)
        print(
            f"num_tests: {eval_result.num_tests} ({len(eval_result.passes)} / "
            f"{len(eval_result.passes) + len(eval_result.failures)} passes)"
        )
        print("")

        print("PASSES")
        for idx, cur_pass in enumerate(eval_result.passes):
            print(f"PASS {idx}")
            print("label:", cur_pass.label)
            print("requirement:")
            print(cur_pass.requirement)
            print("")

        print("FAILURES")
        for idx, cur_failure in enumerate(eval_result.failures):
            print(f"FAILURE {idx}")
            print("label:", cur_failure.label)
            print("requirement:")
            print(cur_failure.requirement)
            print("")

            # Don't show trace for now to reduce text (not sure how helpful it is)
            # print("trace:")
            # print(cur_pass.trace)
            # print("")


def match(
    task_id: str,
    pattern: str | None,
    sample_id: str | None,
) -> bool:
    if sample_id is not None:
        return task_id == sample_id
    elif pattern is not None:
        return task_id == pattern
        # TODO: Implement this
        # return any(bool(re.search(pattern, m.text_content)) for m in episode.chat_history)
        raise NotImplementedError("Not yet supported.")
    else:
        return True


class AppWorldDisplayIterator(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Self:
        return self

    @abstractmethod
    def __next__(self) -> tuple[int, Episode]:
        pass

    @abstractmethod
    def same(self) -> None:
        pass

    @abstractmethod
    def prev(self) -> None:
        pass

    @abstractmethod
    def random(self) -> None:
        pass

    def search(self, pattern: str) -> None:
        return None

    def reverse_search(self, pattern: str) -> None:
        return None

    @abstractmethod
    def next_sample(self) -> None:
        pass

    @abstractmethod
    def prev_sample(self) -> None:
        pass

    @abstractmethod
    def eval_result(self, task_id: str) -> TaskEvalResult | None:
        pass


class EpisodeIterator(AppWorldDisplayIterator):
    """Iterator over the episodes for viewing."""

    def __init__(
        self,
        experiment_name: str,
        current_index: int | None = None,
        regex_pattern: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        """Construct an EpisodeIterator.

        Args:
            experiment_name: AppWorld experiment name.
            current_index: If specified, start from a specific task referenced by index.
            regex_pattern: Find episodes which contain the specified regex pattern in the history.
            sample_id: If specified, view only episodes with the provided task ID.
        """
        self.experiment_name = experiment_name
        experiment_dir = get_experiment_dir(experiment_name)
        tasks_dir = experiment_dir / "tasks"
        self.task_ids: list[str] = [d.parts[-1] for d in tasks_dir.glob("*") if d.is_dir()]

        # Check if eval results exist
        eval_dir = experiment_dir / "evaluations"
        eval_json_paths = fu.glob(eval_dir / "*.json")
        self.metricscollections_by_split: dict[str, TaskMetricsCollection] = {}
        for json_scheme_path in eval_json_paths:
            scheme, _json_path = fu.get_scheme_and_path(json_scheme_path)
            json_path = Path(_json_path)
            assert scheme == "file"
            print(f"LOADING {json_path}")
            experiment_eval_result = load_eval_result(json_path)

            dataset_name = json_path.stem
            self.metricscollections_by_split[dataset_name] = experiment_eval_result.individual

            print(experiment_eval_result.individual.collective_metrics)
            print("")

        if current_index is not None:
            self.current_index = current_index
        else:
            try:
                self.current_index = next(
                    idx
                    for idx, task_id in enumerate(self.task_ids)
                    if match(task_id, pattern=regex_pattern, sample_id=sample_id)
                )
            except StopIteration as e:
                raise RuntimeError("Could not find example!") from e

    def __len__(self) -> int:
        return len(self.task_ids)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[int, Episode]:
        if self.current_index >= len(self.task_ids):
            raise StopIteration
        current_index = self.current_index
        cur_task_id = self.task_ids[self.current_index]
        self.current_index += 1
        current = Episode.load(json_path=get_episode_path(self.experiment_name, cur_task_id))
        return current_index, current

    def same(self) -> None:
        self.current_index = max(self.current_index - 1, 0)

    def prev(self) -> None:
        self.current_index = max(self.current_index - 2, 0)

    def random(self) -> None:
        self.current_index = random.randrange(len(self.task_ids))

    def search(self, pattern: str) -> None:
        self.current_index = next(
            idx
            for idx, example in enumerate(
                self.task_ids[self.current_index :], start=self.current_index
            )
            if match(example, pattern=pattern, sample_id=None)
        )

    def reverse_search(self, pattern: str) -> None:
        reverse_dist = next(
            idx
            for idx, example in enumerate(reversed(self.task_ids[: self.current_index - 1]))
            if match(example, pattern=pattern, sample_id=None)
        )
        self.current_index -= reverse_dist + 2

    def next_sample(self) -> None:
        current_index = max(self.current_index - 1, 0)
        current_sample_id = self.task_ids[current_index]
        while current_index < len(self.task_ids) - 1:
            current_index += 1
            if self.task_ids[current_index] != current_sample_id:
                self.current_index = current_index
                break
        else:
            raise StopIteration

    def prev_sample(self) -> None:
        current_index = self.current_index - 1
        current_sample_id = self.task_ids[current_index]
        if self.task_ids[current_index] != self.task_ids[current_index - 1]:
            # first example from a dialog, so we want to skip all the way back to the start of the
            # previous sample
            current_index -= 1

        while current_index >= 0:
            current_index -= 1
            if self.task_ids[current_index] != current_sample_id:
                self.current_index = current_index
                break
        else:
            raise StopIteration

    def eval_result(self, task_id: str) -> TaskEvalResult | None:
        for metrics_collection in self.metricscollections_by_split.values():
            if task_id in metrics_collection:
                return metrics_collection[task_id]
        return None


def visualize_episodes_with_iter(episode_iter: AppWorldDisplayIterator) -> None:
    """Visualize episodes interactively."""
    console = get_console()

    for current_index, example in episode_iter:
        console.clear()
        display(example, episode_iter.eval_result(example.task.task_id))

        console.print(f"\n[{current_index + 1}/{len(episode_iter)}]", end=" ", markup=False)
        print(
            rf"choose [bold]n[/bold]ext, [bold]p[/bold]revious, [bold]r[/bold]andom, [bold]s[/bold]kip dialog, [bold]b[/bold]ack dialog, [bold]/[/bold]search, [bold]?[/bold]reverse-search, [bold]q[/bold]uit"
        )

        match get_keypress():
            case "q":
                break
            case "n" | "\r":
                pass
            case "p":
                episode_iter.prev()
            case "r":
                episode_iter.random()
            case "/":
                search_term = input("/")
                episode_iter.search(search_term)
            case "?":
                search_term = input("?")
                episode_iter.reverse_search(search_term)
            case "s":
                episode_iter.next_sample()
            case "b":
                episode_iter.prev_sample()
            case _:
                episode_iter.same()


def visualize_episodes(
    experiment_name: str | None = None,
    sample_id: str | None = None,
    s3_experiment_path: Path | None = None,
    exists_ok: bool = True,
) -> None:
    """Visualize episodes from an inference run.

    Args:
        experiment_name: AppWorld experiment name (local). Set to None if loading from S3.
        sample_id: Skip to this task ID, if provided.
        s3_experiment_path: S3 path of AppWorld experiment. Set to None if it is local.
        exists_ok: If True and s3_experiment_path experiment already exists on local, reuse it.
    """
    if s3_experiment_path:  # Download S3 directory to local if applicable
        assert experiment_name is None
        scheme, _ = fu.get_scheme_and_path(s3_experiment_path)
        assert scheme == "s3"

        # Download the directory
        experiment_name = s3_experiment_path.parts[-1]
        download_dir(
            s3_experiment_path, dest_dir=get_experiment_dir(experiment_name), exists_ok=exists_ok
        )
    elif experiment_name:
        assert experiment_name is not None
        assert s3_experiment_path is None
    else:
        raise ValueError("Please specify an experiment to visualize.")

    episode_iter = EpisodeIterator(experiment_name, sample_id=sample_id)
    visualize_episodes_with_iter(episode_iter=episode_iter)
