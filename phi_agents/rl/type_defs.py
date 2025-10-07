#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import datetime
import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import cattrs
import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

    from phi_agents.rl.llm import TrainableLLM


class SystemMessage:
    """System message."""

    def __init__(self, content: str, today_date: datetime.date | None = None):
        self.role = "system"
        self.content = content
        self.eot = True
        if today_date is None:
            today_date = datetime.date.today()
        self.today_date = today_date

    def asdict(self) -> dict[str, Any]:
        return dict(
            role=self.role, content=self.content, today_date=self.today_date.strftime("%Y-%m-%d")
        )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SystemMessage:
        assert d["role"] == "system"
        return SystemMessage(d["content"], datetime.date.fromisoformat(d["today_date"]))

    def __str__(self) -> str:
        return json.dumps(self.asdict())


class UserMessage:
    """User message."""

    def __init__(self, content: str):
        self.role = "user"
        self.content = content
        self.eot = True

    def asdict(self) -> dict[str, Any]:
        return dict(role=self.role, content=self.content)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> UserMessage:
        assert d["role"] == "user"
        return UserMessage(d["content"])

    def __str__(self) -> str:
        return json.dumps(self.asdict())


class IPythonMessage:
    """IPython message (return from python tool call)."""

    def __init__(self, content: str):
        self.role = "ipython"
        self.content = content
        self.eot = True

    def asdict(self) -> dict[str, Any]:
        return dict(role=self.role, content=self.content)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> IPythonMessage:
        assert d["role"] == "ipython"
        return IPythonMessage(d["content"])

    def __str__(self) -> str:
        return json.dumps(self.asdict())


class AssistantMessage:
    """Assistant message that is not a PolicyMessage.

    Args:
        content: .
        ipython: Does the message start with a <|python_tag|>?
    """

    def __init__(
        self, content: str, *, ipython: bool = False, stopped_by_max_tokens_limit: bool = False
    ):
        self.role = "assistant"
        self.content = content
        self.ipython = ipython
        self.eot = not ipython
        self.stopped_by_max_tokens_limit = stopped_by_max_tokens_limit

    def asdict(self) -> dict[str, Any]:
        return dict(
            role=self.role,
            content=self.content,
            ipython=self.ipython,
            stopped_by_max_tokens_limit=self.stopped_by_max_tokens_limit,
        )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> AssistantMessage:
        assert d["role"] == "assistant"
        return AssistantMessage(
            d["content"],
            ipython=d["ipython"],
            stopped_by_max_tokens_limit=d["stopped_by_max_tokens_limit"],
        )

    def __str__(self) -> str:
        return json.dumps(self.asdict())


class PolicyMessage(AssistantMessage):
    r"""An assistant message produced the policy, including token and probability information.

    Args:
        content: Text content of message not including the prompt_tkoens nor termination tokens.
        prompt_tokens: The non-output tokens at the beginning of the message (mostlyheader)
            e.g. '<|im_start|>assistant\n'.
        stopped_by_max_tokens_limit: The message was terminated because the agent exceeded the max
            output token limit.
        generated_tokens: Generated tokens in this message, not including prompt_tokens nor
            termination tokens.
        generated_token_logprobs: For each output token, what was the log probability of being output?
        ipython: Does the message contain raw python code intended for execution?
        rollout_cancelled: Whether the policy response generation was externally cancelled, e.g. on termination.
    """

    def __init__(
        self,
        content: str,
        *,
        prompt_tokens: list[int],
        stopped_by_max_tokens_limit: bool,
        generated_tokens: list[int],
        generated_token_logprobs: list[float],
        ipython: bool = False,
        rollout_cancelled: bool = False,
    ):
        """Construct an assistant message with token information."""
        super().__init__(
            content, ipython=ipython, stopped_by_max_tokens_limit=stopped_by_max_tokens_limit
        )
        if len(generated_tokens) != len(generated_token_logprobs):
            raise ValueError(
                f"Length of generated_tokens ({len(generated_tokens)}) does not match length of generated_token_logprobs ({len(generated_token_logprobs)})"
            )
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.generated_token_logprobs = generated_token_logprobs
        self.rollout_cancelled = rollout_cancelled

    def asdict(self) -> dict[str, Any]:
        return dict(
            role=self.role,
            content=self.content,
            ipython=self.ipython,
            stopped_by_max_tokens_limit=self.stopped_by_max_tokens_limit,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
            generated_token_logprobs=self.generated_token_logprobs,
        )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PolicyMessage:
        assert d["role"] == "assistant"
        return PolicyMessage(
            d["content"],
            ipython=d["ipython"],
            stopped_by_max_tokens_limit=d["stopped_by_max_tokens_limit"],
            prompt_tokens=d["prompt_tokens"],
            generated_tokens=d["generated_tokens"],
            generated_token_logprobs=d["generated_token_logprobs"],
        )


Message = SystemMessage | UserMessage | IPythonMessage | AssistantMessage | PolicyMessage


def message_from_dict(d: dict[str, Any]) -> Message:
    role = d["role"]
    match role:
        case "system":
            return SystemMessage.from_dict(d)
        case "user":
            return UserMessage.from_dict(d)
        case "ipython":
            return IPythonMessage.from_dict(d)
        case "assistant":
            if "generated_tokens" in d:
                return PolicyMessage.from_dict(d)
            else:
                return AssistantMessage.from_dict(d)
        case _:
            raise ValueError(f"Unknown role: {role}")


WHITE = "\033[37m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"


def print_messages(messages: list[Message]) -> str:
    output = ""
    for msg in messages:
        match msg:
            case SystemMessage():
                output += f"\n\n{RED}=========== system ===========\n"
            case UserMessage():
                output += f"\n\n{BLUE}=========== user ===========\n"
            case PolicyMessage():
                output += f"\n\n{CYAN}=========== assistant (ipython={msg.ipython}) ===========\n"
            case AssistantMessage():
                output += (
                    f"\n\n{MAGENTA}=========== assistant (ipython={msg.ipython}) ===========\n"
                )
            case IPythonMessage():
                output += f"\n\n{GREEN}=========== ipython ===========\n"
        output += msg.content
        output += RESET
    return output


@dataclass
class PolicyTokenInfo:
    tokens: list[int] = field(default_factory=list)
    """All tokens (both prompt and generated tokens) through the last policy message."""

    log_probs: list[float] = field(default_factory=list)
    """Logprobs for self.tokens (NaN for prompt tokens)."""

    is_output: list[bool] = field(default_factory=list)
    """Indicator for whether each token in self.tokens is a generated token."""

    def __post_init__(self) -> None:
        assert len(self.tokens) == len(self.log_probs)
        assert len(self.tokens) == len(self.is_output)

    @staticmethod
    def concatenate(infos: list[PolicyTokenInfo]) -> PolicyTokenInfo:
        return PolicyTokenInfo(
            tokens=list(chain.from_iterable(i.tokens for i in infos)),
            log_probs=list(chain.from_iterable(i.log_probs for i in infos)),
            is_output=list(chain.from_iterable(i.is_output for i in infos)),
        )


def generic_python_constructor(
    loader: yaml.SafeLoader, tag_suffix: str, node: yaml.nodes.MappingNode
) -> Any:
    class_path = tag_suffix
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    # Get the instance data
    value = loader.construct_mapping(node, deep=True)
    return cls.from_dict(value)


class RolloutLoader(yaml.SafeLoader):
    """Rollout loader.

    Because Message class subtypes are not dataclasses we need to register a resolver for them.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Remove the resolver for timestamps so we load it as a str and not a datetime
        self.yaml_implicit_resolvers = {
            k: [(tag, regexp) for tag, regexp in v if not tag == "tag:yaml.org,2002:timestamp"]
            for k, v in self.yaml_implicit_resolvers.items()
        }


# Register python objects
RolloutLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", generic_python_constructor)  # type: ignore


@dataclass
class Rollout:
    messages: list[Message]
    """List of messages."""

    ret: float
    """Return for the rollout"""

    elapsed: float
    """Wall clock time to complete the rollout"""

    cancelled: bool
    """True if the rollout was interrupted/cancelled e.g. by the RL algorithm."""

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Rollout:
        return cls(**yaml.load(yaml_str, Loader=RolloutLoader))


@dataclass
class TrainingRollout(Rollout):
    policy_token_info: PolicyTokenInfo
    """Token-level information about policy input and output needed for training."""


class Scenario(ABC):
    @abstractmethod
    def to_yaml(self) -> str:
        pass

    @property
    def today_date(self) -> datetime.date:
        return datetime.date.today()


class ScenarioRunner(ABC):
    @abstractmethod
    def run(self, scenario: Scenario, llm: TrainableLLM) -> TrainingRollout:
        """Run the scenario.

        Must be thread-safe.

        Args:
            scenario: .
            llm: The LLM to use within the agent.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any extra processes, etc. to ensure graceful termination."""
        pass


def _default_score(
    answer: str, answer_scores: dict[str, float], invalid_answer_reward: float
) -> float:
    return answer_scores.get(answer, invalid_answer_reward)


@dataclass
class SimpleScenario(Scenario):
    query: str
    """Textual description of the problem instance. Note that this is not generally the entire prompt."""

    answer_scores: dict[str, float]
    """The expected answer to the problem instance."""

    invalid_answer_reward: float = -1.0
    """The reward for producing an answer not belonging to the set of scored answers (e.g. incorrect format)."""

    score_func: Callable[[str, dict[str, float], float], float] = _default_score

    def score(self, answer: str) -> float:
        return self.score_func(answer, self.answer_scores, self.invalid_answer_reward)

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Scenario:
        return cast(Scenario, cattrs.structure(yaml.safe_load(yaml_str), cls))
