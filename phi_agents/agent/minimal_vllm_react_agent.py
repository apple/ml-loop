#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import datetime
import re
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum

from jinja2 import Template

from phi_agents.agent.react_template import (
    APP_DESCRIPTIONS,
    PROMPT_TEMPLATE,
)
from phi_agents.appworld.interface import AppWorldInterface, AppWorldTaskEvalResult, Task
from phi_agents.evals.appworld_evals import AppworldAgent, RolloutCancelled, execution_failed
from phi_agents.evals.appworld_evals import no_code_found as no_code_found_fn
from phi_agents.rl.llm import TrainableLLM
from phi_agents.rl.llm.base_llm import messages_str
from phi_agents.rl.type_defs import (
    AssistantMessage,
    IPythonMessage,
    Message,
    PolicyMessage,
    SystemMessage,
    UserMessage,
)
from phi_agents.rl.vllm_client import MaxSeqLenExceeded
from phi_agents.utils.appworld import extract_code_format_output
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


@dataclass(frozen=True)
class TestCodeResult:
    output: str  # AppWorld output
    no_code_found: bool
    result_before: AppWorldTaskEvalResult
    result_after: AppWorldTaskEvalResult


def get_pseudoreward(possibility: tuple[PolicyMessage, TestCodeResult]) -> int:
    res = possibility[1]
    return len(res.result_after.passes) - len(res.result_before.passes)


def is_valid_msg(result: TestCodeResult) -> bool:
    return not execution_failed(result.output) and not result.no_code_found


def choose_msg(
    possibilities: Sequence[tuple[PolicyMessage, TestCodeResult]], use_reward: bool
) -> PolicyMessage:
    """Choose the best generated messages out of a list of candidates.

    Args:
        possibilities: Candidate responses and corresponding test code results.
        use_reward: Whether to use the pseudo-reward from TestCodeResult in determining which
            candidate msg to select.

    Returns:
        Chosen PolicyMessage.
    """
    msgs_without_code_exec_error: Sequence[int] = [
        idx for idx, p in enumerate(possibilities) if is_valid_msg(p[1])
    ]

    if len(msgs_without_code_exec_error) > 0:
        if use_reward:
            # Use rewards and choose the ones with highest reward
            selected_idx = msgs_without_code_exec_error[0]
            max_pseudoreward = get_pseudoreward(possibilities[selected_idx])
            for msg_idx in msgs_without_code_exec_error[1:]:
                cur_possibility = possibilities[msg_idx]
                cur_pseudoreward = get_pseudoreward(cur_possibility)
                if cur_pseudoreward > max_pseudoreward:
                    selected_idx = msg_idx
                    max_pseudoreward = cur_pseudoreward
        else:
            selected_idx = msgs_without_code_exec_error[0]  # Choose the first one (random)
    else:
        selected_idx = 0  # Choose the first one  (random)
    return possibilities[selected_idx][0]


def rewrite_task(llm: TrainableLLM, message: str) -> str:
    """Use an llm to rephrase the message (task instruction)."""
    messages: list[Message] = [
        SystemMessage(
            "Your job is to re-write messages by rewording the original message while maintaining the original meaning. Present your answer *only* with the re-written message. Make sure that all details in the original message are captured in the re-written message, and no included details are made up. Be as concise as the original task message."
        ),
        UserMessage(f"Re-write this message/task: {message}"),
    ]
    rewritten_task = llm.generate(messages).content
    logger.info(
        f"-------------------\nOriginal task: {message}\nRewritten task: {rewritten_task}\n--------"
    )
    return rewritten_task


def get_tokens(msg: str, llm: TrainableLLM) -> list[int]:
    r"""Get (approximately) the tokens for the msg str.

    This is approximate because it removes special tokens but not completely.
    For example, the 'user' str will not be discarded in the following tokens:
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    """
    # Token IDs of special tokens
    s_ids = {s.id for s in llm.special_tokens.values()}

    # Estimate where in the text the token cutoff is
    text_tokens = llm.get_tokens([UserMessage(msg)])[0]
    tokens_no_special = [t for t in text_tokens if t not in s_ids]
    return tokens_no_special


def _look_backward_for_newline_or_whitespace(appworld_output: str, cutoff_idx: int) -> int:
    # Look backwards from estimated token cutoff for newline
    idx = appworld_output.rfind("\n", 0, cutoff_idx)

    if idx == -1:
        logger.info(
            "Agent truncation: Did not find newline character prior to the cutoff for truncation"
        )
        logger.info(appworld_output)

        # Truncate at space (' ') character instead
        idx = appworld_output.rfind(" ", 0, cutoff_idx)
        if idx == -1:
            raise ValueError("Agent truncation: could not find a space character for truncation.")
    return idx


def truncate_appworld_output(
    appworld_output: str,
    max_allowed_appworld_output: int,
    llm: TrainableLLM,
    truncate_at_newline_or_whitespace: bool,
) -> str:
    """Truncate the appworld shell output so that the output is fewer tokens.

    This approximately truncates the str so that the num tokens is just under the
    max_allowed_appworld_output number of tokens. Specifically, it finds the last
    newline char under the token limit and cuts off the text there, appending
    a message about execution output being too long to be shown.

    Args:
        appworld_output: Message from the appworld shell.
        max_allowed_appworld_output: Max allowed str length in num tokens.
        llm: .
        truncate_at_newline_or_whitespace: Whether to truncate at newline or whitespace,

    Returns:
        Truncated version of appworld_output.
    """
    max_allowed_appworld_output -= 30  # Allow some buffer depending on the TRUNCATION_MSG
    text_tokens = get_tokens(appworld_output, llm)
    if len(text_tokens) <= max_allowed_appworld_output:
        raise ValueError("Asked to truncate str but it is already below token count limit.")
    truncated_tokens = text_tokens[:max_allowed_appworld_output]

    truncated_approximate_text = llm.tokenizer.decode(truncated_tokens)
    cutoff_idx = len(truncated_approximate_text)

    if truncate_at_newline_or_whitespace:
        try:
            cutoff_idx = _look_backward_for_newline_or_whitespace(appworld_output, cutoff_idx)
        except ValueError:
            logger.warning("Agent truncation: could not find a space character for truncation.")

    TRUNCATION_MSG = "\n...\nExecution output is too long and is not fully shown."
    return appworld_output[:cutoff_idx] + TRUNCATION_MSG


class SearchMethod(StrEnum):
    NONE = "none"
    SYNTAX = "syntax"
    SYNTAX_REWARD = "syntax_reward"


class TrainMode(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


class MinimalReactAgent(AppworldAgent):
    """A minimal ReAct Agent for AppWorld tasks."""

    def __init__(
        self,
        *,
        task: Task,
        llm: TrainableLLM,
        max_character_prompt_len: int,
        max_character_msg_len: int,
        min_truncated_block_idx: int,
        appworld_style_truncation: bool,
        # AppWorld execution shell truncation (num tokens)
        max_allowed_appworld_output: int | None,
        max_seq_len_tokens: int | None,  # early stop if we exceed this length
        truncate_at_newline_or_whitespace: bool,
        recaption: bool,
        search_method: str,
        mode: str,  # train or eval
    ):
        self.mode = TrainMode(mode)

        # Re-write task instruction
        if recaption and self.mode is TrainMode.TRAIN:
            # Use type ignore because of hydra instantiate lack of typing
            task["instruction"] = rewrite_task(llm, task["instruction"])  # type: ignore[index]

        self.task = task
        self.messages, self.reminder_message = self.prompt_messages_and_reminder(
            task.datetime.date()
        )
        self.llm = llm
        self.n_prompt_messages = len(self.messages)
        self.max_character_prompt_len = max_character_prompt_len
        self.max_character_msg_len = max_character_msg_len
        self.min_truncated_block_idx = min_truncated_block_idx
        self.appworld_style_truncation = appworld_style_truncation
        self.max_allowed_appworld_output = max_allowed_appworld_output
        self.max_seq_len_tokens = max_seq_len_tokens
        self.truncate_at_newline_or_whitespace = truncate_at_newline_or_whitespace
        self.search_method = SearchMethod(search_method)
        self.n_search_retries = 5  # NOTE: Hardcoded

    def prompt_messages_and_reminder(self, date: datetime.date) -> tuple[list[Message], Message]:
        """Build prompt messages for the agent to solve self.task.instruction."""
        # Populate the fields of the prompt template with the task details
        dictionary = {
            "main_user": self.task.supervisor,
            "input_str": self.task.instruction,
            "app_descriptions": APP_DESCRIPTIONS,
            "date": self.task.datetime.isoformat(timespec="seconds"),
        }
        prompt = Template(PROMPT_TEMPLATE.lstrip()).render(dictionary)

        # Extract and return the OpenAI JSON formatted messages from the prompt
        messages: list[Message] = []
        last_start = 0
        for match in re.finditer("(USER|ASSISTANT|SYSTEM):\n", prompt):
            last_end = match.span()[0]
            if len(messages) == 0:
                if last_end != 0:
                    raise ValueError(
                        f"Start of the prompt has no assigned role: {prompt[:last_end]}"
                    )
            else:
                messages[-1].content = prompt[last_start:last_end]
            mesg_type = match.group(1).lower()
            mesg: Message
            if mesg_type == "user":
                mesg = UserMessage("")
            elif mesg_type == "system":
                mesg = SystemMessage("", date)
            elif mesg_type == "assistant":
                mesg = AssistantMessage("")
            else:
                raise AssertionError("Unexpected prompt msg type: {mesg_type}. Prompt:\n{prompt}")
            messages.append(mesg)
            last_start = match.span()[1]
        messages[-1].content = prompt[last_start:]
        # check that there's exactly one system message
        # assert sum([isinstance(mesg, SystemMessage) for mesg in messages]) == 1
        return messages, messages[-1]

    def test_code(self, msg: PolicyMessage, world: AppWorldInterface) -> TestCodeResult:
        # Get init state
        initial_state = world.get_state()
        task_id, experiment_name, executed_code = initial_state

        # Do stuff
        result_before = world.evaluate()
        code = extract_code_format_output(msg.content)
        output = world.execute(code)
        result_after = world.evaluate()

        # Reset state
        world.set_state(task_id, experiment_name, executed_code)

        return TestCodeResult(
            output=output,
            no_code_found=no_code_found_fn(code),
            result_before=result_before,
            result_after=result_after,
        )

    def next_code_block(self, last_execution_output: str | None, world: AppWorldInterface) -> str:
        """Ask Agent to generate next code block given last_execution_output and history.
        The code is available at msg.content where msg is the return value.
        This method can throw MaxSeqLenExceeded if the total number of tokens in messages exceeded the limit.
        """
        # Add the last execution output as the user response to the history
        if last_execution_output is not None:
            if (  # Check whether output is too large and if so, truncate
                self.max_allowed_appworld_output is not None
                and (n_tokens_before_trunc := len(get_tokens(last_execution_output, self.llm)))
                > self.max_allowed_appworld_output
            ):
                last_execution_output = truncate_appworld_output(
                    last_execution_output,
                    self.max_allowed_appworld_output,
                    self.llm,
                    self.truncate_at_newline_or_whitespace,
                )
                n_tokens_after_trunc = len(get_tokens(last_execution_output, self.llm))

                logger.info(
                    f"AppWorld output truncated (limit={self.max_allowed_appworld_output}): "
                    f"{n_tokens_before_trunc} -> {n_tokens_after_trunc} tokens."
                )

                # Allow raising an exception here to let us know whether our truncation is insufficient
                if n_tokens_after_trunc > self.max_allowed_appworld_output:
                    raise ValueError(
                        "Failed to truncate appworld output to be short enough: "
                        f"{n_tokens_after_trunc} (limit {self.max_allowed_appworld_output})."
                    )

            # Add instruction to bottom of appworld output
            last_prompt_user_msg = self.reminder_message
            instruction = last_prompt_user_msg.content.split("solve the actual task:")[-1]
            if not instruction.startswith("\n"):
                instruction = "\n" + instruction
            last_execution_output += f"\nAs a reminder{instruction}"
            # logger.info(instruction)

            # TODO consider using role ipython here
            self.messages.append(UserMessage(last_execution_output))

        if self.appworld_style_truncation:
            messages = self.truncate_messages(self.messages)
        else:
            messages = self.messages

        # Get the next code block based on the history.
        match self.search_method:
            case SearchMethod.SYNTAX | SearchMethod.SYNTAX_REWARD:
                if self.mode is TrainMode.TRAIN:
                    possible_msgs = []
                    for attempt_idx in range(self.n_search_retries):
                        start = time.perf_counter()
                        try:
                            possible_msg = self.llm.generate(messages)
                        except MaxSeqLenExceeded:
                            continue
                        generation_time = time.perf_counter() - start

                        start = time.perf_counter()
                        possible_msg_result = self.test_code(possible_msg, world)
                        possible_msgs.append((possible_msg, possible_msg_result))
                        execution_time = time.perf_counter() - start

                        is_valid = is_valid_msg(possible_msg_result)

                        # Break early if possible to reduce compute and speed things up
                        if self.search_method is SearchMethod.SYNTAX and is_valid:
                            break
                        elif not is_valid and attempt_idx >= self.n_search_retries - 1:
                            logger.info(
                                f"Invalid message:\n {possible_msg.content}, result: {possible_msg_result.output}, attempt took {generation_time:.2f}+{execution_time:.2f} s, {attempt_idx + 1}/{self.n_search_retries}"
                            )

                    if len(possible_msgs) == 0:
                        raise MaxSeqLenExceeded(
                            "Max sequence length exceeded on every generate() call."
                        )
                    msg = choose_msg(
                        possible_msgs, use_reward=self.search_method is SearchMethod.SYNTAX_REWARD
                    )
                else:
                    assert self.mode is TrainMode.EVAL
                    msg = self.llm.generate(messages)
            case SearchMethod.NONE:
                msg = self.llm.generate(messages)
            case _:
                raise ValueError("Unsupported search method.")

        # Add this code block to history as the assistant response
        self.messages.append(msg)

        if msg.rollout_cancelled:
            raise RolloutCancelled()

        return extract_code_format_output(msg.content)

    def truncate_messages(self, history: list[Message]) -> list[Message]:
        # only truncate observations in the task messages, not the prompt
        prompt = history[: self.n_prompt_messages]

        task_messages = deepcopy(history[self.n_prompt_messages :])

        def n_characters(history: list[Message]) -> int:
            return sum([len(m.content) for m in history])

        uses_ipython = any([isinstance(m, IPythonMessage) for m in history])

        def is_observation(message: Message) -> bool:
            if uses_ipython:
                return isinstance(message, IPythonMessage)
            else:
                return isinstance(message, UserMessage)

        # First truncation strategy: replace entire observation blocks early in the conversation
        # with a short message, without removing observations from the most recent n blocks.
        for block_idx, message in enumerate(task_messages):
            if (
                n_characters(task_messages) <= self.max_character_prompt_len
                or block_idx >= len(task_messages) - self.min_truncated_block_idx
            ):
                break
            elif is_observation(message):
                message.content = "[NOT SHOWN FOR BREVITY]"

        # second strategy: find very long observations and truncate them.
        truncation_text = "\n[MESSAGE TOO LONG, TRUNCATED]"
        for message in task_messages:
            if n_characters(task_messages) <= self.max_character_prompt_len:
                break

            if is_observation(message) and len(message.content) > self.max_character_msg_len:
                remaining_msg_len = max(1, self.max_character_msg_len - len(truncation_text))
                message.content = message.content[:remaining_msg_len] + truncation_text

        # last resort: remove entire blocks, even assistant blocks
        while n_characters(task_messages) > self.max_character_prompt_len:
            if len(task_messages) > self.min_truncated_block_idx:
                task_messages = task_messages[1:]
            else:
                break

        if n_characters(task_messages) > self.max_character_prompt_len:
            # we've exhausted our options and are risking to exceed max context length, although
            # this should almost never happen
            logger.info(
                f"We've failed to truncate properly, total len {n_characters(task_messages)=}"
            )
            logger.info(messages_str(task_messages))

        return prompt + task_messages

    def history_as_messages(self, last_execution_output: str) -> list[Message]:
        # TODO consider using role ipython here
        return self.messages + [UserMessage(last_execution_output)]
